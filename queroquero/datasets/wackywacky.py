from __future__ import annotations

import hashlib
import heapq
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote, urlsplit

import zstandard as zstd

from queroquero.config import ConfigError, resolve_dataset_root
from queroquero.datasets.base import (
    CheckpointCallback,
    Document,
    ScanResult,
    clean_text,
    stable_hash,
)


_EXPECTED_COLUMNS = (
    "id",
    "domain_id",
    "parent_page_id",
    "same_as",
    "url",
    "url_md5",
    "url_final",
    "url_final_md5",
    "status_code",
    "title",
    "recursion_level",
    "status",
    "retry_count",
    "text",
    "html",
    "text_md5",
    "html_md5",
    "created_at",
    "updated_at",
)
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_MD5_RE = re.compile(r"[0-9a-fA-F]{32}\Z")
_ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"
_TRUNCATED_ZSTD_FRAME_SIZE_BYTES = 65_535
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()


class _TextDecodeError(ValueError):
    """A bounded source-record defect with a safe aggregate metric."""

    def __init__(self, metric: str) -> None:
        super().__init__("WackyWacky text field was rejected")
        self.metric = metric


@dataclass(frozen=True)
class _NormalizedParagraph:
    lines: Tuple[str, ...]
    text: str
    sha256: str


@dataclass(frozen=True)
class _NormalizedDocument:
    paragraphs: Tuple[_NormalizedParagraph, ...]
    text: str


@dataclass(frozen=True)
class _BoilerplateAnalysis:
    normalized_documents: Dict[str, _NormalizedDocument]
    repeated_cross_domain_paragraphs: set[str]
    repeated_within_domain_blocks: set[Tuple[str, str]]
    paragraph_occurrences_considered: int
    distinct_paragraph_hashes: int
    matching_paragraph_occurrences: int
    block_occurrences_considered: int
    distinct_domain_block_hashes: int
    matching_block_occurrences: int


class WackyWackyAdapter:
    """Stream the 19-column WackyWacky TSV without exposing source values."""

    def scan(
        self,
        config: Dict[str, Any],
        resume_cursor: Optional[Dict[str, Any]] = None,
        resume_documents: Optional[List[Document]] = None,
        checkpoint: Optional[CheckpointCallback] = None,
    ) -> ScanResult:
        source = config["dataset"]["source"]
        filters = config["dataset"]["filters"]
        profile = config["profile"]
        _validate_config(config, source, filters)
        page_filter_rules = _page_filter_rules(filters)
        minimum_line_characters = _line_filter_minimum(filters)
        path = _source_path(config)
        columns = _configured_columns(source)
        header_end = _validate_header(path, columns)
        candidate_limit = _positive_int(
            profile.get("candidate_documents"), "profile.candidate_documents"
        )
        checkpoint_interval = _positive_int(
            source.get("checkpoint_interval_records", 50000),
            "source.checkpoint_interval_records",
        )
        max_field_size = _positive_int(
            source.get("max_field_size_bytes", 128 * 1024 * 1024),
            "source.max_field_size_bytes",
        )
        max_decompressed_text_bytes = _positive_int(
            source.get("max_decompressed_text_bytes"),
            "source.max_decompressed_text_bytes",
        )
        truncated_zstd_frame_size = _truncated_zstd_frame_size(source)
        sample_bytes = _positive_int(
            source.get("fingerprint_sample_bytes", 64 * 1024),
            "source.fingerprint_sample_bytes",
        )
        selection = profile.get("selection")
        if selection not in {"engineering_prefix", "representative"}:
            raise ConfigError("unsupported WackyWacky selection strategy")

        guard = _sampled_source_guard(path, sample_bytes)
        cursor = dict(resume_cursor or {})
        if resume_documents and not resume_cursor:
            raise ConfigError(
                "WackyWacky resume documents require a matching resume cursor"
            )
        saved_guard = cursor.get("source_guard_sha256")
        if saved_guard and saved_guard != guard["resume_guard_sha256"]:
            raise ConfigError("WackyWacky source changed since the saved checkpoint")
        saved_local_stat_guard = cursor.get("_local_source_stat_sha256")

        byte_offset = cursor.get("byte_offset", header_end)
        row_number = cursor.get("row_number", 0)
        if (
            not isinstance(byte_offset, int)
            or isinstance(byte_offset, bool)
            or not header_end <= byte_offset <= path.stat().st_size
            or not isinstance(row_number, int)
            or isinstance(row_number, bool)
            or row_number < 0
        ):
            raise ConfigError("invalid WackyWacky resume cursor")

        metrics = _resume_metrics(cursor.get("metrics"))
        documents = list(resume_documents or [])
        if resume_cursor and metrics["selected_documents"] != len(documents):
            raise ConfigError(
                "WackyWacky checkpoint documents do not match the resume cursor"
            )
        seed = config["preparation"]["seed"]
        if selection == "representative":
            reservoir = _resume_reservoir(documents)
            if len(reservoir) > candidate_limit:
                raise ConfigError("WackyWacky checkpoint exceeds the candidate budget")
        else:
            reservoir = []
            if len(documents) > candidate_limit:
                raise ConfigError("WackyWacky checkpoint exceeds the candidate budget")

        saved_prefix = cursor.get("prefix_sha256")
        source_size = path.stat().st_size
        reuse_complete_scan = (
            selection == "representative"
            and cursor.get("complete") is True
            and byte_offset == source_size
            and isinstance(saved_prefix, str)
            and re.fullmatch(r"[0-9a-f]{64}", saved_prefix) is not None
            and saved_local_stat_guard == guard["local_source_stat_sha256"]
        )
        if reuse_complete_scan:
            # The full hash was already computed and the local stat plus
            # sparse-content guards were revalidated above. A post-scan review
            # decision therefore does not trigger another 56 GB read.
            prefix_sha256 = saved_prefix
            final_offset = byte_offset
        else:
            digest = hashlib.sha256()
            with path.open("rb") as raw:
                _hash_prefix(raw, byte_offset, digest)
                if saved_prefix and saved_prefix != digest.hexdigest():
                    raise ConfigError(
                        "WackyWacky source prefix changed since the saved checkpoint"
                    )

                while True:
                    if (
                        selection == "engineering_prefix"
                        and len(documents) >= candidate_limit
                    ):
                        break
                    raw_record = raw.readline()
                    if not raw_record:
                        break
                    digest.update(raw_record)
                    row_number += 1
                    metrics["rows_seen"] += 1
                    row = _split_tsv_record(raw_record, max_field_size, row_number)
                    record = dict(zip(columns, row))
                    document = _eligible_document(
                        record,
                        row_number=row_number,
                        byte_offset=raw.tell(),
                        filters=filters,
                        page_filter_rules=page_filter_rules,
                        seed=seed,
                        metrics=metrics,
                        maximum_text_bytes=max_decompressed_text_bytes,
                        truncated_zstd_frame_size=truncated_zstd_frame_size,
                    )
                    if document is not None:
                        metrics["eligible_records"] += 1
                        if selection == "engineering_prefix":
                            if len(documents) < candidate_limit:
                                documents.append(document)
                        else:
                            _reservoir_add(reservoir, document, candidate_limit)

                    metrics["selected_documents"] = (
                        len(reservoir)
                        if selection == "representative"
                        else len(documents)
                    )
                    if checkpoint and row_number % checkpoint_interval == 0:
                        current_documents = (
                            _ordered_reservoir(reservoir)
                            if selection == "representative"
                            else list(documents)
                        )
                        checkpoint(
                            _with_local_stat_guard(
                                _cursor(
                                    raw.tell(),
                                    row_number,
                                    guard["resume_guard_sha256"],
                                    digest.hexdigest(),
                                    metrics,
                                    complete=False,
                                    finalization_blocked=False,
                                ),
                                guard["local_source_stat_sha256"],
                            ),
                            current_documents,
                        )

                final_offset = raw.tell()
                prefix_sha256 = digest.hexdigest()

        if selection == "representative":
            documents = _ordered_reservoir(reservoir)

        extra_reports: Dict[str, Dict[str, Any]] = {}
        finalization_blocked = False
        final_boilerplate_report: Optional[Dict[str, Any]] = None
        profile_name = str(config.get("profile_name"))
        rules = _boilerplate_rules(filters, profile_name)
        saved_report = cursor.get("boilerplate_report")
        saved_line_filter = (
            _restore_line_filter_report(
                saved_report.get("line_filter"), minimum_line_characters
            )
            if cursor.get("complete") is True and isinstance(saved_report, dict)
            else None
        )
        if saved_line_filter is None:
            documents, line_filter_report = _filter_short_lines(
                documents,
                minimum_line_characters,
                rules,
            )
            metrics["short_lines_considered"] = line_filter_report[
                "lines_considered"
            ]
            metrics["short_lines_removed"] = line_filter_report["lines_removed"]
            metrics["short_line_characters_removed"] = line_filter_report[
                "removed_characters"
            ]
            metrics["documents_affected_by_short_line_filter"] = (
                line_filter_report["affected_documents"]
            )
            metrics["documents_discarded_by_short_line_filter"] = (
                line_filter_report["documents_discarded_total"]
            )
        else:
            line_filter_report = saved_line_filter
        reuse_saved_report = (
            cursor.get("complete") is True
            and isinstance(saved_report, dict)
            and saved_report.get("decision") == rules["decision"]
        )
        if reuse_saved_report:
            report = _restore_boilerplate_report(
                saved_report,
                rules["decision"],
                profile_name,
            )
        else:
            if (
                isinstance(saved_report, dict)
                and saved_report.get("decision") == "remove_exact"
            ):
                raise ConfigError(
                    "cannot change the boilerplate decision after exact removal"
                )
            analysis = _analyze_boilerplate(documents, rules)
            simulated_documents, simulation = _remove_exact_boilerplate(
                documents,
                analysis,
                rules,
            )
            if rules["decision"] == "remove_exact":
                documents = simulated_documents
                applied = dict(simulation)
            else:
                applied = _no_removal_metrics(
                    len(documents), simulation["original_characters"]
                )
            report = _boilerplate_report(
                documents_before=len(analysis.normalized_documents),
                profile_name=profile_name,
                rules=rules,
                analysis=analysis,
                simulation=simulation,
                applied=applied,
                line_filter=line_filter_report,
            )
        finalization_blocked = bool(report["finalization_blocked"])
        final_boilerplate_report = report
        extra_reports["boilerplate_report"] = report

        metrics["selected_documents"] = len(documents)
        complete_source_scan = final_offset == path.stat().st_size
        if selection == "representative" and not complete_source_scan:
            raise RuntimeError("representative WackyWacky selection requires a full pass")
        fingerprint = _source_fingerprint(
            guard,
            prefix_sha256,
            final_offset,
            complete_source_scan,
        )
        final_cursor = _cursor(
            final_offset,
            row_number,
            guard["resume_guard_sha256"],
            prefix_sha256,
            metrics,
            complete=True,
            finalization_blocked=finalization_blocked,
            boilerplate_report=final_boilerplate_report,
        )
        local_resume_cursor = _with_local_stat_guard(
            final_cursor, guard["local_source_stat_sha256"]
        )
        if checkpoint:
            checkpoint(local_resume_cursor, list(documents))

        return ScanResult(
            documents=documents,
            source_fingerprint=fingerprint,
            metrics=metrics,
            cursor=final_cursor,
            extra_reports=extra_reports,
            resume_cursor=local_resume_cursor,
        )


def _source_path(config: Dict[str, Any]) -> Path:
    root = resolve_dataset_root(config)
    value = config["dataset"]["source"].get("path")
    if not isinstance(value, str) or not value:
        raise ConfigError("WackyWacky source.path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ConfigError("WackyWacky source.path must be relative to the dataset root")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ConfigError("WackyWacky source.path escapes the dataset root")
    if not path.is_file():
        raise ConfigError("WackyWacky source file was not found")
    return path


def _validate_config(
    config: Dict[str, Any],
    source: Dict[str, Any],
    filters: Dict[str, Any],
) -> None:
    if source.get("format") != "tsv":
        raise ConfigError("WackyWacky source.format must be 'tsv'")
    if source.get("encoding") != "utf-8":
        raise ConfigError("WackyWacky source.encoding must be 'utf-8'")
    if source.get("text_encoding") != "hex-zstd-utf8":
        raise ConfigError(
            "WackyWacky source.text_encoding must be 'hex-zstd-utf8'"
        )
    _positive_int(
        source.get("max_decompressed_text_bytes"),
        "source.max_decompressed_text_bytes",
    )
    _truncated_zstd_frame_size(source)
    _text_decode_error_policy(source)
    if filters.get("status") != "done":
        raise ConfigError("WackyWacky filters.status must be 'done'")
    for name in ("require_text", "require_text_md5", "exclude_same_as"):
        if filters.get(name) is not True:
            raise ConfigError(f"WackyWacky filters.{name} must be true")
    if filters.get("text_md5_policy") != "count_mismatch":
        raise ConfigError(
            "WackyWacky filters.text_md5_policy must be 'count_mismatch'"
        )
    if filters.get("same_as_null_values") != ["", "NULL"]:
        raise ConfigError(
            "WackyWacky filters.same_as_null_values must be ['', 'NULL']"
        )
    _page_filter_rules(filters)
    _line_filter_minimum(filters)
    profile_name = config.get("profile_name")
    selection = config["profile"].get("selection")
    expected_selection = {
        "smoke": "engineering_prefix",
        "mvp": "representative",
        "real": "representative",
    }.get(profile_name)
    if expected_selection is None:
        raise ConfigError("WackyWacky profile_name must be smoke, mvp, or real")
    if selection != expected_selection:
        raise ConfigError(
            f"WackyWacky {profile_name} selection must be {expected_selection}"
        )
    seed = config["preparation"].get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("preparation.seed must be a non-negative integer")
    _boilerplate_rules(filters, profile_name)


def _configured_columns(source: Dict[str, Any]) -> Tuple[str, ...]:
    value = source.get("columns", list(_EXPECTED_COLUMNS))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError("WackyWacky source.columns must be a list of strings")
    columns = tuple(value)
    if columns != _EXPECTED_COLUMNS:
        raise ConfigError("WackyWacky requires the documented 19-column schema")
    return columns


def _validate_header(path: Path, columns: Sequence[str]) -> int:
    with path.open("rb") as raw:
        value = raw.readline()
        if not value:
            raise ValueError("WackyWacky TSV header is missing or invalid")
        try:
            header = _strip_record_ending(value).decode("utf-8", errors="strict").split("\t")
        except UnicodeDecodeError as exc:
            raise ValueError("WackyWacky TSV header is not strict UTF-8") from exc
        if header:
            header[0] = header[0].lstrip("\ufeff")
        if tuple(header) != tuple(columns):
            raise ValueError("WackyWacky TSV header does not match the 19-column schema")
        return raw.tell()


def _split_tsv_record(
    value: bytes,
    maximum_field_bytes: int,
    row_number: int,
) -> List[str]:
    raw_fields = _strip_record_ending(value).split(b"\t")
    if len(raw_fields) != len(_EXPECTED_COLUMNS):
        raise ValueError(f"WackyWacky row {row_number} does not have 19 columns")
    if any(len(field) > maximum_field_bytes for field in raw_fields):
        raise ValueError(f"WackyWacky row {row_number} exceeds the field-size limit")
    try:
        return [field.decode("utf-8", errors="strict") for field in raw_fields]
    except UnicodeDecodeError as exc:
        raise ValueError(f"WackyWacky row {row_number} is not strict UTF-8") from exc


def _strip_record_ending(value: bytes) -> bytes:
    if value.endswith(b"\n"):
        value = value[:-1]
    if value.endswith(b"\r"):
        value = value[:-1]
    return value


def _eligible_document(
    record: Dict[str, str],
    row_number: int,
    byte_offset: int,
    filters: Dict[str, Any],
    page_filter_rules: Dict[str, Tuple[str, ...]],
    seed: int,
    metrics: Dict[str, int],
    maximum_text_bytes: int,
    truncated_zstd_frame_size: int,
) -> Optional[Document]:
    if record["status"].strip() != filters.get("status", "done"):
        metrics["filtered_status"] += 1
        return None
    text = record["text"]
    if filters.get("require_text", True) and not text.strip():
        metrics["filtered_missing_text"] += 1
        return None
    text_md5 = record["text_md5"].strip()
    if filters.get("require_text_md5", True) and not text_md5:
        metrics["filtered_missing_text_md5"] += 1
        return None
    same_as = record["same_as"].strip()
    same_as_null_values = filters.get("same_as_null_values", [""])
    if filters.get("exclude_same_as", True) and same_as not in same_as_null_values:
        metrics["filtered_same_as"] += 1
        return None
    page_rejection_reason = _page_rejection_reason(record, page_filter_rules)
    if page_rejection_reason is not None:
        metrics[f"filtered_page_{page_rejection_reason}"] += 1
        return None

    try:
        text, text_md5_matches = _decode_text(
            text,
            text_md5,
            maximum_bytes=maximum_text_bytes,
            truncated_zstd_frame_size=truncated_zstd_frame_size,
        )
    except _TextDecodeError as exc:
        metrics[exc.metric] += 1
        return None
    if not text_md5_matches:
        metrics["text_md5_mismatches"] += 1

    record_key = stable_hash(
        "wackywacky",
        record["id"],
        record["url_md5"],
        text_md5,
        row_number,
    )
    score = stable_hash(seed, "wackywacky-reservoir", record_key)
    domain_value = record["domain_id"].strip()
    domain_ref = stable_hash(
        "wackywacky-domain",
        domain_value if domain_value else record_key,
    )
    return Document(
        text=text,
        source_ref=f"wackywacky:{record_key}",
        source_position={
            "row_number": row_number,
            "byte_offset_after_record": byte_offset,
        },
        metadata={
            "domain_ref_sha256": domain_ref,
            "text_ref_sha256": stable_hash("wackywacky-text-md5", text_md5),
            "selection_score": score,
        },
    )


def _page_filter_rules(filters: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    value = filters.get("page_filter")
    if not isinstance(value, dict):
        raise ConfigError("filters.page_filter must be an object")
    expected = {
        "search_title_markers",
        "search_query_parameters",
        "search_query_value_markers",
        "search_path_segments",
        "listing_path_segments",
    }
    if set(value) != expected:
        raise ConfigError(
            "filters.page_filter must define only the documented page rules"
        )
    result: Dict[str, Tuple[str, ...]] = {}
    for name in sorted(expected):
        items = value.get(name)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)
        ):
            raise ConfigError(f"filters.page_filter.{name} must be a string list")
        normalized = tuple(item.strip().casefold() for item in items)
        if len(set(normalized)) != len(normalized):
            raise ConfigError(f"filters.page_filter.{name} contains duplicates")
        result[name] = normalized
    return result


def _line_filter_minimum(filters: Dict[str, Any]) -> int:
    value = filters.get("line_filter")
    if not isinstance(value, dict) or set(value) != {"minimum_characters"}:
        raise ConfigError(
            "filters.line_filter must define only minimum_characters"
        )
    return _positive_int(
        value.get("minimum_characters"),
        "filters.line_filter.minimum_characters",
    )


def _page_rejection_reason(
    record: Dict[str, str], rules: Dict[str, Tuple[str, ...]]
) -> Optional[str]:
    title = " ".join(clean_text(record["title"], strip_html=True).casefold().split())
    if any(
        _title_starts_with_marker(title, marker)
        for marker in rules["search_title_markers"]
    ):
        return "search"

    for raw_url in (record["url_final"], record["url"]):
        if not raw_url.strip():
            continue
        try:
            parsed = urlsplit(raw_url.strip())
            path_segments = tuple(
                unquote(segment).strip().casefold()
                for segment in parsed.path.split("/")
                if segment.strip()
            )
            query = tuple(
                (unquote(key).strip().casefold(), unquote(item).strip().casefold())
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            )
        except (UnicodeError, ValueError):
            continue
        if any(
            segment in rules["listing_path_segments"] for segment in path_segments
        ):
            return "listing"
        if any(
            segment in rules["search_path_segments"] for segment in path_segments
        ):
            return "search"
        if any(
            key in rules["search_query_parameters"] for key, _ in query
        ):
            return "search"
        if any(
            marker in item
            for _, item in query
            for marker in rules["search_query_value_markers"]
        ):
            return "search"
    return None


def _title_starts_with_marker(title: str, marker: str) -> bool:
    if title == marker:
        return True
    return any(
        title.startswith(f"{marker}{separator}")
        for separator in (" ", ":", "-", "–", "—", "|")
    )


def _decode_text(
    encoded: str,
    text_md5: str,
    maximum_bytes: int,
    truncated_zstd_frame_size: int,
) -> tuple[str, bool]:
    if not _MD5_RE.fullmatch(text_md5):
        raise _TextDecodeError("filtered_invalid_text_md5")
    try:
        compressed = bytes.fromhex(encoded)
    except ValueError:
        raise _TextDecodeError("filtered_invalid_text_hex") from None
    if not compressed:
        raise _TextDecodeError("filtered_empty_compressed_texts")
    try:
        declared_size = zstd.frame_content_size(compressed)
    except zstd.ZstdError:
        raise _TextDecodeError("filtered_invalid_zstd_frames") from None
    if declared_size == zstd.CONTENTSIZE_ERROR:
        raise _TextDecodeError("filtered_invalid_zstd_frames")
    if (
        declared_size != zstd.CONTENTSIZE_UNKNOWN
        and declared_size > maximum_bytes
    ):
        raise _TextDecodeError("filtered_oversized_decompressed_texts")
    try:
        decoded_bytes = _ZSTD_DECOMPRESSOR.decompress(
            compressed,
            max_output_size=maximum_bytes,
        )
    except zstd.ZstdError:
        if (
            len(compressed) == truncated_zstd_frame_size
            and compressed.startswith(_ZSTD_FRAME_MAGIC)
        ):
            raise _TextDecodeError(
                "filtered_truncated_zstd_frames_65535_bytes"
            ) from None
        if compressed.startswith(_ZSTD_FRAME_MAGIC):
            raise _TextDecodeError("filtered_corrupt_zstd_frames") from None
        raise _TextDecodeError("filtered_invalid_zstd_frames") from None
    if len(decoded_bytes) > maximum_bytes:
        raise _TextDecodeError("filtered_oversized_decompressed_texts")
    text_md5_matches = (
        hashlib.md5(decoded_bytes, usedforsecurity=False).hexdigest()
        == text_md5.lower()
    )
    try:
        decoded = decoded_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _TextDecodeError("filtered_non_utf8_decompressed_texts") from None
    return decoded, text_md5_matches


def _resume_reservoir(
    documents: Sequence[Document],
) -> List[Tuple[int, int, Document]]:
    heap: List[Tuple[int, int, Document]] = []
    row_numbers: set[int] = set()
    for document in documents:
        score = document.metadata.get("selection_score")
        row_number = document.source_position.get("row_number")
        if (
            not isinstance(score, str)
            or len(score) != 64
            or not isinstance(row_number, int)
        ):
            raise ConfigError("invalid WackyWacky reservoir checkpoint")
        if row_number in row_numbers:
            raise ConfigError("invalid WackyWacky reservoir checkpoint")
        row_numbers.add(row_number)
        try:
            score_value = int(score, 16)
        except ValueError as exc:
            raise ConfigError("invalid WackyWacky reservoir checkpoint") from exc
        heapq.heappush(heap, (-score_value, -row_number, document))
    return heap


def _reservoir_add(
    reservoir: List[Tuple[int, int, Document]],
    document: Document,
    limit: int,
) -> None:
    score_value = int(document.metadata["selection_score"], 16)
    row_number = int(document.source_position["row_number"])
    entry = (-score_value, -row_number, document)
    if len(reservoir) < limit:
        heapq.heappush(reservoir, entry)
    elif entry[:2] > reservoir[0][:2]:
        heapq.heapreplace(reservoir, entry)


def _ordered_reservoir(
    reservoir: Sequence[Tuple[int, int, Document]],
) -> List[Document]:
    return sorted(
        (entry[2] for entry in reservoir),
        key=lambda document: (
            document.metadata["selection_score"],
            document.source_position["row_number"],
        ),
    )


def _boilerplate_rules(
    filters: Dict[str, Any], profile_name: str
) -> Dict[str, Any]:
    value = filters.get("boilerplate")
    if not isinstance(value, dict):
        raise ConfigError("filters.boilerplate must be an object")
    if value.get("schema_version") != 4:
        raise ConfigError("boilerplate.schema_version must be 4")
    decisions = value.get("decision_by_profile")
    if not isinstance(decisions, dict) or set(decisions) != {"smoke", "mvp"}:
        raise ConfigError(
            "boilerplate.decision_by_profile must define smoke and mvp"
        )
    if decisions.get("smoke") != "remove_exact":
        raise ConfigError("boilerplate smoke decision must be remove_exact")
    if decisions.get("mvp") not in {"pending", "keep", "remove_exact"}:
        raise ConfigError(
            "boilerplate mvp decision must be pending, keep, or remove_exact"
        )
    decision_profile = "mvp" if profile_name == "real" else profile_name
    if decision_profile not in decisions:
        raise ConfigError("boilerplate profile must be smoke, mvp, or real")

    cross_domain = value.get("cross_domain_paragraphs")
    within_domain = value.get("within_domain_blocks")
    document_filter = value.get("document_filter")
    if not isinstance(cross_domain, dict):
        raise ConfigError("boilerplate.cross_domain_paragraphs must be an object")
    if not isinstance(within_domain, dict):
        raise ConfigError("boilerplate.within_domain_blocks must be an object")
    if not isinstance(document_filter, dict):
        raise ConfigError("boilerplate.document_filter must be an object")

    paragraph_minimum_characters = _positive_int(
        cross_domain.get("minimum_characters"),
        "boilerplate.cross_domain_paragraphs.minimum_characters",
    )
    paragraph_minimum_documents = _positive_int(
        cross_domain.get("minimum_documents"),
        "boilerplate.cross_domain_paragraphs.minimum_documents",
    )
    paragraph_minimum_domains = _positive_int(
        cross_domain.get("minimum_domains"),
        "boilerplate.cross_domain_paragraphs.minimum_domains",
    )
    if (
        paragraph_minimum_characters < 80
        or paragraph_minimum_documents < 5
        or paragraph_minimum_domains < 3
    ):
        raise ConfigError(
            "cross-domain boilerplate thresholds cannot be below 80 chars, "
            "5 docs, and 3 domains"
        )

    lines_per_block = _positive_int(
        within_domain.get("lines_per_block"),
        "boilerplate.within_domain_blocks.lines_per_block",
    )
    block_minimum_characters = _positive_int(
        within_domain.get("minimum_characters"),
        "boilerplate.within_domain_blocks.minimum_characters",
    )
    block_minimum_documents = _positive_int(
        within_domain.get("minimum_documents"),
        "boilerplate.within_domain_blocks.minimum_documents",
    )
    if lines_per_block != 3:
        raise ConfigError("within-domain boilerplate blocks must use exactly 3 lines")
    if block_minimum_characters < 60 or block_minimum_documents < 5:
        raise ConfigError(
            "within-domain boilerplate thresholds cannot be below 60 chars and 5 docs"
        )

    minimum_remaining_characters = _positive_int(
        document_filter.get("minimum_remaining_characters"),
        "boilerplate.document_filter.minimum_remaining_characters",
    )
    maximum_removed_fraction = document_filter.get("maximum_removed_fraction")
    if (
        not isinstance(maximum_removed_fraction, (int, float))
        or isinstance(maximum_removed_fraction, bool)
        or not 0 < float(maximum_removed_fraction) <= 0.8
    ):
        raise ConfigError(
            "boilerplate.document_filter.maximum_removed_fraction must be "
            "greater than 0 and at most 0.8"
        )
    if minimum_remaining_characters < 300:
        raise ConfigError(
            "boilerplate.document_filter.minimum_remaining_characters cannot "
            "be below 300"
        )
    return {
        "decision": decisions[decision_profile],
        "paragraph_minimum_characters": paragraph_minimum_characters,
        "paragraph_minimum_documents": paragraph_minimum_documents,
        "paragraph_minimum_domains": paragraph_minimum_domains,
        "lines_per_block": lines_per_block,
        "block_minimum_characters": block_minimum_characters,
        "block_minimum_documents": block_minimum_documents,
        "minimum_remaining_characters": minimum_remaining_characters,
        "maximum_removed_fraction": float(maximum_removed_fraction),
    }


def _normalize_document(text: str) -> _NormalizedDocument:
    cleaned = clean_text(text, strip_html=True)
    paragraphs: List[_NormalizedParagraph] = []
    for raw_paragraph in _PARAGRAPH_BREAK_RE.split(cleaned):
        lines = tuple(
            normalized
            for raw_line in raw_paragraph.splitlines()
            if (normalized := " ".join(raw_line.split()))
        )
        if not lines:
            continue
        paragraph_text = " ".join(lines)
        paragraphs.append(
            _NormalizedParagraph(
                lines=lines,
                text=paragraph_text,
                sha256=stable_hash(
                    "wackywacky-cross-domain-paragraph-v2",
                    paragraph_text,
                ),
            )
        )
    normalized = tuple(paragraphs)
    return _NormalizedDocument(
        paragraphs=normalized,
        text="\n\n".join("\n".join(paragraph.lines) for paragraph in normalized),
    )


def _filter_short_lines(
    documents: Sequence[Document],
    minimum_characters: int,
    rules: Dict[str, Any],
) -> Tuple[List[Document], Dict[str, Any]]:
    result: List[Document] = []
    lines_considered = 0
    lines_removed = 0
    affected_documents = 0
    original_characters = 0
    removed_characters = 0
    discarded_short = 0
    discarded_fraction = 0
    discarded_total = 0

    for document in documents:
        normalized = _normalize_document(document.text)
        kept_paragraphs: List[str] = []
        document_lines_removed = 0
        for paragraph in normalized.paragraphs:
            lines_considered += len(paragraph.lines)
            kept_lines = [
                line
                for line in paragraph.lines
                if len(line) >= minimum_characters
            ]
            document_lines_removed += len(paragraph.lines) - len(kept_lines)
            if kept_lines:
                kept_paragraphs.append("\n".join(kept_lines))

        remaining_text = "\n\n".join(kept_paragraphs)
        before = len(normalized.text)
        after = len(remaining_text)
        removed = max(0, before - after)
        original_characters += before
        removed_characters += removed
        lines_removed += document_lines_removed
        if document_lines_removed:
            affected_documents += 1
        removed_fraction = removed / before if before else 0.0
        too_short = (
            not remaining_text
            or (
                document_lines_removed > 0
                and after < rules["minimum_remaining_characters"]
            )
        )
        too_much_removed = (
            document_lines_removed > 0
            and removed_fraction > rules["maximum_removed_fraction"]
        )
        if too_short:
            discarded_short += 1
        if too_much_removed:
            discarded_fraction += 1
        if too_short or too_much_removed:
            discarded_total += 1
            continue
        result.append(replace(document, text=remaining_text))

    return result, {
        "minimum_characters": minimum_characters,
        "input_documents": len(documents),
        "lines_considered": lines_considered,
        "lines_removed": lines_removed,
        "affected_documents": affected_documents,
        "original_characters": original_characters,
        "removed_characters": removed_characters,
        "removed_fraction": (
            removed_characters / original_characters if original_characters else 0.0
        ),
        "documents_discarded_minimum_remaining_characters": discarded_short,
        "documents_discarded_maximum_removed_fraction": discarded_fraction,
        "documents_discarded_total": discarded_total,
        "documents_remaining": len(result),
    }


def _restore_line_filter_report(
    value: Any, minimum_characters: int
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    required_counts = (
        "input_documents",
        "lines_considered",
        "lines_removed",
        "affected_documents",
        "original_characters",
        "removed_characters",
        "documents_discarded_minimum_remaining_characters",
        "documents_discarded_maximum_removed_fraction",
        "documents_discarded_total",
        "documents_remaining",
    )
    if (
        not isinstance(value, dict)
        or value.get("minimum_characters") != minimum_characters
        or any(
            not isinstance(value.get(name), int)
            or isinstance(value.get(name), bool)
            or value[name] < 0
            for name in required_counts
        )
        or not isinstance(value.get("removed_fraction"), (int, float))
        or isinstance(value.get("removed_fraction"), bool)
    ):
        raise ConfigError("invalid WackyWacky line-filter checkpoint")
    return dict(value)


def _document_blocks(
    document: _NormalizedDocument,
    lines_per_block: int,
    minimum_characters: int,
) -> List[Tuple[Tuple[Tuple[int, int], ...], str]]:
    ordered_lines = _ordered_document_lines(document)
    result: List[Tuple[Tuple[Tuple[int, int], ...], str]] = []
    for start in range(max(0, len(ordered_lines) - lines_per_block + 1)):
        window = ordered_lines[start : start + lines_per_block]
        lines = tuple(item[2] for item in window)
        if sum(len(line) for line in lines) < minimum_characters:
            continue
        result.append(_block_record(window))
    return result


def _ordered_document_lines(
    document: _NormalizedDocument,
) -> Tuple[Tuple[int, int, str], ...]:
    return tuple(
        (paragraph_index, line_index, line)
        for paragraph_index, paragraph in enumerate(document.paragraphs)
        for line_index, line in enumerate(paragraph.lines)
    )


def _block_record(
    window: Sequence[Tuple[int, int, str]],
) -> Tuple[Tuple[Tuple[int, int], ...], str]:
    return (
        tuple((item[0], item[1]) for item in window),
        stable_hash(
            "wackywacky-within-domain-block-v2",
            "\n".join(item[2] for item in window),
        ),
    )


def _analyze_boilerplate(
    documents: Sequence[Document],
    rules: Dict[str, Any],
) -> _BoilerplateAnalysis:
    normalized_documents: Dict[str, _NormalizedDocument] = {}
    paragraph_documents: Dict[str, set[str]] = {}
    paragraph_domains: Dict[str, set[str]] = {}
    paragraph_occurrences: Dict[str, int] = {}
    block_documents: Dict[Tuple[str, str], set[str]] = {}
    block_occurrences: Dict[Tuple[str, str], int] = {}

    paragraph_occurrences_considered = 0
    block_occurrences_considered = 0
    for document in documents:
        if document.source_ref in normalized_documents:
            raise ConfigError("WackyWacky candidate source references must be unique")
        normalized = _normalize_document(document.text)
        normalized_documents[document.source_ref] = normalized
        domain_ref = str(document.metadata["domain_ref_sha256"])
        paragraph_hashes: set[str] = set()
        block_keys: set[Tuple[str, str]] = set()
        for paragraph in normalized.paragraphs:
            if len(paragraph.text) >= rules["paragraph_minimum_characters"]:
                paragraph_occurrences_considered += 1
                paragraph_occurrences[paragraph.sha256] = (
                    paragraph_occurrences.get(paragraph.sha256, 0) + 1
                )
                paragraph_hashes.add(paragraph.sha256)
        for _, block_hash in _document_blocks(
            normalized,
            rules["lines_per_block"],
            rules["block_minimum_characters"],
        ):
            block_occurrences_considered += 1
            key = (domain_ref, block_hash)
            block_occurrences[key] = block_occurrences.get(key, 0) + 1
            block_keys.add(key)
        for paragraph_hash in paragraph_hashes:
            paragraph_documents.setdefault(paragraph_hash, set()).add(
                document.source_ref
            )
            paragraph_domains.setdefault(paragraph_hash, set()).add(domain_ref)
        for block_key in block_keys:
            block_documents.setdefault(block_key, set()).add(document.source_ref)

    repeated_paragraphs = {
        paragraph_hash
        for paragraph_hash, document_refs in paragraph_documents.items()
        if len(document_refs) >= rules["paragraph_minimum_documents"]
        and len(paragraph_domains[paragraph_hash])
        >= rules["paragraph_minimum_domains"]
    }
    repeated_blocks = {
        key
        for key, document_refs in block_documents.items()
        if len(document_refs) >= rules["block_minimum_documents"]
    }
    return _BoilerplateAnalysis(
        normalized_documents=normalized_documents,
        repeated_cross_domain_paragraphs=repeated_paragraphs,
        repeated_within_domain_blocks=repeated_blocks,
        paragraph_occurrences_considered=paragraph_occurrences_considered,
        distinct_paragraph_hashes=len(paragraph_documents),
        matching_paragraph_occurrences=sum(
            paragraph_occurrences[value] for value in repeated_paragraphs
        ),
        block_occurrences_considered=block_occurrences_considered,
        distinct_domain_block_hashes=len(block_documents),
        matching_block_occurrences=sum(
            block_occurrences[value] for value in repeated_blocks
        ),
    )


def _remove_exact_boilerplate(
    documents: Sequence[Document],
    analysis: _BoilerplateAnalysis,
    rules: Dict[str, Any],
) -> Tuple[List[Document], Dict[str, Any]]:
    result: List[Document] = []
    affected_documents = 0
    removed_characters = 0
    original_characters = 0
    discarded_short = 0
    discarded_fraction = 0
    discarded_total = 0

    for document in documents:
        normalized = analysis.normalized_documents[document.source_ref]
        domain_ref = str(document.metadata["domain_ref_sha256"])
        removed_line_positions: set[Tuple[int, int]] = set()
        for positions, block_hash in _document_blocks(
            normalized,
            rules["lines_per_block"],
            rules["block_minimum_characters"],
        ):
            if (
                domain_ref,
                block_hash,
            ) in analysis.repeated_within_domain_blocks:
                removed_line_positions.update(positions)
        kept_paragraphs: List[str] = []
        for paragraph_index, paragraph in enumerate(normalized.paragraphs):
            if paragraph.sha256 in analysis.repeated_cross_domain_paragraphs:
                continue
            kept_lines = [
                line
                for index, line in enumerate(paragraph.lines)
                if (paragraph_index, index) not in removed_line_positions
            ]
            if kept_lines:
                kept_paragraphs.append("\n".join(kept_lines))

        remaining_text = "\n\n".join(kept_paragraphs)
        before = len(normalized.text)
        after = len(remaining_text)
        removed = max(0, before - after)
        original_characters += before
        removed_characters += removed
        if removed:
            affected_documents += 1
        removed_fraction = removed / before if before else 0.0
        too_short = removed > 0 and after < rules["minimum_remaining_characters"]
        too_much_removed = (
            removed > 0
            and removed_fraction > rules["maximum_removed_fraction"]
        )
        if too_short:
            discarded_short += 1
        if too_much_removed:
            discarded_fraction += 1
        if too_short or too_much_removed:
            discarded_total += 1
            continue
        result.append(replace(document, text=remaining_text))

    return result, {
        "original_characters": original_characters,
        "affected_documents": affected_documents,
        "removed_characters": removed_characters,
        "removed_fraction": (
            removed_characters / original_characters if original_characters else 0.0
        ),
        "documents_discarded_minimum_remaining_characters": discarded_short,
        "documents_discarded_maximum_removed_fraction": discarded_fraction,
        "documents_discarded_total": discarded_total,
        "documents_remaining": len(result),
    }


def _no_removal_metrics(
    candidate_documents: int, original_characters: int
) -> Dict[str, Any]:
    return {
        "original_characters": original_characters,
        "affected_documents": 0,
        "removed_characters": 0,
        "removed_fraction": 0.0,
        "documents_discarded_minimum_remaining_characters": 0,
        "documents_discarded_maximum_removed_fraction": 0,
        "documents_discarded_total": 0,
        "documents_remaining": candidate_documents,
    }


def _boilerplate_report(
    documents_before: int,
    profile_name: str,
    rules: Dict[str, Any],
    analysis: _BoilerplateAnalysis,
    simulation: Dict[str, Any],
    applied: Dict[str, Any],
    line_filter: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "queroquero-boilerplate-report/v2",
        "profile": profile_name,
        "decision": rules["decision"],
        "contains_examples": False,
        "analysis_scope": "selected_candidate_documents",
        "thresholds": {
            "cross_domain_paragraphs": {
                "minimum_characters": rules["paragraph_minimum_characters"],
                "minimum_documents": rules["paragraph_minimum_documents"],
                "minimum_domains": rules["paragraph_minimum_domains"],
            },
            "within_domain_blocks": {
                "lines_per_block": rules["lines_per_block"],
                "minimum_characters": rules["block_minimum_characters"],
                "minimum_documents": rules["block_minimum_documents"],
            },
            "document_filter": {
                "minimum_remaining_characters": rules[
                    "minimum_remaining_characters"
                ],
                "maximum_removed_fraction": rules["maximum_removed_fraction"],
            },
        },
        "candidate_documents": line_filter["input_documents"],
        "line_filter": dict(line_filter),
        "boilerplate_candidate_documents": documents_before,
        "analysis": {
            "paragraph_occurrences_considered": (
                analysis.paragraph_occurrences_considered
            ),
            "distinct_paragraph_hashes": analysis.distinct_paragraph_hashes,
            "cross_domain_paragraphs_repeated": len(
                analysis.repeated_cross_domain_paragraphs
            ),
            "matching_cross_domain_paragraph_occurrences": (
                analysis.matching_paragraph_occurrences
            ),
            "block_occurrences_considered": analysis.block_occurrences_considered,
            "distinct_domain_block_hashes": analysis.distinct_domain_block_hashes,
            "within_domain_blocks_repeated": len(
                analysis.repeated_within_domain_blocks
            ),
            "matching_within_domain_block_occurrences": (
                analysis.matching_block_occurrences
            ),
        },
        "simulation": dict(simulation),
        "applied": dict(applied),
        "finalization_blocked": rules["decision"] == "pending",
    }


def _restore_boilerplate_report(
    value: Any, decision: str, profile_name: str
) -> Dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "queroquero-boilerplate-report/v2"
        or value.get("profile") != profile_name
        or value.get("decision") != decision
        or value.get("contains_examples") is not False
        or not isinstance(value.get("finalization_blocked"), bool)
        or not isinstance(value.get("analysis"), dict)
        or not isinstance(value.get("simulation"), dict)
        or not isinstance(value.get("applied"), dict)
        or not isinstance(value.get("line_filter"), dict)
    ):
        raise ConfigError("invalid WackyWacky boilerplate checkpoint")
    return dict(value)


def _sampled_source_guard(path: Path, sample_bytes: int) -> Dict[str, Any]:
    stat = path.stat()
    size = stat.st_size
    maximum_start = max(0, size - sample_bytes)
    positions = sorted(
        {
            0,
            min(maximum_start, size // 4),
            min(maximum_start, size // 2),
            min(maximum_start, (size * 3) // 4),
            maximum_start,
        }
    )
    digest = hashlib.sha256()
    digest.update(b"queroquero-wackywacky-sparse-v1\0")
    digest.update(size.to_bytes(8, byteorder="big", signed=False))
    with path.open("rb") as raw:
        for position in positions:
            raw.seek(position)
            value = raw.read(sample_bytes)
            digest.update(position.to_bytes(8, byteorder="big", signed=False))
            digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
            digest.update(value)
    content_sha256 = digest.hexdigest()
    resume_digest = hashlib.sha256()
    resume_digest.update(b"queroquero-wackywacky-resume-guard-v1\0")
    resume_digest.update(content_sha256.encode("ascii"))
    resume_digest.update(stat.st_size.to_bytes(16, byteorder="big", signed=False))
    local_stat_digest = hashlib.sha256()
    local_stat_digest.update(b"queroquero-wackywacky-local-stat-v1\0")
    for value in (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_dev,
        stat.st_ino,
    ):
        local_stat_digest.update(int(value).to_bytes(16, byteorder="big", signed=False))
    return {
        "method": "size-plus-five-sparse-samples-v1",
        "sha256": content_sha256,
        "resume_guard_sha256": resume_digest.hexdigest(),
        "local_source_stat_sha256": local_stat_digest.hexdigest(),
        "size_bytes": size,
        "sample_bytes": sample_bytes,
        "sample_count": len(positions),
    }


def _hash_prefix(raw: BinaryIO, length: int, digest: Any) -> None:
    remaining = length
    while remaining:
        value = raw.read(min(1024 * 1024, remaining))
        if not value:
            raise ConfigError("WackyWacky resume offset exceeds the source size")
        digest.update(value)
        remaining -= len(value)


def _source_fingerprint(
    guard: Dict[str, Any],
    prefix_sha256: str,
    scanned_bytes: int,
    complete_source_scan: bool,
) -> Dict[str, Any]:
    if complete_source_scan:
        return {
            "method": "streamed-full-sha256-v1",
            "sha256": prefix_sha256,
            "size_bytes": guard["size_bytes"],
            "complete_source_scan": True,
        }
    digest = hashlib.sha256()
    digest.update(b"queroquero-wackywacky-prefix-sparse-v1\0")
    digest.update(guard["sha256"].encode("ascii"))
    digest.update(prefix_sha256.encode("ascii"))
    digest.update(scanned_bytes.to_bytes(8, byteorder="big", signed=False))
    return {
        "method": "scanned-prefix-plus-sparse-guard-v1",
        "sha256": digest.hexdigest(),
        "size_bytes": guard["size_bytes"],
        "scanned_bytes": scanned_bytes,
        "complete_source_scan": False,
        "source_guard_sha256": guard["sha256"],
        "scanned_prefix_sha256": prefix_sha256,
    }


def _cursor(
    byte_offset: int,
    row_number: int,
    source_guard_sha256: str,
    prefix_sha256: str,
    metrics: Dict[str, int],
    complete: bool,
    finalization_blocked: bool,
    boilerplate_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "byte_offset": byte_offset,
        "row_number": row_number,
        "source_guard_sha256": source_guard_sha256,
        "prefix_sha256": prefix_sha256,
        "metrics": dict(metrics),
        "complete": complete,
        "finalization_blocked": finalization_blocked,
    }
    if boilerplate_report is not None:
        value["boilerplate_report"] = dict(boilerplate_report)
    return value


def _with_local_stat_guard(
    cursor: Dict[str, Any], local_source_stat_sha256: str
) -> Dict[str, Any]:
    value = dict(cursor)
    value["_local_source_stat_sha256"] = local_source_stat_sha256
    return value


def _resume_metrics(value: Any) -> Dict[str, int]:
    metrics = {
        "rows_seen": 0,
        "eligible_records": 0,
        "filtered_status": 0,
        "filtered_missing_text": 0,
        "filtered_missing_text_md5": 0,
        "filtered_same_as": 0,
        "filtered_page_search": 0,
        "filtered_page_listing": 0,
        "filtered_invalid_text_md5": 0,
        "filtered_invalid_text_hex": 0,
        "filtered_empty_compressed_texts": 0,
        "filtered_invalid_zstd_frames": 0,
        "filtered_truncated_zstd_frames_65535_bytes": 0,
        "filtered_corrupt_zstd_frames": 0,
        "filtered_oversized_decompressed_texts": 0,
        "filtered_non_utf8_decompressed_texts": 0,
        "text_md5_mismatches": 0,
        "selected_documents": 0,
        "short_lines_considered": 0,
        "short_lines_removed": 0,
        "short_line_characters_removed": 0,
        "documents_affected_by_short_line_filter": 0,
        "documents_discarded_by_short_line_filter": 0,
    }
    if value is None:
        return metrics
    if not isinstance(value, dict):
        raise ConfigError("invalid WackyWacky checkpoint metrics")
    for key in metrics:
        saved = value.get(key, 0)
        if not isinstance(saved, int) or isinstance(saved, bool) or saved < 0:
            raise ConfigError("invalid WackyWacky checkpoint metrics")
        metrics[key] = saved
    return metrics


def _truncated_zstd_frame_size(source: Dict[str, Any]) -> int:
    value = _positive_int(
        source.get("discard_truncated_zstd_frame_size_bytes"),
        "source.discard_truncated_zstd_frame_size_bytes",
    )
    if value != _TRUNCATED_ZSTD_FRAME_SIZE_BYTES:
        raise ConfigError(
            "source.discard_truncated_zstd_frame_size_bytes must be 65535"
        )
    return value


def _text_decode_error_policy(source: Dict[str, Any]) -> str:
    value = source.get("text_decode_error_policy")
    if value != "discard":
        raise ConfigError("source.text_decode_error_policy must be 'discard'")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


ADAPTER = WackyWackyAdapter()
