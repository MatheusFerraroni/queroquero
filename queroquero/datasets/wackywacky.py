from __future__ import annotations

import hashlib
import heapq
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple

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
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()


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
                        seed=seed,
                        metrics=metrics,
                        maximum_text_bytes=max_decompressed_text_bytes,
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
        if config.get("profile_name") == "mvp":
            rules = _boilerplate_rules(filters)
            saved_report = cursor.get("boilerplate_report")
            reuse_saved_report = (
                cursor.get("complete") is True
                and isinstance(saved_report, dict)
                and saved_report.get("decision") == rules["decision"]
            )
            if reuse_saved_report:
                report = _restore_boilerplate_report(saved_report, rules["decision"])
            else:
                if (
                    isinstance(saved_report, dict)
                    and saved_report.get("decision") == "remove_exact"
                ):
                    raise ConfigError(
                        "cannot change the boilerplate decision after exact removal"
                    )
                report, repeated_hashes = _boilerplate_report(documents, rules)
                decision = rules["decision"]
                if decision == "remove_exact":
                    documents = _remove_repeated_paragraphs(
                        documents,
                        repeated_hashes,
                        rules["minimum_paragraph_characters"],
                    )
                    report["removed_paragraph_occurrences"] = report[
                        "matching_paragraph_occurrences"
                    ]
                else:
                    report["removed_paragraph_occurrences"] = 0
                report["finalization_blocked"] = decision == "pending"
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
    profile_name = config.get("profile_name")
    selection = config["profile"].get("selection")
    expected_selection = {
        "smoke": "engineering_prefix",
        "mvp": "representative",
    }.get(profile_name)
    if expected_selection is None:
        raise ConfigError("WackyWacky profile_name must be smoke or mvp")
    if selection != expected_selection:
        raise ConfigError(
            f"WackyWacky {profile_name} selection must be {expected_selection}"
        )
    seed = config["preparation"].get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("preparation.seed must be a non-negative integer")
    _boilerplate_rules(filters)


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
    seed: int,
    metrics: Dict[str, int],
    maximum_text_bytes: int,
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

    text, text_md5_matches = _decode_text(
        text,
        text_md5,
        maximum_bytes=maximum_text_bytes,
    )
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


def _decode_text(encoded: str, text_md5: str, maximum_bytes: int) -> tuple[str, bool]:
    if not _MD5_RE.fullmatch(text_md5):
        raise ValueError("WackyWacky text_md5 is not a valid MD5 digest")
    try:
        compressed = bytes.fromhex(encoded)
    except ValueError:
        raise ValueError("WackyWacky text is not valid hexadecimal") from None
    if not compressed:
        raise ValueError("WackyWacky text contains an empty compressed payload")
    try:
        declared_size = zstd.frame_content_size(compressed)
    except zstd.ZstdError:
        raise ValueError("WackyWacky text is not a valid Zstandard frame") from None
    if declared_size == zstd.CONTENTSIZE_ERROR:
        raise ValueError("WackyWacky text is not a valid Zstandard frame")
    if (
        declared_size != zstd.CONTENTSIZE_UNKNOWN
        and declared_size > maximum_bytes
    ):
        raise ValueError("WackyWacky decompressed text exceeds the size limit")
    try:
        decoded_bytes = _ZSTD_DECOMPRESSOR.decompress(
            compressed,
            max_output_size=maximum_bytes,
        )
    except zstd.ZstdError:
        raise ValueError("WackyWacky text is not a valid Zstandard frame") from None
    if len(decoded_bytes) > maximum_bytes:
        raise ValueError("WackyWacky decompressed text exceeds the size limit")
    text_md5_matches = (
        hashlib.md5(decoded_bytes, usedforsecurity=False).hexdigest()
        == text_md5.lower()
    )
    try:
        decoded = decoded_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("WackyWacky decompressed text is not strict UTF-8") from None
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


def _boilerplate_rules(filters: Dict[str, Any]) -> Dict[str, Any]:
    value = filters.get("boilerplate")
    if not isinstance(value, dict):
        raise ConfigError("filters.boilerplate must be an object")
    decision = value.get("decision")
    if decision not in {"pending", "keep", "remove_exact"}:
        raise ConfigError("boilerplate.decision must be pending, keep, or remove_exact")
    minimum_characters = _positive_int(
        value.get("minimum_paragraph_characters", 80),
        "boilerplate.minimum_paragraph_characters",
    )
    minimum_documents = _positive_int(
        value.get("minimum_documents", 5),
        "boilerplate.minimum_documents",
    )
    minimum_domains = _positive_int(
        value.get("minimum_domains", 3),
        "boilerplate.minimum_domains",
    )
    if minimum_characters < 80 or minimum_documents < 5 or minimum_domains < 3:
        raise ConfigError("boilerplate thresholds cannot be below 80 chars, 5 docs, 3 domains")
    return {
        "decision": decision,
        "minimum_paragraph_characters": minimum_characters,
        "minimum_documents": minimum_documents,
        "minimum_domains": minimum_domains,
    }


def _normalized_paragraphs(text: str) -> List[Tuple[str, str]]:
    cleaned = clean_text(text, strip_html=True)
    paragraphs: List[Tuple[str, str]] = []
    for raw_paragraph in _PARAGRAPH_BREAK_RE.split(cleaned):
        paragraph = " ".join(raw_paragraph.split())
        if not paragraph:
            continue
        paragraphs.append((paragraph, stable_hash("wackywacky-boilerplate", paragraph)))
    return paragraphs


def _boilerplate_report(
    documents: Sequence[Document],
    rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], set[str]]:
    minimum_characters = rules["minimum_paragraph_characters"]
    documents_by_hash: Dict[str, set[str]] = {}
    domains_by_hash: Dict[str, set[str]] = {}
    occurrence_counts: Dict[str, int] = {}
    document_paragraphs: Dict[str, List[str]] = {}

    paragraphs_considered = 0
    for document in documents:
        hashes = [
            paragraph_hash
            for paragraph, paragraph_hash in _normalized_paragraphs(document.text)
            if len(paragraph) >= minimum_characters
        ]
        document_paragraphs[document.source_ref] = hashes
        paragraphs_considered += len(hashes)
        domain_ref = str(document.metadata["domain_ref_sha256"])
        for paragraph_hash in hashes:
            occurrence_counts[paragraph_hash] = (
                occurrence_counts.get(paragraph_hash, 0) + 1
            )
        for paragraph_hash in set(hashes):
            documents_by_hash.setdefault(paragraph_hash, set()).add(
                document.source_ref
            )
            domains_by_hash.setdefault(paragraph_hash, set()).add(domain_ref)

    repeated = {
        paragraph_hash
        for paragraph_hash, document_refs in documents_by_hash.items()
        if len(document_refs) >= rules["minimum_documents"]
        and len(domains_by_hash[paragraph_hash]) >= rules["minimum_domains"]
    }
    affected_documents = sum(
        1
        for hashes in document_paragraphs.values()
        if any(paragraph_hash in repeated for paragraph_hash in hashes)
    )
    matching_occurrences = sum(occurrence_counts[value] for value in repeated)
    report = {
        "schema_version": "queroquero-boilerplate-report/v1",
        "decision": rules["decision"],
        "contains_examples": False,
        "analysis_scope": "selected_candidate_documents",
        "thresholds": {
            "minimum_paragraph_characters": rules[
                "minimum_paragraph_characters"
            ],
            "minimum_documents": rules["minimum_documents"],
            "minimum_domains": rules["minimum_domains"],
        },
        "candidate_documents": len(documents),
        "paragraph_occurrences_considered": paragraphs_considered,
        "distinct_paragraph_hashes": len(documents_by_hash),
        "repeated_paragraph_hashes": len(repeated),
        "affected_documents": affected_documents,
        "matching_paragraph_occurrences": matching_occurrences,
    }
    return report, repeated


def _restore_boilerplate_report(value: Any, decision: str) -> Dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "queroquero-boilerplate-report/v1"
        or value.get("decision") != decision
        or not isinstance(value.get("finalization_blocked"), bool)
    ):
        raise ConfigError("invalid WackyWacky boilerplate checkpoint")
    return dict(value)


def _remove_repeated_paragraphs(
    documents: Sequence[Document],
    repeated_hashes: set[str],
    minimum_characters: int,
) -> List[Document]:
    if not repeated_hashes:
        return list(documents)
    result: List[Document] = []
    for document in documents:
        kept = [
            paragraph
            for paragraph, paragraph_hash in _normalized_paragraphs(document.text)
            if len(paragraph) < minimum_characters
            or paragraph_hash not in repeated_hashes
        ]
        text = "\n\n".join(kept).strip()
        if text:
            result.append(replace(document, text=text))
    return result


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
        "text_md5_mismatches": 0,
        "selected_documents": 0,
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


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


ADAPTER = WackyWackyAdapter()
