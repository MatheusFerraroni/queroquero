from __future__ import annotations

import fnmatch
import itertools
import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from queroquero.config import canonical_json_bytes, sha256_bytes

from .base import CheckpointCallback, Document, ScanResult, clean_text, stable_hash


DATASET_ID = "Polygl0t/gigaverbo-v2"
REVISION = "b39dfa703102a20dc609ed6e7aaae22e8e3a233f"
CONFIG_NAME = "default"
SPLIT = "train"
LICENSE_POLICY = "internal_research_only"

_SAFE_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+\-]{0,127}\Z")
_HEX_DIGEST_RE = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")


class GigaverboAdapter:
    """Stream the pinned GigaVerbo split without retaining record identifiers."""

    def __init__(
        self,
        load_dataset_fn: Callable[..., Any] | None = None,
        hf_api: Any | None = None,
    ) -> None:
        self._load_dataset_fn = load_dataset_fn
        self._hf_api = hf_api

    def scan(
        self,
        config: dict[str, Any],
        resume_cursor: dict[str, Any] | None = None,
        resume_documents: list[Document] | None = None,
        checkpoint: CheckpointCallback | None = None,
    ) -> ScanResult:
        source, filters, selection, profile, seed = _read_config(config)
        source_fingerprint = self._source_fingerprint(source)

        documents = list(resume_documents or [])
        candidate_limit = profile["candidate_documents"]
        if len(documents) > candidate_limit:
            raise ValueError("resume documents exceed the configured candidate limit")

        cursor = _restore_cursor(
            resume_cursor,
            len(documents),
            source_fingerprint["sha256"],
        )
        if cursor["source_revision"] != REVISION:
            raise ValueError("resume cursor belongs to another source revision")
        if cursor["documents_selected"] != len(documents):
            raise ValueError("resume cursor and documents disagree")

        max_source_records = profile["max_source_records"]
        if cursor["records_seen"] > max_source_records:
            raise ValueError("resume cursor exceeds the configured source-record limit")

        if len(documents) < candidate_limit and cursor["records_seen"] < max_source_records:
            stream = self._load_stream(source)
            stream = stream.shuffle(
                seed=seed,
                buffer_size=selection["shuffle_buffer_size"],
            )
            records = _skip_records(stream, cursor["records_seen"])
            remaining = max_source_records - cursor["records_seen"]

            for record in itertools.islice(records, remaining):
                stream_index = cursor["records_seen"]
                cursor["records_seen"] += 1

                if not isinstance(record, Mapping):
                    cursor["records_invalid"] += 1
                    continue

                score = record.get("edu_int_score")
                if not _score_passes(score, filters["min_edu_int_score"]):
                    cursor["records_filtered_score"] += 1
                    continue

                text = record.get("text")
                if not isinstance(text, str):
                    cursor["records_invalid"] += 1
                    continue
                text = clean_text(text)
                if not text:
                    cursor["records_invalid"] += 1
                    continue

                documents.append(
                    Document(
                        text=text,
                        source_ref="gigaverbo:"
                        + stable_hash(DATASET_ID, REVISION, CONFIG_NAME, SPLIT, stream_index),
                        source_position={"stream_index": stream_index},
                        metadata=_safe_record_metadata(record),
                    )
                )
                cursor["documents_selected"] = len(documents)

                if (
                    checkpoint is not None
                    and len(documents) % selection["checkpoint_interval"] == 0
                ):
                    checkpoint(dict(cursor), list(documents))

                if len(documents) == candidate_limit:
                    break

        metrics = {
            "source_records_seen": cursor["records_seen"],
            "records_filtered_score": cursor["records_filtered_score"],
            "records_invalid": cursor["records_invalid"],
            "documents_selected": len(documents),
            "candidate_limit_reached": int(len(documents) == candidate_limit),
            "source_record_limit_reached": int(
                cursor["records_seen"] == max_source_records
                and len(documents) < candidate_limit
            ),
        }
        return ScanResult(
            documents=documents,
            source_fingerprint=source_fingerprint,
            metrics=metrics,
            cursor=dict(cursor),
        )

    def _load_stream(self, source: dict[str, Any]) -> Any:
        load_dataset_fn = self._load_dataset_fn
        if load_dataset_fn is None:
            from datasets import load_dataset

            load_dataset_fn = load_dataset
        return load_dataset_fn(
            path=source["dataset_id"],
            name=source["config_name"],
            split=source["split"],
            revision=source["revision"],
            streaming=True,
        )

    def _source_fingerprint(self, source: dict[str, Any]) -> dict[str, Any]:
        hf_api = self._hf_api
        if hf_api is None:
            from huggingface_hub import HfApi

            hf_api = HfApi()

        info = hf_api.dataset_info(
            repo_id=source["dataset_id"],
            revision=source["revision"],
            files_metadata=True,
        )
        resolved_revision = _read_value(info, "sha")
        if resolved_revision is not None and resolved_revision != REVISION:
            raise ValueError("Hugging Face resolved an unexpected dataset revision")

        siblings = _read_value(info, "siblings")
        if not isinstance(siblings, Iterable) or isinstance(siblings, (str, bytes)):
            raise ValueError("dataset metadata does not contain a shard listing")

        shards = []
        for sibling in siblings:
            path = _read_value(sibling, "rfilename")
            if not isinstance(path, str) or not fnmatch.fnmatchcase(
                path, source["shard_glob"]
            ):
                continue
            if not _safe_repo_path(path):
                raise ValueError("dataset metadata contains an unsafe shard path")

            metadata: dict[str, Any] = {"path": path}
            size = _read_value(sibling, "size")
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                metadata["size"] = size

            lfs = _read_value(sibling, "lfs")
            sha256 = _read_value(lfs, "sha256")
            if isinstance(sha256, str) and _HEX_DIGEST_RE.fullmatch(sha256):
                metadata["sha256"] = sha256.lower()
            else:
                blob_id = _read_value(sibling, "blob_id")
                if isinstance(blob_id, str) and _HEX_DIGEST_RE.fullmatch(blob_id):
                    metadata["blob_id"] = blob_id.lower()
            shards.append(metadata)

        shards.sort(key=lambda item: item["path"])
        if not shards:
            raise ValueError("no shards matched the pinned default/train source")

        fingerprint = {
            "kind": "huggingface_dataset",
            "dataset_id": DATASET_ID,
            "revision": REVISION,
            "config_name": CONFIG_NAME,
            "split": SPLIT,
            "shard_count": len(shards),
            "shards": shards,
            "license_policy": LICENSE_POLICY,
        }
        fingerprint["sha256"] = sha256_bytes(canonical_json_bytes(fingerprint))
        return fingerprint


def _read_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], int]:
    try:
        dataset = config["dataset"]
        source = dataset["source"]
        filters = dataset["filters"]
        selection = dataset["selection"]
        profile = config["profile"]
        seed = config["preparation"]["seed"]
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete resolved GigaVerbo configuration") from exc

    expected_source = {
        "provider": "huggingface",
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "config_name": CONFIG_NAME,
        "split": SPLIT,
        "streaming": True,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"GigaVerbo source.{key} must be {expected!r}")

    shard_glob = source.get("shard_glob")
    if shard_glob != "data/default/train-*.parquet":
        raise ValueError("GigaVerbo source.shard_glob must select only default/train")
    min_score = filters.get("min_edu_int_score")
    if not isinstance(min_score, int) or isinstance(min_score, bool) or min_score != 4:
        raise ValueError("GigaVerbo min_edu_int_score must be 4")
    if dataset.get("license_policy") != LICENSE_POLICY:
        raise ValueError(f"GigaVerbo license_policy must be {LICENSE_POLICY!r}")
    if dataset.get("redistribution_status") != LICENSE_POLICY:
        raise ValueError(
            f"GigaVerbo redistribution_status must be {LICENSE_POLICY!r}"
        )

    _positive_integer(selection, "shuffle_buffer_size")
    _positive_integer(selection, "checkpoint_interval")
    _positive_integer(profile, "candidate_documents")
    _positive_integer(profile, "max_source_records")
    profile_name = config.get("profile_name")
    expected_strategy = {
        "smoke": "engineering_prefix",
        "mvp": "representative",
    }.get(profile_name)
    if expected_strategy is None or profile.get("selection") != expected_strategy:
        raise ValueError("GigaVerbo profile selection strategy is invalid")
    if profile["max_source_records"] < profile["candidate_documents"]:
        raise ValueError("max_source_records cannot be smaller than candidate_documents")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("preparation seed must be a non-negative integer")
    return source, filters, selection, profile, seed


def _restore_cursor(
    resume_cursor: dict[str, Any] | None,
    resumed_document_count: int,
    source_fingerprint_sha256: str,
) -> dict[str, Any]:
    defaults = {
        "source_revision": REVISION,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "records_seen": 0,
        "records_filtered_score": 0,
        "records_invalid": 0,
        "documents_selected": resumed_document_count,
    }
    if resume_cursor is None:
        return defaults
    if not isinstance(resume_cursor, dict):
        raise ValueError("resume cursor must be an object")
    if set(resume_cursor) != set(defaults):
        raise ValueError("resume cursor has an invalid schema")
    restored = dict(resume_cursor)
    if restored["source_revision"] != REVISION:
        raise ValueError("resume cursor belongs to another source revision")
    if restored["source_fingerprint_sha256"] != source_fingerprint_sha256:
        raise ValueError("source fingerprint changed; refusing to resume")
    for key in defaults.keys() - {"source_revision", "source_fingerprint_sha256"}:
        value = restored[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"resume cursor {key} must be a non-negative integer")
    return restored


def _skip_records(stream: Any, count: int) -> Iterable[Any]:
    if count == 0:
        return iter(stream)
    skip = getattr(stream, "skip", None)
    if callable(skip):
        return iter(skip(count))
    return itertools.islice(iter(stream), count, None)


def _score_passes(value: Any, minimum: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= minimum


def _safe_record_metadata(record: Mapping[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in ("source", "subset"):
        value = record.get(key)
        if isinstance(value, str) and _SAFE_LABEL_RE.fullmatch(value):
            metadata[key] = value
    return metadata


def _safe_repo_path(path: str) -> bool:
    parts = path.split("/")
    return (
        not path.startswith("/")
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in parts)
    )


def _read_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _positive_integer(mapping: Mapping[str, Any], key: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


ADAPTER = GigaverboAdapter()
