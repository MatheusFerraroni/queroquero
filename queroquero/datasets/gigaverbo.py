from __future__ import annotations

import fnmatch
import itertools
import json
import math
import os
import random
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from queroquero.config import (
    ConfigError,
    canonical_json_bytes,
    resolve_dataset_root,
    sha256_bytes,
)

from .base import CheckpointCallback, Document, ScanResult, clean_text, stable_hash


DATASET_ID = "Polygl0t/gigaverbo-v2"
REVISION = "b39dfa703102a20dc609ed6e7aaae22e8e3a233f"
CONFIG_NAME = "default"
SPLIT = "train"
LICENSE_POLICY = "internal_research_only"

_SAFE_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+\-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


class GigaverboAdapter:
    """Stream the pinned GigaVerbo split without retaining record identifiers."""

    def __init__(
        self,
        load_dataset_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._load_dataset_fn = load_dataset_fn

    def scan(
        self,
        config: dict[str, Any],
        resume_cursor: dict[str, Any] | None = None,
        resume_documents: list[Document] | None = None,
        checkpoint: CheckpointCallback | None = None,
    ) -> ScanResult:
        source, filters, selection, profile, seed = _read_config(config)
        source_directory, shard_paths = _local_shards(config, source)
        source_fingerprint = _source_fingerprint(
            source_directory, shard_paths, source
        )

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
            stream = self._load_stream(shard_paths)
            stream = stream.shuffle(
                seed=seed,
                buffer_size=profile["shuffle_buffer_size"],
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

    def _load_stream(self, shard_paths: list[Path]) -> Any:
        load_dataset_fn = self._load_dataset_fn
        if load_dataset_fn is not None:
            return load_dataset_fn(
                path="parquet",
                data_files={SPLIT: [str(path) for path in shard_paths]},
                split=SPLIT,
                streaming=True,
            )
        return _LocalParquetStream(shard_paths)


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
        "provider": "local",
        "path": "gigaverbo-v2",
        "format": "parquet",
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
    if shard_glob != "default/train-*.parquet":
        raise ValueError("GigaVerbo source.shard_glob must select only default/train")
    if source.get("expected_shards") != 224:
        raise ValueError("GigaVerbo source.expected_shards must be 224")
    min_score = filters.get("min_edu_int_score")
    if not isinstance(min_score, int) or isinstance(min_score, bool) or min_score != 4:
        raise ValueError("GigaVerbo min_edu_int_score must be 4")
    if dataset.get("license_policy") != LICENSE_POLICY:
        raise ValueError(f"GigaVerbo license_policy must be {LICENSE_POLICY!r}")
    if dataset.get("redistribution_status") != LICENSE_POLICY:
        raise ValueError(
            f"GigaVerbo redistribution_status must be {LICENSE_POLICY!r}"
        )

    _positive_integer(selection, "checkpoint_interval")
    _positive_integer(profile, "candidate_documents")
    _positive_integer(profile, "max_source_records")
    _positive_integer(profile, "shuffle_buffer_size")
    profile_name = config.get("profile_name")
    expected_strategy = {
        "smoke": "engineering_prefix",
        "mvp": "representative",
        "real": "representative",
    }.get(profile_name)
    if expected_strategy is None or profile.get("selection") != expected_strategy:
        raise ValueError("GigaVerbo profile selection strategy is invalid")
    if profile["max_source_records"] < profile["candidate_documents"]:
        raise ValueError("max_source_records cannot be smaller than candidate_documents")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("preparation seed must be a non-negative integer")
    return source, filters, selection, profile, seed


class _LocalParquetStream:
    def __init__(
        self,
        shard_paths: list[Path],
        *,
        seed: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self._shard_paths = list(shard_paths)
        self._seed = seed
        self._buffer_size = buffer_size

    def shuffle(self, *, seed: int, buffer_size: int) -> "_LocalParquetStream":
        return _LocalParquetStream(
            self._shard_paths,
            seed=seed,
            buffer_size=buffer_size,
        )

    def __iter__(self) -> Iterable[dict[str, Any]]:
        shard_paths = list(self._shard_paths)
        if self._seed is None or self._buffer_size is None:
            return _iter_parquet_records(shard_paths)

        shard_seed = int(stable_hash(self._seed, "gigaverbo", "shards"), 16)
        random.Random(shard_seed).shuffle(shard_paths)
        records = _iter_parquet_records(shard_paths)
        buffer_seed = int(stable_hash(self._seed, "gigaverbo", "buffer"), 16)
        return _buffered_shuffle(
            records,
            buffer_size=self._buffer_size,
            seed=buffer_seed,
        )


def _iter_parquet_records(shard_paths: list[Path]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    required_columns = {"text", "edu_int_score"}
    optional_columns = ("source", "subset")
    for shard_path in shard_paths:
        try:
            parquet = pq.ParquetFile(shard_path)
        except Exception:
            raise ValueError("GigaVerbo local shard is not readable Parquet") from None
        available = set(parquet.schema_arrow.names)
        if not required_columns.issubset(available):
            raise ValueError("GigaVerbo local shard is missing required columns")
        columns = ["text", "edu_int_score"] + [
            column for column in optional_columns if column in available
        ]
        try:
            batches = parquet.iter_batches(
                batch_size=1024,
                columns=columns,
                use_threads=False,
            )
            for batch in batches:
                values = batch.to_pydict()
                for row_index in range(batch.num_rows):
                    yield {
                        column: values[column][row_index] for column in columns
                    }
        except Exception:
            raise ValueError("GigaVerbo local shard could not be streamed") from None


def _buffered_shuffle(
    records: Iterable[dict[str, Any]], *, buffer_size: int, seed: int
) -> Iterable[dict[str, Any]]:
    iterator = iter(records)
    buffer = list(itertools.islice(iterator, buffer_size))
    generator = random.Random(seed)
    while buffer:
        index = generator.randrange(len(buffer))
        selected = buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)
        yield selected


def _local_shards(
    config: dict[str, Any], source: dict[str, Any]
) -> tuple[Path, list[Path]]:
    root = resolve_dataset_root(config)
    relative = Path(source["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("GigaVerbo source.path must stay inside the dataset root")

    unresolved = root / relative
    if unresolved.is_symlink():
        raise ConfigError("GigaVerbo source directory must not be a symlink")
    source_directory = unresolved.resolve()
    if root != source_directory and root not in source_directory.parents:
        raise ConfigError("GigaVerbo source.path escapes the dataset root")
    if not source_directory.is_dir():
        raise ConfigError("GigaVerbo local source directory was not found")

    data_directory = source_directory / CONFIG_NAME
    if data_directory.is_symlink() or not data_directory.is_dir():
        raise ConfigError("GigaVerbo local default directory was not found")
    shard_paths = []
    with os.scandir(data_directory) as entries:
        for entry in entries:
            relative_path = f"{CONFIG_NAME}/{entry.name}"
            if not fnmatch.fnmatchcase(relative_path, source["shard_glob"]):
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ConfigError("GigaVerbo local shard must be a regular file")
            shard_paths.append(Path(entry.path))
    shard_paths.sort()
    expected_shards = source["expected_shards"]
    if len(shard_paths) != expected_shards:
        raise ConfigError(
            f"GigaVerbo requires exactly {expected_shards} local default/train shards"
        )
    return source_directory, shard_paths


def _source_fingerprint(
    source_directory: Path,
    shard_paths: list[Path],
    source: dict[str, Any],
) -> dict[str, Any]:
    tree_path = (
        source_directory
        / ".cache"
        / "huggingface"
        / "trees"
        / f"{REVISION}.json"
    )
    if tree_path.is_symlink() or not tree_path.is_file():
        raise ConfigError("GigaVerbo pinned download tree metadata is missing")
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ConfigError("GigaVerbo download tree metadata is invalid") from None
    if not isinstance(tree, dict) or tree.get("format_version") != 1:
        raise ConfigError("GigaVerbo download tree metadata has an unknown format")
    files = tree.get("files")
    if not isinstance(files, dict):
        raise ConfigError("GigaVerbo download tree metadata has no file listing")

    shards = []
    for relative_path, metadata in sorted(files.items()):
        if not isinstance(relative_path, str) or not fnmatch.fnmatchcase(
            relative_path, source["shard_glob"]
        ):
            continue
        if not isinstance(metadata, dict):
            raise ConfigError("GigaVerbo download tree contains invalid shard metadata")
        content_sha256 = metadata.get("lfs_sha256")
        size = metadata.get("lfs_size")
        if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(
            content_sha256
        ):
            raise ConfigError("GigaVerbo download tree has an invalid shard SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ConfigError("GigaVerbo download tree has an invalid shard size")
        shards.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": content_sha256.lower(),
            }
        )

    local_paths = [path.relative_to(source_directory).as_posix() for path in shard_paths]
    tree_paths = [shard["path"] for shard in shards]
    if len(shards) != source["expected_shards"] or local_paths != tree_paths:
        raise ConfigError("GigaVerbo local shards differ from the pinned download tree")

    fingerprint = {
        "kind": "local_huggingface_snapshot",
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


def _positive_integer(mapping: Mapping[str, Any], key: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


ADAPTER = GigaverboAdapter()
