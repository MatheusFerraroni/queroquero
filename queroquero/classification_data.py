from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple
from zipfile import BadZipFile, ZipFile, ZipInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import (
    ConfigError,
    PROJECT_ROOT,
    canonical_json_bytes,
    load_json,
    resolve_output_root,
    sha256_bytes,
)
from .datasets.base import clean_text, safe_source_hash, stable_hash
from .manifest import file_sha256, write_json_atomic


LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = "queroquero-classification-data-config/v1"
DATASET_MANIFEST_SCHEMA = "queroquero-classification-dataset-manifest/v1"
AUDIT_SCHEMA = "queroquero-classification-audit/v1"
WORK_SCHEMA = "queroquero-classification-work/v1"
REDISTRIBUTION_STATUS = "internal_research_only"
HTML_ENTITY_POLICY = "html_entities_until_stable_max_8"
MAX_HTML_ENTITY_DECODES = 8
CLASSIFICATION_CLEANING_POLICY = {
    "html_entity_policy": HTML_ENTITY_POLICY,
    "unicode_normalization": "NFC",
    "strip_html": True,
    "strip_control_characters": True,
    "whitespace": "conservative",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID20_RE = re.compile(r"[0-9a-f]{20}\Z")
_CONVERSATION_MEMBER_RE = re.compile(
    r"clear_threads/(?P<thread_id>[0-9]+)(?:_[^/]+)?\.tsv\Z"
)
_MAPPING_MEMBER_RE = re.compile(
    r"(?P<prefix>.+)/categories_threads/"
    r"category_(?P<category>[0-9]+)_subcategory_(?P<subcategory>[0-9]+)\.json\Z"
)
_THREAD_MEMBER_RE = re.compile(
    r"(?P<prefix>.+)/threads/(?P<thread_id>[0-9]+)\.json\Z"
)

FINAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("first_post", pa.string(), nullable=False),
        pa.field("category_id", pa.int32(), nullable=False),
        pa.field("category_name", pa.string(), nullable=False),
        pa.field("subcategory_id", pa.int32(), nullable=False),
        pa.field("subcategory_name", pa.string(), nullable=False),
        pa.field("title_group_id", pa.string(), nullable=False),
        pa.field("title_chars", pa.int32(), nullable=False),
        pa.field("first_post_chars", pa.int32(), nullable=False),
    ]
)

WORK_TABLE_SCHEMA = FINAL_SCHEMA.append(
    pa.field("content_sha256", pa.string(), nullable=False)
)

_FINAL_COLUMNS = [field.name for field in FINAL_SCHEMA]
_CSV_COLUMNS = list(_FINAL_COLUMNS)


def load_classification_config(path: Path) -> tuple[Dict[str, Any], str]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ConfigError("classification configuration must not be a symlink")
    config = load_json(requested.resolve())
    validate_classification_config(config)
    return config, sha256_bytes(canonical_json_bytes(config))


def validate_classification_config(config: Dict[str, Any]) -> None:
    _require_keys(
        config,
        {"schema_version", "dataset_id", "source", "cleaning", "output", "benchmark"},
        "classification configuration",
    )
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ConfigError(f"classification schema must be {CONFIG_SCHEMA!r}")
    if config["dataset_id"] != "adrenaline":
        raise ConfigError("classification dataset_id must be adrenaline")

    source = _object(config, "source")
    _require_keys(
        source,
        {
            "dataset_root_env",
            "forum_archive",
            "forum_prefix",
            "pretraining_archive",
            "cpt_manifest",
            "expected",
        },
        "classification source",
    )
    _safe_relative(source["forum_archive"], "forum archive")
    _safe_relative(source["pretraining_archive"], "pretraining archive")
    if source["forum_prefix"] != "forum.adrenaline.com.br":
        raise ConfigError("forum archive prefix changed")
    if source["dataset_root_env"] != "PTBR_DATASET_ROOT":
        raise ConfigError("dataset root environment changed")

    cpt = _object(source, "cpt_manifest")
    _require_keys(
        cpt,
        {"root_env", "relative_path", "preparation_id", "sha256"},
        "CPT manifest source",
    )
    if cpt["root_env"] != "PTBR_OUTPUT_ROOT":
        raise ConfigError("CPT manifest root environment changed")
    _safe_relative(cpt["relative_path"], "CPT manifest")
    if not isinstance(cpt["preparation_id"], str) or not _ID20_RE.fullmatch(
        cpt["preparation_id"]
    ):
        raise ConfigError("CPT preparation_id must be a 20-character digest")
    _sha256(cpt, "sha256", "CPT manifest")

    expected = _object(source, "expected")
    _require_keys(expected, {"forum", "pretraining"}, "source fingerprints")
    forum = _object(expected, "forum")
    _require_keys(
        forum,
        {
            "archive_size_bytes",
            "central_directory_entries",
            "mapping_entries",
            "thread_json_entries",
            "labeled_thread_references",
            "unlabeled_thread_entries",
            "sha256",
        },
        "forum fingerprint",
    )
    for key in (
        "archive_size_bytes",
        "central_directory_entries",
        "mapping_entries",
        "thread_json_entries",
        "labeled_thread_references",
        "unlabeled_thread_entries",
    ):
        _positive_integer(forum, key, allow_zero=key == "unlabeled_thread_entries")
    _sha256(forum, "sha256", "forum fingerprint")
    pretraining = _object(expected, "pretraining")
    _require_keys(
        pretraining,
        {
            "archive_size_bytes",
            "central_directory_entries",
            "eligible_member_count",
            "sha256",
        },
        "pretraining fingerprint",
    )
    for key in (
        "archive_size_bytes",
        "central_directory_entries",
        "eligible_member_count",
    ):
        _positive_integer(pretraining, key)
    _sha256(pretraining, "sha256", "pretraining fingerprint")

    cleaning = _object(config, "cleaning")
    if cleaning != CLASSIFICATION_CLEANING_POLICY:
        raise ConfigError("classification cleaning policy changed")

    output = _object(config, "output")
    if output != {
        "root_env": "PTBR_CLASSIFICATION_ROOT",
        "compression": "zstd",
        "csv_compression": "gzip-mtime-zero",
    }:
        raise ConfigError("classification output policy changed")

    benchmark = _object(config, "benchmark")
    _require_keys(
        benchmark,
        {"seeds", "input_variants", "split_percentages", "coarse", "fine"},
        "benchmark configuration",
    )
    if benchmark["seeds"] != [42, 43, 44, 45, 46]:
        raise ConfigError("classification seeds must be exactly 42 through 46")
    if benchmark["input_variants"] != ["title", "title_first_post"]:
        raise ConfigError("classification input variants changed")
    if benchmark["split_percentages"] != {
        "train": 70,
        "validation": 15,
        "test": 15,
    }:
        raise ConfigError("classification split must be 70/15/15")
    coarse = _object(benchmark, "coarse")
    fine = _object(benchmark, "fine")
    expected_categories = [3, 8, 19, 23, 26, 32]
    if coarse != {
        "category_ids": expected_categories,
        "maximum_examples_per_class": 2000,
    }:
        raise ConfigError("coarse benchmark contract changed")
    if fine != {
        "category_ids": expected_categories,
        "minimum_unique_title_groups": 1000,
        "examples_per_class": 1000,
    }:
        raise ConfigError("fine benchmark contract changed")


def _canonicalize_classification_text(
    value: str,
    *,
    max_entity_decodes: int = MAX_HTML_ENTITY_DECODES,
) -> str | None:
    """Return stable classification text, or None when decoding does not converge."""
    if not isinstance(value, str):
        raise TypeError("classification text must be a string")
    if (
        not isinstance(max_entity_decodes, int)
        or isinstance(max_entity_decodes, bool)
        or max_entity_decodes < 1
    ):
        raise ValueError("max_entity_decodes must be a positive integer")

    decoded = value
    for _ in range(max_entity_decodes):
        next_value = html.unescape(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    else:
        if html.unescape(decoded) != decoded:
            return None

    canonical = clean_text(decoded, strip_html=True)
    verification_input = canonical
    for _ in range(max_entity_decodes):
        next_value = html.unescape(verification_input)
        if next_value == verification_input:
            break
        verification_input = next_value
    else:
        if html.unescape(verification_input) != verification_input:
            return None
    if clean_text(verification_input, strip_html=True) != canonical:
        return None
    return canonical


def _resolve_env_root(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        raise ConfigError(f"environment variable {name} is required")
    requested = Path(raw.strip()).expanduser()
    if not requested.is_absolute():
        raise ConfigError(f"{name} must contain an absolute path")
    if requested.is_symlink():
        raise ConfigError(f"{name} must not point to a symlink")
    resolved = requested.resolve()
    if not resolved.is_dir():
        raise ConfigError(f"{name} is not a directory")
    return resolved


def _resolve_output_root(config: Dict[str, Any], override: Path | None) -> Path:
    if override is None:
        name = config["output"]["root_env"]
        raw = os.environ.get(name)
        if not raw or not raw.strip():
            raise ConfigError(f"environment variable {name} is required")
        requested = Path(raw.strip()).expanduser()
    else:
        requested = override.expanduser()
    if not requested.is_absolute():
        raise ConfigError("classification output root must be absolute")
    if requested.is_symlink():
        raise ConfigError("classification output root must not be a symlink")
    resolved = requested.resolve()
    unsafe = {
        Path(resolved.anchor).resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
    }
    if resolved in unsafe:
        raise ConfigError("classification output root is unsafe")
    if resolved.exists() and not resolved.is_dir():
        raise ConfigError("classification output root must be a directory")
    return resolved


def _safe_join(root: Path, relative: str, description: str) -> Path:
    _safe_relative(relative, description)
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink")
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise RuntimeError(f"{description} escapes its configured root")
    if not resolved.is_file():
        raise RuntimeError(f"{description} is missing")
    return resolved


def _zip_fingerprint(
    archive_path: Path,
    *,
    forum_prefix: str | None = None,
) -> tuple[Dict[str, Any], List[ZipInfo]]:
    digest = hashlib.sha256()
    size = archive_path.stat().st_size
    digest.update(size.to_bytes(16, "big"))
    mapping_entries = 0
    thread_entries = 0
    eligible_members = 0
    try:
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            for info in infos:
                central_record = json.dumps(
                    [info.filename, info.CRC, info.compress_size, info.file_size],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                digest.update(len(central_record).to_bytes(8, "big"))
                digest.update(central_record)
                if _is_conversation_member(info):
                    eligible_members += 1
                if forum_prefix is not None:
                    mapping_match = _MAPPING_MEMBER_RE.fullmatch(info.filename)
                    thread_match = _THREAD_MEMBER_RE.fullmatch(info.filename)
                    if (
                        mapping_match is not None
                        and mapping_match.group("prefix") == forum_prefix
                    ):
                        mapping_entries += 1
                    if (
                        thread_match is not None
                        and thread_match.group("prefix") == forum_prefix
                    ):
                        thread_entries += 1
    except BadZipFile:
        raise RuntimeError("classification source ZIP is invalid") from None
    result: Dict[str, Any] = {
        "archive_size_bytes": size,
        "central_directory_entries": len(infos),
        "sha256": digest.hexdigest(),
    }
    if forum_prefix is None:
        result["eligible_member_count"] = eligible_members
    else:
        result["mapping_entries"] = mapping_entries
        result["thread_json_entries"] = thread_entries
    return result, infos


def _assert_fingerprint(
    actual: Mapping[str, Any], expected: Mapping[str, Any], keys: Sequence[str], name: str
) -> None:
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"{name} fingerprint changed: {key}")


def _thread_key(thread_id: str) -> str:
    return stable_hash("adrenaline-classification-thread/v1", thread_id)


def _load_cpt_source_hashes(
    manifest_path: Path,
    expected: Mapping[str, Any],
) -> tuple[set[str], set[str], Dict[str, Any]]:
    if file_sha256(manifest_path) != expected["sha256"]:
        raise RuntimeError("CPT manifest hash changed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("CPT manifest is unreadable") from None
    if (
        manifest.get("schema_version") != "queroquero-dataset-manifest/v1"
        or manifest.get("dataset_id") != "adrenaline"
        or manifest.get("profile") != "paired_real"
        or manifest.get("preparation_id") != expected["preparation_id"]
    ):
        raise RuntimeError("CPT manifest identity changed")

    train: set[str] = set()
    evaluation: set[str] = set()
    for split, target in (("train", train), ("eval", evaluation)):
        records = manifest.get("splits", {}).get(split)
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"CPT manifest {split} shards are missing")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise RuntimeError("CPT shard record is invalid")
            shard = _safe_artifact_path(manifest_path.parent, record["path"])
            if (
                shard.stat().st_size != record.get("size_bytes")
                or file_sha256(shard) != record.get("sha256")
            ):
                raise RuntimeError("CPT shard fingerprint changed")
            table = pq.read_table(shard, columns=["source_ref_sha256"])
            for values in table.column("source_ref_sha256").to_pylist():
                if not isinstance(values, list) or not values:
                    raise RuntimeError("CPT shard provenance is invalid")
                for value in values:
                    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                        raise RuntimeError("CPT source hash is invalid")
                    target.add(value)
    source_fingerprint = manifest.get("source", {}).get("fingerprint")
    if not isinstance(source_fingerprint, dict):
        raise RuntimeError("CPT source fingerprint is missing")
    return train, evaluation, source_fingerprint


def _scan_pretraining_overlap(
    archive_path: Path,
    train_hashes: set[str],
    eval_hashes: set[str],
) -> tuple[Dict[str, Any], set[str], set[str]]:
    train_threads: set[str] = set()
    eval_threads: set[str] = set()
    matched_train: set[str] = set()
    matched_eval: set[str] = set()
    digest = hashlib.sha256()
    size = archive_path.stat().st_size
    digest.update(size.to_bytes(16, "big"))
    eligible = 0
    try:
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            for info in infos:
                central_record = json.dumps(
                    [info.filename, info.CRC, info.compress_size, info.file_size],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                digest.update(len(central_record).to_bytes(8, "big"))
                digest.update(central_record)
                if not _is_conversation_member(info):
                    continue
                eligible += 1
                source_ref = f"adrenaline/{archive_path.name}/{info.filename}"
                source_hash = safe_source_hash(source_ref)
                in_train = source_hash in train_hashes
                in_eval = source_hash in eval_hashes
                if not in_train and not in_eval:
                    continue
                match = _CONVERSATION_MEMBER_RE.fullmatch(info.filename)
                if match is None:
                    raise RuntimeError(
                        "CPT source member cannot be mapped to a forum thread"
                    )
                key = _thread_key(match.group("thread_id"))
                if in_train:
                    matched_train.add(source_hash)
                    train_threads.add(key)
                if in_eval:
                    matched_eval.add(source_hash)
                    eval_threads.add(key)
    except BadZipFile:
        raise RuntimeError("pretraining source ZIP is invalid") from None

    missing_train = len(train_hashes - matched_train)
    missing_eval = len(eval_hashes - matched_eval)
    if missing_train or missing_eval:
        raise RuntimeError(
            "CPT source crosswalk is incomplete "
            f"(train_missing={missing_train}, eval_missing={missing_eval})"
        )
    fingerprint = {
        "archive_size_bytes": size,
        "central_directory_entries": len(infos),
        "eligible_member_count": eligible,
        "sha256": digest.hexdigest(),
    }
    return fingerprint, train_threads, eval_threads


def _is_conversation_member(info: ZipInfo) -> bool:
    if info.is_dir():
        return False
    path = PurePosixPath(info.filename)
    return path.parent == PurePosixPath("clear_threads") and path.suffix == ".tsv"


def _safe_artifact_path(root: Path, relative: str) -> Path:
    _safe_relative(relative, "CPT shard")
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError("CPT shard must not be a symlink")
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents or not resolved.is_file():
        raise RuntimeError("CPT shard is missing or unsafe")
    return resolved


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ConfigError(f"{name} keys are incomplete or unknown")


def _object(value: Mapping[str, Any], key: str) -> Dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ConfigError(f"{key} must be an object")
    return nested


def _positive_integer(
    value: Mapping[str, Any], key: str, *, allow_zero: bool = False
) -> None:
    nested = value.get(key)
    minimum = 0 if allow_zero else 1
    if not isinstance(nested, int) or isinstance(nested, bool) or nested < minimum:
        raise ConfigError(f"{key} must be a valid integer")


def _sha256(value: Mapping[str, Any], key: str, name: str) -> None:
    nested = value.get(key)
    if not isinstance(nested, str) or not _SHA256_RE.fullmatch(nested):
        raise ConfigError(f"{name} {key} must be a SHA-256")


def _safe_relative(value: Any, description: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{description} path is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{description} path must be safe and relative")


def _load_forum_labels(
    archive: ZipFile,
    infos: Sequence[ZipInfo],
    prefix: str,
    expected: Mapping[str, Any],
) -> tuple[
    Dict[str, tuple[int, int]],
    Dict[int, str],
    Dict[tuple[int, int], str],
    List[ZipInfo],
    Dict[str, int],
]:
    categories_member = f"{prefix}/categories.json"
    try:
        categories = json.loads(archive.read(categories_member))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("forum category index is unreadable") from None
    if not isinstance(categories, list) or not categories:
        raise RuntimeError("forum category index is invalid")

    category_names: Dict[int, str] = {}
    subcategory_names: Dict[tuple[int, int], str] = {}
    incomplete_subcategory_metadata = 0
    for category in categories:
        if not isinstance(category, dict):
            raise RuntimeError("forum category record is invalid")
        category_id = category.get("id")
        category_name = category.get("title_text")
        subcategories = category.get("subs")
        if (
            not isinstance(category_id, int)
            or isinstance(category_id, bool)
            or not isinstance(category_name, str)
            or not category_name
            or not isinstance(subcategories, list)
            or category_id in category_names
        ):
            raise RuntimeError("forum category contract changed")
        canonical_category_name = _canonicalize_classification_text(category_name)
        if not canonical_category_name:
            raise RuntimeError("forum category name is not canonicalizable")
        category_names[category_id] = canonical_category_name
        for subcategory in subcategories:
            if not isinstance(subcategory, dict):
                raise RuntimeError("forum subcategory record is invalid")
            subcategory_id = subcategory.get("id")
            subcategory_name = subcategory.get("title_text")
            key = (category_id, subcategory_id)
            if (
                not isinstance(subcategory_id, int)
                or isinstance(subcategory_id, bool)
                or not isinstance(subcategory_name, str)
                or not subcategory_name
                or key in subcategory_names
            ):
                raise RuntimeError("forum subcategory contract changed")
            canonical_subcategory_name = _canonicalize_classification_text(
                subcategory_name
            )
            if not canonical_subcategory_name:
                raise RuntimeError("forum subcategory name is not canonicalizable")
            subcategory_names[key] = canonical_subcategory_name
            if subcategory.get("complete") is not True:
                incomplete_subcategory_metadata += 1

    mappings = sorted(
        (
            info
            for info in infos
            if (
                (match := _MAPPING_MEMBER_RE.fullmatch(info.filename)) is not None
                and match.group("prefix") == prefix
            )
        ),
        key=lambda info: info.filename,
    )
    if len(mappings) != expected["mapping_entries"]:
        raise RuntimeError("forum mapping count changed")
    labels: Dict[str, tuple[int, int]] = {}
    declared_threads = 0
    for info in mappings:
        match = _MAPPING_MEMBER_RE.fullmatch(info.filename)
        assert match is not None
        category_id = int(match.group("category"))
        subcategory_id = int(match.group("subcategory"))
        try:
            mapping = json.loads(archive.read(info))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError("forum mapping index is unreadable") from None
        threads = mapping.get("threads") if isinstance(mapping, dict) else None
        if (
            not isinstance(mapping, dict)
            or mapping.get("status") != "complete"
            or mapping.get("category") != category_id
            or mapping.get("subcategory") != subcategory_id
            or (category_id, subcategory_id) not in subcategory_names
            or not isinstance(threads, list)
            or mapping.get("total_threads") != len(threads)
        ):
            raise RuntimeError("forum mapping contract changed")
        declared_threads += len(threads)
        for item in threads:
            if not isinstance(item, dict):
                raise RuntimeError("forum mapping item is invalid")
            thread_id = item.get("id")
            if (
                not isinstance(thread_id, int)
                or isinstance(thread_id, bool)
                or item.get("category") != category_id
                or item.get("subcategory") != subcategory_id
            ):
                raise RuntimeError("forum mapping label changed")
            key = _thread_key(str(thread_id))
            if key in labels:
                raise RuntimeError("forum mapping references a thread more than once")
            labels[key] = (category_id, subcategory_id)

    if declared_threads != expected["labeled_thread_references"]:
        raise RuntimeError("forum labeled thread count changed")

    thread_infos = sorted(
        (
            info
            for info in infos
            if (
                (match := _THREAD_MEMBER_RE.fullmatch(info.filename)) is not None
                and match.group("prefix") == prefix
            )
        ),
        key=lambda info: info.filename,
    )
    if len(thread_infos) != expected["thread_json_entries"]:
        raise RuntimeError("forum thread entry count changed")
    thread_keys: set[str] = set()
    for info in thread_infos:
        match = _THREAD_MEMBER_RE.fullmatch(info.filename)
        assert match is not None
        key = _thread_key(match.group("thread_id"))
        if key in thread_keys:
            raise RuntimeError("forum archive contains a duplicate thread entry")
        thread_keys.add(key)
    if not set(labels).issubset(thread_keys):
        raise RuntimeError("forum mapping references missing thread records")
    unlabeled = len(thread_keys - set(labels))
    if unlabeled != expected["unlabeled_thread_entries"]:
        raise RuntimeError("forum unlabeled thread count changed")
    metadata = {
        "category_count": len(category_names),
        "subcategory_count": len(subcategory_names),
        "subcategory_metadata_incomplete": incomplete_subcategory_metadata,
        "mapping_declared_threads": declared_threads,
        "unlabeled_thread_entries": unlabeled,
    }
    return labels, category_names, subcategory_names, thread_infos, metadata


def _dataset_identity(
    config_sha256: str,
    forum_fingerprint: Mapping[str, Any],
    pretraining_fingerprint: Mapping[str, Any],
    cpt_manifest_sha256: str,
) -> str:
    value = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "config_sha256": config_sha256,
        "forum_fingerprint": dict(forum_fingerprint),
        "pretraining_fingerprint": dict(pretraining_fingerprint),
        "cpt_manifest_sha256": cpt_manifest_sha256,
    }
    return sha256_bytes(canonical_json_bytes(value))[:20]


def build_classification_dataset(
    config_path: Path,
    *,
    output_root: Path | None = None,
    checkpoint_threads: int = 5_000,
) -> Dict[str, Any]:
    if (
        not isinstance(checkpoint_threads, int)
        or isinstance(checkpoint_threads, bool)
        or checkpoint_threads < 1
    ):
        raise ValueError("checkpoint_threads must be a positive integer")
    config, config_sha256 = load_classification_config(config_path)
    dataset_root = _resolve_env_root(config["source"]["dataset_root_env"])
    cpt_root = resolve_output_root("derived")
    resolved_output = _resolve_output_root(config, output_root)
    for source_root in (dataset_root, cpt_root):
        if (
            resolved_output == source_root
            or resolved_output in source_root.parents
            or source_root in resolved_output.parents
        ):
            raise ConfigError("classification output must not overlap source roots")

    forum_path = _safe_join(
        dataset_root, config["source"]["forum_archive"], "forum archive"
    )
    pretraining_path = _safe_join(
        dataset_root,
        config["source"]["pretraining_archive"],
        "pretraining archive",
    )
    cpt_path = _safe_join(
        cpt_root,
        config["source"]["cpt_manifest"]["relative_path"],
        "CPT manifest",
    )

    LOGGER.info("stage=overlap status=started")
    train_hashes, eval_hashes, cpt_source_fingerprint = _load_cpt_source_hashes(
        cpt_path, config["source"]["cpt_manifest"]
    )
    pretraining_fingerprint, train_threads, eval_threads = _scan_pretraining_overlap(
        pretraining_path, train_hashes, eval_hashes
    )
    _assert_fingerprint(
        pretraining_fingerprint,
        config["source"]["expected"]["pretraining"],
        (
            "archive_size_bytes",
            "central_directory_entries",
            "eligible_member_count",
            "sha256",
        ),
        "pretraining archive",
    )
    _assert_fingerprint(
        cpt_source_fingerprint,
        config["source"]["expected"]["pretraining"],
        (
            "archive_size_bytes",
            "central_directory_entries",
            "eligible_member_count",
            "sha256",
        ),
        "CPT manifest source",
    )
    LOGGER.info(
        "stage=overlap status=complete train_sources=%d eval_sources=%d "
        "excluded_threads=%d",
        len(train_hashes),
        len(eval_hashes),
        len(train_threads | eval_threads),
    )

    forum_fingerprint, forum_infos = _zip_fingerprint(
        forum_path, forum_prefix=config["source"]["forum_prefix"]
    )
    _assert_fingerprint(
        forum_fingerprint,
        config["source"]["expected"]["forum"],
        (
            "archive_size_bytes",
            "central_directory_entries",
            "mapping_entries",
            "thread_json_entries",
            "sha256",
        ),
        "forum archive",
    )
    classification_dataset_id = _dataset_identity(
        config_sha256,
        forum_fingerprint,
        pretraining_fingerprint,
        config["source"]["cpt_manifest"]["sha256"],
    )
    resolved_output.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_output.chmod(0o700)
    output_dir = resolved_output / "adrenaline" / classification_dataset_id
    if (output_dir / "dataset_manifest.json").exists():
        manifest = validate_classification_dataset(output_dir)
        return {
            "classification_dataset_id": manifest["classification_dataset_id"],
            "examples": manifest["counts"]["examples"],
            "relative_path": f"adrenaline/{classification_dataset_id}",
            "status": "existing",
        }
    if output_dir.exists():
        raise RuntimeError("incomplete classification output already exists")

    work_root = (
        resolved_output / ".work" / "adrenaline" / classification_dataset_id
    )
    work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    work_root.chmod(0o700)
    _write_overlap_work(work_root, train_threads, eval_threads)

    try:
        with ZipFile(forum_path) as forum_archive:
            (
                labels,
                category_names,
                subcategory_names,
                thread_infos,
                label_metadata,
            ) = _load_forum_labels(
                forum_archive,
                forum_infos,
                config["source"]["forum_prefix"],
                config["source"]["expected"]["forum"],
            )
            parts, extraction_counts = _extract_thread_parts(
                archive=forum_archive,
                thread_infos=thread_infos,
                labels=labels,
                category_names=category_names,
                subcategory_names=subcategory_names,
                train_threads=train_threads,
                eval_threads=eval_threads,
                work_root=work_root,
                classification_dataset_id=classification_dataset_id,
                config_sha256=config_sha256,
                checkpoint_threads=checkpoint_threads,
            )
    except BadZipFile:
        raise RuntimeError("forum source ZIP is invalid") from None

    manifest = _finalize_dataset(
        output_dir=output_dir,
        work_root=work_root,
        parts=parts,
        extraction_counts=extraction_counts,
        label_metadata=label_metadata,
        category_names=category_names,
        subcategory_names=subcategory_names,
        config=config,
        config_sha256=config_sha256,
        classification_dataset_id=classification_dataset_id,
        forum_fingerprint=forum_fingerprint,
        pretraining_fingerprint=pretraining_fingerprint,
        cpt_manifest_sha256=config["source"]["cpt_manifest"]["sha256"],
        train_source_hashes=len(train_hashes),
        eval_source_hashes=len(eval_hashes),
        train_threads=len(train_threads),
        eval_threads=len(eval_threads),
        overlap_threads=len(train_threads | eval_threads),
    )
    validate_classification_dataset(output_dir)
    shutil.rmtree(work_root)
    return {
        "classification_dataset_id": manifest["classification_dataset_id"],
        "examples": manifest["counts"]["examples"],
        "relative_path": f"adrenaline/{classification_dataset_id}",
        "status": "complete",
    }


def _write_overlap_work(
    work_root: Path, train_threads: set[str], eval_threads: set[str]
) -> None:
    value = {
        "schema_version": "queroquero-classification-overlap-work/v1",
        "train_thread_keys": sorted(train_threads),
        "eval_thread_keys": sorted(eval_threads),
    }
    path = work_root / "overlap_keys.json"
    if path.exists():
        if path.is_symlink():
            raise RuntimeError("classification overlap work must not be a symlink")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("classification overlap work is unreadable") from None
        if existing != value:
            raise RuntimeError("classification overlap work changed")
    else:
        write_json_atomic(path, value)
        path.chmod(0o600)


def _extract_thread_parts(
    *,
    archive: ZipFile,
    thread_infos: Sequence[ZipInfo],
    labels: Mapping[str, tuple[int, int]],
    category_names: Mapping[int, str],
    subcategory_names: Mapping[tuple[int, int], str],
    train_threads: set[str],
    eval_threads: set[str],
    work_root: Path,
    classification_dataset_id: str,
    config_sha256: str,
    checkpoint_threads: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    progress_path = work_root / "progress.json"
    parts_root = work_root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    parts_root.chmod(0o700)
    next_index = 0
    parts: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    if progress_path.exists():
        if progress_path.is_symlink():
            raise RuntimeError("classification progress must not be a symlink")
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("classification progress is unreadable") from None
        if (
            progress.get("schema_version") != WORK_SCHEMA
            or progress.get("classification_dataset_id")
            != classification_dataset_id
            or progress.get("config_sha256") != config_sha256
            or not isinstance(progress.get("next_thread_index"), int)
            or not isinstance(progress.get("parts"), list)
            or not isinstance(progress.get("counts"), dict)
        ):
            raise RuntimeError("classification progress contract changed")
        next_index = progress["next_thread_index"]
        if next_index < 0 or next_index > len(thread_infos):
            raise RuntimeError("classification progress cursor is invalid")
        for record in progress["parts"]:
            _validate_work_part_record(record, parts_root)
            if record["start_thread_index"] != (
                parts[-1]["end_thread_index"] if parts else 0
            ):
                raise RuntimeError("classification work part ranges are not contiguous")
            parts.append(record)
        for key, value in progress["counts"].items():
            if (
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise RuntimeError("classification progress counts are invalid")
            counts[key] = value
        LOGGER.info(
            "stage=extract status=resumed thread_index=%d parts=%d",
            next_index,
            len(parts),
        )
    elif any(parts_root.iterdir()):
        raise RuntimeError("classification work parts exist without progress")

    for start in range(next_index, len(thread_infos), checkpoint_threads):
        end = min(start + checkpoint_threads, len(thread_infos))
        rows: List[Dict[str, Any]] = []
        chunk_counts: Counter[str] = Counter()
        for info in thread_infos[start:end]:
            chunk_counts["thread_entries_seen"] += 1
            match = _THREAD_MEMBER_RE.fullmatch(info.filename)
            assert match is not None
            raw_thread_id = match.group("thread_id")
            key = _thread_key(raw_thread_id)
            label = labels.get(key)
            if label is None:
                chunk_counts["discarded_unlabeled"] += 1
                continue
            in_train = key in train_threads
            in_eval = key in eval_threads
            if in_train:
                chunk_counts["discarded_cpt_train_overlap"] += 1
            if in_eval:
                chunk_counts["discarded_cpt_eval_overlap"] += 1
            if in_train or in_eval:
                chunk_counts["discarded_cpt_overlap_union"] += 1
                continue
            try:
                thread = json.loads(archive.read(info))
            except (json.JSONDecodeError, UnicodeDecodeError):
                chunk_counts["discarded_unreadable_thread"] += 1
                continue
            if not isinstance(thread, dict):
                chunk_counts["discarded_invalid_thread"] += 1
                continue
            try:
                thread_id = thread["id"]
                category_id = thread["category"]
                subcategory_id = thread["subcategory"]
            except KeyError:
                chunk_counts["discarded_invalid_thread"] += 1
                continue
            if (
                str(thread_id) != raw_thread_id
                or (category_id, subcategory_id) != label
            ):
                raise RuntimeError("forum thread identity or label changed")
            raw_title = thread.get("title")
            messages = thread.get("messages")
            if not isinstance(raw_title, str):
                chunk_counts["discarded_missing_title"] += 1
                continue
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(messages[0], dict)
                or not isinstance(messages[0].get("message"), str)
            ):
                chunk_counts["discarded_missing_first_post"] += 1
                continue
            title = _canonicalize_classification_text(raw_title)
            first_post = _canonicalize_classification_text(
                messages[0]["message"]
            )
            if title is None or first_post is None:
                if title is None:
                    chunk_counts["discarded_noncanonical_title"] += 1
                if first_post is None:
                    chunk_counts["discarded_noncanonical_first_post"] += 1
                chunk_counts["discarded_noncanonical_text"] += 1
                continue
            if not title:
                chunk_counts["discarded_empty_title"] += 1
                continue
            if not first_post:
                chunk_counts["discarded_empty_first_post"] += 1
                continue
            sample_id = stable_hash(
                "adrenaline-classification-sample/v1",
                classification_dataset_id,
                key,
            )
            title_group_id = stable_hash(
                "adrenaline-classification-title/v1", title
            )
            content_sha256 = stable_hash(
                "adrenaline-classification-content/v1", title, first_post
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "title": title,
                    "first_post": first_post,
                    "category_id": category_id,
                    "category_name": category_names[category_id],
                    "subcategory_id": subcategory_id,
                    "subcategory_name": subcategory_names[
                        (category_id, subcategory_id)
                    ],
                    "title_group_id": title_group_id,
                    "title_chars": len(title),
                    "first_post_chars": len(first_post),
                    "content_sha256": content_sha256,
                }
            )
            chunk_counts["candidate_examples"] += 1

        part_name = f"part-{start:09d}-{end:09d}.parquet"
        part_path = parts_root / part_name
        table = pa.Table.from_pylist(rows, schema=WORK_TABLE_SCHEMA)
        _write_parquet_atomic(table, part_path, WORK_TABLE_SCHEMA)
        part_record = {
            "path": part_name,
            "start_thread_index": start,
            "end_thread_index": end,
            "rows": len(rows),
            "size_bytes": part_path.stat().st_size,
            "sha256": file_sha256(part_path),
        }
        parts.append(part_record)
        counts.update(chunk_counts)
        progress = {
            "schema_version": WORK_SCHEMA,
            "classification_dataset_id": classification_dataset_id,
            "config_sha256": config_sha256,
            "next_thread_index": end,
            "parts": parts,
            "counts": dict(sorted(counts.items())),
        }
        write_json_atomic(progress_path, progress)
        progress_path.chmod(0o600)
        LOGGER.info(
            "stage=extract status=checkpoint thread_index=%d rows=%d parts=%d",
            end,
            counts["candidate_examples"],
            len(parts),
        )
    if next_index == len(thread_infos):
        LOGGER.info("stage=extract status=resumed_complete parts=%d", len(parts))
    if counts["thread_entries_seen"] != len(thread_infos):
        raise RuntimeError("classification extraction did not cover every thread")
    for key in (
        "discarded_noncanonical_title",
        "discarded_noncanonical_first_post",
        "discarded_noncanonical_text",
    ):
        counts.setdefault(key, 0)
    LOGGER.info(
        "stage=extract status=complete threads=%d candidates=%d",
        counts["thread_entries_seen"],
        counts["candidate_examples"],
    )
    return parts, dict(sorted(counts.items()))


def _validate_work_part_record(record: Any, parts_root: Path) -> None:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "start_thread_index",
        "end_thread_index",
        "rows",
        "size_bytes",
        "sha256",
    }:
        raise RuntimeError("classification work part record is invalid")
    _safe_relative(record["path"], "classification work part")
    path = parts_root / record["path"]
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("classification work part is missing or unsafe")
    if (
        path.stat().st_size != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError("classification work part changed")
    table = pq.read_table(path)
    if table.schema != WORK_TABLE_SCHEMA or table.num_rows != record["rows"]:
        raise RuntimeError("classification work part schema changed")


def _write_parquet_atomic(table: pa.Table, path: Path, schema: pa.Schema) -> None:
    if table.schema != schema:
        raise RuntimeError("classification table schema changed")
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        pq.write_table(
            table,
            partial,
            compression="zstd",
            version="2.6",
            use_dictionary=False,
            write_statistics=True,
        )
        if pq.read_schema(partial) != schema:
            raise RuntimeError("classification Parquet read-back failed")
        partial.chmod(0o600)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _finalize_dataset(
    *,
    output_dir: Path,
    work_root: Path,
    parts: Sequence[Mapping[str, Any]],
    extraction_counts: Mapping[str, int],
    label_metadata: Mapping[str, int],
    category_names: Mapping[int, str],
    subcategory_names: Mapping[tuple[int, int], str],
    config: Mapping[str, Any],
    config_sha256: str,
    classification_dataset_id: str,
    forum_fingerprint: Mapping[str, Any],
    pretraining_fingerprint: Mapping[str, Any],
    cpt_manifest_sha256: str,
    train_source_hashes: int,
    eval_source_hashes: int,
    train_threads: int,
    eval_threads: int,
    overlap_threads: int,
) -> Dict[str, Any]:
    LOGGER.info("stage=finalize status=started parts=%d", len(parts))
    tables: List[pa.Table] = []
    for record in parts:
        path = work_root / "parts" / str(record["path"])
        table = pq.read_table(path)
        if table.schema != WORK_TABLE_SCHEMA:
            raise RuntimeError("classification work schema changed before finalization")
        tables.append(table)
    if not tables:
        raise RuntimeError("classification extraction produced no work parts")
    combined = pa.concat_tables(tables)
    if combined.num_rows != extraction_counts.get("candidate_examples"):
        raise RuntimeError("classification candidate total changed")

    content_labels: Dict[str, set[tuple[int, int]]] = defaultdict(set)
    content_samples: Dict[str, List[str]] = defaultdict(list)
    content_hashes = combined.column("content_sha256").to_pylist()
    sample_ids = combined.column("sample_id").to_pylist()
    categories = combined.column("category_id").to_pylist()
    subcategories = combined.column("subcategory_id").to_pylist()
    for content_hash, sample_id, category_id, subcategory_id in zip(
        content_hashes, sample_ids, categories, subcategories, strict=True
    ):
        content_labels[content_hash].add((category_id, subcategory_id))
        content_samples[content_hash].append(sample_id)

    keep: set[str] = set()
    duplicate_same_label = 0
    conflicting_groups = 0
    conflicting_records = 0
    for content_hash in sorted(content_samples):
        ids = content_samples[content_hash]
        labels = content_labels[content_hash]
        if len(labels) != 1:
            conflicting_groups += 1
            conflicting_records += len(ids)
            continue
        keep.add(min(ids))
        duplicate_same_label += len(ids) - 1
    mask = pc.is_in(
        combined.column("sample_id"), value_set=pa.array(sorted(keep), pa.string())
    )
    final_table = combined.filter(mask).select(_FINAL_COLUMNS).sort_by("sample_id")
    if final_table.schema != FINAL_SCHEMA or final_table.num_rows != len(keep):
        raise RuntimeError("classification final table is inconsistent")
    if final_table.num_rows < 1:
        raise RuntimeError("classification final dataset is empty")

    category_counts = Counter(final_table.column("category_id").to_pylist())
    subcategory_counts = Counter(
        zip(
            final_table.column("category_id").to_pylist(),
            final_table.column("subcategory_id").to_pylist(),
            strict=True,
        )
    )
    title_group_count = len(set(final_table.column("title_group_id").to_pylist()))
    row_digest = _table_row_digest(final_table)

    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.parent.chmod(0o700)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{classification_dataset_id}.", dir=output_dir.parent
        )
    )
    temporary.chmod(0o700)
    try:
        examples_path = temporary / "examples.parquet"
        _write_parquet_atomic(final_table, examples_path, FINAL_SCHEMA)
        csv_path = temporary / "examples.csv.gz"
        _write_examples_csv(final_table, csv_path)
        categories_path = temporary / "categories.csv"
        _write_label_csv(
            categories_path,
            ("category_id", "category_name"),
            ((key, category_names[key]) for key in sorted(category_names)),
        )
        subcategories_path = temporary / "subcategories.csv"
        _write_label_csv(
            subcategories_path,
            ("category_id", "subcategory_id", "subcategory_name"),
            (
                (category_id, subcategory_id, subcategory_names[(category_id, subcategory_id)])
                for category_id, subcategory_id in sorted(subcategory_names)
            ),
        )

        audit = {
            "schema_version": AUDIT_SCHEMA,
            "classification_dataset_id": classification_dataset_id,
            "policy": {
                "canonicalization": HTML_ENTITY_POLICY,
                "cpt_overlap": "exclude_train_and_eval",
                "deduplication": "exact_title_first_post/v1",
                "missing_text": "exclude",
                "source_access": "read_only",
            },
            "source_counts": {
                **dict(sorted(label_metadata.items())),
                "cpt_train_source_hashes": train_source_hashes,
                "cpt_eval_source_hashes": eval_source_hashes,
                "cpt_train_threads": train_threads,
                "cpt_eval_threads": eval_threads,
                "cpt_overlap_threads_union": overlap_threads,
            },
            "extraction_counts": dict(sorted(extraction_counts.items())),
            "deduplication_counts": {
                "duplicate_records_same_label": duplicate_same_label,
                "conflicting_content_groups": conflicting_groups,
                "conflicting_content_records": conflicting_records,
            },
            "output_counts": {
                "examples": final_table.num_rows,
                "unique_title_groups": title_group_count,
                "categories": len(category_counts),
                "subcategories": len(subcategory_counts),
            },
            "category_counts": [
                {"category_id": key, "examples": category_counts[key]}
                for key in sorted(category_counts)
            ],
            "subcategory_counts": [
                {
                    "category_id": category_id,
                    "subcategory_id": subcategory_id,
                    "examples": subcategory_counts[(category_id, subcategory_id)],
                }
                for category_id, subcategory_id in sorted(subcategory_counts)
            ],
            "redistribution_status": REDISTRIBUTION_STATUS,
        }
        _assert_private_safe_metadata(audit)
        audit_path = temporary / "audit.json"
        write_json_atomic(audit_path, audit)
        audit_path.chmod(0o600)

        file_paths = (
            examples_path,
            csv_path,
            categories_path,
            subcategories_path,
            audit_path,
        )
        files = [_file_record(path, temporary) for path in file_paths]
        manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "classification_dataset_id": classification_dataset_id,
            "dataset_id": "adrenaline",
            "config_sha256": config_sha256,
            "source": {
                "forum_fingerprint": dict(forum_fingerprint),
                "pretraining_fingerprint": dict(pretraining_fingerprint),
                "cpt_manifest": {
                    "preparation_id": config["source"]["cpt_manifest"][
                        "preparation_id"
                    ],
                    "sha256": cpt_manifest_sha256,
                },
            },
            "cleaning": dict(config["cleaning"]),
            "eligibility": {
                "requires_title": True,
                "requires_first_post": True,
                "cpt_overlap": "exclude_train_and_eval",
            },
            "deduplication": {
                "exact_content": "keep_lowest_sample_id_same_label",
                "conflicting_exact_content": "exclude_all",
                "title_groups": "preserved_for_experiment_splitting",
            },
            "format": {
                "canonical": "parquet-zstd",
                "portable": "csv-gzip-mtime-zero",
                "split_column": False,
            },
            "counts": {
                "examples": final_table.num_rows,
                "unique_title_groups": title_group_count,
                "categories": len(category_counts),
                "subcategories": len(subcategory_counts),
            },
            "examples_row_sha256": row_digest,
            "files": files,
            "redistribution_status": REDISTRIBUTION_STATUS,
        }
        _assert_private_safe_metadata(manifest)
        expected_identity = _dataset_identity(
            config_sha256,
            forum_fingerprint,
            pretraining_fingerprint,
            cpt_manifest_sha256,
        )
        if expected_identity != classification_dataset_id:
            raise RuntimeError("classification dataset identity changed")
        manifest_path = temporary / "dataset_manifest.json"
        write_json_atomic(manifest_path, manifest)
        manifest_path.chmod(0o600)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    LOGGER.info(
        "stage=finalize status=complete examples=%d unique_title_groups=%d",
        final_table.num_rows,
        title_group_count,
    )
    return manifest


def _write_examples_csv(table: pa.Table, path: Path) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as binary:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=binary, mtime=0
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", newline="", write_through=True
                ) as text:
                    writer = csv.writer(text, lineterminator="\n")
                    writer.writerow(_CSV_COLUMNS)
                    for batch in table.to_batches(max_chunksize=4_096):
                        columns = [batch.column(index).to_pylist() for index in range(len(batch.schema))]
                        for row in zip(*columns, strict=True):
                            writer.writerow(row)
        partial.chmod(0o600)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_label_csv(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]
) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        partial.chmod(0o600)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _table_row_digest(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for batch in table.to_batches(max_chunksize=4_096):
        columns = [batch.column(index).to_pylist() for index in range(len(batch.schema))]
        for row in zip(*columns, strict=True):
            value = dict(zip(_FINAL_COLUMNS, row, strict=True))
            encoded = canonical_json_bytes(value)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_classification_dataset(path: Path) -> Dict[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise RuntimeError("classification dataset directory must not be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise RuntimeError("classification dataset directory is missing")
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("classification manifest must not be a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("classification manifest is unreadable") from None
    if not isinstance(manifest, dict):
        raise RuntimeError("classification manifest must be an object")
    if (
        manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA
        or manifest.get("dataset_id") != "adrenaline"
        or not isinstance(manifest.get("classification_dataset_id"), str)
        or not _ID20_RE.fullmatch(manifest["classification_dataset_id"])
        or root.name != manifest["classification_dataset_id"]
        or manifest.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise RuntimeError("classification manifest identity is invalid")
    if manifest.get("cleaning") != CLASSIFICATION_CLEANING_POLICY:
        raise RuntimeError("classification cleaning policy is invalid")
    _assert_private_safe_metadata(manifest)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("classification source provenance is missing")
    cpt = source.get("cpt_manifest")
    if (
        not isinstance(cpt, dict)
        or not isinstance(cpt.get("preparation_id"), str)
        or not _ID20_RE.fullmatch(cpt["preparation_id"])
        or not isinstance(cpt.get("sha256"), str)
        or not _SHA256_RE.fullmatch(cpt["sha256"])
    ):
        raise RuntimeError("classification CPT provenance is invalid")
    for key in ("forum_fingerprint", "pretraining_fingerprint"):
        fingerprint = source.get(key)
        if (
            not isinstance(fingerprint, dict)
            or not isinstance(fingerprint.get("sha256"), str)
            or not _SHA256_RE.fullmatch(fingerprint["sha256"])
        ):
            raise RuntimeError("classification source fingerprint is invalid")
    expected_identity = _dataset_identity(
        manifest.get("config_sha256", ""),
        source["forum_fingerprint"],
        source["pretraining_fingerprint"],
        cpt["sha256"],
    )
    if expected_identity != manifest["classification_dataset_id"]:
        raise RuntimeError("classification dataset identity is invalid")

    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 5:
        raise RuntimeError("classification file inventory is invalid")
    expected_files = {
        "examples.parquet",
        "examples.csv.gz",
        "categories.csv",
        "subcategories.csv",
        "audit.json",
    }
    actual_files = {
        item.name for item in root.iterdir() if item.name != "dataset_manifest.json"
    }
    if actual_files != expected_files:
        raise RuntimeError("classification dataset contains missing or extra files")
    inventoried: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError("classification file record is invalid")
        _safe_relative(record["path"], "classification artifact")
        if record["path"] in inventoried:
            raise RuntimeError("classification file is inventoried more than once")
        inventoried.add(record["path"])
        artifact = root / record["path"]
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError("classification artifact is missing or unsafe")
        if (
            artifact.stat().st_size != record.get("size_bytes")
            or file_sha256(artifact) != record.get("sha256")
        ):
            raise RuntimeError("classification artifact fingerprint changed")
    if inventoried != expected_files:
        raise RuntimeError("classification file inventory changed")

    try:
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("classification audit is unreadable") from None
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("classification_dataset_id")
        != manifest["classification_dataset_id"]
        or audit.get("redistribution_status") != REDISTRIBUTION_STATUS
        or not isinstance(audit.get("policy"), dict)
        or audit["policy"].get("canonicalization") != HTML_ENTITY_POLICY
    ):
        raise RuntimeError("classification audit is invalid")
    extraction_counts = audit.get("extraction_counts")
    if not isinstance(extraction_counts, dict):
        raise RuntimeError("classification extraction counts are invalid")
    for key in (
        "discarded_noncanonical_title",
        "discarded_noncanonical_first_post",
        "discarded_noncanonical_text",
    ):
        value = extraction_counts.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise RuntimeError("classification extraction counts are invalid")
    noncanonical_records = extraction_counts["discarded_noncanonical_text"]
    noncanonical_title = extraction_counts["discarded_noncanonical_title"]
    noncanonical_first_post = extraction_counts[
        "discarded_noncanonical_first_post"
    ]
    if not (
        max(noncanonical_title, noncanonical_first_post)
        <= noncanonical_records
        <= noncanonical_title + noncanonical_first_post
    ):
        raise RuntimeError("classification extraction counts are inconsistent")
    _assert_private_safe_metadata(audit)

    category_names = _read_categories(root / "categories.csv")
    subcategory_names = _read_subcategories(root / "subcategories.csv")
    parquet_path = root / "examples.parquet"
    parquet_file = pq.ParquetFile(parquet_path)
    if parquet_file.schema_arrow != FINAL_SCHEMA:
        raise RuntimeError("classification Parquet schema changed")
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            if row_group.column(column_index).compression != "ZSTD":
                raise RuntimeError("classification Parquet must use ZSTD")
    table = pq.read_table(parquet_path)
    _validate_examples_table(table, category_names, subcategory_names)
    counts = manifest.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("examples") != table.num_rows
        or counts.get("unique_title_groups")
        != len(set(table.column("title_group_id").to_pylist()))
        or counts.get("categories")
        != len(set(table.column("category_id").to_pylist()))
        or counts.get("subcategories")
        != len(
            set(
                zip(
                    table.column("category_id").to_pylist(),
                    table.column("subcategory_id").to_pylist(),
                    strict=True,
                )
            )
        )
    ):
        raise RuntimeError("classification manifest counts changed")
    if manifest.get("examples_row_sha256") != _table_row_digest(table):
        raise RuntimeError("classification row digest changed")
    _validate_examples_csv(root / "examples.csv.gz", table)
    return manifest


def _validate_examples_table(
    table: pa.Table,
    category_names: Mapping[int, str],
    subcategory_names: Mapping[tuple[int, int], str],
) -> None:
    if table.schema != FINAL_SCHEMA or table.num_rows < 1:
        raise RuntimeError("classification examples table is invalid")
    columns = {name: table.column(name).to_pylist() for name in _FINAL_COLUMNS}
    sample_ids = columns["sample_id"]
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("classification sample IDs are duplicated or unordered")
    content_hashes: set[str] = set()
    for index in range(table.num_rows):
        sample_id = sample_ids[index]
        title = columns["title"][index]
        first_post = columns["first_post"][index]
        category_id = columns["category_id"][index]
        subcategory_id = columns["subcategory_id"][index]
        if not isinstance(sample_id, str) or not _SHA256_RE.fullmatch(sample_id):
            raise RuntimeError("classification sample ID is invalid")
        if (
            not isinstance(title, str)
            or not title
            or _canonicalize_classification_text(title) != title
            or not isinstance(first_post, str)
            or not first_post
            or _canonicalize_classification_text(first_post) != first_post
        ):
            raise RuntimeError("classification text is not canonically cleaned")
        if (
            columns["title_chars"][index] != len(title)
            or columns["first_post_chars"][index] != len(first_post)
        ):
            raise RuntimeError("classification character counts changed")
        if columns["title_group_id"][index] != stable_hash(
            "adrenaline-classification-title/v1", title
        ):
            raise RuntimeError("classification title group changed")
        if category_names.get(category_id) != columns["category_name"][index]:
            raise RuntimeError("classification category name changed")
        if (
            subcategory_names.get((category_id, subcategory_id))
            != columns["subcategory_name"][index]
        ):
            raise RuntimeError("classification subcategory name changed")
        content_hash = stable_hash(
            "adrenaline-classification-content/v1", title, first_post
        )
        if content_hash in content_hashes:
            raise RuntimeError("classification exact content is duplicated")
        content_hashes.add(content_hash)


def _read_categories(path: Path) -> Dict[int, str]:
    result: Dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["category_id", "category_name"]:
            raise RuntimeError("classification category table schema changed")
        for row in reader:
            try:
                key = int(row["category_id"])
            except (TypeError, ValueError):
                raise RuntimeError("classification category ID is invalid") from None
            name = row["category_name"]
            if (
                key in result
                or not name
                or _canonicalize_classification_text(name) != name
            ):
                raise RuntimeError("classification category table is invalid")
            result[key] = name
    if not result:
        raise RuntimeError("classification category table is empty")
    return result


def _read_subcategories(path: Path) -> Dict[tuple[int, int], str]:
    result: Dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "category_id",
            "subcategory_id",
            "subcategory_name",
        ]:
            raise RuntimeError("classification subcategory table schema changed")
        for row in reader:
            try:
                key = (int(row["category_id"]), int(row["subcategory_id"]))
            except (TypeError, ValueError):
                raise RuntimeError("classification subcategory ID is invalid") from None
            name = row["subcategory_name"]
            if (
                key in result
                or not name
                or _canonicalize_classification_text(name) != name
            ):
                raise RuntimeError("classification subcategory table is invalid")
            result[key] = name
    if not result:
        raise RuntimeError("classification subcategory table is empty")
    return result


def _validate_examples_csv(path: Path, table: pa.Table) -> None:
    with path.open("rb") as raw:
        header = raw.read(10)
    if len(header) < 10 or header[:2] != b"\x1f\x8b" or header[4:8] != b"\0\0\0\0":
        raise RuntimeError("classification CSV gzip header is not deterministic")
    expected_columns = {
        name: table.column(name).to_pylist() for name in _FINAL_COLUMNS
    }
    row_index = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _CSV_COLUMNS:
            raise RuntimeError("classification CSV schema changed")
        for row in reader:
            if row_index >= table.num_rows:
                raise RuntimeError("classification CSV has extra rows")
            for name in _FINAL_COLUMNS:
                actual: Any = row[name]
                if name in {
                    "category_id",
                    "subcategory_id",
                    "title_chars",
                    "first_post_chars",
                }:
                    try:
                        actual = int(actual)
                    except ValueError:
                        raise RuntimeError("classification CSV integer is invalid") from None
                if actual != expected_columns[name][row_index]:
                    raise RuntimeError("classification CSV and Parquet differ")
            row_index += 1
    if row_index != table.num_rows:
        raise RuntimeError("classification CSV row count changed")


def _assert_private_safe_metadata(value: Any, key: str | None = None) -> None:
    forbidden_keys = {
        "title",
        "first_post",
        "sample_id",
        "thread_id",
        "thread_ids",
        "thread_keys",
        "source_ref",
    }
    if key in forbidden_keys:
        raise RuntimeError("classification metadata contains private record data")
    if isinstance(value, dict):
        for nested_key, nested in value.items():
            _assert_private_safe_metadata(nested, str(nested_key))
    elif isinstance(value, list):
        for nested in value:
            _assert_private_safe_metadata(nested, key)
    elif isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise RuntimeError("classification metadata contains an absolute path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and split the private Adrenaline classification dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the canonical unsplit dataset")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-root", type=Path)
    build.add_argument("--checkpoint-threads", type=int, default=5_000)
    validate_dataset = subparsers.add_parser(
        "validate-dataset", help="validate a canonical classification dataset"
    )
    validate_dataset.add_argument("--path", type=Path, required=True)
    split = subparsers.add_parser(
        "split", help="materialize one deterministic experiment split"
    )
    split.add_argument("--config", type=Path, required=True)
    split.add_argument("--dataset", type=Path, required=True)
    split.add_argument("--task", choices=("coarse", "fine"), required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--output", type=Path, required=True)
    validate_split = subparsers.add_parser(
        "validate-split", help="validate one experiment split manifest"
    )
    validate_split.add_argument("--config", type=Path, required=True)
    validate_split.add_argument("--dataset", type=Path, required=True)
    validate_split.add_argument("--path", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.command == "build":
        result = build_classification_dataset(
            args.config,
            output_root=args.output_root,
            checkpoint_threads=args.checkpoint_threads,
        )
    elif args.command == "validate-dataset":
        manifest = validate_classification_dataset(args.path)
        result = {
            "classification_dataset_id": manifest["classification_dataset_id"],
            "examples": manifest["counts"]["examples"],
            "status": "valid",
        }
    elif args.command == "split":
        from .classification_split import create_classification_split

        manifest = create_classification_split(
            args.config,
            args.dataset,
            task=args.task,
            seed=args.seed,
            output=args.output,
        )
        result = {
            "benchmark_id": manifest["benchmark_id"],
            "classification_dataset_id": manifest["classification_dataset_id"],
            "examples": manifest["counts"]["total"],
            "seed": manifest["seed"],
            "status": "complete",
            "task": manifest["task"],
        }
    else:
        from .classification_split import validate_classification_split

        manifest = validate_classification_split(
            args.config, args.dataset, args.path
        )
        result = {
            "benchmark_id": manifest["benchmark_id"],
            "classification_dataset_id": manifest["classification_dataset_id"],
            "seed": manifest["seed"],
            "status": "valid",
            "task": manifest["task"],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
