from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_SCHEMA = "queroquero-preparation/v1"
DATASET_SCHEMA = "queroquero-dataset-config/v1"
RESOLVED_SCHEMA = "queroquero-resolved-preparation/v1"
MODEL_ID = "Polygl0t/Tucano2-0.6B-Base"
MODEL_REVISION = "dad97dc864a8f9a1d240fb9351d098f3af9511d7"
OUTPUT_ROOT_ENV = "PTBR_OUTPUT_ROOT"
DATASET_IDS = (
    "adrenaline",
    "brwac",
    "gigaverbo",
    "multiwoz_ptbr",
    "outerspace",
    "wackywacky",
)


class ConfigError(ValueError):
    """Raised when a versioned configuration is incomplete or unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be an object: {path}")
    return value


def load_resolved_config(
    dataset_id: str,
    profile_name: str,
    config_root: Path | None = None,
) -> Tuple[Dict[str, Any], str]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset_id):
        raise ConfigError("dataset_id must use lowercase letters, digits, or underscores")
    if dataset_id not in DATASET_IDS:
        raise ConfigError(f"unknown dataset_id: {dataset_id}")
    root = (config_root or (PROJECT_ROOT / "configs")).resolve()
    preparation = load_json(root / "preparation.json")
    dataset = load_json(root / "datasets" / f"{dataset_id}.json")
    validate_preparation_config(preparation)
    validate_dataset_config(dataset, dataset_id)

    profiles = dataset["profiles"]
    if profile_name not in profiles:
        raise ConfigError(
            f"unknown profile {profile_name!r} for {dataset_id}; "
            f"expected one of {sorted(profiles)}"
        )
    profile = profiles[profile_name]
    validate_profile(profile, profile_name)
    resolved = {
        "schema_version": RESOLVED_SCHEMA,
        "dataset_id": dataset_id,
        "profile_name": profile_name,
        "preparation": preparation,
        "dataset": {key: value for key, value in dataset.items() if key != "profiles"},
        "profile": profile,
    }
    return resolved, sha256_bytes(canonical_json_bytes(resolved))


def scan_config_sha256(resolved: Dict[str, Any]) -> str:
    """Hash only options that affect source scanning/candidate selection.

    WackyWacky's boilerplate decision is deliberately post-scan: the expensive
    full-pass candidates may be reused after the required report is reviewed,
    while the final preparation still keeps the complete resolved-config hash.
    """

    scan_config = deepcopy(resolved)
    if scan_config.get("dataset_id") == "wackywacky":
        boilerplate = (
            scan_config.get("dataset", {}).get("filters", {}).get("boilerplate")
        )
        decisions = (
            boilerplate.get("decision_by_profile")
            if isinstance(boilerplate, dict)
            else None
        )
        if isinstance(decisions, dict):
            for profile_name in decisions:
                decisions[profile_name] = "post_scan_review"
    return sha256_bytes(canonical_json_bytes(scan_config))


def validate_preparation_config(config: Dict[str, Any]) -> None:
    if config.get("schema_version") != PREPARATION_SCHEMA:
        raise ConfigError(f"preparation schema must be {PREPARATION_SCHEMA!r}")
    tokenizer = _mapping(config, "tokenizer")
    if tokenizer.get("model_id") != MODEL_ID:
        raise ConfigError(f"tokenizer.model_id must be {MODEL_ID!r}")
    if tokenizer.get("revision") != MODEL_REVISION:
        raise ConfigError(f"tokenizer.revision must be {MODEL_REVISION!r}")
    if tokenizer.get("trust_remote_code") is not False:
        raise ConfigError("tokenizer.trust_remote_code must be false")
    if config.get("sequence_length") != 1024:
        raise ConfigError("sequence_length must be exactly 1024")
    storage = _mapping(config, "storage")
    if storage.get("format") != "parquet" or storage.get("compression") != "zstd":
        raise ConfigError("storage must use parquet with zstd compression")
    _positive_int(storage, "sequences_per_shard")
    if storage["sequences_per_shard"] != 1024:
        raise ConfigError("storage.sequences_per_shard must be exactly 1024")
    _positive_int(config, "seed", allow_zero=True)
    if config["seed"] != 42:
        raise ConfigError("seed must be exactly 42")
    cleaning = _mapping(config, "cleaning")
    if cleaning != {
        "unicode_normalization": "NFC",
        "strip_html": True,
        "strip_control_characters": True,
    }:
        raise ConfigError("cleaning must use the fixed conservative NFC/HTML policy")
    output_root = config.get("output_root")
    if output_root != "derived":
        raise ConfigError("output_root must be 'derived'")
    resolve_project_path(output_root)


def validate_dataset_config(config: Dict[str, Any], expected_id: str) -> None:
    if config.get("schema_version") != DATASET_SCHEMA:
        raise ConfigError(f"dataset schema must be {DATASET_SCHEMA!r}")
    if config.get("dataset_id") != expected_id:
        raise ConfigError(
            f"dataset_id must match the filename: expected {expected_id!r}"
        )
    if config.get("adapter") != expected_id:
        raise ConfigError("adapter must match dataset_id")
    _mapping(config, "source")
    _mapping(config, "filters")
    profiles = _mapping(config, "profiles")
    if set(profiles) != {"smoke", "mvp"}:
        raise ConfigError("profiles must contain exactly smoke and mvp")


def validate_profile(profile: Dict[str, Any], name: str) -> None:
    _positive_int(profile, "train_sequences")
    _positive_int(profile, "eval_sequences")
    _positive_int(profile, "candidate_documents")
    selection = profile.get("selection")
    if selection not in {"engineering_prefix", "representative"}:
        raise ConfigError(f"invalid selection for profile {name!r}: {selection!r}")
    if name == "smoke" and (
        profile["train_sequences"] != 8 or profile["eval_sequences"] != 2
    ):
        raise ConfigError("smoke profile must use 8 train and 2 eval sequences")
    if name == "mvp" and (
        profile["train_sequences"] != 256 or profile["eval_sequences"] != 32
    ):
        raise ConfigError("mvp profile must use 256 train and 32 eval sequences")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ConfigError(f"project path must be relative: {value}")
    resolved = (PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ConfigError(f"project path escapes repository: {value}")
    return resolved


def resolve_output_root(default_value: str) -> Path:
    """Resolve the local output location without changing preparation identity."""

    raw = os.environ.get(OUTPUT_ROOT_ENV)
    if raw is None:
        resolved = resolve_project_path(default_value)
    else:
        value = raw.strip()
        if not value:
            raise ConfigError(f"environment variable {OUTPUT_ROOT_ENV} cannot be empty")
        configured = Path(value).expanduser()
        if configured.is_symlink():
            raise ConfigError(f"{OUTPUT_ROOT_ENV} must not point to a symlink")
        resolved = (
            configured.resolve()
            if configured.is_absolute()
            else resolve_project_path(value)
        )

    unsafe_roots = {
        Path(resolved.anchor).resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
    }
    if resolved in unsafe_roots:
        raise ConfigError(f"{OUTPUT_ROOT_ENV} points to an unsafe broad directory")
    if resolved.exists() and not resolved.is_dir():
        raise ConfigError(f"{OUTPUT_ROOT_ENV} must point to a directory")

    dataset_raw = os.environ.get("PTBR_DATASET_ROOT")
    if dataset_raw:
        dataset_root = Path(dataset_raw).expanduser().resolve()
        if (
            resolved == dataset_root
            or resolved in dataset_root.parents
            or dataset_root in resolved.parents
        ):
            raise ConfigError(
                f"{OUTPUT_ROOT_ENV} must not overlap PTBR_DATASET_ROOT"
            )
    return resolved


def resolve_dataset_root(config: Dict[str, Any]) -> Path:
    env_name = config["dataset"]["source"].get("root_env", "PTBR_DATASET_ROOT")
    raw = os.environ.get(env_name)
    if not raw:
        raise ConfigError(f"environment variable {env_name} is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ConfigError(f"{env_name} must contain an absolute path")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ConfigError(f"dataset root is not a directory: {resolved}")
    return resolved


def _mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _positive_int(config: Dict[str, Any], key: str, allow_zero: bool = False) -> None:
    value = config.get(key)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        description = "a non-negative" if allow_zero else "a positive"
        raise ConfigError(f"{key} must be {description} integer")
