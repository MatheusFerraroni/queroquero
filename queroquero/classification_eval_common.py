from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from .classification_data import (
    load_classification_config,
    validate_classification_dataset,
)
from .classification_split import validate_classification_split
from .config import (
    ConfigError,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ROOT,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
)
from .experiment_report import validate_paired_experiment_report
from .manifest import file_sha256, write_json_atomic
from .model_artifact import validate_model_artifact


EVALUATION_CONFIG_SCHEMA = "queroquero-classification-evaluation-config/v1"
RESOLVED_EVALUATION_SCHEMA = "queroquero-resolved-classification-evaluation/v1"
PREFLIGHT_SCHEMA = "queroquero-classification-evaluation-preflight/v1"
EMBEDDING_MANIFEST_SCHEMA = "queroquero-classification-embedding-manifest/v1"
EMBEDDING_VALIDATION_SCHEMA = "queroquero-classification-embedding-validation/v1"
TUNING_UNIT_SCHEMA = "queroquero-classification-tuning-unit/v1"
SELECTION_SCHEMA = "queroquero-classification-selection/v1"
EVALUATION_UNIT_SCHEMA = "queroquero-classification-evaluation-unit/v1"
REPORT_SCHEMA = "queroquero-classification-evaluation-report/v1"
REPORT_FILES_SCHEMA = "queroquero-classification-evaluation-report-files/v1"
REDISTRIBUTION_STATUS = "internal_research_only"

MODEL_NAMES = ("base", "general", "forum")
TASKS = ("coarse", "fine")
INPUT_VARIANTS = ("title", "title_first_post")
POOLINGS = ("masked_mean", "last_content")
SEEDS = (42, 43, 44, 45, 46)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID20_RE = re.compile(r"[0-9a-f]{20}\Z")


def load_evaluation_config(path: Path) -> tuple[Dict[str, Any], str]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ConfigError("classification evaluation config must not be a symlink")
    config = load_json(requested.resolve())
    validate_evaluation_config(config)
    return config, sha256_bytes(canonical_json_bytes(config))


def validate_evaluation_config(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {
            "schema_version",
            "dataset",
            "splits",
            "paired_report",
            "models",
            "embedding",
            "classifier",
            "statistics",
            "output",
        },
        "classification evaluation config",
    )
    if config["schema_version"] != EVALUATION_CONFIG_SCHEMA:
        raise ConfigError("classification evaluation config schema changed")

    dataset = _mapping(config, "dataset")
    _keys(
        dataset,
        {"classification_dataset_id", "relative_path", "root_env"},
        "classification evaluation dataset",
    )
    _id20(dataset, "classification_dataset_id", "classification dataset")
    _relative(dataset["relative_path"], "classification dataset")
    if dataset["root_env"] != "PTBR_CLASSIFICATION_ROOT":
        raise ConfigError("classification evaluation dataset root changed")

    splits = config["splits"]
    if not isinstance(splits, list) or len(splits) != 10:
        raise ConfigError("classification evaluation requires ten splits")
    seen: set[tuple[str, int]] = set()
    for split in splits:
        if not isinstance(split, dict):
            raise ConfigError("classification split record must be an object")
        _keys(
            split,
            {"task", "seed", "benchmark_id", "relative_path"},
            "classification split record",
        )
        key = (split["task"], split["seed"])
        if split["task"] not in TASKS or split["seed"] not in SEEDS or key in seen:
            raise ConfigError("classification split task or seed changed")
        seen.add(key)
        _id20(split, "benchmark_id", "classification benchmark")
        _relative(split["relative_path"], "classification split")
    if seen != {(task, seed) for task in TASKS for seed in SEEDS}:
        raise ConfigError("classification split matrix is incomplete")

    report = _mapping(config, "paired_report")
    _keys(report, {"relative_path", "report_id"}, "paired report")
    _relative(report["relative_path"], "paired report")
    _id20(report, "report_id", "paired report")

    models = _mapping(config, "models")
    _keys(models, set(MODEL_NAMES), "classification models")
    base = _mapping(models, "base")
    if base != {
        "kind": "huggingface",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }:
        raise ConfigError("classification baseline changed")
    for name, arm in (("general", "general"), ("forum", "forum_tech")):
        model = _mapping(models, name)
        _keys(
            model,
            {
                "kind",
                "artifact_id",
                "artifact_sha256",
                "expected_arm",
                "relative_path",
            },
            f"classification {name} model",
        )
        if model["kind"] != "local_artifact" or model["expected_arm"] != arm:
            raise ConfigError(f"classification {name} model identity changed")
        _id20(model, "artifact_id", f"classification {name} artifact")
        _sha256(model, "artifact_sha256", f"classification {name} artifact")
        _relative(model["relative_path"], f"classification {name} artifact")
        if Path(model["relative_path"]).name != model["artifact_id"]:
            raise ConfigError(f"classification {name} artifact path changed")

    embedding = _mapping(config, "embedding")
    if embedding != {
        "batch_size_per_rank": 8,
        "chunk_size_per_rank": 4096,
        "exclude_special_tokens": True,
        "input_variants": {
            "title": "title",
            "title_first_post": "title_double_newline_first_post",
        },
        "layer": "last_hidden_state",
        "max_length": 1024,
        "output_dtype": "float32",
        "poolings": ["masked_mean", "last_content"],
        "runtime_dtype": "bfloat16",
        "truncation_side": "right",
        "world_size": 2,
    }:
        raise ConfigError("classification embedding policy changed")

    classifier = _mapping(config, "classifier")
    if classifier != {
        "c_grid": [0.01, 0.1, 1.0, 10.0],
        "class_weight": None,
        "fit_intercept": True,
        "l1_ratio": 0.0,
        "max_iter": 2000,
        "refit": "train_validation",
        "scaler": "standard",
        "selection_metric": "macro_f1",
        "selection_scope": "shared_models_seeds_per_task_input",
        "solver": "lbfgs",
        "tolerance": 1e-5,
    }:
        raise ConfigError("classification linear probe policy changed")

    statistics = _mapping(config, "statistics")
    _keys(
        statistics,
        {
            "confidence_interval",
            "pairwise_contrasts",
            "primary_endpoint",
            "report_p_values",
            "seeds",
        },
        "classification statistics",
    )
    if (
        statistics["confidence_interval"] != "student_t_95"
        or statistics["report_p_values"] is not False
        or statistics["seeds"] != list(SEEDS)
        or statistics["primary_endpoint"]
        != {
            "task": "coarse",
            "input_variant": "title_first_post",
            "metric": "macro_f1",
        }
        or statistics["pairwise_contrasts"]
        != [
            {
                "name": "continual_pretraining",
                "first": "base",
                "second": "general",
            },
            {
                "name": "domain_proximity",
                "first": "general",
                "second": "forum",
            },
            {
                "name": "total_practical_gain",
                "first": "base",
                "second": "forum",
            },
        ]
    ):
        raise ConfigError("classification statistical policy changed")

    output = _mapping(config, "output")
    if output != {
        "root_env": "PTBR_CLASSIFICATION_ROOT",
        "relative_path": "evaluations",
        "status": REDISTRIBUTION_STATUS,
    }:
        raise ConfigError("classification evaluation output policy changed")


def resolve_evaluation_inputs(
    config_path: Path,
    *,
    require_clean_git: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    config, config_sha256 = load_evaluation_config(config_path)
    git_commit = clean_git_commit() if require_clean_git else "0" * 40
    classification_root = _absolute_env_root(config["dataset"]["root_env"])
    dataset_path = _safe_join(
        classification_root,
        config["dataset"]["relative_path"],
        "classification dataset",
    )
    dataset_manifest = validate_classification_dataset(dataset_path)
    if (
        dataset_manifest["classification_dataset_id"]
        != config["dataset"]["classification_dataset_id"]
    ):
        raise RuntimeError("classification evaluation dataset identity changed")

    classification_config = PROJECT_ROOT / "configs/classification/adrenaline-v1.json"
    _, classification_config_sha256 = load_classification_config(
        classification_config
    )
    split_manifests: Dict[tuple[str, int], Dict[str, Any]] = {}
    split_paths: Dict[tuple[str, int], Path] = {}
    split_records = []
    sample_ids: set[str] = set()
    for split in config["splits"]:
        key = (split["task"], split["seed"])
        split_path = _safe_join(
            classification_root, split["relative_path"], "classification split"
        )
        manifest = validate_classification_split(
            classification_config, dataset_path, split_path
        )
        if manifest["benchmark_id"] != split["benchmark_id"]:
            raise RuntimeError("classification benchmark identity changed")
        split_manifests[key] = manifest
        split_paths[key] = split_path
        for values in manifest["sample_ids"].values():
            sample_ids.update(values)
        split_records.append(
            {
                "task": split["task"],
                "seed": split["seed"],
                "benchmark_id": manifest["benchmark_id"],
                "sha256": file_sha256(split_path),
            }
        )

    report_path = _safe_join(
        PROJECT_ROOT, config["paired_report"]["relative_path"], "paired report"
    )
    report = _read_json(report_path, "paired experiment report")
    validate_paired_experiment_report(report)
    if report["report_id"] != config["paired_report"]["report_id"]:
        raise RuntimeError("paired experiment report identity changed")

    model_manifests: Dict[str, Dict[str, Any]] = {}
    model_paths: Dict[str, Path | None] = {"base": None}
    for name in ("general", "forum"):
        expected = config["models"][name]
        artifact_path = _safe_join(
            PROJECT_ROOT, expected["relative_path"], f"{name} artifact"
        )
        manifest = validate_model_artifact(artifact_path, load_model=False)
        if (
            manifest["artifact_id"] != expected["artifact_id"]
            or manifest["artifact_sha256"] != expected["artifact_sha256"]
            or manifest.get("training", {}).get("experiment", {}).get("arm")
            != expected["expected_arm"]
        ):
            raise RuntimeError(f"classification {name} artifact identity changed")
        report_arm = "general" if name == "general" else "forum_tech"
        report_model = report["arms"][report_arm]
        if (
            report_model["artifact_id"] != manifest["artifact_id"]
            or report_model["artifact_sha256"] != manifest["artifact_sha256"]
        ):
            raise RuntimeError(f"classification {name} report artifact changed")
        model_manifests[name] = manifest
        model_paths[name] = artifact_path
    if (
        model_manifests["general"]["tokenizer"]["prepared_fingerprint_sha256"]
        != model_manifests["forum"]["tokenizer"][
            "prepared_fingerprint_sha256"
        ]
    ):
        raise RuntimeError("classification artifact tokenizers differ")

    if any(not _SHA256_RE.fullmatch(value) for value in sample_ids):
        raise RuntimeError("classification split contains an invalid sample ID")
    ordered_ids = sorted(sample_ids)
    sample_ids_sha256 = digest_strings(ordered_ids)
    identity = {
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "classification_config_sha256": classification_config_sha256,
        "classification_dataset_id": dataset_manifest[
            "classification_dataset_id"
        ],
        "dataset_manifest_sha256": file_sha256(
            dataset_path / "dataset_manifest.json"
        ),
        "splits": sorted(split_records, key=lambda item: (item["seed"], item["task"])),
        "paired_report_id": report["report_id"],
        "paired_report_sha256": file_sha256(report_path),
        "models": {
            "base": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "general": {
                "artifact_id": model_manifests["general"]["artifact_id"],
                "artifact_sha256": model_manifests["general"]["artifact_sha256"],
            },
            "forum": {
                "artifact_id": model_manifests["forum"]["artifact_id"],
                "artifact_sha256": model_manifests["forum"]["artifact_sha256"],
            },
        },
        "sample_ids_sha256": sample_ids_sha256,
        "sample_count": len(ordered_ids),
    }
    evaluation_id = sha256_bytes(canonical_json_bytes(identity))[:20]
    resolved = {
        "schema_version": RESOLVED_EVALUATION_SCHEMA,
        "evaluation_id": evaluation_id,
        **identity,
        "embedding": dict(config["embedding"]),
        "classifier": dict(config["classifier"]),
        "statistics": dict(config["statistics"]),
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    _assert_safe_metadata(resolved)
    output_root = _safe_join(
        _absolute_env_root(config["output"]["root_env"]),
        config["output"]["relative_path"],
        "classification evaluation output",
        create=True,
    )
    evaluation_dir = _safe_join(
        output_root,
        evaluation_id,
        "classification evaluation directory",
        create=True,
    )
    runtime = {
        "config": config,
        "dataset_path": dataset_path,
        "dataset_manifest": dataset_manifest,
        "split_manifests": split_manifests,
        "split_paths": split_paths,
        "paired_report": report,
        "paired_report_path": report_path,
        "model_manifests": model_manifests,
        "model_paths": model_paths,
        "sample_ids": ordered_ids,
        "evaluation_dir": evaluation_dir,
    }
    return resolved, runtime


def write_resolved_evaluation(path: Path, resolved: Mapping[str, Any]) -> None:
    target = path / "resolved_evaluation.json"
    if target.exists():
        if _read_json(target, "resolved evaluation") != resolved:
            raise RuntimeError("resolved classification evaluation changed")
        return
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    write_json_atomic(target, dict(resolved))
    target.chmod(0o600)


def locate_existing_evaluation(config_path: Path) -> tuple[Dict[str, Any], Path]:
    config, config_sha256 = load_evaluation_config(config_path)
    git_commit = clean_git_commit()
    root = _safe_join(
        _absolute_env_root(config["output"]["root_env"]),
        config["output"]["relative_path"],
        "classification evaluation output",
        create=False,
    )
    matches: list[tuple[Dict[str, Any], Path]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            resolved_path = child / "resolved_evaluation.json"
            if child.is_symlink() or not child.is_dir() or not resolved_path.is_file():
                continue
            value = _read_json(resolved_path, "resolved evaluation")
            if (
                value.get("schema_version") == RESOLVED_EVALUATION_SCHEMA
                and value.get("config_sha256") == config_sha256
                and value.get("git_commit") == git_commit
                and child.name == value.get("evaluation_id")
            ):
                matches.append((value, child))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one preflighted classification evaluation for this commit"
        )
    resolved, evaluation_dir = matches[0]
    _validate_resolved_shape(resolved, config)
    preflight = _read_json(evaluation_dir / "preflight.json", "evaluation preflight")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA
        or preflight.get("evaluation_id") != resolved["evaluation_id"]
        or preflight.get("status") != "ok"
    ):
        raise RuntimeError("classification evaluation preflight is incomplete")
    return resolved, evaluation_dir


def iter_units(config: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    index = 0
    for seed in config["statistics"]["seeds"]:
        for task in TASKS:
            for input_variant in INPUT_VARIANTS:
                yield {
                    "unit_index": index,
                    "seed": seed,
                    "task": task,
                    "input_variant": input_variant,
                }
                index += 1


def unit_by_index(config: Mapping[str, Any], index: int) -> Dict[str, Any]:
    units = list(iter_units(config))
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(units):
        raise ValueError("classification evaluation unit index must be 0 through 19")
    return units[index]


def split_config_record(
    config: Mapping[str, Any], task: str, seed: int
) -> Mapping[str, Any]:
    matches = [
        item
        for item in config["splits"]
        if item["task"] == task and item["seed"] == seed
    ]
    if len(matches) != 1:
        raise RuntimeError("classification split config is incomplete")
    return matches[0]


def digest_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def clean_git_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("classification evaluation requires a clean Git checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Git HEAD is not a full commit SHA")
    return commit


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def read_json(path: Path, description: str) -> Dict[str, Any]:
    return _read_json(path, description)


def classification_root(config: Mapping[str, Any]) -> Path:
    return _absolute_env_root(config["dataset"]["root_env"])


def classification_dataset_path(config: Mapping[str, Any]) -> Path:
    return _safe_join(
        classification_root(config),
        config["dataset"]["relative_path"],
        "classification dataset",
    )


def classification_split_path(
    config: Mapping[str, Any], task: str, seed: int
) -> Path:
    record = split_config_record(config, task, seed)
    return _safe_join(
        classification_root(config),
        record["relative_path"],
        "classification split",
    )


def classification_model_path(
    config: Mapping[str, Any], model_name: str
) -> Path | None:
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown classification model")
    if model_name == "base":
        return None
    return _safe_join(
        PROJECT_ROOT,
        config["models"][model_name]["relative_path"],
        f"{model_name} artifact",
    )


def load_pinned_split(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    task: str,
    seed: int,
) -> Dict[str, Any]:
    path = classification_split_path(config, task, seed)
    record = split_config_record(config, task, seed)
    resolved_records = [
        item
        for item in resolved["splits"]
        if item["task"] == task and item["seed"] == seed
    ]
    if (
        len(resolved_records) != 1
        or resolved_records[0]["benchmark_id"] != record["benchmark_id"]
        or resolved_records[0]["sha256"] != file_sha256(path)
    ):
        raise RuntimeError("classification split identity changed after preflight")
    manifest = _read_json(path, "classification split")
    if manifest.get("benchmark_id") != record["benchmark_id"]:
        raise RuntimeError("classification split benchmark changed after preflight")
    return manifest


def assert_safe_metadata(value: Any) -> None:
    _assert_safe_metadata(value)


def _validate_resolved_shape(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "evaluation_id",
        "config_sha256",
        "git_commit",
        "classification_config_sha256",
        "classification_dataset_id",
        "dataset_manifest_sha256",
        "splits",
        "paired_report_id",
        "paired_report_sha256",
        "models",
        "sample_ids_sha256",
        "sample_count",
        "embedding",
        "classifier",
        "statistics",
        "redistribution_status",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != RESOLVED_EVALUATION_SCHEMA
        or not _ID20_RE.fullmatch(str(value.get("evaluation_id", "")))
        or not _SHA256_RE.fullmatch(str(value.get("config_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("git_commit", "")))
        or value.get("embedding") != config["embedding"]
        or value.get("classifier") != config["classifier"]
        or value.get("statistics") != config["statistics"]
        or value.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise RuntimeError("resolved classification evaluation is invalid")
    identity_keys = {
        "config_sha256",
        "git_commit",
        "classification_config_sha256",
        "classification_dataset_id",
        "dataset_manifest_sha256",
        "splits",
        "paired_report_id",
        "paired_report_sha256",
        "models",
        "sample_ids_sha256",
        "sample_count",
    }
    identity = {key: value[key] for key in identity_keys}
    if sha256_bytes(canonical_json_bytes(identity))[:20] != value["evaluation_id"]:
        raise RuntimeError("resolved classification evaluation identity changed")
    _assert_safe_metadata(value)


def _absolute_env_root(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        raise ConfigError(f"{name} is required")
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise ConfigError(f"{name} must be an absolute non-symlink path")
    root = requested.resolve()
    if root == Path("/") or len(root.parts) < 3:
        raise ConfigError(f"{name} is unsafe")
    return root


def _safe_join(
    root: Path,
    relative: str,
    description: str,
    *,
    create: bool = False,
) -> Path:
    _relative(relative, description)
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"{description} escapes its root")
    if target.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink")
    if create:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.chmod(0o700)
    return target


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{description} is missing") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be an object")
    return value


def _assert_safe_metadata(
    value: Any,
    key: str | None = None,
    path: tuple[str, ...] = (),
) -> None:
    forbidden = {
        "title",
        "first_post",
        "sample_id",
        "sample_ids",
        "text",
        "input_ids",
        "predictions",
    }
    policy_title = (
        path[-2:] == ("input_variants", "title") and value == "title"
    )
    if key in forbidden and not policy_title:
        raise RuntimeError("classification evaluation metadata contains private data")
    if isinstance(value, Mapping):
        for nested_key, nested in value.items():
            nested_name = str(nested_key)
            _assert_safe_metadata(
                nested,
                nested_name,
                (*path, nested_name),
            )
    elif isinstance(value, list):
        for nested in value:
            _assert_safe_metadata(nested, key, path)
    elif isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise RuntimeError("classification evaluation metadata contains an absolute path")


def _keys(value: Mapping[str, Any], expected: set[str], description: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConfigError(f"{description} keys are incomplete or unknown")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ConfigError(f"{key} must be an object")
    return nested


def _relative(value: Any, description: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{description} path is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{description} path must be safe and relative")


def _id20(value: Mapping[str, Any], key: str, description: str) -> None:
    if not isinstance(value.get(key), str) or not _ID20_RE.fullmatch(value[key]):
        raise ConfigError(f"{description} ID must be a 20-character digest")


def _sha256(value: Mapping[str, Any], key: str, description: str) -> None:
    if not isinstance(value.get(key), str) or not _SHA256_RE.fullmatch(value[key]):
        raise ConfigError(f"{description} SHA-256 is invalid")
