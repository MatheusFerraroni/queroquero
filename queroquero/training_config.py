from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import (
    DATASET_IDS,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ROOT,
    canonical_json_bytes,
    resolve_project_path,
    sha256_bytes,
)


TRAINING_CONFIG_SCHEMA = "queroquero-training-config/v1"
TRAINING_METHOD = "full_parameter_continual_pretraining"
TRAINING_DATASET_IDS = tuple(sorted(DATASET_IDS))


class TrainingConfigError(ValueError):
    """Raised when a continual-pretraining configuration violates the contract."""


def load_training_config(path: str | Path) -> tuple[Dict[str, Any], str]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise TrainingConfigError("training configuration must not be a symlink")
    resolved_path = (
        requested.resolve()
        if requested.is_absolute()
        else (PROJECT_ROOT / requested).resolve()
    )
    if PROJECT_ROOT != resolved_path and PROJECT_ROOT not in resolved_path.parents:
        raise TrainingConfigError("training configuration must stay inside the project")
    try:
        config = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingConfigError(
            f"training configuration not found: {resolved_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TrainingConfigError(f"invalid training configuration JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise TrainingConfigError("training configuration must be an object")
    validate_training_config(config)
    return config, sha256_bytes(canonical_json_bytes(config))


def validate_training_config(config: Dict[str, Any]) -> None:
    expected_top_level = {
        "schema_version",
        "profile",
        "model",
        "datasets",
        "training",
        "hardware",
        "output",
    }
    if set(config) != expected_top_level:
        raise TrainingConfigError("training configuration keys are incomplete or unknown")
    if config.get("schema_version") != TRAINING_CONFIG_SCHEMA:
        raise TrainingConfigError(
            f"training schema must be {TRAINING_CONFIG_SCHEMA!r}"
        )
    profile = config.get("profile")
    if profile not in {"smoke", "mvp"}:
        raise TrainingConfigError("training profile must be smoke or mvp")

    model = _mapping(config, "model")
    expected_model = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "trust_remote_code": False,
        "model_type": "llama",
        "parameter_count": 670127616,
        "native_context_length": 4096,
    }
    if not _matches_exact(model, expected_model):
        raise TrainingConfigError("model must match the pinned Tucano2 baseline")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != len(TRAINING_DATASET_IDS):
        raise TrainingConfigError("datasets must list all six sources exactly once")
    expected_sequences = (8, 2) if profile == "smoke" else (256, 32)
    seen = set()
    train_total = 0
    for entry in datasets:
        if not isinstance(entry, dict):
            raise TrainingConfigError("each dataset entry must be an object")
        if set(entry) != {
            "dataset_id",
            "weight",
            "train_sequences",
            "eval_sequences",
        }:
            raise TrainingConfigError("dataset entry keys are incomplete or unknown")
        dataset_id = entry.get("dataset_id")
        if dataset_id not in TRAINING_DATASET_IDS or dataset_id in seen:
            raise TrainingConfigError("dataset IDs must be unique and known")
        seen.add(dataset_id)
        if not _matches_exact(entry.get("weight"), 1):
            raise TrainingConfigError("all dataset weights must be exactly one")
        if (
            not _matches_exact(entry.get("train_sequences"), expected_sequences[0])
            or not _matches_exact(
                entry.get("eval_sequences"), expected_sequences[1]
            )
        ):
            raise TrainingConfigError(
                f"dataset budgets do not match the {profile} preparation profile"
            )
        train_total += entry["train_sequences"]
    if seen != set(TRAINING_DATASET_IDS):
        raise TrainingConfigError("training configuration omits a dataset")

    training = _mapping(config, "training")
    fixed = {
        "method": TRAINING_METHOD,
        "sequence_length": 1024,
        "seed": 42,
        "epochs": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw8bit",
        "learning_rate": 0.000005,
        "scheduler": "linear",
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "epsilon": 0.00000001,
        "max_grad_norm": 1.0,
        "precision": "fp16",
        "gradient_checkpointing": True,
    }
    expected_training_keys = set(fixed) | {
        "warmup_steps",
        "checkpoint_steps",
        "total_optimizer_steps",
    }
    if set(training) != expected_training_keys:
        raise TrainingConfigError("training keys are incomplete or unknown")
    for key, expected in fixed.items():
        if not _matches_exact(training.get(key), expected):
            raise TrainingConfigError(f"training.{key} must be {expected!r}")
    expected_steps = train_total // training["gradient_accumulation_steps"]
    if train_total % training["gradient_accumulation_steps"]:
        raise TrainingConfigError("training rows must divide the accumulation interval")
    if not _matches_exact(training.get("total_optimizer_steps"), expected_steps):
        raise TrainingConfigError("total_optimizer_steps does not match dataset budgets")
    expected_warmup = 1 if profile == "smoke" else 20
    if not _matches_exact(training.get("warmup_steps"), expected_warmup):
        raise TrainingConfigError("warmup_steps does not match the fixed profile")
    expected_checkpoints = [expected_steps // 2]
    if not _matches_exact(training.get("checkpoint_steps"), expected_checkpoints):
        raise TrainingConfigError("checkpoint must be exactly at the half epoch")

    hardware = _mapping(config, "hardware")
    if not _matches_exact(hardware, {
        "accelerator": "cuda",
        "visible_gpus": 1,
        "gpu_name_contains": "P100",
        "compute_capability": [6, 0],
        "minimum_memory_gib": 11,
        "torch_cuda": "11.8",
    }):
        raise TrainingConfigError("hardware must target one P100 with CUDA 11.8")

    output = _mapping(config, "output")
    if set(output) != {"runs_root", "checkpoints_root", "artifacts_root"}:
        raise TrainingConfigError("output roots are incomplete")
    if len(set(output.values())) != 3:
        raise TrainingConfigError("output roots must be distinct")
    for value in output.values():
        if not isinstance(value, str) or not value:
            raise TrainingConfigError("output roots must be non-empty paths")
        resolved = resolve_project_path(value)
        if resolved == PROJECT_ROOT:
            raise TrainingConfigError("output root must not be the project root")


def resolve_training_output_roots(config: Dict[str, Any]) -> Dict[str, Path]:
    validate_training_config(config)
    return {
        key: resolve_project_path(value)
        for key, value in config["output"].items()
    }


def _mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{key} must be an object")
    return value


def _matches_exact(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(expected)
            and all(_matches_exact(value[key], nested) for key, nested in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _matches_exact(actual, nested)
                for actual, nested in zip(value, expected)
            )
        )
    return type(value) is type(expected) and value == expected
