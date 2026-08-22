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
from .paired_plan import validate_paired_mixture


TRAINING_CONFIG_SCHEMA = "queroquero-training-config/v2"
REAL_TRAINING_CONFIG_SCHEMA = "queroquero-training-config/v3"
PAIRED_REAL_TRAINING_CONFIG_SCHEMA = "queroquero-training-config/v4"
TRAINING_METHOD = "full_parameter_continual_pretraining"
TRAINING_DATASET_IDS = tuple(sorted(DATASET_IDS))
REAL_TRAINING_SEQUENCES = 416_000
REAL_EVAL_SEQUENCES_PER_DATASET = 256
REAL_OPTIMIZER_STEPS = 52_000


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
    schema = config.get("schema_version")
    legacy = schema == TRAINING_CONFIG_SCHEMA
    single_real = schema == REAL_TRAINING_CONFIG_SCHEMA
    paired_real = schema == PAIRED_REAL_TRAINING_CONFIG_SCHEMA
    real = single_real or paired_real
    if not legacy and not real:
        raise TrainingConfigError(
            "training schema must be "
            f"{TRAINING_CONFIG_SCHEMA!r}, {REAL_TRAINING_CONFIG_SCHEMA!r}, or "
            f"{PAIRED_REAL_TRAINING_CONFIG_SCHEMA!r}"
        )
    expected_top_level = {
        "schema_version",
        "profile",
        "model",
        "datasets",
        "training",
        "execution",
        "hardware",
        "output",
    }
    if real:
        expected_top_level.add("data_mixture")
    if set(config) != expected_top_level:
        raise TrainingConfigError("training configuration keys are incomplete or unknown")
    profile = config.get("profile")
    if legacy and profile not in {"smoke", "mvp"}:
        raise TrainingConfigError("v2 training profile must be smoke or mvp")
    if real and profile != "real":
        raise TrainingConfigError("v3/v4 training profile must be real")

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
        expected_entry_keys = {
            "dataset_id",
            "train_sequences",
            "eval_sequences",
        }
        if legacy:
            expected_entry_keys.add("weight")
        if paired_real:
            expected_entry_keys.add("prepared_train_sequences")
        if set(entry) != expected_entry_keys:
            raise TrainingConfigError("dataset entry keys are incomplete or unknown")
        dataset_id = entry.get("dataset_id")
        if dataset_id not in TRAINING_DATASET_IDS or dataset_id in seen:
            raise TrainingConfigError("dataset IDs must be unique and known")
        seen.add(dataset_id)
        if legacy and not _matches_exact(entry.get("weight"), 1):
            raise TrainingConfigError("all dataset weights must be exactly one")
        if legacy:
            if (
                not _matches_exact(
                    entry.get("train_sequences"), expected_sequences[0]
                )
                or not _matches_exact(
                    entry.get("eval_sequences"), expected_sequences[1]
                )
            ):
                raise TrainingConfigError(
                    f"dataset budgets do not match the {profile} preparation profile"
                )
        elif single_real:
            _positive_int(entry, "train_sequences")
            if not _matches_exact(
                entry.get("eval_sequences"), REAL_EVAL_SEQUENCES_PER_DATASET
            ):
                raise TrainingConfigError(
                    "real evaluation budget must be 256 sequences per dataset"
                )
        else:
            _nonnegative_int(entry, "train_sequences")
            _positive_int(entry, "prepared_train_sequences")
            if entry["train_sequences"] > entry["prepared_train_sequences"]:
                raise TrainingConfigError(
                    "paired train budget exceeds its prepared dataset"
                )
            if not _matches_exact(
                entry.get("eval_sequences"), REAL_EVAL_SEQUENCES_PER_DATASET
            ):
                raise TrainingConfigError(
                    "paired evaluation budget must be 256 sequences per dataset"
                )
        train_total += entry["train_sequences"]
    if seen != set(TRAINING_DATASET_IDS):
        raise TrainingConfigError("training configuration omits a dataset")
    if single_real:
        mixture = _mapping(config, "data_mixture")
        if set(mixture) != {
            "policy",
            "without_replacement",
            "allocation_sha256",
        }:
            raise TrainingConfigError("real data_mixture keys are incomplete or unknown")
        if mixture.get("policy") != "equal_share_without_replacement":
            raise TrainingConfigError(
                "real data mixture must use equal_share_without_replacement"
            )
        if mixture.get("without_replacement") is not True:
            raise TrainingConfigError("real training must be without replacement")
        allocation_sha256 = mixture.get("allocation_sha256")
        if not isinstance(allocation_sha256, str) or not _is_sha256(
            allocation_sha256
        ):
            raise TrainingConfigError("real allocation_sha256 must be a SHA-256")
        if train_total != REAL_TRAINING_SEQUENCES:
            raise TrainingConfigError(
                f"real training allocation must total {REAL_TRAINING_SEQUENCES}"
            )
    elif paired_real:
        mixture = _mapping(config, "data_mixture")
        try:
            validate_paired_mixture(mixture)
        except RuntimeError as exc:
            raise TrainingConfigError(str(exc)) from exc
        prepared_expected = {dataset_id: 0 for dataset_id in DATASET_IDS}
        used_expected = {dataset_id: 0 for dataset_id in DATASET_IDS}
        for pool in mixture["pools"]:
            dataset_id = pool["dataset_id"]
            count = pool["train_sequences"]
            prepared_expected[dataset_id] += count
            if mixture["arm"] == "forum_tech":
                if pool["role"] != "replacement":
                    used_expected[dataset_id] += count
            elif pool["role"] != "domain":
                used_expected[dataset_id] += count
        configured_prepared = {
            entry["dataset_id"]: entry["prepared_train_sequences"]
            for entry in datasets
        }
        configured_used = {
            entry["dataset_id"]: entry["train_sequences"] for entry in datasets
        }
        if configured_prepared != prepared_expected:
            raise TrainingConfigError(
                "paired prepared dataset budgets do not match the pools"
            )
        if configured_used != used_expected:
            raise TrainingConfigError(
                "paired arm dataset budgets do not match the pools"
            )
        if train_total != REAL_TRAINING_SEQUENCES:
            raise TrainingConfigError(
                f"paired training allocation must total {REAL_TRAINING_SEQUENCES}"
            )

    execution = _mapping(config, "execution")
    hardware = _mapping(config, "hardware")
    target = _hardware_target(execution, hardware)
    if real and target != "l40s":
        raise TrainingConfigError("real training is restricted to 2x L40S")
    training = _mapping(config, "training")
    common = {
        "method": TRAINING_METHOD,
        "sequence_length": 1024,
        "seed": 42,
        "epochs": 1,
        "micro_batch_size_per_rank": 1,
        "global_batch_sequences": 8,
        "global_batch_tokens": 8192,
        "learning_rate": 0.000005,
        "scheduler": "linear",
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "epsilon": 0.00000001,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
    }
    target_training = {
        "p100": {
            "gradient_accumulation_steps_per_rank": 8,
            "optimizer": "adamw8bit",
            "optimizer_implementation": "bitsandbytes",
            "precision": "fp16",
            "use_grad_scaler": True,
        },
        "l40s": {
            "gradient_accumulation_steps_per_rank": 4,
            "optimizer": "adamw",
            "optimizer_implementation": "torch_fused",
            "precision": "bf16",
            "use_grad_scaler": False,
        },
    }[target]
    fixed = {**common, **target_training}
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

    expected_global_batch = (
        execution["world_size"]
        * training["micro_batch_size_per_rank"]
        * training["gradient_accumulation_steps_per_rank"]
    )
    if training["global_batch_sequences"] != expected_global_batch:
        raise TrainingConfigError("global batch does not match world size and accumulation")
    if training["global_batch_tokens"] != (
        training["global_batch_sequences"] * training["sequence_length"]
    ):
        raise TrainingConfigError("global token batch is inconsistent")
    if train_total % training["global_batch_sequences"]:
        raise TrainingConfigError("training rows must divide the global batch")
    expected_steps = train_total // training["global_batch_sequences"]
    if not _matches_exact(training.get("total_optimizer_steps"), expected_steps):
        raise TrainingConfigError("total_optimizer_steps does not match dataset budgets")
    expected_warmup = (
        520 if real else (1 if profile == "smoke" else 20)
    )
    if not _matches_exact(training.get("warmup_steps"), expected_warmup):
        raise TrainingConfigError("warmup_steps does not match the fixed profile")
    expected_checkpoints = (
        [13_000, 26_000, 39_000] if real else [expected_steps // 2]
    )
    if not _matches_exact(training.get("checkpoint_steps"), expected_checkpoints):
        raise TrainingConfigError("checkpoint steps do not match the fixed profile")
    if real and expected_steps != REAL_OPTIMIZER_STEPS:
        raise TrainingConfigError(
            f"real training must use exactly {REAL_OPTIMIZER_STEPS} optimizer steps"
        )

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


def _hardware_target(execution: Dict[str, Any], hardware: Dict[str, Any]) -> str:
    targets = {
        "p100": (
            {
                "strategy": "single_process",
                "backend": "none",
                "world_size": 1,
                "processes_per_node": 1,
            },
            {
                "accelerator": "cuda",
                "visible_gpus": 1,
                "gpu_name_contains": "P100",
                "compute_capability": [6, 0],
                "minimum_memory_gib": 11,
                "torch_cuda": "11.8",
                "required_arch": "sm_60",
            },
        ),
        "l40s": (
            {
                "strategy": "ddp",
                "backend": "nccl",
                "world_size": 2,
                "processes_per_node": 2,
            },
            {
                "accelerator": "cuda",
                "visible_gpus": 2,
                "gpu_name_contains": "L40S",
                "compute_capability": [8, 9],
                "minimum_memory_gib": 44,
                "torch_cuda": "11.8",
                "required_arch": "sm_89",
            },
        ),
    }
    for name, (expected_execution, expected_hardware) in targets.items():
        if _matches_exact(execution, expected_execution) and _matches_exact(
            hardware, expected_hardware
        ):
            return name
    raise TrainingConfigError("execution and hardware must target P100 or 2x L40S")


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
            and all(
                _matches_exact(value[key], nested)
                for key, nested in expected.items()
            )
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


def _positive_int(config: Dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainingConfigError(f"{key} must be a positive integer")


def _nonnegative_int(config: Dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingConfigError(f"{key} must be a non-negative integer")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
