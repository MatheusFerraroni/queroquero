from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import statistics
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .classification_data import (
    load_classification_config,
    validate_classification_dataset,
)
from .classification_embeddings import (
    load_embedding_rows,
    load_private_texts,
    require_validated_embeddings,
    verify_embedding_manifests,
)
from .classification_eval_common import (
    MODEL_NAMES,
    assert_safe_metadata,
    clean_git_commit,
    digest_strings,
    load_evaluation_config,
    load_pinned_split,
    read_json,
    validate_resolved_evaluation,
)
from .classification_probe import (
    load_private_labels,
    validate_report as validate_source_report,
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
from .manifest import file_sha256, write_json_atomic
from .model_artifact import validate_model_artifact
from .packing import tokenizer_fingerprint
from .train import CHECKPOINT_SCHEMA
from .training_config import load_training_config


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/classification/diagnostics-v2.json")
CONFIG_SCHEMA_V1 = "queroquero-cpt-diagnostics-config/v1"
CONFIG_SCHEMA_V2 = "queroquero-cpt-diagnostics-config/v2"
CONFIG_SCHEMA = CONFIG_SCHEMA_V1
RESOLVED_SCHEMA_V1 = "queroquero-resolved-cpt-diagnostics/v1"
RESOLVED_SCHEMA_V2 = "queroquero-resolved-cpt-diagnostics/v2"
COHORT_SCHEMA_V1 = "queroquero-cpt-diagnostic-cohort/v1"
COHORT_SCHEMA_V2 = "queroquero-cpt-diagnostic-cohort/v2"
COHORT_CAPACITY_SCHEMA = "queroquero-cpt-diagnostic-cohort-capacity/v1"
PREFLIGHT_SCHEMA = "queroquero-cpt-diagnostic-preflight/v1"
LOW_SHOT_UNIT_SCHEMA = "queroquero-cpt-low-shot-unit/v1"
SCORE_CHUNK_SCHEMA = "queroquero-cpt-nll-score-chunk/v1"
SCORE_MANIFEST_SCHEMA = "queroquero-cpt-nll-score-manifest/v1"
SCORE_VALIDATION_SCHEMA = "queroquero-cpt-nll-score-validation/v1"
REPORT_SCHEMA_V1 = "queroquero-cpt-diagnostics-report/v1"
REPORT_SCHEMA_V2 = "queroquero-cpt-diagnostics-report/v2"
REPORT_SCHEMA = REPORT_SCHEMA_V1
REPORT_FILES_SCHEMA = "queroquero-cpt-diagnostics-report-files/v1"
REDISTRIBUTION_STATUS = "internal_research_only"

LOW_SHOT_BUDGETS = (16, 64, 256, 1400)
SEEDS = (42, 43, 44, 45, 46)
CATEGORIES = (3, 8, 19, 23, 26, 32)
STATE_STEPS = (0, 13000, 26000, 39000, 52000, 13000, 26000, 39000, 52000)
STATE_TOKENS = (
    0,
    106_496_000,
    212_992_000,
    319_488_000,
    425_984_000,
    106_496_000,
    212_992_000,
    319_488_000,
    425_984_000,
)
STATE_NAMES = (
    "base-000000",
    "general-013000",
    "general-026000",
    "general-039000",
    "general-052000",
    "forum-013000",
    "forum-026000",
    "forum-039000",
    "forum-052000",
)
_ID20_RE = re.compile(r"[0-9a-f]{20}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def load_diagnostics_config(path: Path) -> tuple[Dict[str, Any], str]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ConfigError("diagnostics configuration must not be a symlink")
    config = load_json(requested.resolve())
    validate_diagnostics_config(config)
    return config, sha256_bytes(canonical_json_bytes(config))


def validate_diagnostics_config(config: Mapping[str, Any]) -> None:
    _keys(
        config,
        {
            "schema_version",
            "source_evaluation",
            "dataset",
            "splits",
            "models",
            "training_runs",
            "states",
            "low_shot",
            "nll",
            "statistics",
            "output",
        },
        "diagnostics configuration",
    )
    schema_version = config["schema_version"]
    if schema_version not in {CONFIG_SCHEMA_V1, CONFIG_SCHEMA_V2}:
        raise ConfigError("diagnostics schema is unsupported")

    source = _mapping(config, "source_evaluation")
    _keys(
        source,
        {"evaluation_id", "git_commit", "config_sha256", "report_sha256", "relative_path"},
        "diagnostics source evaluation",
    )
    _id20(source.get("evaluation_id"), "source evaluation ID")
    _commit(source.get("git_commit"), "source evaluation Git commit")
    _sha256(source.get("config_sha256"), "source evaluation config")
    _sha256(source.get("report_sha256"), "source evaluation report")
    _relative(source.get("relative_path"), "source evaluation")
    if source["relative_path"] != f"evaluations/{source['evaluation_id']}":
        raise ConfigError("diagnostics source evaluation path changed")

    dataset = _mapping(config, "dataset")
    _keys(dataset, {"classification_dataset_id", "relative_path", "root_env"}, "diagnostics dataset")
    _id20(dataset.get("classification_dataset_id"), "classification dataset ID")
    _relative(dataset.get("relative_path"), "classification dataset")
    if dataset["relative_path"] != f"adrenaline/{dataset['classification_dataset_id']}":
        raise ConfigError("diagnostics classification dataset path changed")
    if dataset.get("root_env") != "PTBR_CLASSIFICATION_ROOT":
        raise ConfigError("diagnostics dataset root changed")

    split_records = config.get("splits")
    if not isinstance(split_records, list) or len(split_records) != 10:
        raise ConfigError("diagnostics must pin ten classification splits")
    expected_matrix = {(task, seed) for task in ("coarse", "fine") for seed in SEEDS}
    actual_matrix = set()
    for record in split_records:
        _keys(record, {"benchmark_id", "relative_path", "seed", "task"}, "diagnostics split")
        _id20(record.get("benchmark_id"), "classification benchmark ID")
        _relative(record.get("relative_path"), "classification split")
        if record.get("task") not in {"coarse", "fine"} or record.get("seed") not in SEEDS:
            raise ConfigError("diagnostics split matrix changed")
        if record["relative_path"] != (
            f"splits/{record['task']}/seed-{record['seed']}/split_manifest.json"
        ):
            raise ConfigError("diagnostics split path changed")
        actual_matrix.add((record["task"], record["seed"]))
    if actual_matrix != expected_matrix:
        raise ConfigError("diagnostics split matrix is incomplete")

    models = _mapping(config, "models")
    _keys(models, {"base", "general", "forum"}, "diagnostics models")
    base = _mapping(models, "base")
    _keys(base, {"model_id", "revision"}, "diagnostics base model")
    if base != {"model_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ConfigError("diagnostics base model changed")
    for name, arm in (("general", "general"), ("forum", "forum_tech")):
        model = _mapping(models, name)
        _keys(
            model,
            {"artifact_id", "artifact_sha256", "expected_arm", "relative_path", "run_id"},
            f"diagnostics {name} model",
        )
        _id20(model.get("artifact_id"), f"diagnostics {name} artifact ID")
        _sha256(model.get("artifact_sha256"), f"diagnostics {name} artifact")
        _id20(model.get("run_id"), f"diagnostics {name} run ID")
        _relative(model.get("relative_path"), f"diagnostics {name} artifact")
        if model.get("expected_arm") != arm:
            raise ConfigError(f"diagnostics {name} arm changed")
        if model["relative_path"] != f"artifacts/{model['artifact_id']}":
            raise ConfigError(f"diagnostics {name} artifact path changed")

    runs = _mapping(config, "training_runs")
    _keys(runs, {"general", "forum"}, "diagnostics training runs")
    for name, arm in (("general", "general"), ("forum", "forum_tech")):
        run = _mapping(runs, name)
        _keys(
            run,
            {"run_id", "expected_arm", "config_sha256", "inputs_sha256", "git_commit"},
            f"diagnostics {name} run",
        )
        _id20(run.get("run_id"), f"diagnostics {name} run ID")
        _sha256(run.get("config_sha256"), f"diagnostics {name} config")
        _sha256(run.get("inputs_sha256"), f"diagnostics {name} inputs")
        _commit(run.get("git_commit"), f"diagnostics {name} Git commit")
        if run.get("expected_arm") != arm or run.get("run_id") != models[name]["run_id"]:
            raise ConfigError(f"diagnostics {name} run identity changed")

    states = config.get("states")
    if not isinstance(states, list) or len(states) != 9:
        raise ConfigError("diagnostics must define exactly nine model states")
    expected_kinds = ("huggingface", "checkpoint", "checkpoint", "checkpoint", "artifact", "checkpoint", "checkpoint", "checkpoint", "artifact")
    for index, state in enumerate(states):
        required = {
            "state_index", "state_name", "model", "arm", "kind", "optimizer_step", "training_tokens"
        }
        if state.get("kind") != "huggingface":
            required.add("relative_path")
        _keys(state, required, "diagnostics model state")
        if (
            state.get("state_index") != index
            or state.get("kind") != expected_kinds[index]
            or state.get("optimizer_step") != STATE_STEPS[index]
            or state.get("training_tokens") != STATE_TOKENS[index]
            or state.get("state_name") != STATE_NAMES[index]
        ):
            raise ConfigError("diagnostics model state schedule changed")
        if index == 0:
            if state.get("model") != "base" or state.get("arm") != "base":
                raise ConfigError("diagnostics baseline state changed")
        else:
            expected_model = "general" if index < 5 else "forum"
            expected_arm = "general" if index < 5 else "forum_tech"
            if state.get("model") != expected_model or state.get("arm") != expected_arm:
                raise ConfigError("diagnostics state arm changed")
            _relative(state.get("relative_path"), "diagnostics model state")
            expected_relative = (
                f"checkpoints/{runs[expected_model]['run_id']}/"
                f"step-{state['optimizer_step']:06d}"
                if state["kind"] == "checkpoint"
                else models[expected_model]["relative_path"]
            )
            if state["relative_path"] != expected_relative:
                raise ConfigError("diagnostics model state path changed")

    low_shot = _mapping(config, "low_shot")
    expected_low_shot = {
        "task": "coarse",
        "input_variant": "title_first_post",
        "pooling": "masked_mean",
        "c": 0.01,
        "budgets_per_class": list(LOW_SHOT_BUDGETS),
        "seeds": list(SEEDS),
        "test_examples_per_class": 300,
        "selection": "nested_sha256_seed_class_sample_id",
        "scaler": "standard_train_only",
        "solver": "lbfgs",
        "l1_ratio": 0.0,
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": 2000,
        "tolerance": 0.00001,
        "primary_budget_per_class": 64,
        "primary_metric": "macro_f1",
        "primary_contrast": "general_minus_base",
        "status": "exploratory_post_hoc",
    }
    if schema_version == CONFIG_SCHEMA_V2:
        expected_low_shot["categories"] = list(CATEGORIES)
    if dict(low_shot) != expected_low_shot:
        raise ConfigError("diagnostics low-shot policy changed")

    nll = _mapping(config, "nll")
    expected_nll = {
        "task": "coarse",
        "categories": list(CATEGORIES),
        "examples_per_category": 300,
        "selection_seed": 47,
        "selection": "one_per_title_group_sha256",
        "input": "title_double_newline_first_post",
        "target": "first_post_only",
        "max_length": 1024,
        "truncation_side": "right",
        "minimum_target_tokens": 32,
        "batch_size": 4,
        "runtime_dtype": "bfloat16",
        "loss_dtype": "float32",
        "chunk_size": 256,
        "primary_contrast": "forum_minus_general",
    }
    if schema_version == CONFIG_SCHEMA_V2:
        expected_nll["prior_split_exclusion"] = "test_only"
    if dict(nll) != expected_nll:
        raise ConfigError("diagnostics NLL policy changed")

    statistics_config = _mapping(config, "statistics")
    if dict(statistics_config) != {
        "low_shot_confidence_interval": "student_t_95_descriptive",
        "nll_confidence_interval": "paired_stratified_percentile_bootstrap_95",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20260829,
        "report_p_values": False,
    }:
        raise ConfigError("diagnostics statistical policy changed")
    output = _mapping(config, "output")
    if dict(output) != {
        "root_env": "PTBR_CLASSIFICATION_ROOT",
        "relative_path": "diagnostics",
        "status": REDISTRIBUTION_STATUS,
    }:
        raise ConfigError("diagnostics output policy changed")


def state_by_index(config: Mapping[str, Any], index: int) -> Dict[str, Any]:
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(config["states"])
    ):
        raise ValueError("diagnostics state index must be between 0 and 8")
    return dict(config["states"][index])


def low_shot_seed(config: Mapping[str, Any], index: int) -> int:
    seeds = tuple(int(value) for value in config["low_shot"]["seeds"])
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(seeds)
    ):
        raise ValueError("low-shot unit index must be between 0 and 4")
    return seeds[index]


def _prior_split_exclusion(config: Mapping[str, Any]) -> str:
    if config["schema_version"] == CONFIG_SCHEMA_V1:
        return "all_partitions"
    return str(config["nll"]["prior_split_exclusion"])


def _resolved_schema(config: Mapping[str, Any]) -> str:
    return (
        RESOLVED_SCHEMA_V1
        if config["schema_version"] == CONFIG_SCHEMA_V1
        else RESOLVED_SCHEMA_V2
    )


def _cohort_schema(config: Mapping[str, Any]) -> str:
    return (
        COHORT_SCHEMA_V1
        if config["schema_version"] == CONFIG_SCHEMA_V1
        else COHORT_SCHEMA_V2
    )


def _report_schema(config: Mapping[str, Any]) -> str:
    return (
        REPORT_SCHEMA_V1
        if config["schema_version"] == CONFIG_SCHEMA_V1
        else REPORT_SCHEMA_V2
    )


def _low_shot_categories(config: Mapping[str, Any]) -> tuple[int, ...]:
    values = config["low_shot"].get("categories", CATEGORIES)
    return tuple(int(value) for value in values)


def _low_shot_budgets(config: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["low_shot"]["budgets_per_class"])


def _low_shot_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["low_shot"]["seeds"])


def _low_shot_fits(config: Mapping[str, Any]) -> int:
    return (
        len(MODEL_NAMES)
        * len(_low_shot_budgets(config))
        * len(_low_shot_seeds(config))
    )


def _nll_examples_per_state(config: Mapping[str, Any]) -> int:
    return len(config["nll"]["categories"]) * int(
        config["nll"]["examples_per_category"]
    )


def _nll_score_total(config: Mapping[str, Any]) -> int:
    return len(config["states"]) * _nll_examples_per_state(config)


def _low_shot_test_examples(config: Mapping[str, Any]) -> int:
    return len(_low_shot_categories(config)) * int(
        config["low_shot"]["test_examples_per_class"]
    )


def _cohort_report_metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = {
        "threads": _nll_examples_per_state(config),
        "categories": len(config["nll"]["categories"]),
        "threads_per_category": config["nll"]["examples_per_category"],
    }
    if config["schema_version"] == CONFIG_SCHEMA_V1:
        value["fresh_against_prior_splits"] = True
    else:
        value.update(
            {
                "novelty_definition": "not_used_in_any_prior_test_partition",
                "prior_split_exclusion": _prior_split_exclusion(config),
            }
        )
    return value


def first_post_target_mask(
    offsets: Sequence[Sequence[int]],
    special_tokens_mask: Sequence[int],
    attention_mask: Sequence[int],
    boundary: int,
) -> list[bool]:
    if not (len(offsets) == len(special_tokens_mask) == len(attention_mask)):
        raise ValueError("token mask inputs have different lengths")
    result = []
    for offset, special, attended in zip(offsets, special_tokens_mask, attention_mask):
        if len(offset) != 2:
            raise ValueError("token offset must contain start and end")
        start, end = int(offset[0]), int(offset[1])
        result.append(
            bool(attended)
            and not bool(special)
            and end > start
            and start >= boundary
        )
    return result


def nested_low_shot_ids(
    train_ids: Sequence[str],
    labels: Sequence[str],
    *,
    seed: int,
    budgets: Sequence[int] = LOW_SHOT_BUDGETS,
) -> Dict[int, list[str]]:
    if len(train_ids) != len(labels) or len(set(train_ids)) != len(train_ids):
        raise RuntimeError("low-shot training IDs are invalid")
    grouped: Dict[str, list[str]] = defaultdict(list)
    for sample_id, label in zip(train_ids, labels):
        grouped[str(label)].append(sample_id)
    maximum = max(budgets)
    ordered: Dict[str, list[str]] = {}
    for label, values in grouped.items():
        if len(values) < maximum:
            raise RuntimeError("low-shot class has insufficient training examples")
        ordered[label] = sorted(
            values,
            key=lambda sample_id: hashlib.sha256(
                f"low-shot-v1:{seed}:{label}:{sample_id}".encode("ascii")
            ).hexdigest(),
        )
    result: Dict[int, list[str]] = {}
    for budget in budgets:
        selected = [sample_id for label in sorted(ordered) for sample_id in ordered[label][:budget]]
        result[int(budget)] = selected
    return result


def _keys(value: Any, expected: set[str], description: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConfigError(f"{description} keys are incomplete or unknown")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ConfigError(f"diagnostics {key} must be an object")
    return nested


def _id20(value: Any, description: str) -> None:
    if not isinstance(value, str) or not _ID20_RE.fullmatch(value):
        raise ConfigError(f"{description} is invalid")


def _sha256(value: Any, description: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ConfigError(f"{description} is invalid")


def _commit(value: Any, description: str) -> None:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ConfigError(f"{description} is invalid")


def _relative(value: Any, description: str) -> None:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ConfigError(f"{description} path is unsafe")


def _absolute_root(env_name: str) -> Path:
    raw = os.environ.get(env_name)
    if not raw or not raw.strip():
        raise ConfigError(f"{env_name} is required")
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise ConfigError(f"{env_name} must be an absolute non-symlink path")
    root = requested.resolve()
    if root == Path("/") or len(root.parts) < 3:
        raise ConfigError(f"{env_name} is unsafe")
    return root


def _safe_join(root: Path, relative: str, description: str, *, create: bool = False) -> Path:
    _relative(relative, description)
    normalized_root = root.resolve()
    requested = normalized_root
    for part in Path(relative).parts:
        requested = requested / part
        if requested.is_symlink():
            raise RuntimeError(f"{description} must not traverse a symlink")
    target = requested.resolve()
    if normalized_root not in target.parents:
        raise RuntimeError(f"{description} escapes its root")
    if target.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink")
    if create:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.chmod(0o700)
    return target


def _project_path(relative: str, description: str) -> Path:
    return _safe_join(PROJECT_ROOT, relative, description)


def _classification_root(config: Mapping[str, Any]) -> Path:
    return _absolute_root(config["dataset"]["root_env"])


def _dataset_path(config: Mapping[str, Any]) -> Path:
    return _safe_join(_classification_root(config), config["dataset"]["relative_path"], "diagnostics dataset")


def _source_evaluation_path(config: Mapping[str, Any]) -> Path:
    return _safe_join(
        _classification_root(config),
        config["source_evaluation"]["relative_path"],
        "diagnostics source evaluation",
    )


def _output_root(config: Mapping[str, Any], *, create: bool = False) -> Path:
    return _safe_join(
        _classification_root(config),
        config["output"]["relative_path"],
        "diagnostics output",
        create=create,
    )


def _private_artifact_path(output: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError(f"{description} path is invalid")
    _relative(relative, description)
    root = output.resolve()
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{description} must not traverse a symlink")
    target = current.resolve()
    if root not in target.parents:
        raise RuntimeError(f"{description} escapes its root")
    return target


def _output_subdirectory(output: Path, relative: str, description: str) -> Path:
    _relative(relative, description)
    root = output.resolve()
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{description} must not traverse a symlink")
        current.mkdir(exist_ok=True, mode=0o700)
        if not current.is_dir():
            raise RuntimeError(f"{description} is not a directory")
        current.chmod(0o700)
    target = current.resolve()
    if root not in target.parents:
        raise RuntimeError(f"{description} escapes its root")
    return target


def _read_source_evaluation(
    config: Mapping[str, Any],
    *,
    full_validation: bool,
    verify_embeddings: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any], Path, Dict[str, Any]]:
    source_path = _source_evaluation_path(config)
    if source_path.is_symlink() or not source_path.is_dir():
        raise RuntimeError("diagnostics source evaluation is missing or unsafe")
    source_config_path = PROJECT_ROOT / "configs/classification/evaluation-v1.json"
    source_config, source_config_sha256 = load_evaluation_config(source_config_path)
    expected = config["source_evaluation"]
    if source_config_sha256 != expected["config_sha256"]:
        raise RuntimeError("diagnostics source evaluation config changed")
    resolved_path = source_path / "resolved_evaluation.json"
    resolved = read_json(resolved_path, "source resolved evaluation")
    validate_resolved_evaluation(resolved, source_config)
    if (
        resolved.get("evaluation_id") != expected["evaluation_id"]
        or resolved.get("git_commit") != expected["git_commit"]
        or resolved.get("config_sha256") != expected["config_sha256"]
        or resolved.get("classification_dataset_id")
        != config["dataset"]["classification_dataset_id"]
    ):
        raise RuntimeError("diagnostics source evaluation identity changed")
    preflight = read_json(source_path / "preflight.json", "source evaluation preflight")
    if preflight.get("evaluation_id") != expected["evaluation_id"] or preflight.get("status") != "ok":
        raise RuntimeError("diagnostics source evaluation preflight is invalid")
    recorded_embedding_validation = require_validated_embeddings(resolved, source_path)
    verified_embeddings = (
        verify_embedding_manifests(source_config, resolved, source_path)
        if verify_embeddings
        else recorded_embedding_validation
    )
    if verified_embeddings != recorded_embedding_validation:
        raise RuntimeError("diagnostics source embedding validation changed")
    report = read_json(source_path / "report/report.json", "source evaluation report")
    if report.get("report_sha256") != expected["report_sha256"] or report.get("status") != "complete":
        raise RuntimeError("diagnostics source evaluation report changed")
    if full_validation:
        result = validate_source_report(source_config, resolved, source_path)
        if result.get("report_sha256") != expected["report_sha256"] or result.get("status") != "valid":
            raise RuntimeError("diagnostics source evaluation report is invalid")
    embedding_manifests = {
        model: file_sha256(source_path / "embeddings" / model / "embedding_manifest.json")
        for model in MODEL_NAMES
    }
    fingerprints = {
        "resolved_evaluation_sha256": file_sha256(resolved_path),
        "preflight_sha256": file_sha256(source_path / "preflight.json"),
        "report_file_sha256": file_sha256(source_path / "report/report.json"),
        "report_files_sha256": file_sha256(source_path / "report/report_files.json"),
        "embedding_manifests": embedding_manifests,
        "embedding_validation_sha256": sha256_bytes(
            canonical_json_bytes(verified_embeddings)
        ),
    }
    return source_config, resolved, source_path, fingerprints


def _validate_cpt_exclusion_contract(
    dataset_path: Path,
    dataset_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    classification_config, classification_config_sha256 = load_classification_config(
        PROJECT_ROOT / "configs/classification/adrenaline-v1.json"
    )
    expected_cpt = classification_config["source"]["cpt_manifest"]
    source_cpt = dataset_manifest.get("source", {}).get("cpt_manifest", {})
    if (
        dataset_manifest.get("config_sha256") != classification_config_sha256
        or source_cpt.get("preparation_id") != expected_cpt["preparation_id"]
        or source_cpt.get("sha256") != expected_cpt["sha256"]
        or dataset_manifest.get("eligibility", {}).get("cpt_overlap")
        != "exclude_train_and_eval"
    ):
        raise RuntimeError("diagnostics classification CPT exclusion contract changed")
    audit = read_json(dataset_path / "audit.json", "classification dataset audit")
    source_counts = audit.get("source_counts", {})
    required_counts = (
        "cpt_train_source_hashes",
        "cpt_eval_source_hashes",
        "cpt_train_threads",
        "cpt_eval_threads",
        "cpt_overlap_threads_union",
    )
    if (
        audit.get("policy", {}).get("cpt_overlap") != "exclude_train_and_eval"
        or any(
            not isinstance(source_counts.get(key), int)
            or isinstance(source_counts.get(key), bool)
            or source_counts[key] < 1
            for key in required_counts
        )
    ):
        raise RuntimeError("diagnostics classification CPT exclusion audit changed")
    return {
        "preparation_id": source_cpt["preparation_id"],
        "manifest_sha256": source_cpt["sha256"],
        "audit_sha256": file_sha256(dataset_path / "audit.json"),
        "policy": "exclude_train_and_eval",
    }


def validate_checkpoint_model_for_inference(
    path: Path,
    run: Mapping[str, Any],
    expected_step: int,
    *,
    verify_model_hashes: bool,
) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("diagnostics checkpoint must be a real directory")
    manifest_path = path / "checkpoint_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("diagnostics checkpoint manifest must not be a symlink")
    manifest = read_json(manifest_path, "diagnostics checkpoint manifest")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or manifest.get("checkpoint_id") != f"step-{expected_step:06d}"
        or path.name != manifest.get("checkpoint_id")
        or manifest.get("run_id") != run["run_id"]
        or manifest.get("optimizer_step") != expected_step
        or manifest.get("sequences_consumed") != expected_step * 8
        or manifest.get("world_size") != 2
        or manifest.get("global_batch_sequences") != 8
        or manifest.get("config_sha256") != run["config_sha256"]
        or manifest.get("inputs_sha256") != run["inputs_sha256"]
        or manifest.get("git_commit") != run["git_commit"]
    ):
        raise RuntimeError("diagnostics checkpoint identity changed")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("diagnostics checkpoint file inventory is missing")
    if manifest.get("files_sha256") != sha256_bytes(canonical_json_bytes(records)):
        raise RuntimeError("diagnostics checkpoint inventory digest changed")
    listed: set[str] = set()
    model_records = []
    has_state = False
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "size_bytes", "sha256"}:
            raise RuntimeError("diagnostics checkpoint file record is invalid")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in listed:
            raise RuntimeError("diagnostics checkpoint file path is unsafe")
        listed.add(relative.as_posix())
        _sha256_runtime(record.get("sha256"), "diagnostics checkpoint file hash")
        if not isinstance(record.get("size_bytes"), int) or isinstance(record.get("size_bytes"), bool) or record["size_bytes"] < 1:
            raise RuntimeError("diagnostics checkpoint file size is invalid")
        candidate = path / relative
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size != record["size_bytes"]:
            raise RuntimeError("diagnostics checkpoint file is missing or changed")
        if relative.parts[0] == "model":
            model_records.append(dict(record))
            if verify_model_hashes and file_sha256(candidate) != record["sha256"]:
                raise RuntimeError("diagnostics checkpoint model file hash changed")
        elif relative.as_posix() == "training_state.pt":
            has_state = True
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "checkpoint_manifest.json"
    }
    if actual != listed or any(item.is_symlink() for item in path.rglob("*")):
        raise RuntimeError("diagnostics checkpoint contains unexpected files")
    model_names = {record["path"] for record in model_records}
    if "model/config.json" not in model_names or not any(name.endswith(".safetensors") for name in model_names) or not has_state:
        raise RuntimeError("diagnostics checkpoint inference files are incomplete")
    model_digest = sha256_bytes(canonical_json_bytes(sorted(model_records, key=lambda value: value["path"])))
    return {
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "model_files_sha256": model_digest,
        "model_files": len(model_records),
    }


def _sha256_runtime(value: Any, description: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RuntimeError(f"{description} is invalid")


def _state_identity(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    verify_model_hashes: bool,
) -> Dict[str, Any]:
    record = {
        "state_index": state["state_index"],
        "state_name": state["state_name"],
        "model": state["model"],
        "arm": state["arm"],
        "kind": state["kind"],
        "optimizer_step": state["optimizer_step"],
        "training_tokens": state["training_tokens"],
    }
    if state["kind"] == "huggingface":
        record.update(config["models"]["base"])
        return record
    path = _project_path(state["relative_path"], "diagnostics model state")
    record["relative_path"] = state["relative_path"]
    if state["kind"] == "checkpoint":
        run = config["training_runs"][state["model"]]
        record.update(
            validate_checkpoint_model_for_inference(
                path,
                run,
                state["optimizer_step"],
                verify_model_hashes=verify_model_hashes,
            )
        )
        record["run_id"] = run["run_id"]
        return record
    expected = config["models"][state["model"]]
    if verify_model_hashes:
        manifest = validate_model_artifact(path, load_model=False)
    else:
        manifest = read_json(path / "model_artifact_manifest.json", "diagnostics model artifact manifest")
    training = manifest.get("training", {})
    if (
        manifest.get("artifact_id") != expected["artifact_id"]
        or manifest.get("artifact_sha256") != expected["artifact_sha256"]
        or training.get("run_id") != expected["run_id"]
        or training.get("optimizer_steps") != state["optimizer_step"]
        or training.get("experiment", {}).get("arm") != expected["expected_arm"]
    ):
        raise RuntimeError("diagnostics final artifact identity changed")
    record.update(
        {
            "artifact_id": manifest["artifact_id"],
            "artifact_sha256": manifest["artifact_sha256"],
            "artifact_manifest_sha256": file_sha256(path / "model_artifact_manifest.json"),
            "run_id": training["run_id"],
        }
    )
    return record


def _run_identity(config: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    expected = config["training_runs"][model_name]
    training_config_path = PROJECT_ROOT / "configs/training" / (
        "l40s-real-general.json"
        if model_name == "general"
        else "l40s-real-forum-tech.json"
    )
    training_config, training_config_sha256 = load_training_config(
        training_config_path
    )
    if (
        training_config_sha256 != expected["config_sha256"]
        or training_config.get("data_mixture", {}).get("arm")
        != expected["expected_arm"]
    ):
        raise RuntimeError("diagnostics versioned training configuration changed")
    run_dir = _project_path(f"runs/{expected['run_id']}", "diagnostics training run")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("diagnostics training run is missing or unsafe")
    resolved_path = run_dir / "resolved_training.json"
    manifest_path = run_dir / "run_manifest.json"
    resolved = read_json(resolved_path, "diagnostics resolved training")
    manifest = read_json(manifest_path, "diagnostics run manifest")
    if (
        resolved.get("run_id") != expected["run_id"]
        or resolved.get("config_sha256") != expected["config_sha256"]
        or resolved.get("inputs_sha256") != expected["inputs_sha256"]
        or resolved.get("git_commit") != expected["git_commit"]
        or resolved.get("profile") != "real"
        or resolved.get("training", {}).get("total_optimizer_steps") != 52_000
        or resolved.get("execution", {}).get("world_size") != 2
        or resolved.get("training", {}).get("global_batch_sequences") != 8
    ):
        raise RuntimeError("diagnostics resolved training identity changed")
    manifest_experiment = manifest.get("experiment", {})
    if (
        manifest.get("run_id") != expected["run_id"]
        or manifest.get("status") != "complete"
        or manifest.get("optimizer_steps_completed") != 52_000
        or manifest_experiment.get("arm") != expected["expected_arm"]
    ):
        raise RuntimeError("diagnostics completed run manifest changed")
    return {
        "run_id": expected["run_id"],
        "arm": expected["expected_arm"],
        "config_sha256": expected["config_sha256"],
        "inputs_sha256": expected["inputs_sha256"],
        "git_commit": expected["git_commit"],
        "training_config_sha256": training_config_sha256,
        "training_arm": training_config["data_mixture"]["arm"],
        "resolved_training_sha256": file_sha256(resolved_path),
        "run_manifest_sha256": file_sha256(manifest_path),
    }


def initialize_diagnostics(config_path: Path) -> tuple[Dict[str, Any], Path]:
    config, config_sha256 = load_diagnostics_config(config_path)
    try:
        return locate_diagnostics(config_path)
    except RuntimeError as exc:
        if "expected exactly one diagnostics run" not in str(exc):
            raise
    git_commit = clean_git_commit()
    dataset_path = _dataset_path(config)
    dataset_manifest = validate_classification_dataset(dataset_path)
    if dataset_manifest.get("classification_dataset_id") != config["dataset"]["classification_dataset_id"]:
        raise RuntimeError("diagnostics classification dataset identity changed")
    cpt_exclusion = _validate_cpt_exclusion_contract(dataset_path, dataset_manifest)
    _, source_resolved, _, source_files = _read_source_evaluation(
        config, full_validation=True
    )
    split_records = []
    classification_config = PROJECT_ROOT / "configs/classification/adrenaline-v1.json"
    for split in config["splits"]:
        split_path = _safe_join(_classification_root(config), split["relative_path"], "diagnostics split")
        manifest = validate_classification_split(classification_config, dataset_path, split_path)
        if manifest.get("benchmark_id") != split["benchmark_id"]:
            raise RuntimeError("diagnostics classification split identity changed")
        split_records.append(
            {
                "task": split["task"],
                "seed": split["seed"],
                "benchmark_id": split["benchmark_id"],
                "sha256": file_sha256(split_path),
            }
        )
    runs = {name: _run_identity(config, name) for name in ("general", "forum")}
    states = [
        _state_identity(config, state, verify_model_hashes=False)
        for state in config["states"]
    ]
    final_manifests = {}
    for name in ("general", "forum"):
        path = _project_path(config["models"][name]["relative_path"], f"diagnostics {name} artifact")
        manifest = validate_model_artifact(path, load_model=False)
        final_manifests[name] = {
            "artifact_id": manifest["artifact_id"],
            "artifact_sha256": manifest["artifact_sha256"],
            "artifact_manifest_sha256": file_sha256(path / "model_artifact_manifest.json"),
            "run_id": manifest["training"]["run_id"],
        }
    identity = {
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "source_evaluation": {
            "evaluation_id": source_resolved["evaluation_id"],
            "git_commit": source_resolved["git_commit"],
            "config_sha256": source_resolved["config_sha256"],
            "report_sha256": config["source_evaluation"]["report_sha256"],
            **source_files,
        },
        "classification_dataset_id": dataset_manifest["classification_dataset_id"],
        "dataset_manifest_sha256": file_sha256(dataset_path / "dataset_manifest.json"),
        "cpt_exclusion": cpt_exclusion,
        "splits": sorted(split_records, key=lambda value: (value["seed"], value["task"])),
        "models": {
            "base": dict(config["models"]["base"]),
            "general": final_manifests["general"],
            "forum": final_manifests["forum"],
        },
        "runs": runs,
        "states": states,
    }
    diagnostic_id = sha256_bytes(canonical_json_bytes(identity))[:20]
    resolved = {
        "schema_version": _resolved_schema(config),
        "diagnostic_id": diagnostic_id,
        **identity,
        "low_shot": dict(config["low_shot"]),
        "nll": dict(config["nll"]),
        "statistics": dict(config["statistics"]),
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    assert_safe_metadata(resolved)
    root = _output_root(config, create=True)
    output = _safe_join(root, diagnostic_id, "diagnostics run", create=True)
    resolved_path = output / "resolved_diagnostics.json"
    if resolved_path.exists():
        if read_json(resolved_path, "resolved diagnostics") != resolved:
            raise RuntimeError("existing resolved diagnostics changed")
    else:
        write_json_atomic(resolved_path, resolved)
        resolved_path.chmod(0o600)
    return resolved, output


def locate_diagnostics(config_path: Path) -> tuple[Dict[str, Any], Path]:
    config, config_sha256 = load_diagnostics_config(config_path)
    git_commit = clean_git_commit()
    root = _output_root(config, create=True)
    matches = []
    for child in root.iterdir():
        if child.is_symlink() or not child.is_dir() or not _ID20_RE.fullmatch(child.name):
            continue
        path = child / "resolved_diagnostics.json"
        if not path.is_file() or path.is_symlink():
            continue
        value = read_json(path, "resolved diagnostics")
        if value.get("config_sha256") == config_sha256 and value.get("git_commit") == git_commit:
            matches.append((value, child))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one diagnostics run for this commit; found {len(matches)}")
    resolved, output = matches[0]
    _validate_resolved_diagnostics(resolved, config, output)
    return resolved, output


def _validate_resolved_diagnostics(
    resolved: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
) -> None:
    required = {
        "schema_version", "diagnostic_id", "config_sha256", "git_commit",
        "source_evaluation", "classification_dataset_id", "dataset_manifest_sha256",
        "cpt_exclusion", "splits", "models", "runs", "states", "low_shot", "nll", "statistics",
        "redistribution_status",
    }
    if (
        set(resolved) != required
        or resolved.get("schema_version") != _resolved_schema(config)
        or not _ID20_RE.fullmatch(str(resolved.get("diagnostic_id", "")))
        or output.name != resolved.get("diagnostic_id")
        or resolved.get("low_shot") != config["low_shot"]
        or resolved.get("nll") != config["nll"]
        or resolved.get("statistics") != config["statistics"]
        or resolved.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise RuntimeError("resolved diagnostics are invalid")
    identity_keys = {
        "config_sha256", "git_commit", "source_evaluation", "classification_dataset_id",
        "dataset_manifest_sha256", "cpt_exclusion", "splits", "models", "runs", "states",
    }
    identity = {key: resolved[key] for key in identity_keys}
    if sha256_bytes(canonical_json_bytes(identity))[:20] != resolved["diagnostic_id"]:
        raise RuntimeError("resolved diagnostics identity changed")
    dataset_path = _dataset_path(config)
    if file_sha256(dataset_path / "dataset_manifest.json") != resolved["dataset_manifest_sha256"]:
        raise RuntimeError("diagnostics dataset changed")
    dataset_manifest = read_json(
        dataset_path / "dataset_manifest.json",
        "diagnostics classification dataset manifest",
    )
    if _validate_cpt_exclusion_contract(dataset_path, dataset_manifest) != resolved["cpt_exclusion"]:
        raise RuntimeError("diagnostics classification CPT exclusion changed")
    _, source_resolved, _, source_files = _read_source_evaluation(
        config, full_validation=False
    )
    if source_resolved["evaluation_id"] != resolved["source_evaluation"]["evaluation_id"]:
        raise RuntimeError("diagnostics source evaluation changed")
    for key, value in source_files.items():
        if resolved["source_evaluation"].get(key) != value:
            raise RuntimeError("diagnostics source evaluation files changed")
    split_lookup = {(item["task"], item["seed"]): item for item in resolved["splits"]}
    for split in config["splits"]:
        path = _safe_join(_classification_root(config), split["relative_path"], "diagnostics split")
        expected = split_lookup[(split["task"], split["seed"])]
        if file_sha256(path) != expected["sha256"]:
            raise RuntimeError("diagnostics split changed")
    for name in ("general", "forum"):
        if _run_identity(config, name) != resolved["runs"][name]:
            raise RuntimeError("diagnostics training run changed")
    for state, stored in zip(config["states"], resolved["states"]):
        current = _state_identity(config, state, verify_model_hashes=False)
        if current != stored:
            raise RuntimeError("diagnostics model state changed")
    assert_safe_metadata(resolved)


def _excluded_title_groups(config: Mapping[str, Any], dataset_path: Path) -> set[str]:
    import pyarrow.parquet as pq

    policy = _prior_split_exclusion(config)
    selected_partitions = (
        ("train", "validation", "test")
        if policy == "all_partitions"
        else ("test",)
    )
    excluded_ids: set[str] = set()
    for split in config["splits"]:
        path = _safe_join(_classification_root(config), split["relative_path"], "diagnostics split")
        manifest = read_json(path, "diagnostics split")
        if manifest.get("benchmark_id") != split["benchmark_id"]:
            raise RuntimeError("diagnostics split benchmark changed")
        sample_ids = manifest.get("sample_ids")
        if not isinstance(sample_ids, Mapping) or set(sample_ids) != {"train", "validation", "test"}:
            raise RuntimeError("diagnostics split sample IDs are invalid")
        for values in sample_ids.values():
            if not isinstance(values, list):
                raise RuntimeError("diagnostics split sample IDs are invalid")
        for partition in selected_partitions:
            values = sample_ids[partition]
            excluded_ids.update(values)
    found_ids: set[str] = set()
    groups: set[str] = set()
    parquet = pq.ParquetFile(dataset_path / "examples.parquet")
    for batch in parquet.iter_batches(batch_size=16_384, columns=["sample_id", "title_group_id"]):
        values = batch.to_pydict()
        for sample_id, group_id in zip(values["sample_id"], values["title_group_id"]):
            if sample_id in excluded_ids:
                found_ids.add(sample_id)
                groups.add(group_id)
    if found_ids != excluded_ids:
        raise RuntimeError("diagnostics split IDs cannot be resolved to title groups")
    return groups


def _load_nll_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"]["model_id"],
        revision=config["models"]["base"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("diagnostics NLL requires the pinned fast tokenizer")
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise RuntimeError("diagnostics tokenizer has no padding token")
    return tokenizer


def _target_counts_for_texts(
    tokenizer: Any,
    texts: Sequence[tuple[str, str]],
    *,
    max_length: int,
) -> list[int]:
    if not texts:
        return []
    combined = [f"{title}\n\n{post}" for title, post in texts]
    boundaries = [len(f"{title}\n\n") for title, _ in texts]
    encoded = tokenizer(
        combined,
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
    )
    counts = []
    for offsets, special, attention, boundary in zip(
        encoded["offset_mapping"],
        encoded["special_tokens_mask"],
        encoded["attention_mask"],
        boundaries,
    ):
        counts.append(sum(first_post_target_mask(offsets, special, attention, boundary)[1:]))
    return counts


def _cohort_eligible_rows(
    config: Mapping[str, Any], dataset_path: Path
) -> tuple[Dict[int, list[Dict[str, Any]]], Dict[str, Any]]:
    import pyarrow.parquet as pq

    excluded_groups = _excluded_title_groups(config, dataset_path)
    selection_seed = config["nll"]["selection_seed"]
    representatives: Dict[str, Dict[str, Any]] = {}
    group_labels: Dict[str, set[int]] = defaultdict(set)
    parquet = pq.ParquetFile(dataset_path / "examples.parquet")
    columns = ["sample_id", "title", "first_post", "category_id", "title_group_id"]
    for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
        values = batch.to_pydict()
        for row in zip(*(values[column] for column in columns)):
            record = dict(zip(columns, row))
            category = int(record["category_id"])
            group_id = record["title_group_id"]
            if category not in CATEGORIES or group_id in excluded_groups:
                continue
            group_labels[group_id].add(category)
            rank = hashlib.sha256(
                (
                    f"cohort-representative-v1:{selection_seed}:"
                    f"{group_id}:{record['sample_id']}"
                ).encode("ascii")
            ).hexdigest()
            previous = representatives.get(group_id)
            if previous is None or rank < previous["representative_rank"]:
                representatives[group_id] = {**record, "representative_rank": rank}

    candidates = [
        value
        for group_id, value in representatives.items()
        if len(group_labels[group_id]) == 1
    ]
    candidate_counts = {
        str(category): sum(
            int(value["category_id"]) == category for value in candidates
        )
        for category in CATEGORIES
    }
    tokenizer = _load_nll_tokenizer(config)
    eligible: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    for start in range(0, len(candidates), 256):
        batch = candidates[start : start + 256]
        counts = _target_counts_for_texts(
            tokenizer,
            [(value["title"], value["first_post"]) for value in batch],
            max_length=config["nll"]["max_length"],
        )
        for value, count in zip(batch, counts):
            if count < config["nll"]["minimum_target_tokens"]:
                continue
            category = int(value["category_id"])
            ranked = dict(value)
            ranked["target_tokens"] = count
            ranked["selection_hash"] = hashlib.sha256(
                (
                    f"cohort-selection-v1:{selection_seed}:{category}:"
                    f"{value['title_group_id']}:{value['sample_id']}"
                ).encode("ascii")
            ).hexdigest()
            eligible[category].append(ranked)

    eligible_counts = {
        str(category): len(eligible[category]) for category in CATEGORIES
    }
    return eligible, {
        "prior_split_exclusion": _prior_split_exclusion(config),
        "excluded_prior_title_groups": len(excluded_groups),
        "after_prior_split_exclusion_and_deduplication": candidate_counts,
        "after_minimum_target_tokens": eligible_counts,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint(tokenizer),
    }


def _expected_cohort_rows(
    config: Mapping[str, Any], dataset_path: Path
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    eligible, metadata = _cohort_eligible_rows(config, dataset_path)

    selected = []
    expected_per_category = config["nll"]["examples_per_category"]
    insufficient = {
        str(category): len(eligible[category])
        for category in CATEGORIES
        if len(eligible[category]) < expected_per_category
    }
    if insufficient:
        raise RuntimeError(
            "diagnostics NLL capacity is insufficient: "
            + json.dumps(insufficient, sort_keys=True)
        )
    for category in CATEGORIES:
        values = sorted(eligible[category], key=lambda value: value["selection_hash"])
        for rank, value in enumerate(values[:expected_per_category]):
            selected.append(
                {
                    "sample_id": value["sample_id"],
                    "title_group_id": value["title_group_id"],
                    "category_id": category,
                    "target_tokens": value["target_tokens"],
                    "selection_rank": rank,
                }
            )
    selected.sort(key=lambda value: (value["category_id"], value["selection_rank"]))
    return selected, metadata


def _capacity_payload(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    required = int(config["nll"]["examples_per_category"])
    eligible = dict(metadata["after_minimum_target_tokens"])
    sufficient = all(
        eligible.get(str(category), -1) >= required for category in CATEGORIES
    )
    value = {
        "schema_version": COHORT_CAPACITY_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "config_sha256": resolved["config_sha256"],
        "classification_dataset_id": resolved["classification_dataset_id"],
        "dataset_manifest_sha256": resolved["dataset_manifest_sha256"],
        "split_manifests_sha256": sha256_bytes(
            canonical_json_bytes(resolved["splits"])
        ),
        "prior_split_exclusion": metadata["prior_split_exclusion"],
        "categories": list(CATEGORIES),
        "required_per_category": required,
        "expected_examples": _nll_examples_per_state(config),
        "excluded_prior_title_groups": metadata[
            "excluded_prior_title_groups"
        ],
        "after_prior_split_exclusion_and_deduplication": dict(
            metadata["after_prior_split_exclusion_and_deduplication"]
        ),
        "after_minimum_target_tokens": eligible,
        "minimum_target_tokens": int(config["nll"]["minimum_target_tokens"]),
        "tokenizer_fingerprint_sha256": metadata[
            "tokenizer_fingerprint_sha256"
        ],
        "redistribution_status": REDISTRIBUTION_STATUS,
        "status": "sufficient" if sufficient else "insufficient",
    }
    assert_safe_metadata(value)
    return value


def _validate_capacity_report(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    value: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "diagnostic_id",
        "config_sha256",
        "classification_dataset_id",
        "dataset_manifest_sha256",
        "split_manifests_sha256",
        "prior_split_exclusion",
        "categories",
        "required_per_category",
        "expected_examples",
        "excluded_prior_title_groups",
        "after_prior_split_exclusion_and_deduplication",
        "after_minimum_target_tokens",
        "minimum_target_tokens",
        "tokenizer_fingerprint_sha256",
        "redistribution_status",
        "status",
    }
    category_keys = {str(category) for category in CATEGORIES}
    before = value.get("after_prior_split_exclusion_and_deduplication")
    after = value.get("after_minimum_target_tokens")
    required = int(config["nll"]["examples_per_category"])
    if (
        set(value) != expected_keys
        or value.get("schema_version") != COHORT_CAPACITY_SCHEMA
        or value.get("diagnostic_id") != resolved["diagnostic_id"]
        or value.get("config_sha256") != resolved["config_sha256"]
        or value.get("classification_dataset_id")
        != resolved["classification_dataset_id"]
        or value.get("dataset_manifest_sha256")
        != resolved["dataset_manifest_sha256"]
        or value.get("split_manifests_sha256")
        != sha256_bytes(canonical_json_bytes(resolved["splits"]))
        or value.get("prior_split_exclusion")
        != _prior_split_exclusion(config)
        or value.get("categories") != list(CATEGORIES)
        or value.get("required_per_category") != required
        or value.get("expected_examples") != _nll_examples_per_state(config)
        or value.get("minimum_target_tokens")
        != config["nll"]["minimum_target_tokens"]
        or not isinstance(value.get("excluded_prior_title_groups"), int)
        or value["excluded_prior_title_groups"] < 0
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or set(before) != category_keys
        or set(after) != category_keys
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in [*before.values(), *after.values()]
        )
        or any(after[key] > before[key] for key in category_keys)
        or value.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise RuntimeError("diagnostics cohort capacity report is invalid")
    _sha256_runtime(
        value.get("tokenizer_fingerprint_sha256"),
        "diagnostics cohort tokenizer fingerprint",
    )
    expected_status = (
        "sufficient"
        if all(after[str(category)] >= required for category in CATEGORIES)
        else "insufficient"
    )
    if value.get("status") != expected_status:
        raise RuntimeError("diagnostics cohort capacity status changed")
    assert_safe_metadata(value)


def audit_cohort(config_path: Path) -> Dict[str, Any]:
    config, _ = load_diagnostics_config(config_path)
    resolved, output = initialize_diagnostics(config_path)
    target = output / "cohort_capacity.json"
    if target.is_file():
        value = read_json(target, "diagnostics cohort capacity")
        _validate_capacity_report(config, resolved, value)
    else:
        _, metadata = _cohort_eligible_rows(config, _dataset_path(config))
        value = _capacity_payload(config, resolved, metadata)
        write_json_atomic(target, value)
        target.chmod(0o600)
    if value["status"] != "sufficient":
        raise RuntimeError(
            "diagnostics NLL capacity is insufficient: "
            + json.dumps(value["after_minimum_target_tokens"], sort_keys=True)
        )
    return dict(value)


def _require_capacity_audit(
    config: Mapping[str, Any], resolved: Mapping[str, Any], output: Path
) -> Dict[str, Any]:
    value = read_json(
        output / "cohort_capacity.json", "diagnostics cohort capacity"
    )
    _validate_capacity_report(config, resolved, value)
    if value["status"] != "sufficient":
        raise RuntimeError("diagnostics cohort capacity is insufficient")
    return dict(value)


def prepare_cohort(config_path: Path) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    config, _ = load_diagnostics_config(config_path)
    resolved, output = initialize_diagnostics(config_path)
    capacity = _require_capacity_audit(config, resolved, output)
    target = output / "cohort_manifest.json"
    if target.is_file():
        return validate_cohort(config_path)
    dataset_path = _dataset_path(config)
    selected, selection_metadata = _expected_cohort_rows(config, dataset_path)
    if _capacity_payload(config, resolved, selection_metadata) != capacity:
        raise RuntimeError("diagnostics cohort capacity changed after audit")
    selection_seed = config["nll"]["selection_seed"]
    expected_per_category = config["nll"]["examples_per_category"]
    private_root = _output_subdirectory(output, "private", "diagnostics private output")
    private_path = private_root / "cohort.parquet"
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("title_group_id", pa.string(), nullable=False),
            pa.field("category_id", pa.int32(), nullable=False),
            pa.field("target_tokens", pa.int32(), nullable=False),
            pa.field("selection_rank", pa.int32(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(selected, schema=schema)
    partial = private_path.with_name(f".{private_path.name}.partial")
    partial.unlink(missing_ok=True)
    pq.write_table(table, partial, compression="zstd", use_dictionary=False)
    partial.chmod(0o600)
    partial.replace(private_path)
    ids = [value["sample_id"] for value in selected]
    groups = [value["title_group_id"] for value in selected]
    manifest = {
        "schema_version": _cohort_schema(config),
        "diagnostic_id": resolved["diagnostic_id"],
        "classification_dataset_id": resolved["classification_dataset_id"],
        "selection": config["nll"]["selection"],
        "selection_seed": selection_seed,
        "categories": list(CATEGORIES),
        "examples_per_category": expected_per_category,
        "examples": len(selected),
        "unique_title_groups": len(set(groups)),
        "excluded_prior_title_groups": selection_metadata[
            "excluded_prior_title_groups"
        ],
        "eligible_per_category": selection_metadata[
            "after_minimum_target_tokens"
        ],
        "cohort_ids_sha256": digest_strings(ids),
        "cohort_title_groups_sha256": digest_strings(groups),
        "tokenizer_fingerprint_sha256": selection_metadata[
            "tokenizer_fingerprint_sha256"
        ],
        "minimum_target_tokens": config["nll"]["minimum_target_tokens"],
        "private_output": {
            "relative_path": "private/cohort.parquet",
            "rows": len(selected),
            "size_bytes": private_path.stat().st_size,
            "sha256": file_sha256(private_path),
        },
        "redistribution_status": REDISTRIBUTION_STATUS,
        "status": "complete",
    }
    if config["schema_version"] == CONFIG_SCHEMA_V2:
        manifest.update(
            {
                "prior_split_exclusion": selection_metadata[
                    "prior_split_exclusion"
                ],
                "after_prior_split_exclusion_and_deduplication": selection_metadata[
                    "after_prior_split_exclusion_and_deduplication"
                ],
                "capacity_report_sha256": file_sha256(
                    output / "cohort_capacity.json"
                ),
            }
        )
    assert_safe_metadata(manifest)
    write_json_atomic(target, manifest)
    target.chmod(0o600)
    return {
        "diagnostic_id": resolved["diagnostic_id"],
        "examples": len(selected),
        "categories": len(CATEGORIES),
        "status": "complete",
    }


def _load_cohort(output: Path) -> tuple[Dict[str, Any], Any]:
    import pyarrow.parquet as pq

    manifest = read_json(output / "cohort_manifest.json", "diagnostics cohort manifest")
    private = manifest.get("private_output", {})
    relative = private.get("relative_path")
    if relative != "private/cohort.parquet":
        raise RuntimeError("diagnostics cohort private path changed")
    path = _private_artifact_path(output, relative, "diagnostics cohort private output")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("diagnostics cohort private output is missing")
    if path.stat().st_size != private.get("size_bytes") or file_sha256(path) != private.get("sha256"):
        raise RuntimeError("diagnostics cohort private output changed")
    return manifest, pq.read_table(path)


def validate_cohort(config_path: Path) -> Dict[str, Any]:
    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    capacity = _require_capacity_audit(config, resolved, output)
    dataset_path = _dataset_path(config)
    validate_classification_dataset(dataset_path)
    manifest, table = _load_cohort(output)
    expected_columns = ["sample_id", "title_group_id", "category_id", "target_tokens", "selection_rank"]
    if (
        manifest.get("schema_version") != _cohort_schema(config)
        or manifest.get("diagnostic_id") != resolved["diagnostic_id"]
        or manifest.get("classification_dataset_id") != resolved["classification_dataset_id"]
        or manifest.get("status") != "complete"
        or table.column_names != expected_columns
        or table.num_rows != _nll_examples_per_state(config)
    ):
        raise RuntimeError("diagnostics cohort is invalid")
    values = table.to_pydict()
    ids = values["sample_id"]
    groups = values["title_group_id"]
    categories = [int(value) for value in values["category_id"]]
    target_tokens = [int(value) for value in values["target_tokens"]]
    if (
        len(set(ids)) != _nll_examples_per_state(config)
        or len(set(groups)) != _nll_examples_per_state(config)
        or digest_strings(ids) != manifest.get("cohort_ids_sha256")
        or digest_strings(groups) != manifest.get("cohort_title_groups_sha256")
        or any(
            categories.count(category)
            != config["nll"]["examples_per_category"]
            for category in CATEGORIES
        )
        or any(value < config["nll"]["minimum_target_tokens"] for value in target_tokens)
    ):
        raise RuntimeError("diagnostics cohort balance or identity changed")
    expected_rows, selection_metadata = _expected_cohort_rows(config, dataset_path)
    if table.to_pylist() != expected_rows:
        raise RuntimeError("diagnostics cohort deterministic selection changed")
    expected_metadata = {
        "excluded_prior_title_groups": selection_metadata[
            "excluded_prior_title_groups"
        ],
        "eligible_per_category": selection_metadata[
            "after_minimum_target_tokens"
        ],
        "tokenizer_fingerprint_sha256": selection_metadata[
            "tokenizer_fingerprint_sha256"
        ],
    }
    if config["schema_version"] == CONFIG_SCHEMA_V2:
        expected_metadata.update(
            {
                "prior_split_exclusion": selection_metadata[
                    "prior_split_exclusion"
                ],
                "after_prior_split_exclusion_and_deduplication": selection_metadata[
                    "after_prior_split_exclusion_and_deduplication"
                ],
                "capacity_report_sha256": file_sha256(
                    output / "cohort_capacity.json"
                ),
            }
        )
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("diagnostics cohort selection metadata changed")
    if _capacity_payload(config, resolved, selection_metadata) != capacity:
        raise RuntimeError("diagnostics cohort capacity changed during validation")
    result = {
        "diagnostic_id": resolved["diagnostic_id"],
        "examples": table.num_rows,
        "categories": len(CATEGORIES),
        "cohort_manifest_sha256": file_sha256(output / "cohort_manifest.json"),
        "status": "valid",
    }
    write_json_atomic(output / "cohort_validation.json", result)
    (output / "cohort_validation.json").chmod(0o600)
    return result


def _scale_embeddings(train: Any, test: Any) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=True)
    scaled_train = scaler.fit_transform(train).astype(np.float32, copy=False)
    scaled_test = scaler.transform(test).astype(np.float32, copy=False)
    if not np.isfinite(scaled_train).all() or not np.isfinite(scaled_test).all():
        raise RuntimeError("low-shot scaled embeddings are non-finite")
    return scaled_train, scaled_test


def _fit_low_shot_classifier(train: Any, labels: Sequence[str], policy: Mapping[str, Any], seed: int) -> Any:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        C=policy["c"],
        solver=policy["solver"],
        l1_ratio=policy["l1_ratio"],
        fit_intercept=policy["fit_intercept"],
        class_weight=policy["class_weight"],
        tol=policy["tolerance"],
        max_iter=policy["max_iter"],
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train, labels)
    return classifier


def _classification_metrics(expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    accuracy = float(accuracy_score(expected, predicted))
    macro_f1 = float(f1_score(expected, predicted, average="macro"))
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=list(labels), zero_division=0
    )
    values = [accuracy, macro_f1, *precision.tolist(), *recall.tolist(), *f1.tolist()]
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("low-shot metrics are non-finite")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": list(labels),
        "by_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(labels)).astype(int).tolist(),
    }


def run_low_shot_unit(config_path: Path, unit_index: int) -> Dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    source_config, source_resolved, source_path, _ = _read_source_evaluation(
        config, full_validation=False, verify_embeddings=False
    )
    require_validated_embeddings(source_resolved, source_path)
    seed = low_shot_seed(config, unit_index)
    units_root = _output_subdirectory(
        output, "low_shot/units", "low-shot unit output"
    )
    target = units_root / f"seed-{seed}.json"
    split = load_pinned_split(source_config, source_resolved, "coarse", seed)
    train_ids = list(split["sample_ids"]["train"])
    validation_ids = list(split["sample_ids"]["validation"])
    test_ids = list(split["sample_ids"]["test"])
    if (
        set(train_ids) & set(validation_ids)
        or set(train_ids) & set(test_ids)
        or set(validation_ids) & set(test_ids)
    ):
        raise RuntimeError("low-shot split partitions overlap")
    dataset_path = _dataset_path(config)
    labels_all = load_private_labels(dataset_path, train_ids + test_ids, "coarse")
    train_labels = labels_all[: len(train_ids)]
    test_labels = labels_all[len(train_ids) :]
    label_values = sorted(split["labels"], key=int)
    if set(train_labels) != set(label_values) or set(test_labels) != set(label_values):
        raise RuntimeError("low-shot split has absent classes")
    for label in label_values:
        if (
            train_labels.count(label) != max(config["low_shot"]["budgets_per_class"])
            or test_labels.count(label)
            != config["low_shot"]["test_examples_per_class"]
        ):
            raise RuntimeError("low-shot split class balance changed")
    budgets = _low_shot_budgets(config)
    selected = nested_low_shot_ids(
        train_ids, train_labels, seed=seed, budgets=budgets
    )
    maximum_ids = selected[max(budgets)]
    label_by_id = dict(zip(train_ids + test_ids, train_labels + test_labels))
    selection_records = [
        {
            "budget_per_class": budget,
            "examples": len(selected[budget]),
            "selection_sha256": digest_strings(selected[budget]),
        }
        for budget in budgets
    ]
    test_set_sha256 = digest_strings(test_ids)
    if target.is_file():
        existing = read_json(target, "low-shot unit")
        _validate_low_shot_unit(
            existing,
            resolved,
            output,
            unit_index,
            seed,
            expected_selection=selection_records,
            expected_test_set_sha256=test_set_sha256,
        )
        return existing
    results = []
    private_rows = []
    for model_name in MODEL_NAMES:
        all_ids = maximum_ids + test_ids
        embeddings = load_embedding_rows(
            source_path,
            model_name,
            config["low_shot"]["input_variant"],
            config["low_shot"]["pooling"],
            all_ids,
        )
        position = {sample_id: index for index, sample_id in enumerate(all_ids)}
        test_matrix = embeddings[len(maximum_ids) :]
        for budget in budgets:
            budget_ids = selected[budget]
            train_matrix = np.stack([embeddings[position[sample_id]] for sample_id in budget_ids]).astype(np.float32)
            budget_labels = [label_by_id[sample_id] for sample_id in budget_ids]
            scaled_train, scaled_test = _scale_embeddings(train_matrix, test_matrix)
            classifier = _fit_low_shot_classifier(scaled_train, budget_labels, config["low_shot"], seed)
            predicted = classifier.predict(scaled_test).tolist()
            metrics = _classification_metrics(test_labels, predicted, label_values)
            results.append({"model": model_name, "budget_per_class": budget, **metrics})
            private_rows.extend(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "model": model_name,
                    "seed": seed,
                    "budget_per_class": budget,
                }
                for sample_id, true_label, predicted_label in zip(test_ids, test_labels, predicted)
            )
    private_parent = _output_subdirectory(
        output, "private/low_shot", "low-shot private output"
    )
    private_path = private_parent / f"seed-{seed}.parquet"
    partial = private_path.with_name(f".{private_path.name}.partial")
    partial.unlink(missing_ok=True)
    pq.write_table(pa.Table.from_pylist(private_rows), partial, compression="zstd", use_dictionary=False)
    partial.chmod(0o600)
    partial.replace(private_path)
    value = {
        "schema_version": LOW_SHOT_UNIT_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "source_evaluation_id": source_resolved["evaluation_id"],
        "unit_index": unit_index,
        "seed": seed,
        "task": "coarse",
        "input_variant": config["low_shot"]["input_variant"],
        "pooling": config["low_shot"]["pooling"],
        "c": config["low_shot"]["c"],
        "selection": selection_records,
        "counts": {
            "validation_accessed": 0,
            "test_examples": len(test_ids),
            "test_examples_per_class": config["low_shot"][
                "test_examples_per_class"
            ],
            "fits": len(results),
        },
        "test_set_sha256": test_set_sha256,
        "results": results,
        "private_output": {
            "relative_path": private_path.relative_to(output).as_posix(),
            "rows": len(private_rows),
            "size_bytes": private_path.stat().st_size,
            "sha256": file_sha256(private_path),
        },
        "status": "complete",
    }
    assert_safe_metadata(value)
    write_json_atomic(target, value)
    target.chmod(0o600)
    return value


def _validate_low_shot_unit(
    value: Mapping[str, Any],
    resolved: Mapping[str, Any],
    output: Path,
    unit_index: int,
    seed: int,
    *,
    expected_selection: Sequence[Mapping[str, Any]] | None = None,
    expected_test_set_sha256: str | None = None,
) -> None:
    low_shot = resolved["low_shot"]
    categories = _low_shot_categories(resolved)
    budgets = _low_shot_budgets(resolved)
    test_examples = _low_shot_test_examples(resolved)
    fits = len(MODEL_NAMES) * len(budgets)
    private_rows_expected = test_examples * fits
    expected_pairs = {
        (model, budget) for model in MODEL_NAMES for budget in budgets
    }
    results = value.get("results")
    selection = value.get("selection")
    counts = value.get("counts", {})
    if (
        value.get("schema_version") != LOW_SHOT_UNIT_SCHEMA
        or value.get("diagnostic_id") != resolved["diagnostic_id"]
        or value.get("unit_index") != unit_index
        or value.get("seed") != seed
        or value.get("task") != "coarse"
        or value.get("input_variant") != "title_first_post"
        or value.get("pooling") != "masked_mean"
        or value.get("c") != 0.01
        or value.get("status") != "complete"
        or not isinstance(results, list)
        or len(results) != 12
        or {(item.get("model"), item.get("budget_per_class")) for item in results}
        != expected_pairs
        or not isinstance(selection, list)
        or [item.get("budget_per_class") for item in selection]
        != list(budgets)
        or counts
        != {
            "validation_accessed": 0,
            "test_examples": test_examples,
            "test_examples_per_class": low_shot[
                "test_examples_per_class"
            ],
            "fits": fits,
        }
    ):
        raise RuntimeError("low-shot unit matrix is invalid")
    _sha256_runtime(value.get("test_set_sha256"), "low-shot test-set digest")
    if (
        expected_selection is not None
        and selection != list(expected_selection)
    ):
        raise RuntimeError("low-shot deterministic selection changed")
    if (
        expected_test_set_sha256 is not None
        and value.get("test_set_sha256") != expected_test_set_sha256
    ):
        raise RuntimeError("low-shot fixed test set changed")
    for item in selection:
        budget = item["budget_per_class"]
        if item.get("examples") != budget * len(categories):
            raise RuntimeError("low-shot selection count is invalid")
        _sha256_runtime(item.get("selection_sha256"), "low-shot selection digest")
    expected_labels = {str(category) for category in categories}
    for result in results:
        if (
            not math.isfinite(float(result.get("accuracy", math.nan)))
            or not math.isfinite(float(result.get("macro_f1", math.nan)))
            or result.get("labels") != [str(category) for category in categories]
            or set(result.get("by_class", {})) != expected_labels
            or len(result.get("confusion_matrix", [])) != len(categories)
            or any(len(row) != len(categories) for row in result["confusion_matrix"])
        ):
            raise RuntimeError("low-shot unit metrics are invalid")
        for metrics in result["by_class"].values():
            if (
                set(metrics) != {"precision", "recall", "f1", "support"}
                or metrics["support"]
                != low_shot["test_examples_per_class"]
                or not all(
                    math.isfinite(float(metrics[key]))
                    for key in ("precision", "recall", "f1")
                )
            ):
                raise RuntimeError("low-shot class metrics are invalid")
    private = value.get("private_output", {})
    private_path = _private_artifact_path(
        output, private.get("relative_path"), "low-shot private output"
    )
    if (
        private.get("rows") != private_rows_expected
        or private_path.is_symlink()
        or not private_path.is_file()
        or private_path.stat().st_size != private.get("size_bytes")
        or file_sha256(private_path) != private.get("sha256")
    ):
        raise RuntimeError("low-shot private output is invalid")
    import pyarrow.parquet as pq

    table = pq.read_table(private_path)
    expected_columns = [
        "sample_id",
        "true_label",
        "predicted_label",
        "model",
        "seed",
        "budget_per_class",
    ]
    if (
        table.column_names != expected_columns
        or table.num_rows != private_rows_expected
    ):
        raise RuntimeError("low-shot private prediction schema is invalid")
    rows = table.to_pylist()
    result_lookup = {
        (item["model"], item["budget_per_class"]): item for item in results
    }
    reference_ids = None
    reference_truth = None
    labels = [str(category) for category in categories]
    for model, budget in sorted(expected_pairs, key=lambda pair: (pair[0], pair[1])):
        matching = [
            row
            for row in rows
            if row["model"] == model and row["budget_per_class"] == budget
        ]
        ids = [row["sample_id"] for row in matching]
        truth = [row["true_label"] for row in matching]
        predicted = [row["predicted_label"] for row in matching]
        if (
            len(matching) != test_examples
            or len(set(ids)) != test_examples
            or any(row["seed"] != seed for row in matching)
            or digest_strings(ids) != value["test_set_sha256"]
            or set(truth) != set(labels)
            or set(predicted) - set(labels)
        ):
            raise RuntimeError("low-shot private predictions are invalid")
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        elif ids != reference_ids or truth != reference_truth:
            raise RuntimeError("low-shot models used different fixed tests")
        expected_metrics = _classification_metrics(truth, predicted, labels)
        stored = dict(result_lookup[(model, budget)])
        stored.pop("model")
        stored.pop("budget_per_class")
        if stored != expected_metrics:
            raise RuntimeError("low-shot metrics do not match private predictions")


def _load_low_shot_units(
    config: Mapping[str, Any], resolved: Mapping[str, Any], output: Path
) -> list[Dict[str, Any]]:
    source_config, source_resolved, _, _ = _read_source_evaluation(
        config, full_validation=False, verify_embeddings=False
    )
    dataset_path = _dataset_path(config)
    values = []
    budgets = _low_shot_budgets(config)
    for index, seed in enumerate(_low_shot_seeds(config)):
        path = _private_artifact_path(
            output,
            f"low_shot/units/seed-{seed}.json",
            "low-shot unit",
        )
        value = read_json(path, "low-shot unit")
        split = load_pinned_split(source_config, source_resolved, "coarse", seed)
        train_ids = list(split["sample_ids"]["train"])
        train_labels = load_private_labels(dataset_path, train_ids, "coarse")
        selected = nested_low_shot_ids(
            train_ids, train_labels, seed=seed, budgets=budgets
        )
        expected_selection = [
            {
                "budget_per_class": budget,
                "examples": len(selected[budget]),
                "selection_sha256": digest_strings(selected[budget]),
            }
            for budget in budgets
        ]
        _validate_low_shot_unit(
            value,
            resolved,
            output,
            index,
            seed,
            expected_selection=expected_selection,
            expected_test_set_sha256=digest_strings(
                list(split["sample_ids"]["test"])
            ),
        )
        values.append(value)
    return values


def _require_cohort_validation(resolved: Mapping[str, Any], output: Path) -> None:
    value = read_json(output / "cohort_validation.json", "diagnostics cohort validation")
    if (
        value.get("diagnostic_id") != resolved["diagnostic_id"]
        or value.get("examples") != _nll_examples_per_state(resolved)
        or value.get("categories") != len(resolved["nll"]["categories"])
        or value.get("cohort_manifest_sha256") != file_sha256(output / "cohort_manifest.json")
        or value.get("status") != "valid"
    ):
        raise RuntimeError("diagnostics cohort has not been validated")


def _load_state_model(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    state: Mapping[str, Any],
    device: Any,
    *,
    verify_files: bool,
) -> tuple[Any, Any, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = _load_nll_tokenizer(config)
    expected_tokenizer = None
    for model_name in ("general", "forum"):
        manifest_path = _project_path(
            f"{config['models'][model_name]['relative_path']}/model_artifact_manifest.json",
            f"diagnostics {model_name} artifact manifest",
        )
        manifest = read_json(manifest_path, f"diagnostics {model_name} artifact manifest")
        fingerprint = manifest.get("tokenizer", {}).get("prepared_fingerprint_sha256")
        if expected_tokenizer is None:
            expected_tokenizer = fingerprint
        elif fingerprint != expected_tokenizer:
            raise RuntimeError("diagnostics final artifact tokenizers differ")
    actual_fingerprint = tokenizer_fingerprint(tokenizer)
    if actual_fingerprint != expected_tokenizer:
        raise RuntimeError("diagnostics tokenizer fingerprint changed")

    arguments: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    if state["kind"] == "huggingface":
        source: str | Path = config["models"]["base"]["model_id"]
        arguments["revision"] = config["models"]["base"]["revision"]
    else:
        root = _project_path(state["relative_path"], "diagnostics model state")
        if state["kind"] == "checkpoint":
            run = config["training_runs"][state["model"]]
            current = validate_checkpoint_model_for_inference(
                root,
                run,
                state["optimizer_step"],
                verify_model_hashes=verify_files,
            )
            stored = resolved["states"][state["state_index"]]
            if any(current.get(key) != stored.get(key) for key in ("checkpoint_manifest_sha256", "model_files_sha256", "model_files")):
                raise RuntimeError("diagnostics checkpoint model identity changed")
            source = root / "model"
        else:
            if verify_files:
                manifest = validate_model_artifact(root, load_model=False)
                expected = config["models"][state["model"]]
                if manifest.get("artifact_sha256") != expected["artifact_sha256"]:
                    raise RuntimeError("diagnostics artifact hash changed")
            source = root
    model = AutoModelForCausalLM.from_pretrained(source, **arguments)
    if (
        model.config.model_type != "llama"
        or int(model.config.max_position_embeddings) != 4096
        or int(model.config.vocab_size) != 49_152
        or sum(parameter.numel() for parameter in model.parameters()) != 670_127_616
    ):
        raise RuntimeError("diagnostics model architecture changed")
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    metadata = {
        "state_index": state["state_index"],
        "state_name": state["state_name"],
        "model": state["model"],
        "arm": state["arm"],
        "kind": state["kind"],
        "optimizer_step": state["optimizer_step"],
        "training_tokens": state["training_tokens"],
        "parameter_count": 670_127_616,
        "tokenizer_fingerprint_sha256": actual_fingerprint,
    }
    return tokenizer, model, metadata


def _encode_nll_batch(
    tokenizer: Any,
    texts: Sequence[tuple[str, str]],
    *,
    max_length: int,
    device: Any,
) -> tuple[Dict[str, Any], Any, list[int]]:
    import torch

    combined = [f"{title}\n\n{post}" for title, post in texts]
    boundaries = [len(f"{title}\n\n") for title, _ in texts]
    encoded = tokenizer(
        combined,
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    offsets = encoded.pop("offset_mapping").tolist()
    special = encoded.pop("special_tokens_mask").tolist()
    attention = encoded["attention_mask"].tolist()
    masks = [
        first_post_target_mask(row_offsets, row_special, row_attention, boundary)
        for row_offsets, row_special, row_attention, boundary in zip(offsets, special, attention, boundaries)
    ]
    target_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    counts = target_mask[:, 1:].sum(dim=1).to("cpu").tolist()
    inputs = {key: value.to(device) for key, value in encoded.items()}
    return inputs, target_mask, [int(value) for value in counts]


def _score_nll_batch(
    tokenizer: Any,
    model: Any,
    texts: Sequence[tuple[str, str]],
    *,
    max_length: int,
    device: Any,
) -> tuple[list[float], list[int]]:
    import torch
    import torch.nn.functional as functional

    inputs, target_mask, counts = _encode_nll_batch(
        tokenizer, texts, max_length=max_length, device=device
    )
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        logits = model(**inputs, use_cache=False, return_dict=True).logits
    shifted_logits = logits[:, :-1, :].float().transpose(1, 2)
    shifted_labels = inputs["input_ids"][:, 1:]
    losses = functional.cross_entropy(shifted_logits, shifted_labels, reduction="none")
    mask = target_mask[:, 1:]
    sums = (losses * mask).sum(dim=1)
    if any(count < 1 for count in counts) or not bool(torch.isfinite(sums).all().item()):
        raise RuntimeError("diagnostics NLL produced invalid values")
    result = [float(value) for value in sums.detach().cpu().tolist()]
    return result, counts


def run_preflight(config_path: Path) -> Dict[str, Any]:
    import importlib.metadata
    import torch

    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostics preflight requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (8, 9) or not torch.cuda.is_bf16_supported():
        raise RuntimeError("diagnostics preflight requires an L40S with BF16 support")
    _, cohort = _load_cohort(output)
    sample_id = cohort.column("sample_id")[0].as_py()
    text = load_private_texts(
        _dataset_path(config) / "examples.parquet",
        [sample_id],
    )[sample_id]
    states = []
    for state in config["states"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer, model, metadata = _load_state_model(
            config, resolved, state, device, verify_files=True
        )
        sums, counts = _score_nll_batch(
            tokenizer,
            model,
            [text],
            max_length=config["nll"]["max_length"],
            device=device,
        )
        if counts[0] < config["nll"]["minimum_target_tokens"] or not math.isfinite(sums[0]):
            raise RuntimeError("diagnostics preflight NLL is invalid")
        torch.cuda.synchronize(device)
        states.append(
            {
                **metadata,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "probe_target_tokens": counts[0],
                "probe_nll_finite": True,
            }
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers")
    }
    if (
        dependencies["scikit-learn"] != "1.9.0"
        or dependencies["torch"] != "2.7.1+cu118"
        or dependencies["transformers"] != "5.14.1"
    ):
        raise RuntimeError("diagnostics dependency versions changed")
    value = {
        "schema_version": PREFLIGHT_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "bf16_supported": True,
        },
        "dependencies": dependencies,
        "cohort_manifest_sha256": file_sha256(output / "cohort_manifest.json"),
        "states": states,
        "status": "ok",
    }
    assert_safe_metadata(value)
    write_json_atomic(output / "preflight.json", value)
    (output / "preflight.json").chmod(0o600)
    return value


def _require_preflight(resolved: Mapping[str, Any], output: Path) -> None:
    value = read_json(output / "preflight.json", "diagnostics preflight")
    if (
        value.get("schema_version") != PREFLIGHT_SCHEMA
        or value.get("diagnostic_id") != resolved["diagnostic_id"]
        or value.get("cohort_manifest_sha256") != file_sha256(output / "cohort_manifest.json")
        or len(value.get("states", [])) != 9
        or value.get("status") != "ok"
    ):
        raise RuntimeError("diagnostics preflight is missing or invalid")


def _valid_score_chunk(
    metadata_path: Path,
    private_path: Path,
    resolved: Mapping[str, Any],
    state: Mapping[str, Any],
    output: Path,
    *,
    start: int,
    expected_ids: Sequence[str],
    expected_categories: Sequence[int],
    expected_target_tokens: Sequence[int],
) -> bool:
    if not metadata_path.is_file() or not private_path.is_file() or metadata_path.is_symlink() or private_path.is_symlink():
        return False
    try:
        import pyarrow.parquet as pq

        value = read_json(metadata_path, "diagnostics score chunk")
        private = value["private_output"]
        if not (
            value.get("schema_version") == SCORE_CHUNK_SCHEMA
            and value.get("diagnostic_id") == resolved["diagnostic_id"]
            and value.get("state_index") == state["state_index"]
            and value.get("state_name") == state["state_name"]
            and value.get("start") == start
            and value.get("count") == len(expected_ids)
            and value.get("ids_sha256") == digest_strings(expected_ids)
            and value.get("target_tokens") == sum(expected_target_tokens)
            and value.get("status") == "complete"
            and private.get("relative_path")
            == private_path.relative_to(output).as_posix()
            and private_path.stat().st_size == private.get("size_bytes")
            and file_sha256(private_path) == private.get("sha256")
        ):
            return False
        table = pq.read_table(private_path)
        if table.column_names != [
            "sample_id",
            "category_id",
            "target_tokens",
            "nll_sum",
            "mean_nll",
        ]:
            return False
        rows = table.to_pydict()
        return (
            rows["sample_id"] == list(expected_ids)
            and [int(value) for value in rows["category_id"]]
            == list(expected_categories)
            and [int(value) for value in rows["target_tokens"]]
            == list(expected_target_tokens)
            and all(
                math.isfinite(float(nll_sum))
                and math.isfinite(float(mean_nll))
                and math.isclose(
                    float(nll_sum) / target_tokens,
                    float(mean_nll),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                for target_tokens, nll_sum, mean_nll in zip(
                    rows["target_tokens"], rows["nll_sum"], rows["mean_nll"]
                )
            )
        )
    except (KeyError, RuntimeError, OSError, ValueError):
        return False


def run_score_unit(config_path: Path, unit_index: int) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    _require_preflight(resolved, output)
    state = state_by_index(config, unit_index)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostics scoring requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    manifest, cohort = _load_cohort(output)
    values = cohort.to_pydict()
    ids = list(values["sample_id"])
    categories = [int(value) for value in values["category_id"]]
    expected_counts = [int(value) for value in values["target_tokens"]]
    texts = load_private_texts(
        _dataset_path(config) / "examples.parquet",
        ids,
    )
    public_root = _output_subdirectory(
        output,
        f"scores/{state['state_name']}/chunks",
        "diagnostics score output",
    )
    private_root = _output_subdirectory(
        output,
        f"private/nll/{state['state_name']}",
        "diagnostics private score output",
    )
    tokenizer, model, model_metadata = _load_state_model(
        config, resolved, state, device, verify_files=True
    )
    chunk_records = []
    chunk_size = config["nll"]["chunk_size"]
    for start in range(0, len(ids), chunk_size):
        stop = min(start + chunk_size, len(ids))
        name = f"chunk-{start:06d}"
        metadata_path = public_root / f"{name}.json"
        private_path = private_root / f"{name}.parquet"
        if not _valid_score_chunk(
            metadata_path,
            private_path,
            resolved,
            state,
            output,
            start=start,
            expected_ids=ids[start:stop],
            expected_categories=categories[start:stop],
            expected_target_tokens=expected_counts[start:stop],
        ):
            rows = []
            for batch_start in range(start, stop, config["nll"]["batch_size"]):
                batch_stop = min(batch_start + config["nll"]["batch_size"], stop)
                batch_ids = ids[batch_start:batch_stop]
                sums, counts = _score_nll_batch(
                    tokenizer,
                    model,
                    [texts[sample_id] for sample_id in batch_ids],
                    max_length=config["nll"]["max_length"],
                    device=device,
                )
                for offset, (nll_sum, count) in enumerate(zip(sums, counts)):
                    index = batch_start + offset
                    if count != expected_counts[index] or count < config["nll"]["minimum_target_tokens"]:
                        raise RuntimeError("diagnostics scoring target-token count changed")
                    mean_nll = nll_sum / count
                    if not math.isfinite(mean_nll):
                        raise RuntimeError("diagnostics scoring produced non-finite NLL")
                    rows.append(
                        {
                            "sample_id": ids[index],
                            "category_id": categories[index],
                            "target_tokens": count,
                            "nll_sum": nll_sum,
                            "mean_nll": mean_nll,
                        }
                    )
            partial = private_path.with_name(f".{private_path.name}.partial")
            partial.unlink(missing_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), partial, compression="zstd", use_dictionary=False)
            partial.chmod(0o600)
            partial.replace(private_path)
            metadata = {
                "schema_version": SCORE_CHUNK_SCHEMA,
                "diagnostic_id": resolved["diagnostic_id"],
                "state_index": state["state_index"],
                "state_name": state["state_name"],
                "start": start,
                "count": len(rows),
                "ids_sha256": digest_strings(ids[start:stop]),
                "target_tokens": sum(row["target_tokens"] for row in rows),
                "private_output": {
                    "relative_path": private_path.relative_to(output).as_posix(),
                    "size_bytes": private_path.stat().st_size,
                    "sha256": file_sha256(private_path),
                },
                "status": "complete",
            }
            assert_safe_metadata(metadata)
            write_json_atomic(metadata_path, metadata)
            metadata_path.chmod(0o600)
        metadata = read_json(metadata_path, "diagnostics score chunk")
        chunk_records.append(
            {
                "path": metadata_path.relative_to(output / "scores" / state["state_name"]).as_posix(),
                "sha256": file_sha256(metadata_path),
                "count": metadata["count"],
                "start": metadata["start"],
            }
        )
    score_manifest = {
        "schema_version": SCORE_MANIFEST_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "state": model_metadata,
        "cohort_manifest_sha256": file_sha256(output / "cohort_manifest.json"),
        "examples": len(ids),
        "cohort_ids_sha256": manifest["cohort_ids_sha256"],
        "chunks": chunk_records,
        "status": "complete",
    }
    assert_safe_metadata(score_manifest)
    manifest_path = output / "scores" / state["state_name"] / "score_manifest.json"
    write_json_atomic(manifest_path, score_manifest)
    manifest_path.chmod(0o600)
    return {
        "diagnostic_id": resolved["diagnostic_id"],
        "state_index": state["state_index"],
        "state_name": state["state_name"],
        "examples": len(ids),
        "status": "complete",
    }


def _load_state_scores(
    resolved: Mapping[str, Any],
    output: Path,
    state: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, list[Any]]]:
    import pyarrow.parquet as pq

    root = _private_artifact_path(
        output,
        f"scores/{state['state_name']}",
        "diagnostics score state",
    )
    manifest_path = root / "score_manifest.json"
    manifest = read_json(manifest_path, "diagnostics score manifest")
    if (
        manifest.get("schema_version") != SCORE_MANIFEST_SCHEMA
        or manifest.get("diagnostic_id") != resolved["diagnostic_id"]
        or manifest.get("state", {}).get("state_index") != state["state_index"]
        or any(
            manifest.get("state", {}).get(key) != state[key]
            for key in (
                "state_name",
                "model",
                "arm",
                "kind",
                "optimizer_step",
                "training_tokens",
            )
        )
        or manifest.get("cohort_manifest_sha256")
        != file_sha256(output / "cohort_manifest.json")
        or manifest.get("examples") != _nll_examples_per_state(resolved)
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("diagnostics score manifest is invalid")
    records = manifest.get("chunks")
    if not isinstance(records, list) or not records:
        raise RuntimeError("diagnostics score chunks are missing")
    chunks_root = root / "chunks"
    if chunks_root.is_symlink() or not chunks_root.is_dir():
        raise RuntimeError("diagnostics score chunk directory is unsafe")
    expected_metadata = {record.get("path") for record in records}
    actual_metadata = {
        path.relative_to(root).as_posix()
        for path in chunks_root.glob("chunk-*.json")
    }
    if expected_metadata != actual_metadata:
        raise RuntimeError("diagnostics score chunk set changed")
    combined: Dict[str, list[Any]] = {
        "sample_id": [],
        "category_id": [],
        "target_tokens": [],
        "nll_sum": [],
        "mean_nll": [],
    }
    expected_start = 0
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("diagnostics score chunk path is unsafe")
        metadata_path = root / relative
        if metadata_path.is_symlink() or file_sha256(metadata_path) != record.get("sha256"):
            raise RuntimeError("diagnostics score chunk metadata changed")
        metadata = read_json(metadata_path, "diagnostics score chunk")
        if (
            metadata.get("schema_version") != SCORE_CHUNK_SCHEMA
            or metadata.get("diagnostic_id") != resolved["diagnostic_id"]
            or metadata.get("state_index") != state["state_index"]
            or metadata.get("state_name") != state["state_name"]
            or metadata.get("start") != expected_start
            or metadata.get("count") != record.get("count")
            or metadata.get("status") != "complete"
        ):
            raise RuntimeError("diagnostics score chunk coverage changed")
        private = metadata.get("private_output", {})
        private_relative = private.get("relative_path")
        if not isinstance(private_relative, str) or Path(private_relative).is_absolute() or ".." in Path(private_relative).parts:
            raise RuntimeError("diagnostics private score path is unsafe")
        private_path = _private_artifact_path(
            output, private_relative, "diagnostics private score chunk"
        )
        if (
            private_path.is_symlink()
            or not private_path.is_file()
            or private_path.stat().st_size != private.get("size_bytes")
            or file_sha256(private_path) != private.get("sha256")
        ):
            raise RuntimeError("diagnostics private score chunk changed")
        table = pq.read_table(private_path)
        if table.column_names != list(combined) or table.num_rows != metadata["count"]:
            raise RuntimeError("diagnostics private score schema changed")
        values = table.to_pydict()
        if digest_strings(values["sample_id"]) != metadata.get("ids_sha256"):
            raise RuntimeError("diagnostics private score IDs changed")
        for key in combined:
            combined[key].extend(values[key])
        expected_start += metadata["count"]
    if (
        expected_start != _nll_examples_per_state(resolved)
        or digest_strings(combined["sample_id"])
        != manifest.get("cohort_ids_sha256")
    ):
        raise RuntimeError("diagnostics score coverage is incomplete")
    for count, nll_sum, mean_nll in zip(combined["target_tokens"], combined["nll_sum"], combined["mean_nll"]):
        if (
            not isinstance(count, int)
            or count < resolved["nll"]["minimum_target_tokens"]
            or not math.isfinite(float(nll_sum))
            or not math.isfinite(float(mean_nll))
            or not math.isclose(float(nll_sum) / count, float(mean_nll), rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise RuntimeError("diagnostics score values are invalid")
    return manifest, combined


def validate_scores(config_path: Path) -> Dict[str, Any]:
    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    cohort_manifest, cohort = _load_cohort(output)
    cohort_values = cohort.to_pydict()
    reference = (
        list(cohort_values["sample_id"]),
        [int(value) for value in cohort_values["category_id"]],
        [int(value) for value in cohort_values["target_tokens"]],
    )
    manifests = {}
    total = 0
    for state in config["states"]:
        manifest, values = _load_state_scores(resolved, output, state)
        identity = (
            values["sample_id"],
            [int(value) for value in values["category_id"]],
            [int(value) for value in values["target_tokens"]],
        )
        if (
            identity != reference
            or manifest.get("cohort_ids_sha256")
            != cohort_manifest["cohort_ids_sha256"]
        ):
            raise RuntimeError("diagnostics model states scored different examples")
        manifests[state["state_name"]] = file_sha256(
            output / "scores" / state["state_name"] / "score_manifest.json"
        )
        total += len(values["sample_id"])
    value = {
        "schema_version": SCORE_VALIDATION_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "states": len(config["states"]),
        "examples_per_state": _nll_examples_per_state(config),
        "scores": total,
        "score_manifests": manifests,
        "status": "valid",
    }
    assert_safe_metadata(value)
    write_json_atomic(output / "scores_validation.json", value)
    (output / "scores_validation.json").chmod(0o600)
    return value


def _require_scores_validation(resolved: Mapping[str, Any], output: Path) -> None:
    value = read_json(output / "scores_validation.json", "diagnostics score validation")
    if (
        value.get("schema_version") != SCORE_VALIDATION_SCHEMA
        or value.get("diagnostic_id") != resolved["diagnostic_id"]
        or value.get("states") != len(resolved["states"])
        or value.get("examples_per_state")
        != _nll_examples_per_state(resolved)
        or value.get("scores") != _nll_score_total(resolved)
        or value.get("status") != "valid"
    ):
        raise RuntimeError("diagnostics scores have not been validated")
    expected = {
        state["state_name"]: file_sha256(output / "scores" / state["state_name"] / "score_manifest.json")
        for state in resolved["states"]
    }
    if value.get("score_manifests") != expected:
        raise RuntimeError("diagnostics score manifests changed after validation")


def _student_summary(values: Sequence[float]) -> Dict[str, Any]:
    if len(values) != 5 or not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("diagnostics Student interval requires five finite values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = 2.7764451051977987 * standard_deviation / math.sqrt(5)
    return {
        "n": 5,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "confidence_interval_95": [mean - half_width, mean + half_width],
    }


def _bootstrap_state_means(
    categories: Sequence[int],
    matrix: Any,
    *,
    repetitions: int,
    seed: int,
) -> Any:
    import numpy as np

    category_array = np.asarray(categories, dtype=np.int32)
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != len(category_array)
        or values.shape[1] != len(STATE_NAMES)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("diagnostics bootstrap matrix is invalid")
    if len(category_array) % len(CATEGORIES):
        raise RuntimeError("diagnostics bootstrap categories are unbalanced")
    examples_per_category = len(category_array) // len(CATEGORIES)
    result = np.zeros((repetitions, values.shape[1]), dtype=np.float64)
    generator = np.random.default_rng(seed)
    block_size = 500
    for category in CATEGORIES:
        category_values = values[category_array == category]
        if category_values.shape != (
            examples_per_category,
            len(STATE_NAMES),
        ):
            raise RuntimeError("diagnostics bootstrap categories are unbalanced")
        for start in range(0, repetitions, block_size):
            stop = min(start + block_size, repetitions)
            indices = generator.integers(
                0,
                examples_per_category,
                size=(stop - start, examples_per_category),
            )
            result[start:stop] += category_values[indices].mean(axis=1) / len(CATEGORIES)
    return result


def _percentile_interval(values: Any) -> list[float]:
    import numpy as np

    if not np.isfinite(values).all():
        raise RuntimeError("diagnostics bootstrap values are non-finite")
    low, high = np.percentile(values, [2.5, 97.5]).tolist()
    return [float(low), float(high)]


def _perplexity(nll: float) -> float:
    if not math.isfinite(float(nll)):
        raise RuntimeError("diagnostics NLL is non-finite")
    try:
        value = math.exp(float(nll))
    except OverflowError:
        raise RuntimeError("diagnostics perplexity overflowed") from None
    if not math.isfinite(value):
        raise RuntimeError("diagnostics perplexity is non-finite")
    return value


def terminal_checkpoint_decision(interval: Sequence[float]) -> str:
    if len(interval) != 2 or not all(math.isfinite(float(value)) for value in interval):
        raise ValueError("terminal checkpoint interval is invalid")
    if interval[1] < 0:
        return "still_improving_at_52000"
    if interval[0] > 0:
        return "terminal_regression"
    return "no_clear_additional_improvement"


def _build_report_payload(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    output: Path,
) -> tuple[Dict[str, Any], Dict[str, list[Dict[str, Any]]]]:
    import numpy as np

    low_shot_budgets = _low_shot_budgets(config)
    low_shot_seeds = _low_shot_seeds(config)
    low_units = _load_low_shot_units(config, resolved, output)
    low_metrics = []
    low_class_metrics = []
    for unit in low_units:
        for result in unit["results"]:
            record = {
                "seed": unit["seed"],
                "budget_per_class": result["budget_per_class"],
                "model": result["model"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
            }
            low_metrics.append(record)
            for label, values in result["by_class"].items():
                low_class_metrics.append(
                    {
                        "seed": unit["seed"],
                        "budget_per_class": result["budget_per_class"],
                        "model": result["model"],
                        "label": label,
                        **values,
                    }
                )
    if len(low_metrics) != _low_shot_fits(config):
        raise RuntimeError("diagnostics low-shot result matrix is incomplete")
    low_summary = []
    for budget in low_shot_budgets:
        for model in MODEL_NAMES:
            matching = [
                value
                for value in low_metrics
                if value["budget_per_class"] == budget and value["model"] == model
            ]
            for metric in ("accuracy", "macro_f1"):
                low_summary.append(
                    {
                        "budget_per_class": budget,
                        "model": model,
                        "metric": metric,
                        **_student_summary([value[metric] for value in matching]),
                        "endpoint": (
                            "primary_exploratory"
                            if budget == 64 and model in {"base", "general"} and metric == "macro_f1"
                            else "secondary_exploratory"
                        ),
                    }
                )
    contrast_specs = (
        ("continual_pretraining", "base", "general"),
        ("domain_proximity", "general", "forum"),
        ("total_practical_gain", "base", "forum"),
    )
    low_contrasts = []
    for budget in low_shot_budgets:
        for metric in ("accuracy", "macro_f1"):
            for name, first, second in contrast_specs:
                first_values = {
                    value["seed"]: value[metric]
                    for value in low_metrics
                    if value["budget_per_class"] == budget and value["model"] == first
                }
                second_values = {
                    value["seed"]: value[metric]
                    for value in low_metrics
                    if value["budget_per_class"] == budget and value["model"] == second
                }
                if (
                    set(first_values) != set(low_shot_seeds)
                    or set(second_values) != set(low_shot_seeds)
                ):
                    raise RuntimeError("diagnostics low-shot pairing is incomplete")
                deltas = [
                    second_values[seed] - first_values[seed]
                    for seed in low_shot_seeds
                ]
                low_contrasts.append(
                    {
                        "budget_per_class": budget,
                        "metric": metric,
                        "contrast": name,
                        "first": first,
                        "second": second,
                        "direction": "second_minus_first",
                        "values_by_seed": [
                            {"seed": seed, "delta": delta}
                            for seed, delta in zip(low_shot_seeds, deltas)
                        ],
                        **_student_summary(deltas),
                        "endpoint": (
                            "primary_exploratory"
                            if budget == 64 and metric == "macro_f1" and name == "continual_pretraining"
                            else "secondary_exploratory"
                        ),
                    }
                )

    score_values = []
    for state in config["states"]:
        _, values = _load_state_scores(resolved, output, state)
        score_values.append(values)
    reference_ids = score_values[0]["sample_id"]
    categories = [int(value) for value in score_values[0]["category_id"]]
    target_tokens = np.asarray(score_values[0]["target_tokens"], dtype=np.int64)
    for values in score_values[1:]:
        if (
            values["sample_id"] != reference_ids
            or [int(value) for value in values["category_id"]] != categories
            or np.asarray(values["target_tokens"], dtype=np.int64).tolist() != target_tokens.tolist()
        ):
            raise RuntimeError("diagnostics score matrix is not paired")
    matrix = np.column_stack(
        [np.asarray(values["mean_nll"], dtype=np.float64) for values in score_values]
    )
    sums_matrix = np.column_stack(
        [np.asarray(values["nll_sum"], dtype=np.float64) for values in score_values]
    )
    bootstraps = _bootstrap_state_means(
        categories,
        matrix,
        repetitions=config["statistics"]["bootstrap_repetitions"],
        seed=config["statistics"]["bootstrap_seed"],
    )
    category_array = np.asarray(categories, dtype=np.int32)
    state_summary = []
    nll_by_category = []
    for state in config["states"]:
        index = state["state_index"]
        category_means = []
        for category in CATEGORIES:
            mean = float(matrix[category_array == category, index].mean())
            category_means.append(mean)
            nll_by_category.append(
                {
                    "state_index": index,
                    "state_name": state["state_name"],
                    "model": state["model"],
                    "arm": state["arm"],
                    "optimizer_step": state["optimizer_step"],
                    "category_id": category,
                    "threads": config["nll"]["examples_per_category"],
                    "mean_thread_nll": mean,
                    "perplexity": _perplexity(mean),
                }
            )
        macro_nll = statistics.fmean(category_means)
        token_nll = float(sums_matrix[:, index].sum() / target_tokens.sum())
        state_summary.append(
            {
                "state_index": index,
                "state_name": state["state_name"],
                "model": state["model"],
                "arm": state["arm"],
                "optimizer_step": state["optimizer_step"],
                "training_tokens": state["training_tokens"],
                "threads": _nll_examples_per_state(config),
                "target_tokens": int(target_tokens.sum()),
                "macro_mean_thread_nll": macro_nll,
                "confidence_interval_95": _percentile_interval(bootstraps[:, index]),
                "perplexity": _perplexity(macro_nll),
                "token_weighted_nll": token_nll,
                "token_weighted_perplexity": _perplexity(token_nll),
            }
        )
    nll_contrast_specs = (
        ("continual_pretraining", 0, 4, "secondary"),
        ("domain_proximity", 4, 8, "primary"),
        ("total_practical_gain", 0, 8, "secondary"),
    )
    nll_contrasts = []
    for name, first, second, endpoint in nll_contrast_specs:
        deltas = matrix[:, second] - matrix[:, first]
        bootstrap_deltas = bootstraps[:, second] - bootstraps[:, first]
        category_deltas = {
            str(category): float(deltas[category_array == category].mean())
            for category in CATEGORIES
        }
        nll_contrasts.append(
            {
                "contrast": name,
                "first_state": config["states"][first]["state_name"],
                "second_state": config["states"][second]["state_name"],
                "direction": "second_minus_first",
                "mean_delta_nll": float(
                    statistics.fmean(category_deltas.values())
                ),
                "confidence_interval_95": _percentile_interval(bootstrap_deltas),
                "threads_improved_fraction": float((deltas < 0).mean()),
                "by_category": category_deltas,
                "endpoint": endpoint,
            }
        )
    checkpoint_accumulated_gains = []
    for arm, indices in (
        ("general", (1, 2, 3, 4)),
        ("forum_tech", (5, 6, 7, 8)),
    ):
        for state_index in indices:
            deltas = matrix[:, state_index] - matrix[:, 0]
            bootstrap_deltas = bootstraps[:, state_index] - bootstraps[:, 0]
            category_deltas = [
                float(deltas[category_array == category].mean())
                for category in CATEGORIES
            ]
            mean_delta = statistics.fmean(category_deltas)
            checkpoint_accumulated_gains.append(
                {
                    "arm": arm,
                    "base_state": config["states"][0]["state_name"],
                    "state_name": config["states"][state_index]["state_name"],
                    "optimizer_step": config["states"][state_index][
                        "optimizer_step"
                    ],
                    "training_tokens": config["states"][state_index][
                        "training_tokens"
                    ],
                    "direction": "state_minus_base",
                    "mean_delta_nll": mean_delta,
                    "mean_nll_reduction_vs_base": -mean_delta,
                    "confidence_interval_95": _percentile_interval(
                        bootstrap_deltas
                    ),
                    "perplexity_ratio_vs_base": _perplexity(mean_delta),
                    "threads_improved_fraction": float((deltas < 0).mean()),
                }
            )
    checkpoint_increments = []
    terminal_decisions = []
    for arm, indices in (("general", (0, 1, 2, 3, 4)), ("forum_tech", (0, 5, 6, 7, 8))):
        for first, second in zip(indices, indices[1:]):
            deltas = matrix[:, second] - matrix[:, first]
            bootstrap_deltas = bootstraps[:, second] - bootstraps[:, first]
            interval = _percentile_interval(bootstrap_deltas)
            record = {
                "arm": arm,
                "first_state": config["states"][first]["state_name"],
                "second_state": config["states"][second]["state_name"],
                "first_step": config["states"][first]["optimizer_step"],
                "second_step": config["states"][second]["optimizer_step"],
                "first_tokens": config["states"][first]["training_tokens"],
                "second_tokens": config["states"][second]["training_tokens"],
                "direction": "second_minus_first",
                "mean_delta_nll": float(
                    statistics.fmean(
                        float(deltas[category_array == category].mean())
                        for category in CATEGORIES
                    )
                ),
                "confidence_interval_95": interval,
                "threads_improved_fraction": float((deltas < 0).mean()),
            }
            checkpoint_increments.append(record)
            if config["states"][second]["optimizer_step"] == 52_000:
                decision = terminal_checkpoint_decision(interval)
                terminal_decisions.append({**record, "decision": decision})

    cohort_report = _cohort_report_metadata(config)
    report = {
        "schema_version": _report_schema(config),
        "diagnostic_id": resolved["diagnostic_id"],
        "identity": {
            "git_commit": resolved["git_commit"],
            "config_sha256": resolved["config_sha256"],
            "classification_dataset_id": resolved["classification_dataset_id"],
            "source_evaluation_id": resolved["source_evaluation"]["evaluation_id"],
            "models": resolved["models"],
            "runs": resolved["runs"],
        },
        "low_shot": {
            "status": "exploratory_post_hoc",
            "test_reused": True,
            "validation_accessed": False,
            "metrics": low_metrics,
            "summary": low_summary,
            "paired_contrasts": low_contrasts,
            "class_metrics": low_class_metrics,
        },
        "conditional_nll": {
            "cohort": cohort_report,
            "state_summary": [state_summary[index] for index in (0, 4, 8)],
            "paired_contrasts": nll_contrasts,
            "by_category": [
                value for value in nll_by_category if value["state_index"] in {0, 4, 8}
            ],
        },
        "checkpoint_curve": {
            "state_summary": state_summary,
            "accumulated_gains_vs_base": checkpoint_accumulated_gains,
            "increments": checkpoint_increments,
            "terminal_decisions": terminal_decisions,
        },
        "statistics": {
            **config["statistics"],
            "nll_unit": "thread_then_equal_category_weight",
            "training_run_replications_per_arm": 1,
            "p_values_reported": False,
        },
        "redistribution_status": REDISTRIBUTION_STATUS,
        "status": "complete",
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    assert_safe_metadata(report)
    csvs = {
        "low_shot_metrics.csv": low_metrics,
        "low_shot_summary.csv": low_summary,
        "low_shot_contrasts.csv": low_contrasts,
        "low_shot_class_metrics.csv": low_class_metrics,
        "nll_summary.csv": [state_summary[index] for index in (0, 4, 8)],
        "nll_contrasts.csv": nll_contrasts,
        "nll_by_category.csv": [
            value for value in nll_by_category if value["state_index"] in {0, 4, 8}
        ],
        "checkpoint_curve.csv": state_summary,
        "checkpoint_accumulated_gains.csv": checkpoint_accumulated_gains,
        "checkpoint_increments.csv": checkpoint_increments,
    }
    return report, csvs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    partial.chmod(0o600)
    partial.replace(path)


def build_report(config_path: Path) -> Dict[str, Any]:
    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    _require_scores_validation(resolved, output)
    report, csvs = _build_report_payload(config, resolved, output)
    report_root = _output_subdirectory(output, "report", "diagnostics report output")
    write_json_atomic(report_root / "report.json", report)
    (report_root / "report.json").chmod(0o600)
    for name, rows in csvs.items():
        _write_csv(report_root / name, rows)
    files = sorted(
        path for path in report_root.iterdir() if path.is_file() and path.name != "report_files.json"
    )
    manifest = {
        "schema_version": REPORT_FILES_SCHEMA,
        "diagnostic_id": resolved["diagnostic_id"],
        "report_sha256": report["report_sha256"],
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        ],
        "status": "complete",
    }
    assert_safe_metadata(manifest)
    write_json_atomic(report_root / "report_files.json", manifest)
    (report_root / "report_files.json").chmod(0o600)
    return {
        "diagnostic_id": resolved["diagnostic_id"],
        "low_shot_fits": _low_shot_fits(config),
        "nll_scores": _nll_score_total(config),
        "report_sha256": report["report_sha256"],
        "status": "complete",
    }


def validate_diagnostics_report(config_path: Path) -> Dict[str, Any]:
    config, _ = load_diagnostics_config(config_path)
    resolved, output = locate_diagnostics(config_path)
    _require_cohort_validation(resolved, output)
    _require_scores_validation(resolved, output)
    expected, csvs = _build_report_payload(config, resolved, output)
    report_root = _private_artifact_path(
        output, "report", "diagnostics report output"
    )
    actual = read_json(report_root / "report.json", "diagnostics report")
    if actual != expected:
        raise RuntimeError("diagnostics report changed")
    manifest = read_json(report_root / "report_files.json", "diagnostics report files")
    expected_names = {"report.json", *csvs.keys()}
    records = manifest.get("files")
    if (
        manifest.get("schema_version") != REPORT_FILES_SCHEMA
        or manifest.get("diagnostic_id") != resolved["diagnostic_id"]
        or manifest.get("report_sha256") != expected["report_sha256"]
        or manifest.get("status") != "complete"
        or not isinstance(records, list)
        or {record.get("path") for record in records} != expected_names
    ):
        raise RuntimeError("diagnostics report file manifest is invalid")
    actual_names = {
        path.name for path in report_root.iterdir() if path.is_file() and path.name != "report_files.json"
    }
    if actual_names != expected_names:
        raise RuntimeError("diagnostics report file set changed")
    for record in records:
        path = report_root / record["path"]
        if path.is_symlink() or path.stat().st_size != record.get("size_bytes") or file_sha256(path) != record.get("sha256"):
            raise RuntimeError("diagnostics report file changed")
    return {
        "diagnostic_id": resolved["diagnostic_id"],
        "low_shot_fits": _low_shot_fits(config),
        "nll_scores": _nll_score_total(config),
        "report_sha256": expected["report_sha256"],
        "status": "valid",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CPT diagnostic evaluations")
    parser.add_argument(
        "command",
        choices=(
            "audit-cohort",
            "prepare-cohort",
            "validate-cohort",
            "preflight",
            "low-shot-unit",
            "score-unit",
            "validate-scores",
            "report",
            "validate-report",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--unit-index", type=int)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parser().parse_args()
    if args.command in {"low-shot-unit", "score-unit"}:
        if args.unit_index is None:
            raise SystemExit(f"{args.command} requires --unit-index")
    elif args.unit_index is not None:
        raise SystemExit("--unit-index is only valid for unit commands")
    if args.command == "audit-cohort":
        result = audit_cohort(args.config)
    elif args.command == "prepare-cohort":
        result = prepare_cohort(args.config)
    elif args.command == "validate-cohort":
        result = validate_cohort(args.config)
    elif args.command == "preflight":
        result = run_preflight(args.config)
    elif args.command == "low-shot-unit":
        result = run_low_shot_unit(args.config, args.unit_index)
    elif args.command == "score-unit":
        result = run_score_unit(args.config, args.unit_index)
    elif args.command == "validate-scores":
        result = validate_scores(args.config)
    elif args.command == "report":
        result = build_report(args.config)
    else:
        result = validate_diagnostics_report(args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
