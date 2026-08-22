from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

from .config import (
    DATASET_IDS,
    MODEL_ID,
    MODEL_REVISION,
    canonical_json_bytes,
    sha256_bytes,
)
from .manifest import write_json_atomic
from .model_artifact import validate_model_artifact
from .paired_plan import PAIRED_REAL_POLICY


PAIRED_EXPERIMENT_REPORT_SCHEMA = "queroquero-paired-experiment-report/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[0-9a-f]{20}\Z")


def build_paired_experiment_report(
    *,
    general_run_dir: Path,
    forum_tech_run_dir: Path,
    general_artifact: Path,
    forum_tech_artifact: Path,
    general_elapsed_seconds: int,
    forum_tech_elapsed_seconds: int,
) -> Dict[str, Any]:
    elapsed = {
        "general": _positive_int(general_elapsed_seconds, "general elapsed seconds"),
        "forum_tech": _positive_int(
            forum_tech_elapsed_seconds, "forum_tech elapsed seconds"
        ),
    }
    runs = {
        "general": _load_complete_run(general_run_dir),
        "forum_tech": _load_complete_run(forum_tech_run_dir),
    }
    artifacts = {
        "general": validate_model_artifact(general_artifact, load_model=False),
        "forum_tech": validate_model_artifact(
            forum_tech_artifact, load_model=False
        ),
    }
    for arm in ("general", "forum_tech"):
        run = runs[arm]
        resolved = run["resolved"]
        manifest = run["manifest"]
        artifact = artifacts[arm]
        experiment = artifact.get("training", {}).get("experiment", {})
        if experiment.get("arm") != arm:
            raise RuntimeError(f"{arm} artifact identifies the wrong paired arm")
        if artifact["training"].get("run_id") != manifest.get("run_id"):
            raise RuntimeError(f"{arm} artifact and run IDs differ")
        if manifest.get("artifact") != {
            "artifact_id": artifact["artifact_id"],
            "artifact_sha256": artifact["artifact_sha256"],
            "path": manifest["artifact"].get("path"),
        }:
            raise RuntimeError(f"{arm} run artifact metadata changed")
        resolved_mixture = resolved.get("inputs", {}).get("data_mixture", {})
        run_experiment = manifest.get("experiment")
        if (
            resolved_mixture.get("policy") != PAIRED_REAL_POLICY
            or resolved_mixture.get("arm") != arm
            or resolved_mixture.get("experiment_id")
            != experiment.get("experiment_id")
            or resolved_mixture.get("allocation_sha256")
            != experiment.get("allocation_sha256")
            or resolved_mixture.get("schedule_template_sha256")
            != experiment.get("schedule_template_sha256")
            or resolved.get("inputs", {}).get("paired_inputs_sha256")
            != experiment.get("paired_inputs_sha256")
            or run_experiment != experiment
        ):
            raise RuntimeError(f"{arm} resolved inputs do not match the artifact")

    general_resolved = runs["general"]["resolved"]
    forum_resolved = runs["forum_tech"]["resolved"]
    for key in ("model", "training", "execution"):
        if general_resolved.get(key) != forum_resolved.get(key):
            raise RuntimeError(f"paired completed runs differ in {key}")
    general_experiment = artifacts["general"]["training"]["experiment"]
    forum_experiment = artifacts["forum_tech"]["training"]["experiment"]
    for key in (
        "experiment_id",
        "allocation_sha256",
        "schedule_template_sha256",
        "paired_inputs_sha256",
    ):
        if general_experiment.get(key) != forum_experiment.get(key):
            raise RuntimeError(f"paired completed runs differ in {key}")

    value: Dict[str, Any] = {
        "schema_version": PAIRED_EXPERIMENT_REPORT_SCHEMA,
        "experiment_id": general_experiment["experiment_id"],
        "baseline": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "continual_pretraining": False,
        },
        "allocation_sha256": general_experiment["allocation_sha256"],
        "schedule_template_sha256": general_experiment[
            "schedule_template_sha256"
        ],
        "paired_inputs_sha256": general_experiment["paired_inputs_sha256"],
        "training": general_resolved["training"],
        "execution": general_resolved["execution"],
        "arms": {
            arm: {
                "run_id": runs[arm]["manifest"]["run_id"],
                "artifact_id": artifacts[arm]["artifact_id"],
                "artifact_sha256": artifacts[arm]["artifact_sha256"],
                "elapsed_seconds": elapsed[arm],
                "optimizer_steps": artifacts[arm]["training"][
                    "optimizer_steps"
                ],
                "baseline_evaluation": runs[arm]["manifest"][
                    "baseline_evaluation"
                ],
                "final_evaluation": runs[arm]["manifest"]["final_evaluation"],
                "quality_gate_passed": runs[arm]["manifest"][
                    "quality_gate_passed"
                ],
                "promotion_status": runs[arm]["manifest"]["promotion_status"],
            }
            for arm in ("general", "forum_tech")
        },
        "redistribution_status": "internal_research_only",
    }
    value["report_id"] = sha256_bytes(canonical_json_bytes(value))[:20]
    return validate_paired_experiment_report(value)


def validate_paired_experiment_report(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "report_id",
        "experiment_id",
        "baseline",
        "allocation_sha256",
        "schedule_template_sha256",
        "paired_inputs_sha256",
        "training",
        "execution",
        "arms",
        "redistribution_status",
    }:
        raise RuntimeError("paired experiment report keys are incomplete or unknown")
    if value.get("schema_version") != PAIRED_EXPERIMENT_REPORT_SCHEMA:
        raise RuntimeError("unknown paired experiment report schema")
    if value.get("baseline") != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "continual_pretraining": False,
    }:
        raise RuntimeError("paired experiment baseline changed")
    if value.get("redistribution_status") != "internal_research_only":
        raise RuntimeError("paired experiment redistribution status changed")
    if not _ID_RE.fullmatch(value.get("experiment_id", "")):
        raise RuntimeError("paired experiment ID is invalid")
    for key in (
        "allocation_sha256",
        "schedule_template_sha256",
        "paired_inputs_sha256",
    ):
        if not _SHA256_RE.fullmatch(value.get(key, "")):
            raise RuntimeError(f"paired experiment {key} is invalid")
    arms = value.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"general", "forum_tech"}:
        raise RuntimeError("paired experiment arms are incomplete")
    for arm, record in arms.items():
        if (
            not isinstance(record, dict)
            or not _ID_RE.fullmatch(record.get("run_id", ""))
            or not _ID_RE.fullmatch(record.get("artifact_id", ""))
            or not _SHA256_RE.fullmatch(record.get("artifact_sha256", ""))
            or not _positive_int(record.get("elapsed_seconds"), f"{arm} elapsed")
            or record.get("optimizer_steps") != 52_000
            or not isinstance(record.get("baseline_evaluation"), dict)
            or not isinstance(record.get("final_evaluation"), dict)
            or not isinstance(record.get("quality_gate_passed"), bool)
            or record.get("promotion_status") not in {"eligible", "blocked"}
        ):
            raise RuntimeError(f"paired experiment {arm} result is invalid")
        _validate_evaluation(record["baseline_evaluation"], arm, "baseline")
        _validate_evaluation(record["final_evaluation"], arm, "final")
    without_id = {key: nested for key, nested in value.items() if key != "report_id"}
    if value.get("report_id") != sha256_bytes(canonical_json_bytes(without_id))[:20]:
        raise RuntimeError("paired experiment report ID changed")
    _assert_no_absolute_path_strings(value)
    return value


def _load_complete_run(path: Path) -> Dict[str, Dict[str, Any]]:
    root = path.expanduser()
    if root.is_symlink():
        raise RuntimeError("paired run directory must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError("paired run directory is missing")
    resolved = _read_json(root / "resolved_training.json")
    manifest = _read_json(root / "run_manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("optimizer_steps_completed") != 52_000
        or manifest.get("baseline_evaluation") is None
        or manifest.get("final_evaluation") is None
        or not isinstance(manifest.get("artifact"), dict)
        or resolved.get("run_id") != manifest.get("run_id")
    ):
        raise RuntimeError("paired run is incomplete")
    return {"resolved": resolved, "manifest": manifest}


def _read_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError("paired experiment input must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"paired experiment input is missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"paired experiment input is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("paired experiment input must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _validate_evaluation(value: Dict[str, Any], arm: str, phase: str) -> None:
    macro = value.get("macro")
    datasets = value.get("datasets")
    if (
        not isinstance(macro, dict)
        or not isinstance(datasets, dict)
        or set(datasets) != set(DATASET_IDS)
        or not _finite_number(macro.get("loss"))
        or not _finite_number(macro.get("perplexity"))
    ):
        raise RuntimeError(f"paired experiment {arm} {phase} evaluation is invalid")
    for metrics in datasets.values():
        if (
            not isinstance(metrics, dict)
            or not _finite_number(metrics.get("loss"))
            or not _finite_number(metrics.get("perplexity"))
        ):
            raise RuntimeError(
                f"paired experiment {arm} {phase} evaluation is invalid"
            )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _assert_no_absolute_path_strings(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError("paired experiment report contains an absolute path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a private-safe report for a completed paired CPT experiment"
    )
    parser.add_argument("--general-run-dir", type=Path, required=True)
    parser.add_argument("--forum-tech-run-dir", type=Path, required=True)
    parser.add_argument("--general-artifact", type=Path, required=True)
    parser.add_argument("--forum-tech-artifact", type=Path, required=True)
    parser.add_argument("--general-elapsed-seconds", type=int, required=True)
    parser.add_argument("--forum-tech-elapsed-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_paired_experiment_report(
        general_run_dir=args.general_run_dir,
        forum_tech_run_dir=args.forum_tech_run_dir,
        general_artifact=args.general_artifact,
        forum_tech_artifact=args.forum_tech_artifact,
        general_elapsed_seconds=args.general_elapsed_seconds,
        forum_tech_elapsed_seconds=args.forum_tech_elapsed_seconds,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
