from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import DATASET_IDS, canonical_json_bytes, sha256_bytes
from .manifest import write_json_atomic


CAPACITY_REPORT_SCHEMA = "queroquero-capacity-report/v1"
REAL_ALLOCATION_SCHEMA = "queroquero-real-allocation/v1"
REAL_ALLOCATION_POLICY = "equal_share_without_replacement"
REAL_TARGET_TRAIN_SEQUENCES = 416_000
REAL_EVAL_SEQUENCES_PER_DATASET = 256
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def capacity_report_id(report: Dict[str, Any]) -> str:
    value = {
        key: report[key]
        for key in (
            "schema_version",
            "dataset_id",
            "source_fingerprint",
            "scan_config_sha256",
            "tokenizer_fingerprint_sha256",
            "candidate_documents",
            "eval_sequences_requested",
            "train_sequence_capacity",
            "capacity_kind",
        )
    }
    return sha256_bytes(canonical_json_bytes(value))[:20]


def validate_capacity_report(report: Any) -> Dict[str, Any]:
    if not isinstance(report, dict):
        raise RuntimeError("capacity report must be an object")
    if report.get("schema_version") != CAPACITY_REPORT_SCHEMA:
        raise RuntimeError("unknown capacity report schema")
    dataset_id = report.get("dataset_id")
    if dataset_id not in DATASET_IDS:
        raise RuntimeError("capacity report dataset is unknown")
    if report.get("source_profile") != "mvp":
        raise RuntimeError("capacity report must use the full MVP source")
    if report.get("redistribution_status") != "internal_research_only":
        raise RuntimeError("capacity report redistribution policy changed")
    if report.get("eval_sequences_requested") != REAL_EVAL_SEQUENCES_PER_DATASET:
        raise RuntimeError("capacity report evaluation budget changed")
    for key in (
        "candidate_documents",
        "documents_tokenized",
        "eval_sequences_available",
        "train_sequence_capacity",
    ):
        value = report.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"capacity report {key} is invalid")
    if report["candidate_documents"] < 1:
        raise RuntimeError("capacity report candidate budget is invalid")
    if report["eval_sequences_available"] < REAL_EVAL_SEQUENCES_PER_DATASET:
        raise RuntimeError("capacity report cannot reserve the real evaluation split")
    if report.get("capacity_kind") not in {"exact", "lower_bound"}:
        raise RuntimeError("capacity report kind is invalid")
    for key in (
        "scan_config_sha256",
        "tokenizer_fingerprint_sha256",
        "source_fingerprint_sha256",
    ):
        value = report.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise RuntimeError(f"capacity report {key} is invalid")
    fingerprint = report.get("source_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("capacity report source fingerprint is missing")
    if report["source_fingerprint_sha256"] != sha256_bytes(
        canonical_json_bytes(fingerprint)
    ):
        raise RuntimeError("capacity report source fingerprint digest changed")
    report_id = report.get("capacity_report_id")
    if not isinstance(report_id, str) or not re.fullmatch(r"[0-9a-f]{20}", report_id):
        raise RuntimeError("capacity report ID is invalid")
    if report_id != capacity_report_id(report):
        raise RuntimeError("capacity report ID does not match its contents")
    _assert_no_absolute_path_strings(report)
    return report


def load_capacity_report(path: str | Path) -> Dict[str, Any]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise RuntimeError("capacity report must not be a symlink")
    try:
        report = json.loads(requested.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"capacity report is missing: {requested}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("capacity report is invalid JSON") from exc
    return validate_capacity_report(report)


def allocate_real_training(
    reports: Iterable[Dict[str, Any]],
    *,
    target_train_sequences: int = REAL_TARGET_TRAIN_SEQUENCES,
) -> Dict[str, Any]:
    if (
        not isinstance(target_train_sequences, int)
        or isinstance(target_train_sequences, bool)
        or target_train_sequences < 1
        or target_train_sequences % 8
    ):
        raise RuntimeError("real target must be a positive multiple of the global batch")
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for raw_report in reports:
        report = validate_capacity_report(raw_report)
        dataset_id = report["dataset_id"]
        if dataset_id in by_dataset:
            raise RuntimeError("real allocation received duplicate capacity reports")
        by_dataset[dataset_id] = report
    if set(by_dataset) != set(DATASET_IDS):
        raise RuntimeError("real allocation requires one capacity report per dataset")

    capacities = {
        dataset_id: by_dataset[dataset_id]["train_sequence_capacity"]
        for dataset_id in DATASET_IDS
    }
    if sum(capacities.values()) < target_train_sequences:
        incomplete = sorted(
            dataset_id
            for dataset_id, report in by_dataset.items()
            if report["capacity_kind"] == "lower_bound"
        )
        if incomplete:
            raise RuntimeError(
                "capacity audit is incomplete; expand lower-bound scans for: "
                + ", ".join(incomplete)
            )
        available = sum(capacities.values())
        maximum_steps = available // 8
        estimated_hours = 12.0 * available / REAL_TARGET_TRAIN_SEQUENCES
        raise RuntimeError(
            "unique dataset capacity is smaller than the requested real training "
            f"budget: available_sequences={available} "
            f"maximum_optimizer_steps={maximum_steps} "
            f"estimated_maximum_job_hours={estimated_hours:.2f}"
        )
    allocations = _waterfill(capacities, target_train_sequences)
    datasets = []
    for dataset_id in DATASET_IDS:
        report = by_dataset[dataset_id]
        datasets.append(
            {
                "dataset_id": dataset_id,
                "train_sequences": allocations[dataset_id],
                "eval_sequences": REAL_EVAL_SEQUENCES_PER_DATASET,
                "capacity_report_id": report["capacity_report_id"],
                "capacity_report_sha256": sha256_bytes(
                    canonical_json_bytes(report)
                ),
                "capacity_kind": report["capacity_kind"],
                "measured_train_capacity": report["train_sequence_capacity"],
                "candidate_documents": report["candidate_documents"],
            }
        )
    value = {
        "schema_version": REAL_ALLOCATION_SCHEMA,
        "policy": REAL_ALLOCATION_POLICY,
        "oversampling": False,
        "epochs": 1,
        "target_train_sequences": target_train_sequences,
        "target_train_tokens": target_train_sequences * 1024,
        "global_batch_sequences": 8,
        "total_optimizer_steps": target_train_sequences // 8,
        "eval_sequences_per_dataset": REAL_EVAL_SEQUENCES_PER_DATASET,
        "datasets": datasets,
    }
    value["allocation_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def validate_real_allocation(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("real allocation must be an object")
    if value.get("schema_version") != REAL_ALLOCATION_SCHEMA:
        raise RuntimeError("unknown real allocation schema")
    if (
        value.get("policy") != REAL_ALLOCATION_POLICY
        or value.get("oversampling") is not False
    ):
        raise RuntimeError("real allocation policy changed")
    if value.get("epochs") != 1:
        raise RuntimeError("real allocation must use one epoch")
    target = value.get("target_train_sequences")
    if target != REAL_TARGET_TRAIN_SEQUENCES:
        raise RuntimeError("real allocation training budget changed")
    if value.get("target_train_tokens") != target * 1024:
        raise RuntimeError("real allocation token budget changed")
    if value.get("global_batch_sequences") != 8:
        raise RuntimeError("real allocation global batch changed")
    if value.get("total_optimizer_steps") != target // 8:
        raise RuntimeError("real allocation optimizer steps changed")
    if value.get("eval_sequences_per_dataset") != REAL_EVAL_SEQUENCES_PER_DATASET:
        raise RuntimeError("real allocation evaluation budget changed")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != len(DATASET_IDS):
        raise RuntimeError("real allocation datasets are incomplete")
    if [item.get("dataset_id") for item in datasets if isinstance(item, dict)] != list(
        DATASET_IDS
    ):
        raise RuntimeError("real allocation dataset order changed")
    if sum(item.get("train_sequences", 0) for item in datasets) != target:
        raise RuntimeError("real allocation does not fill the target")
    for item in datasets:
        if (
            not isinstance(item.get("train_sequences"), int)
            or isinstance(item["train_sequences"], bool)
            or item["train_sequences"] < 1
            or item.get("eval_sequences") != REAL_EVAL_SEQUENCES_PER_DATASET
            or item.get("measured_train_capacity", -1) < item["train_sequences"]
            or item.get("capacity_kind") not in {"exact", "lower_bound"}
            or not isinstance(item.get("candidate_documents"), int)
            or item["candidate_documents"] < 1
        ):
            raise RuntimeError("real allocation dataset budget is invalid")
        for key in ("capacity_report_id",):
            if not isinstance(item.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{20}", item[key]
            ):
                raise RuntimeError("real allocation capacity report ID is invalid")
        digest = item.get("capacity_report_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RuntimeError("real allocation capacity report digest is invalid")
    expected_digest = value.get("allocation_sha256")
    without_digest = {
        key: nested
        for key, nested in value.items()
        if key != "allocation_sha256"
    }
    if not isinstance(expected_digest, str) or expected_digest != sha256_bytes(
        canonical_json_bytes(without_digest)
    ):
        raise RuntimeError("real allocation digest changed")
    _assert_no_absolute_path_strings(value)
    return value


def write_real_allocation(path: Path, value: Dict[str, Any]) -> None:
    validate_real_allocation(value)
    write_json_atomic(path, value)


def _waterfill(capacities: Dict[str, int], target: int) -> Dict[str, int]:
    allocations = {dataset_id: 0 for dataset_id in DATASET_IDS}
    active = list(DATASET_IDS)
    remaining = target
    while remaining:
        if not active:
            raise RuntimeError("unique dataset capacity was exhausted during allocation")
        share, remainder = divmod(remaining, len(active))
        exhausted = [
            dataset_id
            for dataset_id in active
            if capacities[dataset_id] - allocations[dataset_id] <= share
        ]
        if exhausted:
            for dataset_id in exhausted:
                available = capacities[dataset_id] - allocations[dataset_id]
                allocations[dataset_id] += available
                remaining -= available
            active = [dataset_id for dataset_id in active if dataset_id not in exhausted]
            continue
        for index, dataset_id in enumerate(active):
            addition = share + (1 if index < remainder else 0)
            allocations[dataset_id] += addition
            remaining -= addition
    return allocations


def _assert_no_absolute_path_strings(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError("real planning metadata contains an absolute path")
