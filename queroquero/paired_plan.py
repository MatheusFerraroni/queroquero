from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from .config import DATASET_IDS, canonical_json_bytes, sha256_bytes
from .manifest import write_json_atomic
from .real_plan import validate_capacity_report


PAIRED_REAL_ALLOCATION_SCHEMA = "queroquero-paired-real-allocation/v1"
PAIRED_REAL_POLICY = "matched_domain_substitution_without_replacement"
PAIRED_PREPARATION_PROFILE = "paired_real"
PAIRED_ARMS = ("general", "forum_tech")
SHARED_DATASET_IDS = (
    "brwac",
    "gigaverbo",
    "multiwoz_ptbr",
    "wackywacky",
)
DOMAIN_DATASET_IDS = ("adrenaline", "outerspace")
TARGET_TRAIN_SEQUENCES = 416_000
EVAL_SEQUENCES_PER_DATASET = 256
GLOBAL_BATCH_SEQUENCES = 8
SEQUENCE_LENGTH = 1024
SEED = 42
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[0-9a-f]{20}\Z")


def allocate_paired_real_training(
    reports: Iterable[Dict[str, Any]],
    *,
    target_train_sequences: int = TARGET_TRAIN_SEQUENCES,
) -> Dict[str, Any]:
    if (
        not isinstance(target_train_sequences, int)
        or isinstance(target_train_sequences, bool)
        or target_train_sequences < 1
        or target_train_sequences % GLOBAL_BATCH_SEQUENCES
    ):
        raise RuntimeError(
            "paired real target must be a positive multiple of the global batch"
        )
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for raw_report in reports:
        report = validate_capacity_report(raw_report)
        dataset_id = report["dataset_id"]
        if dataset_id in by_dataset:
            raise RuntimeError(
                "paired allocation received duplicate capacity reports"
            )
        by_dataset[dataset_id] = report
    if set(by_dataset) != set(DATASET_IDS):
        raise RuntimeError(
            "paired allocation requires one capacity report per dataset"
        )

    forum_allocations = _allocate_forum_tech(by_dataset, target_train_sequences)
    if any(value < 1 for value in forum_allocations.values()):
        raise RuntimeError(
            "paired allocation requires a positive forum_tech budget for every dataset"
        )
    replacement_sequences = sum(
        forum_allocations[dataset_id] for dataset_id in DOMAIN_DATASET_IDS
    )
    brwac_common = forum_allocations["brwac"]
    brwac_required = brwac_common + replacement_sequences
    brwac_report = by_dataset["brwac"]
    brwac_capacity = brwac_report["train_sequence_capacity"]
    if brwac_capacity < brwac_required:
        if brwac_report["capacity_kind"] == "lower_bound":
            raise RuntimeError(
                "paired BrWaC capacity audit is incomplete; expand brwac scan: "
                f"proven_sequences={brwac_capacity} "
                f"required_sequences={brwac_required}"
            )
        maximum_replacement = max(0, brwac_capacity - brwac_common)
        maximum_total = (
            sum(forum_allocations[dataset_id] for dataset_id in SHARED_DATASET_IDS)
            + min(replacement_sequences, maximum_replacement)
        )
        raise RuntimeError(
            "unique BrWaC capacity cannot provide the common and replacement pools: "
            f"available_sequences={brwac_capacity} "
            f"required_sequences={brwac_required} "
            f"maximum_replacement_sequences={maximum_replacement} "
            f"maximum_paired_optimizer_steps={maximum_total // GLOBAL_BATCH_SEQUENCES}"
        )

    general_allocations = {
        dataset_id: (
            forum_allocations[dataset_id]
            if dataset_id in SHARED_DATASET_IDS
            else 0
        )
        for dataset_id in DATASET_IDS
    }
    general_allocations["brwac"] += replacement_sequences
    prepared_train_sequences = dict(forum_allocations)
    prepared_train_sequences["brwac"] = brwac_required

    pools = [
        _pool("brwac_common", "brwac", "shared", 0, brwac_common),
        _pool(
            "gigaverbo_shared",
            "gigaverbo",
            "shared",
            0,
            forum_allocations["gigaverbo"],
        ),
        _pool(
            "multiwoz_ptbr_shared",
            "multiwoz_ptbr",
            "shared",
            0,
            forum_allocations["multiwoz_ptbr"],
        ),
        _pool(
            "wackywacky_shared",
            "wackywacky",
            "shared",
            0,
            forum_allocations["wackywacky"],
        ),
        _pool(
            "adrenaline_domain",
            "adrenaline",
            "domain",
            0,
            forum_allocations["adrenaline"],
        ),
        _pool(
            "outerspace_domain",
            "outerspace",
            "domain",
            0,
            forum_allocations["outerspace"],
        ),
        _pool(
            "brwac_extra",
            "brwac",
            "replacement",
            brwac_common,
            replacement_sequences,
        ),
    ]
    slot_counts = {
        _forum_pool_id(dataset_id): forum_allocations[dataset_id]
        for dataset_id in DATASET_IDS
    }
    schedule_template_sha256 = _schedule_template_sha256(slot_counts)
    report_records = [
        {
            "dataset_id": dataset_id,
            "capacity_report_id": by_dataset[dataset_id]["capacity_report_id"],
            "capacity_report_sha256": sha256_bytes(
                canonical_json_bytes(by_dataset[dataset_id])
            ),
            "capacity_kind": by_dataset[dataset_id]["capacity_kind"],
            "measured_train_capacity": by_dataset[dataset_id][
                "train_sequence_capacity"
            ],
            "candidate_documents": by_dataset[dataset_id]["candidate_documents"],
        }
        for dataset_id in DATASET_IDS
    ]
    value: Dict[str, Any] = {
        "schema_version": PAIRED_REAL_ALLOCATION_SCHEMA,
        "policy": PAIRED_REAL_POLICY,
        "oversampling": False,
        "epochs": 1,
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "target_train_sequences_per_arm": target_train_sequences,
        "target_train_tokens_per_arm": target_train_sequences * SEQUENCE_LENGTH,
        "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
        "total_optimizer_steps_per_arm": (
            target_train_sequences // GLOBAL_BATCH_SEQUENCES
        ),
        "eval_sequences_per_dataset": EVAL_SEQUENCES_PER_DATASET,
        "capacity_reports": report_records,
        "prepared_train_sequences": prepared_train_sequences,
        "forum_tech_allocations": forum_allocations,
        "general_allocations": general_allocations,
        "pools": pools,
        "schedule": {
            "policy": "smooth_proportional_paired_slots/v1",
            "slot_counts": slot_counts,
            "domain_replacement_pool_id": "brwac_extra",
            "schedule_template_sha256": schedule_template_sha256,
        },
    }
    allocation_sha256 = sha256_bytes(canonical_json_bytes(value))
    value["allocation_sha256"] = allocation_sha256
    value["experiment_id"] = allocation_sha256[:20]
    return validate_paired_real_allocation(value)


def validate_paired_real_allocation(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("paired allocation must be an object")
    required = {
        "schema_version",
        "experiment_id",
        "policy",
        "oversampling",
        "epochs",
        "seed",
        "sequence_length",
        "target_train_sequences_per_arm",
        "target_train_tokens_per_arm",
        "global_batch_sequences",
        "total_optimizer_steps_per_arm",
        "eval_sequences_per_dataset",
        "capacity_reports",
        "prepared_train_sequences",
        "forum_tech_allocations",
        "general_allocations",
        "pools",
        "schedule",
        "allocation_sha256",
    }
    if set(value) != required:
        raise RuntimeError("paired allocation keys are incomplete or unknown")
    if value.get("schema_version") != PAIRED_REAL_ALLOCATION_SCHEMA:
        raise RuntimeError("unknown paired allocation schema")
    if (
        value.get("policy") != PAIRED_REAL_POLICY
        or value.get("oversampling") is not False
        or value.get("epochs") != 1
        or value.get("seed") != SEED
        or value.get("sequence_length") != SEQUENCE_LENGTH
    ):
        raise RuntimeError("paired allocation policy changed")
    target = value.get("target_train_sequences_per_arm")
    if target != TARGET_TRAIN_SEQUENCES:
        raise RuntimeError("paired allocation training budget changed")
    if (
        value.get("target_train_tokens_per_arm") != target * SEQUENCE_LENGTH
        or value.get("global_batch_sequences") != GLOBAL_BATCH_SEQUENCES
        or value.get("total_optimizer_steps_per_arm")
        != target // GLOBAL_BATCH_SEQUENCES
        or value.get("eval_sequences_per_dataset")
        != EVAL_SEQUENCES_PER_DATASET
    ):
        raise RuntimeError("paired allocation execution budget changed")

    forum = _allocation_map(value.get("forum_tech_allocations"), positive=True)
    general = _allocation_map(value.get("general_allocations"), positive=False)
    prepared = _allocation_map(value.get("prepared_train_sequences"), positive=True)
    if sum(forum.values()) != target or sum(general.values()) != target:
        raise RuntimeError("paired allocation does not fill both arms")
    replacement = sum(forum[dataset_id] for dataset_id in DOMAIN_DATASET_IDS)
    expected_general = {
        dataset_id: (forum[dataset_id] if dataset_id in SHARED_DATASET_IDS else 0)
        for dataset_id in DATASET_IDS
    }
    expected_general["brwac"] += replacement
    if general != expected_general:
        raise RuntimeError("paired general arm is not a token-for-token replacement")
    expected_prepared = dict(forum)
    expected_prepared["brwac"] += replacement
    if prepared != expected_prepared:
        raise RuntimeError("paired prepared budgets do not cover both arms")

    reports = value.get("capacity_reports")
    if not isinstance(reports, list) or len(reports) != len(DATASET_IDS):
        raise RuntimeError("paired capacity report records are incomplete")
    if [item.get("dataset_id") for item in reports if isinstance(item, dict)] != list(
        DATASET_IDS
    ):
        raise RuntimeError("paired capacity report order changed")
    for item in reports:
        if set(item) != {
            "dataset_id",
            "capacity_report_id",
            "capacity_report_sha256",
            "capacity_kind",
            "measured_train_capacity",
            "candidate_documents",
        }:
            raise RuntimeError("paired capacity report record is invalid")
        if (
            not _ID_RE.fullmatch(item["capacity_report_id"])
            or not _SHA256_RE.fullmatch(item["capacity_report_sha256"])
            or item["capacity_kind"] not in {"exact", "lower_bound"}
            or not _positive_integer(item["measured_train_capacity"])
            or not _positive_integer(item["candidate_documents"])
            or item["measured_train_capacity"]
            < prepared[item["dataset_id"]]
        ):
            raise RuntimeError("paired capacity report does not prove its budget")

    pools = value.get("pools")
    if not isinstance(pools, list) or len(pools) != 7:
        raise RuntimeError("paired pools are incomplete")
    expected_pools = [
        _pool("brwac_common", "brwac", "shared", 0, forum["brwac"]),
        _pool("gigaverbo_shared", "gigaverbo", "shared", 0, forum["gigaverbo"]),
        _pool(
            "multiwoz_ptbr_shared",
            "multiwoz_ptbr",
            "shared",
            0,
            forum["multiwoz_ptbr"],
        ),
        _pool("wackywacky_shared", "wackywacky", "shared", 0, forum["wackywacky"]),
        _pool("adrenaline_domain", "adrenaline", "domain", 0, forum["adrenaline"]),
        _pool("outerspace_domain", "outerspace", "domain", 0, forum["outerspace"]),
        _pool("brwac_extra", "brwac", "replacement", forum["brwac"], replacement),
    ]
    if pools != expected_pools:
        raise RuntimeError("paired pool ranges changed")

    schedule = value.get("schedule")
    expected_slot_counts = {
        _forum_pool_id(dataset_id): forum[dataset_id]
        for dataset_id in DATASET_IDS
    }
    if not isinstance(schedule, dict) or set(schedule) != {
        "policy",
        "slot_counts",
        "domain_replacement_pool_id",
        "schedule_template_sha256",
    }:
        raise RuntimeError("paired schedule metadata is invalid")
    if (
        schedule["policy"] != "smooth_proportional_paired_slots/v1"
        or schedule["slot_counts"] != expected_slot_counts
        or schedule["domain_replacement_pool_id"] != "brwac_extra"
        or schedule["schedule_template_sha256"]
        != _schedule_template_sha256(expected_slot_counts)
    ):
        raise RuntimeError("paired schedule template changed")

    digest = value.get("allocation_sha256")
    without_identity = {
        key: nested
        for key, nested in value.items()
        if key not in {"allocation_sha256", "experiment_id"}
    }
    if (
        not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
        or digest != sha256_bytes(canonical_json_bytes(without_identity))
        or value.get("experiment_id") != digest[:20]
    ):
        raise RuntimeError("paired allocation identity changed")
    _assert_no_absolute_path_strings(value)
    return value


def paired_mixture_for_arm(
    allocation: Dict[str, Any], arm: str
) -> Dict[str, Any]:
    value = validate_paired_real_allocation(allocation)
    if arm not in PAIRED_ARMS:
        raise RuntimeError("paired training arm is unknown")
    return {
        "policy": PAIRED_REAL_POLICY,
        "without_replacement": True,
        "preparation_profile": PAIRED_PREPARATION_PROFILE,
        "experiment_id": value["experiment_id"],
        "arm": arm,
        "allocation_sha256": value["allocation_sha256"],
        "schedule_template_sha256": value["schedule"][
            "schedule_template_sha256"
        ],
        "pools": value["pools"],
    }


def validate_paired_mixture(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "without_replacement",
        "preparation_profile",
        "experiment_id",
        "arm",
        "allocation_sha256",
        "schedule_template_sha256",
        "pools",
    }:
        raise RuntimeError("paired data mixture keys are incomplete or unknown")
    if (
        value.get("policy") != PAIRED_REAL_POLICY
        or value.get("without_replacement") is not True
        or value.get("preparation_profile") != PAIRED_PREPARATION_PROFILE
        or value.get("arm") not in PAIRED_ARMS
        or not isinstance(value.get("experiment_id"), str)
        or not _ID_RE.fullmatch(value["experiment_id"])
        or not isinstance(value.get("allocation_sha256"), str)
        or not _SHA256_RE.fullmatch(value["allocation_sha256"])
        or value["experiment_id"] != value["allocation_sha256"][:20]
        or not isinstance(value.get("schedule_template_sha256"), str)
        or not _SHA256_RE.fullmatch(value["schedule_template_sha256"])
    ):
        raise RuntimeError("paired data mixture contract changed")
    pools = value.get("pools")
    if not isinstance(pools, list) or len(pools) != 7:
        raise RuntimeError("paired data mixture pools are incomplete")
    pool_ids = set()
    for pool in pools:
        if (
            not isinstance(pool, dict)
            or set(pool)
            != {"pool_id", "dataset_id", "role", "start_row", "train_sequences"}
            or pool.get("dataset_id") not in DATASET_IDS
            or pool.get("role") not in {"shared", "domain", "replacement"}
            or not isinstance(pool.get("pool_id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", pool["pool_id"])
            or pool["pool_id"] in pool_ids
            or not isinstance(pool.get("start_row"), int)
            or isinstance(pool.get("start_row"), bool)
            or pool["start_row"] < 0
            or not _positive_integer(pool.get("train_sequences"))
        ):
            raise RuntimeError("paired data mixture pool is invalid")
        pool_ids.add(pool["pool_id"])
    expected_ids = {
        "brwac_common",
        "gigaverbo_shared",
        "multiwoz_ptbr_shared",
        "wackywacky_shared",
        "adrenaline_domain",
        "outerspace_domain",
        "brwac_extra",
    }
    if pool_ids != expected_ids:
        raise RuntimeError("paired data mixture pool IDs changed")
    by_id = {pool["pool_id"]: pool for pool in pools}
    expected_contracts = {
        "brwac_common": ("brwac", "shared", 0),
        "gigaverbo_shared": ("gigaverbo", "shared", 0),
        "multiwoz_ptbr_shared": ("multiwoz_ptbr", "shared", 0),
        "wackywacky_shared": ("wackywacky", "shared", 0),
        "adrenaline_domain": ("adrenaline", "domain", 0),
        "outerspace_domain": ("outerspace", "domain", 0),
    }
    if any(
        (
            by_id[pool_id]["dataset_id"],
            by_id[pool_id]["role"],
            by_id[pool_id]["start_row"],
        )
        != contract
        for pool_id, contract in expected_contracts.items()
    ):
        raise RuntimeError("paired data mixture pool contract changed")
    if (
        by_id["brwac_extra"]["dataset_id"] != "brwac"
        or by_id["brwac_extra"]["role"] != "replacement"
        or by_id["brwac_extra"]["start_row"]
        != by_id["brwac_common"]["train_sequences"]
    ):
        raise RuntimeError("paired BrWaC pool ranges overlap or changed")
    domain_total = sum(
        by_id[f"{dataset_id}_domain"]["train_sequences"]
        for dataset_id in DOMAIN_DATASET_IDS
    )
    if by_id["brwac_extra"]["train_sequences"] != domain_total:
        raise RuntimeError("paired replacement pool is not token-for-token")
    slot_counts = {
        pool_id: pool["train_sequences"]
        for pool_id, pool in by_id.items()
        if pool["role"] != "replacement"
    }
    if sum(slot_counts.values()) != TARGET_TRAIN_SEQUENCES:
        raise RuntimeError("paired mixture does not contain 416000 slots")
    if value["schedule_template_sha256"] != _schedule_template_sha256(slot_counts):
        raise RuntimeError("paired mixture schedule digest changed")
    _assert_no_absolute_path_strings(value)
    return value


def iter_paired_schedule_slots(value: Dict[str, Any]) -> Iterator[str]:
    allocation = validate_paired_real_allocation(value)
    yield from _smooth_interleave(allocation["schedule"]["slot_counts"])


def iter_paired_mixture_slots(value: Dict[str, Any]) -> Iterator[str]:
    mixture = validate_paired_mixture(value)
    counts = {
        pool["pool_id"]: pool["train_sequences"]
        for pool in mixture["pools"]
        if pool["role"] != "replacement"
    }
    yield from _smooth_interleave(counts)


def write_paired_real_allocation(path: Path, value: Dict[str, Any]) -> None:
    validate_paired_real_allocation(value)
    write_json_atomic(path, value)


def _allocate_forum_tech(
    reports: Dict[str, Dict[str, Any]], target: int
) -> Dict[str, int]:
    allocations = {dataset_id: 0 for dataset_id in DATASET_IDS}
    active = list(DATASET_IDS)
    remaining = target
    while remaining:
        if not active:
            available = sum(allocations.values())
            raise RuntimeError(
                "unique dataset capacity is smaller than the paired training budget: "
                f"available_sequences={available} "
                f"maximum_optimizer_steps={available // GLOBAL_BATCH_SEQUENCES}"
            )
        share, remainder = divmod(remaining, len(active))
        proposed = {
            dataset_id: allocations[dataset_id]
            + share
            + (1 if index < remainder else 0)
            for index, dataset_id in enumerate(active)
        }
        incomplete = [
            dataset_id
            for dataset_id in active
            if reports[dataset_id]["train_sequence_capacity"]
            < proposed[dataset_id]
            and reports[dataset_id]["capacity_kind"] == "lower_bound"
        ]
        if incomplete:
            requirements = ", ".join(
                f"{dataset_id}>={proposed[dataset_id]}"
                for dataset_id in incomplete
            )
            raise RuntimeError(
                "paired capacity audit is incomplete; expand lower-bound scans for: "
                + requirements
            )
        exhausted = [
            dataset_id
            for dataset_id in active
            if reports[dataset_id]["train_sequence_capacity"]
            < proposed[dataset_id]
        ]
        if exhausted:
            for dataset_id in exhausted:
                capacity = reports[dataset_id]["train_sequence_capacity"]
                remaining -= capacity - allocations[dataset_id]
                allocations[dataset_id] = capacity
            active = [
                dataset_id for dataset_id in active if dataset_id not in exhausted
            ]
            continue
        for dataset_id in active:
            addition = proposed[dataset_id] - allocations[dataset_id]
            allocations[dataset_id] += addition
            remaining -= addition
    return allocations


def _smooth_interleave(counts: Dict[str, int]) -> Iterator[str]:
    total = sum(counts.values())
    current = {pool_id: 0 for pool_id in counts}
    consumed = {pool_id: 0 for pool_id in counts}
    tie_order = sorted(
        counts,
        key=lambda pool_id: sha256_bytes(
            canonical_json_bytes([SEED, "paired_interleave", pool_id])
        ),
    )
    tie_rank = {pool_id: index for index, pool_id in enumerate(tie_order)}
    for _ in range(total):
        active = [
            pool_id
            for pool_id in counts
            if consumed[pool_id] < counts[pool_id]
        ]
        for pool_id in active:
            current[pool_id] += counts[pool_id]
        selected = max(
            active,
            key=lambda pool_id: (current[pool_id], -tie_rank[pool_id]),
        )
        current[selected] -= total
        consumed[selected] += 1
        yield selected


def _schedule_template_sha256(counts: Dict[str, int]) -> str:
    digest = hashlib.sha256()
    for pool_id in _smooth_interleave(counts):
        digest.update(pool_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _forum_pool_id(dataset_id: str) -> str:
    if dataset_id == "brwac":
        return "brwac_common"
    if dataset_id in DOMAIN_DATASET_IDS:
        return f"{dataset_id}_domain"
    return f"{dataset_id}_shared"


def _pool(
    pool_id: str,
    dataset_id: str,
    role: str,
    start_row: int,
    train_sequences: int,
) -> Dict[str, Any]:
    return {
        "pool_id": pool_id,
        "dataset_id": dataset_id,
        "role": role,
        "start_row": start_row,
        "train_sequences": train_sequences,
    }


def _allocation_map(value: Any, *, positive: bool) -> Dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(DATASET_IDS):
        raise RuntimeError("paired dataset allocation is incomplete")
    for item in value.values():
        valid = (
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= (1 if positive else 0)
        )
        if not valid:
            raise RuntimeError("paired dataset allocation is invalid")
    return value


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _assert_no_absolute_path_strings(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError("paired planning metadata contains an absolute path")
