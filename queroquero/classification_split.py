from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pyarrow.parquet as pq

from .classification_data import (
    REDISTRIBUTION_STATUS,
    load_classification_config,
    validate_classification_dataset,
)
from .config import canonical_json_bytes, sha256_bytes
from .datasets.base import stable_hash
from .manifest import file_sha256, write_json_atomic


SPLIT_MANIFEST_SCHEMA = "queroquero-classification-split-manifest/v1"


def create_classification_split(
    config_path: Path,
    dataset_path: Path,
    *,
    task: str,
    seed: int,
    output: Path,
) -> Dict[str, Any]:
    config, config_sha256 = load_classification_config(config_path)
    dataset_manifest = validate_classification_dataset(dataset_path)
    if dataset_manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("classification dataset and benchmark config differ")
    manifest = _build_split_manifest(
        config=config,
        config_sha256=config_sha256,
        dataset_path=dataset_path.resolve(),
        dataset_manifest=dataset_manifest,
        task=task,
        seed=seed,
    )
    requested = output.expanduser()
    if requested.is_symlink():
        raise RuntimeError("classification split output must not be a symlink")
    if requested.exists():
        try:
            existing = json.loads(requested.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("existing classification split is unreadable") from None
        if existing != manifest:
            raise RuntimeError("existing classification split changed")
    else:
        requested.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        requested.parent.chmod(0o700)
        write_json_atomic(requested, manifest)
        requested.chmod(0o600)
    return manifest


def _build_split_manifest(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    dataset_path: Path,
    dataset_manifest: Mapping[str, Any],
    task: str,
    seed: int,
) -> Dict[str, Any]:
    if task not in {"coarse", "fine"}:
        raise ValueError("classification task must be coarse or fine")
    if seed not in config["benchmark"]["seeds"]:
        raise ValueError("classification seed is not configured")
    table = pq.read_table(
        dataset_path / "examples.parquet",
        columns=[
            "sample_id",
            "category_id",
            "subcategory_id",
            "title_group_id",
        ],
    )
    task_config = config["benchmark"][task]
    allowed_categories = set(task_config["category_ids"])
    groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for row in table.to_pylist():
        category_id = row["category_id"]
        if category_id not in allowed_categories:
            continue
        label = (
            str(category_id)
            if task == "coarse"
            else f"{category_id}:{row['subcategory_id']}"
        )
        groups[row["title_group_id"]][label].append(row["sample_id"])

    representatives: Dict[str, List[str]] = defaultdict(list)
    conflicting_title_groups = 0
    for title_group_id in sorted(groups):
        labels = groups[title_group_id]
        if len(labels) != 1:
            conflicting_title_groups += 1
            continue
        label, sample_ids = next(iter(labels.items()))
        representative = min(
            sample_ids,
            key=lambda sample_id: (
                stable_hash(
                    "classification-representative/v1", seed, task, sample_id
                ),
                sample_id,
            ),
        )
        representatives[label].append(representative)

    support = {label: len(values) for label, values in representatives.items()}
    if task == "coarse":
        labels = [str(value) for value in task_config["category_ids"]]
        if any(support.get(label, 0) < 1 for label in labels):
            raise RuntimeError("coarse classification class has no eligible examples")
        class_budget = min(
            task_config["maximum_examples_per_class"],
            *(support[label] for label in labels),
        )
    else:
        labels = sorted(
            (
                label
                for label, count in support.items()
                if count >= task_config["minimum_unique_title_groups"]
            ),
            key=_fine_label_key,
        )
        if not labels:
            raise RuntimeError("fine classification has no eligible classes")
        class_budget = task_config["examples_per_class"]
        if any(support[label] < class_budget for label in labels):
            raise RuntimeError("fine classification class budget is unavailable")

    selected_by_label: Dict[str, List[str]] = {}
    for label in labels:
        selected_by_label[label] = sorted(
            representatives[label],
            key=lambda sample_id: (
                stable_hash("classification-select/v1", seed, task, sample_id),
                sample_id,
            ),
        )[:class_budget]

    percentages = config["benchmark"]["split_percentages"]
    train_count = class_budget * percentages["train"] // 100
    validation_count = class_budget * percentages["validation"] // 100
    split_ids: Dict[str, List[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    per_label_counts: Dict[str, Dict[str, int]] = {}
    for label in labels:
        ordered = sorted(
            selected_by_label[label],
            key=lambda sample_id: (
                stable_hash("classification-split/v1", seed, task, sample_id),
                sample_id,
            ),
        )
        train = ordered[:train_count]
        validation = ordered[train_count : train_count + validation_count]
        test = ordered[train_count + validation_count :]
        split_ids["train"].extend(train)
        split_ids["validation"].extend(validation)
        split_ids["test"].extend(test)
        per_label_counts[label] = {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "total": len(ordered),
        }
    for values in split_ids.values():
        values.sort()
    if any(len(values) != len(set(values)) for values in split_ids.values()):
        raise RuntimeError("classification split contains duplicate sample IDs")
    if (
        set(split_ids["train"]) & set(split_ids["validation"])
        or set(split_ids["train"]) & set(split_ids["test"])
        or set(split_ids["validation"]) & set(split_ids["test"])
    ):
        raise RuntimeError("classification splits overlap")

    value: Dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "classification_dataset_id": dataset_manifest["classification_dataset_id"],
        "dataset_manifest_sha256": file_sha256(
            dataset_path / "dataset_manifest.json"
        ),
        "config_sha256": config_sha256,
        "task": task,
        "seed": seed,
        "label_encoding": (
            "category_id" if task == "coarse" else "category_id:subcategory_id"
        ),
        "labels": labels,
        "class_budget": class_budget,
        "selection": {
            "title_group_policy": "one_deterministic_representative/v1",
            "balance_policy": "equal_examples_per_class/v1",
            "conflicting_title_groups_excluded": conflicting_title_groups,
            "eligible_unique_title_groups_by_label": {
                label: support[label] for label in labels
            },
        },
        "split_percentages": dict(percentages),
        "per_label_counts": per_label_counts,
        "counts": {
            "train": len(split_ids["train"]),
            "validation": len(split_ids["validation"]),
            "test": len(split_ids["test"]),
            "total": sum(len(values) for values in split_ids.values()),
        },
        "sample_ids": split_ids,
        "input_variants": list(config["benchmark"]["input_variants"]),
        "compatible_models": ["base", "general", "forum"],
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    benchmark_id = sha256_bytes(canonical_json_bytes(value))[:20]
    return {"benchmark_id": benchmark_id, **value}


def validate_classification_split(
    config_path: Path, dataset_path: Path, split_path: Path
) -> Dict[str, Any]:
    config, config_sha256 = load_classification_config(config_path)
    dataset_manifest = validate_classification_dataset(dataset_path)
    requested = split_path.expanduser()
    if requested.is_symlink():
        raise RuntimeError("classification split manifest must not be a symlink")
    try:
        manifest = json.loads(requested.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("classification split manifest is unreadable") from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("classification_dataset_id")
        != dataset_manifest["classification_dataset_id"]
        or manifest.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise RuntimeError("classification split identity is invalid")
    expected = _build_split_manifest(
        config=config,
        config_sha256=config_sha256,
        dataset_path=dataset_path.resolve(),
        dataset_manifest=dataset_manifest,
        task=manifest.get("task"),
        seed=manifest.get("seed"),
    )
    if manifest != expected:
        raise RuntimeError("classification split manifest changed")
    return manifest


def _fine_label_key(value: str) -> tuple[int, int]:
    category, subcategory = value.split(":", 1)
    return int(category), int(subcategory)
