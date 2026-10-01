from __future__ import annotations

import csv
import json
import math
import os
import statistics
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .classification_embeddings import (
    load_embedding_rows,
    require_validated_embeddings,
)
from .classification_eval_common import (
    EVALUATION_UNIT_SCHEMA,
    INPUT_VARIANTS,
    MODEL_NAMES,
    POOLINGS,
    REPORT_SCHEMA,
    REPORT_FILES_SCHEMA,
    SEEDS,
    SELECTION_SCHEMA,
    TASKS,
    TUNING_UNIT_SCHEMA,
    assert_safe_metadata,
    classification_dataset_path,
    finite_number,
    load_pinned_split,
    read_json,
    unit_by_index,
)
from .config import canonical_json_bytes, sha256_bytes
from .manifest import file_sha256, write_json_atomic


def load_private_labels(
    dataset_path: Path,
    requested_ids: Sequence[str],
    task: str,
) -> list[str]:
    import pyarrow.parquet as pq

    if task not in TASKS:
        raise ValueError("unknown classification task")
    needed = set(requested_ids)
    labels: Dict[str, str] = {}
    parquet = pq.ParquetFile(dataset_path / "examples.parquet")
    for batch in parquet.iter_batches(
        batch_size=16_384,
        columns=["sample_id", "category_id", "subcategory_id"],
    ):
        values = batch.to_pydict()
        for sample_id, category_id, subcategory_id in zip(
            values["sample_id"],
            values["category_id"],
            values["subcategory_id"],
        ):
            if sample_id in needed:
                labels[sample_id] = (
                    str(category_id)
                    if task == "coarse"
                    else f"{category_id}:{subcategory_id}"
                )
    if set(labels) != needed:
        raise RuntimeError("classification labels are missing")
    return [labels[value] for value in requested_ids]


def tune_unit(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
    unit_index: int,
) -> Dict[str, Any]:
    require_validated_embeddings(resolved, evaluation_dir)
    output_path = evaluation_dir / "tuning" / f"unit-{unit_index:02d}.json"
    if output_path.is_file():
        existing = read_json(output_path, "tuning unit")
        if (
            existing.get("schema_version") != TUNING_UNIT_SCHEMA
            or existing.get("evaluation_id") != resolved["evaluation_id"]
            or existing.get("unit_index") != unit_index
            or existing.get("counts", {}).get("test_accessed") != 0
            or existing.get("status") != "complete"
        ):
            raise RuntimeError("existing classification tuning unit is invalid")
        return existing
    unit = unit_by_index(config, unit_index)
    split = load_pinned_split(config, resolved, unit["task"], unit["seed"])
    train_ids = split["sample_ids"]["train"]
    validation_ids = split["sample_ids"]["validation"]
    dataset_path = classification_dataset_path(config)
    _verify_examples_file(dataset_path, resolved)
    combined_labels = load_private_labels(
        dataset_path, train_ids + validation_ids, unit["task"]
    )
    train_labels = combined_labels[: len(train_ids)]
    validation_labels = combined_labels[len(train_ids) :]
    if set(train_labels) != set(split["labels"]) or set(validation_labels) != set(
        split["labels"]
    ):
        raise RuntimeError("classification tuning split has absent classes")

    results = []
    for model_name in MODEL_NAMES:
        for pooling in POOLINGS:
            combined = load_embedding_rows(
                evaluation_dir,
                model_name,
                unit["input_variant"],
                pooling,
                train_ids + validation_ids,
            )
            train = combined[: len(train_ids)]
            validation = combined[len(train_ids) :]
            scaled_train, scaled_validation = _scale_train_validation(
                train, validation
            )
            for c_value in config["classifier"]["c_grid"]:
                model = _fit_classifier(
                    scaled_train,
                    train_labels,
                    c_value,
                    config["classifier"],
                    seed=unit["seed"],
                )
                predicted = model.predict(scaled_validation)
                metrics = _summary_metrics(validation_labels, predicted)
                results.append(
                    {
                        "model": model_name,
                        "pooling": pooling,
                        "c": c_value,
                        "validation": {
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                        },
                    }
                )
    value = {
        "schema_version": TUNING_UNIT_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        **unit,
        "benchmark_id": split["benchmark_id"],
        "counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test_accessed": 0,
        },
        "results": results,
        "status": "complete",
    }
    assert_safe_metadata(value)
    _write_identity_bound_json(output_path, value)
    return value


def select_hyperparameters(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
) -> Dict[str, Any]:
    require_validated_embeddings(resolved, evaluation_dir)
    units = _load_all_units(
        evaluation_dir / "tuning",
        "tuning unit",
        TUNING_UNIT_SCHEMA,
        resolved["evaluation_id"],
    )
    selected = []
    scores = []
    for task in TASKS:
        for input_variant in INPUT_VARIANTS:
            matching = [
                unit
                for unit in units
                if unit["task"] == task and unit["input_variant"] == input_variant
            ]
            if len(matching) != len(SEEDS):
                raise RuntimeError("classification tuning unit matrix is incomplete")
            if any(
                unit.get("counts", {}).get("test_accessed") != 0
                for unit in matching
            ):
                raise RuntimeError("classification tuning accessed the test split")
            candidates = []
            for pooling in POOLINGS:
                for c_value in config["classifier"]["c_grid"]:
                    records = [
                        result
                        for unit in matching
                        for result in unit["results"]
                        if result["pooling"] == pooling and result["c"] == c_value
                    ]
                    if len(records) != len(MODEL_NAMES) * len(SEEDS):
                        raise RuntimeError("classification tuning scores are incomplete")
                    macro_f1 = statistics.fmean(
                        record["validation"]["macro_f1"] for record in records
                    )
                    accuracy = statistics.fmean(
                        record["validation"]["accuracy"] for record in records
                    )
                    candidate = {
                        "task": task,
                        "input_variant": input_variant,
                        "pooling": pooling,
                        "c": c_value,
                        "mean_validation_macro_f1": macro_f1,
                        "mean_validation_accuracy": accuracy,
                        "observations": len(records),
                    }
                    candidates.append(candidate)
                    scores.append(candidate)
            winner = sorted(
                candidates,
                key=lambda value: (
                    -value["mean_validation_macro_f1"],
                    -value["mean_validation_accuracy"],
                    0 if value["pooling"] == "masked_mean" else 1,
                    value["c"],
                ),
            )[0]
            selected.append(
                {
                    "task": task,
                    "input_variant": input_variant,
                    "pooling": winner["pooling"],
                    "c": winner["c"],
                }
            )
    value = {
        "schema_version": SELECTION_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        "selection_scope": config["classifier"]["selection_scope"],
        "test_accessed": False,
        "scores": scores,
        "selected": selected,
        "status": "complete",
    }
    assert_safe_metadata(value)
    _write_identity_bound_json(evaluation_dir / "selection.json", value)
    return value


def evaluate_unit(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
    unit_index: int,
) -> Dict[str, Any]:
    require_validated_embeddings(resolved, evaluation_dir)
    output_path = (
        evaluation_dir / "evaluation_units" / f"unit-{unit_index:02d}.json"
    )
    if output_path.is_file():
        existing = read_json(output_path, "evaluation unit")
        private = existing.get("private_output", {})
        private_path = _private_output_path(evaluation_dir, private)
        if (
            existing.get("schema_version") != EVALUATION_UNIT_SCHEMA
            or existing.get("evaluation_id") != resolved["evaluation_id"]
            or existing.get("unit_index") != unit_index
            or existing.get("test_accessed") is not True
            or existing.get("status") != "complete"
            or not private_path.is_file()
            or file_sha256(private_path) != private.get("sha256")
        ):
            raise RuntimeError("existing classification evaluation unit is invalid")
        return existing
    selection = read_json(evaluation_dir / "selection.json", "selection")
    _validate_selection(selection, resolved["evaluation_id"])
    unit = unit_by_index(config, unit_index)
    selected = _selected_for(
        selection, unit["task"], unit["input_variant"]
    )
    split = load_pinned_split(config, resolved, unit["task"], unit["seed"])
    fit_ids = split["sample_ids"]["train"] + split["sample_ids"]["validation"]
    test_ids = split["sample_ids"]["test"]
    dataset_path = classification_dataset_path(config)
    _verify_examples_file(dataset_path, resolved)
    combined_labels = load_private_labels(
        dataset_path, fit_ids + test_ids, unit["task"]
    )
    fit_labels = combined_labels[: len(fit_ids)]
    test_labels = combined_labels[len(fit_ids) :]
    if set(fit_labels) != set(split["labels"]) or set(test_labels) != set(
        split["labels"]
    ):
        raise RuntimeError("classification final split has absent classes")

    predictions_by_model: Dict[str, Sequence[str]] = {}
    results = []
    for model_name in MODEL_NAMES:
        combined = load_embedding_rows(
            evaluation_dir,
            model_name,
            unit["input_variant"],
            selected["pooling"],
            fit_ids + test_ids,
        )
        fit = combined[: len(fit_ids)]
        test = combined[len(fit_ids) :]
        scaled_fit, scaled_test = _scale_train_validation(fit, test)
        classifier = _fit_classifier(
            scaled_fit,
            fit_labels,
            selected["c"],
            config["classifier"],
            seed=unit["seed"],
        )
        predicted = classifier.predict(scaled_test).tolist()
        predictions_by_model[model_name] = predicted
        metrics = _full_metrics(test_labels, predicted, split["labels"])
        results.append({"model": model_name, **metrics})
    private_path = (
        evaluation_dir / "private" / "predictions" / f"unit-{unit_index:02d}.parquet"
    )
    _write_private_predictions(
        private_path,
        unit,
        test_ids,
        test_labels,
        predictions_by_model,
    )
    value = {
        "schema_version": EVALUATION_UNIT_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        **unit,
        "benchmark_id": split["benchmark_id"],
        "selected": selected,
        "counts": {
            "train_validation": len(fit_ids),
            "test": len(test_ids),
        },
        "private_output": {
            "relative_path": str(private_path.relative_to(evaluation_dir)),
            "sha256": file_sha256(private_path),
            "rows": len(test_ids) * len(MODEL_NAMES),
        },
        "results": results,
        "test_accessed": True,
        "status": "complete",
    }
    assert_safe_metadata(value)
    _write_identity_bound_json(output_path, value)
    return value


def build_report(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
    *,
    write: bool = True,
) -> Dict[str, Any]:
    require_validated_embeddings(resolved, evaluation_dir)
    selection = read_json(evaluation_dir / "selection.json", "selection")
    _validate_selection(selection, resolved["evaluation_id"])
    units = _load_all_units(
        evaluation_dir / "evaluation_units",
        "evaluation unit",
        EVALUATION_UNIT_SCHEMA,
        resolved["evaluation_id"],
    )
    metrics_by_seed = []
    class_metrics = []
    confusion_matrices = []
    for unit in units:
        private = unit.get("private_output", {})
        private_path = _private_output_path(evaluation_dir, private)
        if (
            unit.get("test_accessed") is not True
            or not private_path.is_file()
            or file_sha256(private_path) != private.get("sha256")
        ):
            raise RuntimeError("classification private predictions changed")
        for result in unit["results"]:
            record = {
                "seed": unit["seed"],
                "task": unit["task"],
                "input_variant": unit["input_variant"],
                "model": result["model"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
            }
            metrics_by_seed.append(record)
            for label, values in result["by_class"].items():
                class_metrics.append(
                    {**record, "label": label, **values}
                )
            confusion_matrices.append(
                {
                    "seed": unit["seed"],
                    "task": unit["task"],
                    "input_variant": unit["input_variant"],
                    "model": result["model"],
                    "labels": result["labels"],
                    "matrix": result["confusion_matrix"],
                }
            )
    if len(metrics_by_seed) != 60:
        raise RuntimeError("classification final evaluation matrix is incomplete")
    summaries = []
    for task in TASKS:
        for input_variant in INPUT_VARIANTS:
            for model_name in MODEL_NAMES:
                matching = [
                    value
                    for value in metrics_by_seed
                    if value["task"] == task
                    and value["input_variant"] == input_variant
                    and value["model"] == model_name
                ]
                for metric in ("accuracy", "macro_f1"):
                    summaries.append(
                        {
                            "task": task,
                            "input_variant": input_variant,
                            "model": model_name,
                            "metric": metric,
                            **_student_summary([value[metric] for value in matching]),
                            "endpoint": _endpoint_kind(config, task, input_variant, metric),
                        }
                    )
    contrasts = []
    for task in TASKS:
        for input_variant in INPUT_VARIANTS:
            for metric in ("accuracy", "macro_f1"):
                for contrast in config["statistics"]["pairwise_contrasts"]:
                    first = _seed_metric_map(
                        metrics_by_seed,
                        task,
                        input_variant,
                        contrast["first"],
                        metric,
                    )
                    second = _seed_metric_map(
                        metrics_by_seed,
                        task,
                        input_variant,
                        contrast["second"],
                        metric,
                    )
                    deltas = [second[seed] - first[seed] for seed in SEEDS]
                    contrasts.append(
                        {
                            "task": task,
                            "input_variant": input_variant,
                            "metric": metric,
                            "contrast": contrast["name"],
                            "first": contrast["first"],
                            "second": contrast["second"],
                            "direction": "second_minus_first",
                            "values_by_seed": [
                                {"seed": seed, "delta": delta}
                                for seed, delta in zip(SEEDS, deltas)
                            ],
                            **_student_summary(deltas),
                            "endpoint": _endpoint_kind(
                                config, task, input_variant, metric
                            ),
                        }
                    )
    preflight = read_json(evaluation_dir / "preflight.json", "preflight")
    report = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        "identity": {
            "git_commit": resolved["git_commit"],
            "config_sha256": resolved["config_sha256"],
            "classification_dataset_id": resolved["classification_dataset_id"],
            "paired_report_id": resolved["paired_report_id"],
            "models": resolved["models"],
        },
        "selection": selection["selected"],
        "software": preflight["dependencies"],
        "metrics_by_seed": metrics_by_seed,
        "summary": summaries,
        "paired_contrasts": contrasts,
        "class_metrics": class_metrics,
        "confusion_matrices": confusion_matrices,
        "test_policy": {
            "selection_accessed_test": False,
            "final_evaluations": len(metrics_by_seed),
            "p_values_reported": False,
        },
        "primary_endpoint": config["statistics"]["primary_endpoint"],
        "redistribution_status": config["output"]["status"],
        "status": "complete",
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    assert_safe_metadata(report)
    if write:
        report_dir = evaluation_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        report_dir.chmod(0o700)
        _write_canonical_json(report_dir / "report.json", report)
        _write_csv(report_dir / "metrics_by_seed.csv", metrics_by_seed)
        _write_csv(report_dir / "class_metrics.csv", class_metrics)
        _write_csv(
            report_dir / "confusion_matrices.csv",
            _confusion_rows(confusion_matrices),
        )
        _write_csv(report_dir / "summary.csv", summaries)
        _write_csv(report_dir / "paired_contrasts.csv", contrasts)
        _write_csv(
            report_dir / "hyperparameter_scores.csv", selection["scores"]
        )
        _write_csv(
            report_dir / "selected_hyperparameters.csv", selection["selected"]
        )
        _write_canonical_json(
            report_dir / "confusion_matrices.json",
            {"matrices": confusion_matrices},
        )
        _write_canonical_json(
            report_dir / "selected_hyperparameters.json",
            {"selected": selection["selected"], "scores": selection["scores"]},
        )
        report_files = sorted(
            path
            for path in report_dir.iterdir()
            if path.is_file() and path.name != "report_files.json"
        )
        files_manifest = {
            "schema_version": REPORT_FILES_SCHEMA,
            "evaluation_id": resolved["evaluation_id"],
            "report_sha256": report["report_sha256"],
            "files": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in report_files
            ],
            "status": "complete",
        }
        _write_canonical_json(report_dir / "report_files.json", files_manifest)
    return report


def validate_report(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
) -> Dict[str, Any]:
    expected = build_report(config, resolved, evaluation_dir, write=False)
    actual = read_json(evaluation_dir / "report/report.json", "evaluation report")
    if actual != expected:
        raise RuntimeError("classification evaluation report changed")
    report_dir = evaluation_dir / "report"
    files_manifest = read_json(report_dir / "report_files.json", "report files")
    if (
        files_manifest.get("schema_version") != REPORT_FILES_SCHEMA
        or files_manifest.get("evaluation_id") != resolved["evaluation_id"]
        or files_manifest.get("report_sha256") != actual["report_sha256"]
        or files_manifest.get("status") != "complete"
    ):
        raise RuntimeError("classification evaluation report file manifest is invalid")
    expected_files = {
        "class_metrics.csv",
        "confusion_matrices.csv",
        "confusion_matrices.json",
        "hyperparameter_scores.csv",
        "metrics_by_seed.csv",
        "paired_contrasts.csv",
        "report.json",
        "selected_hyperparameters.csv",
        "selected_hyperparameters.json",
        "summary.csv",
    }
    actual_files = {
        path.name
        for path in report_dir.iterdir()
        if path.is_file() and path.name != "report_files.json"
    }
    records = files_manifest.get("files")
    if (
        actual_files != expected_files
        or not isinstance(records, list)
        or {record.get("path") for record in records} != expected_files
    ):
        raise RuntimeError("classification evaluation report file set changed")
    for record in records:
        path = report_dir / record["path"]
        if (
            path.stat().st_size != record.get("size_bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise RuntimeError("classification evaluation report file changed")
    return {
        "evaluation_id": resolved["evaluation_id"],
        "final_evaluations": actual["test_policy"]["final_evaluations"],
        "report_sha256": actual["report_sha256"],
        "status": "valid",
    }


def _scale_train_validation(train: Any, validation: Any) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=True)
    scaled_train = scaler.fit_transform(train).astype(np.float32, copy=False)
    scaled_validation = scaler.transform(validation).astype(np.float32, copy=False)
    if not np.isfinite(scaled_train).all() or not np.isfinite(scaled_validation).all():
        raise RuntimeError("classification scaled embeddings are non-finite")
    return scaled_train, scaled_validation


def _verify_examples_file(
    dataset_path: Path, resolved: Mapping[str, Any]
) -> None:
    manifest_path = dataset_path / "dataset_manifest.json"
    if file_sha256(manifest_path) != resolved["dataset_manifest_sha256"]:
        raise RuntimeError("classification dataset manifest changed after preflight")
    manifest = read_json(manifest_path, "classification dataset manifest")
    records = [
        record
        for record in manifest.get("files", [])
        if record.get("path") == "examples.parquet"
    ]
    examples_path = dataset_path / "examples.parquet"
    if (
        len(records) != 1
        or examples_path.stat().st_size != records[0].get("size_bytes")
        or file_sha256(examples_path) != records[0].get("sha256")
    ):
        raise RuntimeError("classification examples changed after preflight")


def _fit_classifier(
    train: Any,
    labels: Sequence[str],
    c_value: float,
    policy: Mapping[str, Any],
    *,
    seed: int,
) -> Any:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        C=c_value,
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


def _summary_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    value = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
    }
    if not all(finite_number(item) for item in value.values()):
        raise RuntimeError("classification metrics are non-finite")
    return value


def _full_metrics(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    summary = _summary_metrics(expected, predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(labels),
        zero_division=0,
    )
    by_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    result = {
        **summary,
        "labels": list(labels),
        "by_class": by_class,
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=list(labels)
        ).astype(int).tolist(),
    }
    if not all(
        finite_number(value)
        for metrics in by_class.values()
        for key, value in metrics.items()
        if key != "support"
    ):
        raise RuntimeError("classification class metrics are non-finite")
    return result


def _write_private_predictions(
    path: Path,
    unit: Mapping[str, Any],
    test_ids: Sequence[str],
    expected: Sequence[str],
    predicted: Mapping[str, Sequence[str]],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    rows = []
    for model_name in MODEL_NAMES:
        rows.extend(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "model": model_name,
                "seed": unit["seed"],
                "task": unit["task"],
                "input_variant": unit["input_variant"],
            }
            for sample_id, true_label, predicted_label in zip(
                test_ids, expected, predicted[model_name]
            )
        )
    table = pa.Table.from_pylist(rows)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        pq.write_table(table, partial, compression="zstd")
        partial.chmod(0o600)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _private_output_path(
    evaluation_dir: Path, private: Mapping[str, Any]
) -> Path:
    relative = private.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("classification private output path is missing")
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise RuntimeError("classification private output path is unsafe")
    root = evaluation_dir.resolve()
    path = (root / requested).resolve()
    if root not in path.parents:
        raise RuntimeError("classification private output escapes its root")
    return path


def _write_identity_bound_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.exists():
        if read_json(path, "classification evaluation output") != value:
            raise RuntimeError("existing classification evaluation output changed")
        return
    _write_canonical_json(path, value)


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, dict(value))
    path.chmod(0o600)


def _load_all_units(
    root: Path,
    description: str,
    schema: str,
    evaluation_id: str,
) -> list[Dict[str, Any]]:
    units = []
    for index in range(20):
        unit = read_json(root / f"unit-{index:02d}.json", description)
        if (
            unit.get("schema_version") != schema
            or unit.get("evaluation_id") != evaluation_id
            or unit.get("unit_index") != index
            or unit.get("status") != "complete"
        ):
            raise RuntimeError(f"{description} is invalid")
        units.append(unit)
    return units


def _validate_selection(value: Mapping[str, Any], evaluation_id: str) -> None:
    if (
        value.get("schema_version") != SELECTION_SCHEMA
        or value.get("evaluation_id") != evaluation_id
        or value.get("test_accessed") is not False
        or value.get("status") != "complete"
        or len(value.get("selected", [])) != 4
    ):
        raise RuntimeError("classification hyperparameter selection is invalid")


def _selected_for(
    selection: Mapping[str, Any], task: str, input_variant: str
) -> Mapping[str, Any]:
    matches = [
        value
        for value in selection["selected"]
        if value["task"] == task and value["input_variant"] == input_variant
    ]
    if len(matches) != 1:
        raise RuntimeError("classification selected hyperparameters are incomplete")
    return matches[0]


def _student_summary(values: Sequence[float]) -> Dict[str, Any]:
    if len(values) != 5 or not all(finite_number(value) for value in values):
        raise RuntimeError("classification Student interval requires five finite values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    critical = 2.7764451051977987
    half_width = critical * standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "confidence_interval_95": [mean - half_width, mean + half_width],
    }


def _seed_metric_map(
    values: Sequence[Mapping[str, Any]],
    task: str,
    input_variant: str,
    model: str,
    metric: str,
) -> Dict[int, float]:
    result = {
        value["seed"]: value[metric]
        for value in values
        if value["task"] == task
        and value["input_variant"] == input_variant
        and value["model"] == model
    }
    if set(result) != set(SEEDS):
        raise RuntimeError("classification paired seed matrix is incomplete")
    return result


def _endpoint_kind(
    config: Mapping[str, Any], task: str, input_variant: str, metric: str
) -> str:
    primary = config["statistics"]["primary_endpoint"]
    return (
        "primary"
        if (task, input_variant, metric)
        == (primary["task"], primary["input_variant"], primary["metric"])
        else "secondary"
    )


def _confusion_rows(
    matrices: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for value in matrices:
        labels = value["labels"]
        matrix = value["matrix"]
        if len(matrix) != len(labels) or any(
            len(row) != len(labels) for row in matrix
        ):
            raise RuntimeError("classification confusion matrix shape changed")
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                rows.append(
                    {
                        "seed": value["seed"],
                        "task": value["task"],
                        "input_variant": value["input_variant"],
                        "model": value["model"],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": matrix[true_index][predicted_index],
                    }
                )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("classification report CSV cannot be empty")
    values = []
    for row in rows:
        values.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
        )
    fieldnames = sorted({key for row in values for key in row})
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())
        partial.chmod(0o600)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
