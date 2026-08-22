from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pyarrow.parquet as pq

from .config import (
    DATASET_IDS,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ROOT,
    canonical_json_bytes,
    load_resolved_config,
    resolve_output_root,
    scan_config_sha256,
    sha256_bytes,
    validate_dataset_config,
)
from .manifest import (
    DATASET_MANIFEST_SCHEMA,
    METRICS_SCHEMA,
    PROGRESS_SCHEMA,
    file_sha256,
    preparation_id,
    write_json_atomic,
)
from .packing import (
    clean_deduplicate_and_tokenize,
    measure_unique_sequence_capacity,
    pack_for_budgets,
    plan_incremental_packing,
    tokenizer_fingerprint,
)
from .real_plan import (
    CAPACITY_REPORT_SCHEMA,
    REAL_EVAL_SEQUENCES_PER_DATASET,
    REAL_TARGET_TRAIN_SEQUENCES,
    allocate_real_training,
    capacity_report_id,
    load_capacity_report,
    validate_capacity_report,
    write_real_allocation,
)
from .paired_plan import (
    PAIRED_PREPARATION_PROFILE,
    allocate_paired_real_training,
    paired_mixture_for_arm,
    validate_paired_real_allocation,
    write_paired_real_allocation,
)
from .storage import WorkStore, validate_shard, write_split, write_split_incremental


LOGGER = logging.getLogger("queroquero.prepare")
_PROGRESS_CURSOR_KEYS = (
    "next_selection_index",
    "next_member_index",
    "next_file_index",
    "next_dialogue_index",
    "dialogues_seen",
    "conversations_seen",
    "messages_seen",
    "records_seen",
    "row_number",
    "documents_selected",
    "documents_emitted",
    "complete",
)


class ReviewRequired(RuntimeError):
    exit_code = 20


def load_adapter(name: str) -> Any:
    module = importlib.import_module(f"queroquero.datasets.{name}")
    try:
        return module.ADAPTER
    except AttributeError as exc:
        raise RuntimeError(f"dataset adapter {name!r} does not expose ADAPTER") from exc


def run_preparation(dataset_id: str, profile: str, config_root: Path | None = None) -> Path:
    LOGGER.info("stage=config status=started dataset=%s profile=%s", dataset_id, profile)
    resolved, resolved_sha256 = load_resolved_config(dataset_id, profile, config_root)
    output_root = resolve_output_root(resolved["preparation"]["output_root"])
    LOGGER.info(
        "stage=config status=complete dataset=%s profile=%s output_root=%s",
        dataset_id,
        profile,
        _project_relative(output_root),
    )
    work = WorkStore(output_root, dataset_id, scan_config_sha256(resolved))
    resume_cursor, resume_documents = work.load()
    if resume_cursor is not None:
        LOGGER.info(
            "stage=scan status=resumed dataset=%s documents=%d%s",
            dataset_id,
            len(resume_documents),
            _cursor_progress(resume_cursor),
        )

    def save_checkpoint(cursor: Dict[str, Any], documents: list[Any]) -> None:
        work.checkpoint(cursor, documents)
        LOGGER.info(
            "stage=scan status=checkpoint dataset=%s documents=%d%s",
            dataset_id,
            len(documents),
            _cursor_progress(cursor),
        )

    adapter = load_adapter(resolved["dataset"]["adapter"])
    LOGGER.info("stage=scan status=started dataset=%s", dataset_id)
    scan = adapter.scan(
        resolved,
        resume_cursor=resume_cursor,
        resume_documents=resume_documents,
        checkpoint=save_checkpoint,
    )
    save_checkpoint(scan.resume_cursor or scan.cursor, scan.documents)
    LOGGER.info(
        "stage=scan status=complete dataset=%s documents=%d",
        dataset_id,
        len(scan.documents),
    )

    tokenizer_config = resolved["preparation"]["tokenizer"]
    tokenizer_identity_sha256 = sha256_bytes(canonical_json_bytes(tokenizer_config))
    run_id = preparation_id(
        resolved_sha256, tokenizer_identity_sha256, scan.source_fingerprint
    )
    output_dir = output_root / dataset_id / run_id
    manifest_path = output_dir / "dataset_manifest.json"
    if manifest_path.exists():
        LOGGER.info("stage=validate status=started dataset=%s existing=true", dataset_id)
        validate_preparation(output_dir)
        work.cleanup()
        LOGGER.info("stage=validate status=complete dataset=%s existing=true", dataset_id)
        return manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)

    for report_name, report in sorted(scan.extra_reports.items()):
        write_json_atomic(output_dir / f"{report_name}.json", report)
    write_json_atomic(
        output_dir / "progress.json",
        {
            "schema_version": PROGRESS_SCHEMA,
            "status": "review_required"
            if scan.cursor.get("finalization_blocked")
            else "packing",
            "config_sha256": resolved_sha256,
            "source_fingerprint": scan.source_fingerprint,
            "cursor": scan.cursor,
        },
    )
    if scan.cursor.get("finalization_blocked"):
        raise ReviewRequired(
            f"{dataset_id} requires an explicit boilerplate decision; "
            f"review {_project_relative(output_dir / 'boilerplate_report.json')}"
        )

    LOGGER.info("stage=tokenizer status=loading dataset=%s", dataset_id)
    tokenizer = _load_pinned_tokenizer(tokenizer_config)
    _validate_loaded_tokenizer(tokenizer)
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
    LOGGER.info("stage=tokenizer status=ready dataset=%s", dataset_id)

    min_characters = int(resolved["dataset"]["filters"].get("min_characters", 1))
    punctuation_spacing = resolved["dataset"]["filters"].get(
        "punctuation_spacing", "preserve"
    )
    shard_size = resolved["preparation"]["storage"]["sequences_per_shard"]
    LOGGER.info("stage=tokenize status=started dataset=%s", dataset_id)
    if profile in {"real", PAIRED_PREPARATION_PROFILE}:
        incremental = plan_incremental_packing(
            scan.documents,
            tokenizer,
            dataset_id=dataset_id,
            seed=resolved["preparation"]["seed"],
            sequence_length=resolved["preparation"]["sequence_length"],
            train_sequences=resolved["profile"]["train_sequences"],
            eval_sequences=resolved["profile"]["eval_sequences"],
            min_characters=min_characters,
            punctuation_spacing=punctuation_spacing,
        )
        tokenization_metrics = incremental.tokenization_metrics
        LOGGER.info("stage=write status=started dataset=%s split=train", dataset_id)
        train_shards = write_split_incremental(
            output_dir, "train", incremental.train, shard_size
        )
        LOGGER.info("stage=write status=complete dataset=%s split=train", dataset_id)
        LOGGER.info("stage=write status=started dataset=%s split=eval", dataset_id)
        eval_shards = write_split_incremental(
            output_dir, "eval", incremental.evaluation, shard_size
        )
        LOGGER.info("stage=write status=complete dataset=%s split=eval", dataset_id)
        packing_metrics = {
            **incremental.packing_metrics,
            "train_discarded_tail_tokens": incremental.train.discarded_tail_tokens,
            "eval_discarded_tail_tokens": incremental.evaluation.discarded_tail_tokens,
            "train_tokens_not_selected_by_sequence_budget": (
                incremental.train.tokens_not_selected_by_sequence_budget
            ),
            "eval_tokens_not_selected_by_sequence_budget": (
                incremental.evaluation.tokens_not_selected_by_sequence_budget
            ),
        }
        train_sequence_count = sum(record["rows"] for record in train_shards)
        eval_sequence_count = sum(record["rows"] for record in eval_shards)
    else:
        tokenized, tokenization_metrics = clean_deduplicate_and_tokenize(
            scan.documents,
            tokenizer,
            dataset_id=dataset_id,
            seed=resolved["preparation"]["seed"],
            min_characters=min_characters,
            punctuation_spacing=punctuation_spacing,
        )
        packed = pack_for_budgets(
            tokenized,
            dataset_id=dataset_id,
            seed=resolved["preparation"]["seed"],
            sequence_length=resolved["preparation"]["sequence_length"],
            train_sequences=resolved["profile"]["train_sequences"],
            eval_sequences=resolved["profile"]["eval_sequences"],
        )
        LOGGER.info("stage=write status=started dataset=%s split=train", dataset_id)
        train_shards = write_split(output_dir, "train", packed.train, shard_size)
        LOGGER.info("stage=write status=complete dataset=%s split=train", dataset_id)
        LOGGER.info("stage=write status=started dataset=%s split=eval", dataset_id)
        eval_shards = write_split(output_dir, "eval", packed.evaluation, shard_size)
        LOGGER.info("stage=write status=complete dataset=%s split=eval", dataset_id)
        packing_metrics = packed.metrics
        train_sequence_count = len(packed.train)
        eval_sequence_count = len(packed.evaluation)
    LOGGER.info(
        "stage=pack status=complete dataset=%s train_sequences=%d eval_sequences=%d",
        dataset_id,
        train_sequence_count,
        eval_sequence_count,
    )

    metrics: Dict[str, Any] = {
        "schema_version": METRICS_SCHEMA,
        "dataset_id": dataset_id,
        "profile": profile,
        "adapter": scan.metrics,
        "tokenization": tokenization_metrics,
        "packing": packing_metrics,
    }
    metrics_path = output_dir / "preparation_metrics.json"
    write_json_atomic(metrics_path, metrics)
    report_records = [
        _artifact_record(output_dir, output_dir / f"{name}.json")
        for name in sorted(scan.extra_reports)
    ]
    counts = {
        "documents_selected": len(scan.documents),
        "documents_tokenized": tokenization_metrics["documents_tokenized"],
        "exact_duplicates_removed": tokenization_metrics[
            "documents_exact_duplicates"
        ],
        "train_sequences": train_sequence_count,
        "eval_sequences": eval_sequence_count,
        "train_tokens": train_sequence_count * 1024,
        "eval_tokens": eval_sequence_count * 1024,
    }
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "preparation_id": run_id,
        "dataset_id": dataset_id,
        "profile": profile,
        "resolved_config_sha256": resolved_sha256,
        "source": {
            "configuration": resolved["dataset"]["source"],
            "fingerprint": scan.source_fingerprint,
            "cursor": scan.cursor,
        },
        "filters": resolved["dataset"]["filters"],
        "selection": {
            "dataset": resolved["dataset"].get("selection", {}),
            "profile": resolved["profile"],
        },
        "tokenizer": {
            "model_id": tokenizer_config["model_id"],
            "revision": tokenizer_config["revision"],
            "identity_sha256": tokenizer_identity_sha256,
            "fingerprint_sha256": tokenizer_sha256,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        },
        "sequence_length": 1024,
        "format": {"kind": "parquet", "compression": "zstd"},
        "splits": {"train": train_shards, "eval": eval_shards},
        "counts": counts,
        "discarded_tail_tokens": {
            "train": packing_metrics["train_discarded_tail_tokens"],
            "eval": packing_metrics["eval_discarded_tail_tokens"],
        },
        "tokens_not_selected_by_sequence_budget": {
            "train": packing_metrics[
                "train_tokens_not_selected_by_sequence_budget"
            ],
            "eval": packing_metrics["eval_tokens_not_selected_by_sequence_budget"],
        },
        "metrics": _artifact_record(output_dir, metrics_path),
        "reports": report_records,
        "redistribution_status": resolved["dataset"].get(
            "redistribution_status", "internal_research_only"
        ),
        "license_policy": resolved["dataset"].get(
            "license_policy", "internal_research_only"
        ),
    }
    write_json_atomic(
        output_dir / "progress.json",
        {
            "schema_version": PROGRESS_SCHEMA,
            "status": "complete",
            "config_sha256": resolved_sha256,
            "source_fingerprint": scan.source_fingerprint,
            "cursor": scan.cursor,
            "preparation_id": run_id,
            "counts": counts,
        },
    )
    # The manifest is the completion marker and is therefore written last.
    write_json_atomic(manifest_path, manifest)
    LOGGER.info("stage=validate status=started dataset=%s existing=false", dataset_id)
    validate_preparation(output_dir)
    work.cleanup()
    LOGGER.info("stage=validate status=complete dataset=%s existing=false", dataset_id)
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "manifest": _project_relative(manifest_path),
                "preparation_id": run_id,
                "train_sequences": train_sequence_count,
                "eval_sequences": eval_sequence_count,
            },
            sort_keys=True,
        )
    )
    return manifest_path


def run_capacity_audit(
    dataset_id: str,
    *,
    candidate_documents: int,
    eval_sequences: int = REAL_EVAL_SEQUENCES_PER_DATASET,
    config_root: Path | None = None,
) -> Path:
    if (
        not isinstance(candidate_documents, int)
        or isinstance(candidate_documents, bool)
        or candidate_documents < 1
    ):
        raise RuntimeError("capacity candidate_documents must be positive")
    if eval_sequences != REAL_EVAL_SEQUENCES_PER_DATASET:
        raise RuntimeError("real capacity audit must reserve 256 evaluation sequences")

    LOGGER.info("stage=capacity status=started dataset=%s", dataset_id)
    base_resolved, _ = load_resolved_config(dataset_id, "mvp", config_root)
    resolved = deepcopy(base_resolved)
    resolved["profile"]["candidate_documents"] = candidate_documents
    resolved["capacity_audit"] = {
        "schema_version": CAPACITY_REPORT_SCHEMA,
        "eval_sequences": eval_sequences,
    }
    if dataset_id == "gigaverbo":
        resolved["profile"]["max_source_records"] = max(
            resolved["profile"]["max_source_records"],
            candidate_documents * 64,
        )
    audit_config_sha256 = sha256_bytes(canonical_json_bytes(resolved))
    output_root = resolve_output_root(resolved["preparation"]["output_root"])
    work_identity = deepcopy(resolved)
    if dataset_id != "wackywacky":
        work_identity["profile"].pop("candidate_documents", None)
        work_identity["profile"].pop("max_source_records", None)
    work = WorkStore(
        output_root,
        dataset_id,
        sha256_bytes(canonical_json_bytes(work_identity)),
    )
    resume_cursor, resume_documents = work.load()
    persisted_documents = len(resume_documents)

    def save_checkpoint(cursor: Dict[str, Any], documents: list[Any]) -> None:
        nonlocal persisted_documents
        complete = bool(cursor.get("complete"))
        if not complete and len(documents) - persisted_documents < 10_000:
            return
        work.checkpoint(cursor, documents)
        persisted_documents = len(documents)
        LOGGER.info(
            "stage=capacity_scan status=checkpoint dataset=%s documents=%d%s",
            dataset_id,
            len(documents),
            _cursor_progress(cursor),
        )

    adapter = load_adapter(resolved["dataset"]["adapter"])
    scan = adapter.scan(
        resolved,
        resume_cursor=resume_cursor,
        resume_documents=resume_documents,
        checkpoint=save_checkpoint,
    )
    work.checkpoint(scan.resume_cursor or scan.cursor, scan.documents)
    if scan.cursor.get("finalization_blocked"):
        raise ReviewRequired(
            f"{dataset_id} capacity audit requires the configured review decision"
        )
    tokenizer_config = resolved["preparation"]["tokenizer"]
    tokenizer = _load_pinned_tokenizer(tokenizer_config)
    _validate_loaded_tokenizer(tokenizer)
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
    min_characters = int(resolved["dataset"]["filters"].get("min_characters", 1))
    punctuation_spacing = resolved["dataset"]["filters"].get(
        "punctuation_spacing", "preserve"
    )
    measured = measure_unique_sequence_capacity(
        scan.documents,
        tokenizer,
        dataset_id=dataset_id,
        seed=resolved["preparation"]["seed"],
        sequence_length=resolved["preparation"]["sequence_length"],
        eval_sequences=eval_sequences,
        min_characters=min_characters,
        punctuation_spacing=punctuation_spacing,
    )
    if measured["eval_sequences_available"] < eval_sequences:
        raise RuntimeError(
            f"{dataset_id} cannot reserve {eval_sequences} unique evaluation sequences"
        )
    reached_artificial_source_limit = (
        dataset_id == "gigaverbo"
        and scan.metrics.get("source_record_limit_reached") == 1
    )
    capacity_kind = (
        "exact"
        if len(scan.documents) < candidate_documents
        and not reached_artificial_source_limit
        else "lower_bound"
    )
    fingerprint_sha256 = sha256_bytes(canonical_json_bytes(scan.source_fingerprint))
    report: Dict[str, Any] = {
        "schema_version": CAPACITY_REPORT_SCHEMA,
        "dataset_id": dataset_id,
        "source_profile": "mvp",
        "scan_config_sha256": audit_config_sha256,
        "source_fingerprint": scan.source_fingerprint,
        "source_fingerprint_sha256": fingerprint_sha256,
        "tokenizer_fingerprint_sha256": tokenizer_sha256,
        "candidate_documents": candidate_documents,
        "documents_selected": len(scan.documents),
        "documents_tokenized": measured["documents_tokenized"],
        "documents_exact_duplicates": measured["documents_exact_duplicates"],
        "eval_sequences_requested": eval_sequences,
        "eval_sequences_available": measured["eval_sequences_available"],
        "train_sequence_capacity": measured["train_sequence_capacity"],
        "capacity_kind": capacity_kind,
        "redistribution_status": "internal_research_only",
    }
    report["capacity_report_id"] = capacity_report_id(report)
    validate_capacity_report(report)
    report_dir = output_root / ".capacity" / dataset_id / report["capacity_report_id"]
    report_path = report_dir / "capacity_report.json"
    write_json_atomic(report_path, report)
    LOGGER.info(
        "stage=capacity status=complete dataset=%s train_sequence_capacity=%d "
        "capacity_kind=%s",
        dataset_id,
        measured["train_sequence_capacity"],
        capacity_kind,
    )
    print(
        json.dumps(
            {
                "capacity_kind": capacity_kind,
                "capacity_report": report_path.relative_to(output_root).as_posix(),
                "capacity_report_id": report["capacity_report_id"],
                "dataset_id": dataset_id,
                "eval_sequences": eval_sequences,
                "train_sequence_capacity": measured["train_sequence_capacity"],
            },
            sort_keys=True,
        )
    )
    return report_path


def run_real_allocation(
    report_paths: list[Path], *, output: Path | None = None
) -> Dict[str, Any]:
    reports = [load_capacity_report(path) for path in report_paths]
    allocation = allocate_real_training(
        reports, target_train_sequences=REAL_TARGET_TRAIN_SEQUENCES
    )
    if output is not None:
        write_real_allocation(output, allocation)
    print(json.dumps(allocation, ensure_ascii=False, sort_keys=True))
    return allocation


def run_paired_real_allocation(
    report_paths: list[Path], *, output: Path | None = None
) -> Dict[str, Any]:
    reports = [load_capacity_report(path) for path in report_paths]
    allocation = allocate_paired_real_training(reports)
    if output is not None:
        write_paired_real_allocation(output, allocation)
    print(json.dumps(allocation, ensure_ascii=False, sort_keys=True))
    return allocation


def materialize_paired_real_configs(
    allocation_path: Path,
    *,
    output_config_root: Path,
    base_config_root: Path | None = None,
) -> Dict[str, Any]:
    from .training_config import (
        PAIRED_REAL_TRAINING_CONFIG_SCHEMA,
        validate_training_config,
    )

    allocation = _read_paired_allocation(allocation_path)
    base_root = (base_config_root or (PROJECT_ROOT / "configs")).resolve()
    output_root = output_config_root.expanduser()
    if output_root.is_symlink():
        raise RuntimeError("paired config output root must not be a symlink")
    output_root = output_root.resolve()

    capacity_by_dataset = {
        record["dataset_id"]: record for record in allocation["capacity_reports"]
    }
    dataset_configs: Dict[str, Dict[str, Any]] = {}
    for dataset_id in DATASET_IDS:
        config = deepcopy(
            _read_json_object(base_root / "datasets" / f"{dataset_id}.json")
        )
        profile = deepcopy(config["profiles"]["mvp"])
        profile.update(
            {
                "train_sequences": allocation["prepared_train_sequences"][
                    dataset_id
                ],
                "eval_sequences": 256,
                "candidate_documents": capacity_by_dataset[dataset_id][
                    "candidate_documents"
                ],
                "selection": "representative",
                "allocation_policy": allocation["policy"],
                "without_replacement": True,
                "allocation_sha256": allocation["allocation_sha256"],
                "pools": [
                    {
                        key: value
                        for key, value in pool.items()
                        if key != "dataset_id"
                    }
                    for pool in allocation["pools"]
                    if pool["dataset_id"] == dataset_id
                ],
            }
        )
        if dataset_id == "gigaverbo":
            profile["max_source_records"] = max(
                profile["max_source_records"],
                profile["candidate_documents"] * 64,
            )
        config["profiles"][PAIRED_PREPARATION_PROFILE] = profile
        validate_dataset_config(config, dataset_id)
        dataset_configs[dataset_id] = config

    training_template = _read_json_object(
        base_root / "training" / "l40s-mvp.json"
    )
    training_configs: Dict[str, Dict[str, Any]] = {}
    for arm in ("general", "forum_tech"):
        config = deepcopy(training_template)
        config["schema_version"] = PAIRED_REAL_TRAINING_CONFIG_SCHEMA
        config["profile"] = "real"
        config["data_mixture"] = paired_mixture_for_arm(allocation, arm)
        used = allocation[f"{arm}_allocations"]
        for entry in config["datasets"]:
            entry.pop("weight")
            dataset_id = entry["dataset_id"]
            entry.update(
                {
                    "prepared_train_sequences": allocation[
                        "prepared_train_sequences"
                    ][dataset_id],
                    "train_sequences": used[dataset_id],
                    "eval_sequences": 256,
                }
            )
        config["training"].update(
            {
                "warmup_steps": 520,
                "checkpoint_steps": [13_000, 26_000, 39_000],
                "total_optimizer_steps": 52_000,
            }
        )
        validate_training_config(config)
        training_configs[arm] = config

    written = []
    for dataset_id, config in dataset_configs.items():
        relative = Path("datasets") / f"{dataset_id}.json"
        write_json_atomic(output_root / relative, config)
        written.append(relative.as_posix())
    for arm, config in training_configs.items():
        relative = Path("training") / f"l40s-real-{arm.replace('_', '-')}.json"
        write_json_atomic(output_root / relative, config)
        written.append(relative.as_posix())
    allocation_relative = Path("allocations") / "paired-real-allocation.json"
    write_json_atomic(output_root / allocation_relative, allocation)
    written.append(allocation_relative.as_posix())
    return {
        "status": "materialized",
        "experiment_id": allocation["experiment_id"],
        "allocation_sha256": allocation["allocation_sha256"],
        "files": sorted(written),
    }


def run_materialize_paired_real_configs(
    allocation_path: Path,
    *,
    output_config_root: Path,
    base_config_root: Path | None = None,
) -> Dict[str, Any]:
    result = materialize_paired_real_configs(
        allocation_path,
        output_config_root=output_config_root,
        base_config_root=base_config_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def run_verify_paired_real(
    general_config_path: Path,
    forum_tech_config_path: Path,
    *,
    output_root: Path | None = None,
    output: Path | None = None,
) -> Dict[str, Any]:
    from .paired_plan import (
        DOMAIN_DATASET_IDS,
        PAIRED_REAL_POLICY,
        iter_paired_mixture_slots,
    )
    from .training_config import load_training_config
    from .training_data import (
        build_paired_training_references,
        resolve_training_inputs,
    )

    general_config, _ = load_training_config(general_config_path)
    forum_config, _ = load_training_config(forum_tech_config_path)
    general_mixture = general_config.get("data_mixture", {})
    forum_mixture = forum_config.get("data_mixture", {})
    if (
        general_mixture.get("policy") != PAIRED_REAL_POLICY
        or forum_mixture.get("policy") != PAIRED_REAL_POLICY
        or general_mixture.get("arm") != "general"
        or forum_mixture.get("arm") != "forum_tech"
    ):
        raise RuntimeError("paired verification requires general and forum_tech configs")
    for key in ("model", "training", "execution", "hardware"):
        if general_config[key] != forum_config[key]:
            raise RuntimeError(f"paired configs differ in {key}")
    comparable_general = {
        key: value for key, value in general_mixture.items() if key != "arm"
    }
    comparable_forum = {
        key: value for key, value in forum_mixture.items() if key != "arm"
    }
    if comparable_general != comparable_forum:
        raise RuntimeError("paired configs do not use the same allocation and pools")

    general_inputs = resolve_training_inputs(general_config, output_root)
    forum_inputs = resolve_training_inputs(forum_config, output_root)
    if general_inputs.paired_inputs_sha256() != forum_inputs.paired_inputs_sha256():
        raise RuntimeError("paired configs did not resolve the same prepared inputs")
    for general_dataset, forum_dataset in zip(
        general_inputs.datasets, forum_inputs.datasets
    ):
        if (
            general_dataset.dataset_id != forum_dataset.dataset_id
            or general_dataset.manifest_sha256 != forum_dataset.manifest_sha256
            or general_dataset.manifest["splits"]["eval"]
            != forum_dataset.manifest["splits"]["eval"]
        ):
            raise RuntimeError("paired evaluation manifests changed between arms")

    general_references = build_paired_training_references(general_inputs, 42)
    forum_references = build_paired_training_references(forum_inputs, 42)
    slots = iter_paired_mixture_slots(general_mixture)
    shared_positions = 0
    replacement_positions = 0
    for slot, general_reference, forum_reference in zip(
        slots, general_references, forum_references
    ):
        if slot in {f"{dataset_id}_domain" for dataset_id in DOMAIN_DATASET_IDS}:
            replacement_positions += 1
            if (
                general_reference.dataset_id != "brwac"
                or forum_reference.dataset_id != slot.removesuffix("_domain")
            ):
                raise RuntimeError("paired domain substitution changed")
        else:
            shared_positions += 1
            if general_reference != forum_reference:
                raise RuntimeError("paired shared reference changed position")
    if shared_positions + replacement_positions != 416_000:
        raise RuntimeError("paired verification schedule length changed")

    report = {
        "schema_version": "queroquero-paired-real-verification/v1",
        "status": "valid",
        "experiment_id": general_mixture["experiment_id"],
        "allocation_sha256": general_mixture["allocation_sha256"],
        "schedule_template_sha256": general_mixture[
            "schedule_template_sha256"
        ],
        "paired_inputs_sha256": general_inputs.paired_inputs_sha256(),
        "train_sequences_per_arm": 416_000,
        "train_tokens_per_arm": 416_000 * 1024,
        "shared_positions": shared_positions,
        "replacement_positions": replacement_positions,
        "eval_sequences_per_dataset": 256,
        "eval_datasets": len(DATASET_IDS),
        "general_references_sha256": _references_sha256(general_references),
        "forum_tech_references_sha256": _references_sha256(forum_references),
    }
    _assert_no_absolute_path_strings(report)
    if output is not None:
        write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def validate_preparation(path: str | Path) -> Dict[str, Any]:
    requested_root = Path(path).expanduser()
    if requested_root.is_symlink():
        raise RuntimeError("preparation path must not be a symlink")
    root = requested_root.resolve()
    if not root.is_dir():
        raise RuntimeError("preparation path must be a real directory")
    manifest_path = root / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"dataset manifest is missing: {manifest_path}") from exc
    if manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise RuntimeError("unknown dataset manifest schema")
    _assert_no_absolute_path_strings(manifest)
    if manifest.get("dataset_id") not in DATASET_IDS:
        raise RuntimeError("manifest dataset_id is unknown")
    expected_budgets = {"smoke": (8, 2), "mvp": (256, 32)}
    if manifest.get("profile") not in {
        *expected_budgets,
        "real",
        PAIRED_PREPARATION_PROFILE,
    }:
        raise RuntimeError("manifest profile is unknown")
    if manifest.get("redistribution_status") != "internal_research_only":
        raise RuntimeError("manifest redistribution policy must remain internal")
    if manifest.get("license_policy") != "internal_research_only":
        raise RuntimeError("manifest license policy must remain internal")
    run_id = manifest.get("preparation_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{20}", run_id):
        raise RuntimeError("invalid preparation_id")
    if root.name != run_id:
        raise RuntimeError("preparation directory does not match preparation_id")
    if manifest.get("sequence_length") != 1024:
        raise RuntimeError("manifest sequence length must be 1024")
    if manifest.get("format") != {"kind": "parquet", "compression": "zstd"}:
        raise RuntimeError("manifest storage format must be parquet/zstd")
    source = manifest.get("source")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(source, dict) or not isinstance(source.get("fingerprint"), dict):
        raise RuntimeError("manifest source provenance is incomplete")
    if not isinstance(tokenizer, dict):
        raise RuntimeError("manifest tokenizer provenance is incomplete")
    if tokenizer.get("model_id") != MODEL_ID or tokenizer.get("revision") != MODEL_REVISION:
        raise RuntimeError("manifest tokenizer is not the pinned model revision")
    for digest_name in (
        "resolved_config_sha256",
        "identity_sha256",
        "fingerprint_sha256",
    ):
        digest = (
            manifest.get(digest_name)
            if digest_name == "resolved_config_sha256"
            else tokenizer.get(digest_name)
        )
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"invalid manifest digest: {digest_name}")
    expected_tokenizer_contract = {
        "vocab_size": 49152,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 49109,
        "unk_token_id": 0,
    }
    if any(
        tokenizer.get(key) != expected
        for key, expected in expected_tokenizer_contract.items()
    ):
        raise RuntimeError("manifest tokenizer IDs or vocabulary changed")
    expected_run_id = preparation_id(
        manifest.get("resolved_config_sha256", ""),
        tokenizer.get("identity_sha256", ""),
        source["fingerprint"],
    )
    if expected_run_id != run_id:
        raise RuntimeError("preparation_id does not match manifest provenance")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("manifest counts are missing")
    listed_shards = set()
    global_sequence_ids = set()
    split_totals: Dict[str, int] = {}
    split_sources: Dict[str, set[str]] = {}
    for split in ("train", "eval"):
        records = manifest.get("splits", {}).get(split)
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"manifest split is empty or invalid: {split}")
        total_rows = 0
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError("manifest shard record must be an object")
            relative = Path(record["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 2
                or relative.parts[0] != split
                or not re.fullmatch(r"shard-[0-9]{5}\.parquet", relative.name)
            ):
                raise RuntimeError("manifest contains an unsafe shard path")
            relative_text = relative.as_posix()
            if relative_text in listed_shards:
                raise RuntimeError("manifest lists a shard more than once")
            listed_shards.add(relative_text)
            shard = _resolved_regular_file(root, relative)
            if shard.stat().st_size != record["size_bytes"]:
                raise RuntimeError(f"shard size mismatch: {relative}")
            if file_sha256(shard) != record["sha256"]:
                raise RuntimeError(f"shard hash mismatch: {relative}")
            rows = validate_shard(shard)
            if record.get("rows") != rows or record.get("tokens") != rows * 1024:
                raise RuntimeError(f"shard row or token count mismatch: {relative}")
            total_rows += rows
            provenance = pq.read_table(
                shard, columns=["sequence_id", "source_ref_sha256"]
            )
            for sequence_id in provenance.column("sequence_id").to_pylist():
                if sequence_id in global_sequence_ids:
                    raise RuntimeError("duplicate sequence_id across shards")
                global_sequence_ids.add(sequence_id)
            split_sources.setdefault(split, set()).update(
                source_hash
                for row in provenance.column("source_ref_sha256").to_pylist()
                for source_hash in row
            )
        split_totals[split] = total_rows
        if counts.get(f"{split}_sequences") != total_rows:
            raise RuntimeError(f"manifest {split} sequence total is inconsistent")
        if counts.get(f"{split}_tokens") != total_rows * 1024:
            raise RuntimeError(f"manifest {split} token total is inconsistent")

    profile = manifest.get("selection", {}).get("profile", {})
    if manifest["profile"] in {"real", PAIRED_PREPARATION_PROFILE}:
        expected_train = profile.get("train_sequences")
        expected_eval = 256
        expected_policy = (
            "matched_domain_substitution_without_replacement"
            if manifest["profile"] == PAIRED_PREPARATION_PROFILE
            else "equal_share_without_replacement"
        )
        if (
            not isinstance(expected_train, int)
            or isinstance(expected_train, bool)
            or expected_train < 1
            or profile.get("allocation_policy")
            != expected_policy
            or profile.get("without_replacement") is not True
            or not isinstance(profile.get("allocation_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", profile["allocation_sha256"])
        ):
            raise RuntimeError("real manifest allocation policy is invalid")
        if manifest["profile"] == PAIRED_PREPARATION_PROFILE:
            pools = profile.get("pools")
            if not isinstance(pools, list) or not pools:
                raise RuntimeError("paired manifest pools are missing")
            expected_pool_contracts = {
                "brwac": [
                    ("brwac_common", "shared"),
                    ("brwac_extra", "replacement"),
                ],
                "gigaverbo": [("gigaverbo_shared", "shared")],
                "multiwoz_ptbr": [
                    ("multiwoz_ptbr_shared", "shared")
                ],
                "wackywacky": [("wackywacky_shared", "shared")],
                "adrenaline": [("adrenaline_domain", "domain")],
                "outerspace": [("outerspace_domain", "domain")],
            }[manifest["dataset_id"]]
            actual_pool_contracts = []
            expected_start = 0
            for pool in pools:
                if (
                    not isinstance(pool, dict)
                    or set(pool)
                    != {"pool_id", "role", "start_row", "train_sequences"}
                    or not isinstance(pool.get("pool_id"), str)
                    or not isinstance(pool.get("role"), str)
                    or pool.get("start_row") != expected_start
                    or not isinstance(pool.get("train_sequences"), int)
                    or isinstance(pool.get("train_sequences"), bool)
                    or pool["train_sequences"] < 1
                ):
                    raise RuntimeError("paired manifest pool ranges are invalid")
                actual_pool_contracts.append((pool["pool_id"], pool["role"]))
                expected_start += pool["train_sequences"]
            if actual_pool_contracts != expected_pool_contracts:
                raise RuntimeError("paired manifest pool contract changed")
            if expected_start != expected_train:
                raise RuntimeError("paired manifest pools do not fill the train split")
    else:
        expected_train, expected_eval = expected_budgets[manifest["profile"]]
    if (
        profile.get("train_sequences") != expected_train
        or profile.get("eval_sequences") != expected_eval
    ):
        raise RuntimeError("manifest profile budgets are invalid")
    if profile.get("train_sequences") != split_totals["train"]:
        raise RuntimeError("train budget does not match prepared rows")
    if profile.get("eval_sequences") != split_totals["eval"]:
        raise RuntimeError("eval budget does not match prepared rows")
    if split_sources["train"] & split_sources["eval"]:
        raise RuntimeError("a source document appears in both splits")

    tails = manifest.get("discarded_tail_tokens")
    excess = manifest.get("tokens_not_selected_by_sequence_budget")
    if not isinstance(tails, dict) or not isinstance(excess, dict):
        raise RuntimeError("manifest packing discard metrics are missing")
    for split in ("train", "eval"):
        tail = tails.get(split)
        not_selected = excess.get(split)
        if (
            not isinstance(tail, int)
            or isinstance(tail, bool)
            or not 0 <= tail < 1024
            or not isinstance(not_selected, int)
            or isinstance(not_selected, bool)
            or not_selected < tail
            or not_selected % 1024 != tail
        ):
            raise RuntimeError("invalid packing discard metrics")

    actual_shards = {
        path.relative_to(root).as_posix() for path in root.rglob("*.parquet")
    }
    if actual_shards != listed_shards:
        raise RuntimeError("preparation contains missing or unlisted shards")
    if any(root.rglob("*.partial")):
        raise RuntimeError("preparation contains an incomplete partial file")

    metrics_record = manifest.get("metrics")
    metrics = _validate_artifact(root, metrics_record, "preparation_metrics.json")
    _assert_no_absolute_path_strings(metrics)
    if metrics.get("schema_version") != METRICS_SCHEMA:
        raise RuntimeError("unknown preparation metrics schema")
    if metrics.get("dataset_id") != manifest.get("dataset_id"):
        raise RuntimeError("metrics dataset does not match manifest")
    if metrics.get("profile") != manifest.get("profile"):
        raise RuntimeError("metrics profile does not match manifest")
    tokenization_metrics = metrics.get("tokenization", {})
    if (
        counts.get("documents_selected")
        != tokenization_metrics.get("documents_received")
        or counts.get("documents_tokenized")
        != tokenization_metrics.get("documents_tokenized")
        or counts.get("exact_duplicates_removed")
        != tokenization_metrics.get("documents_exact_duplicates")
    ):
        raise RuntimeError("tokenization counts do not match manifest")
    packing_metrics = metrics.get("packing", {})
    if packing_metrics.get("train_sequences") != split_totals["train"]:
        raise RuntimeError("metrics train total does not match manifest")
    if packing_metrics.get("eval_sequences") != split_totals["eval"]:
        raise RuntimeError("metrics eval total does not match manifest")
    for split in ("train", "eval"):
        if packing_metrics.get(f"{split}_discarded_tail_tokens") != tails[split]:
            raise RuntimeError("metrics tail count does not match manifest")
        if (
            packing_metrics.get(
                f"{split}_tokens_not_selected_by_sequence_budget"
            )
            != excess[split]
        ):
            raise RuntimeError("metrics budget excess does not match manifest")
    reports = manifest.get("reports")
    if not isinstance(reports, list):
        raise RuntimeError("manifest reports must be an array")
    for report in reports:
        _assert_no_absolute_path_strings(_validate_artifact(root, report))

    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    _assert_no_absolute_path_strings(progress)
    if progress.get("schema_version") != PROGRESS_SCHEMA:
        raise RuntimeError("unknown completion progress schema")
    if progress.get("status") != "complete":
        raise RuntimeError("preparation is not complete")
    if progress.get("config_sha256") != manifest.get("resolved_config_sha256"):
        raise RuntimeError("progress configuration does not match manifest")
    if progress.get("source_fingerprint") != source["fingerprint"]:
        raise RuntimeError("progress source does not match manifest")
    if progress.get("cursor") != source.get("cursor"):
        raise RuntimeError("progress cursor does not match manifest")
    if progress.get("preparation_id") != run_id or progress.get("counts") != counts:
        raise RuntimeError("progress counts do not match manifest")

    expected_json = {
        "dataset_manifest.json",
        "preparation_metrics.json",
        "progress.json",
        *(Path(record["path"]).name for record in manifest.get("reports", [])),
    }
    actual_json = {path.name for path in root.glob("*.json")}
    if actual_json != expected_json:
        raise RuntimeError("preparation contains an unlisted JSON artifact")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("preparation contains a symlink")
    expected_files = listed_shards | expected_json
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError("preparation contains an unexpected file")
    return manifest


def _artifact_record(root: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _read_json_object(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError("paired configuration input must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"paired configuration input is missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"paired configuration input is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("paired configuration input must be an object")
    return value


def _read_paired_allocation(path: Path) -> Dict[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise RuntimeError("paired allocation input must not be a symlink")
    return validate_paired_real_allocation(_read_json_object(requested.resolve()))


def _validate_artifact(
    root: Path, record: Any, expected_name: str | None = None
) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError("manifest artifact record must be an object")
    relative = Path(record.get("path", ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 1
        or relative.suffix != ".json"
        or (expected_name is not None and relative.name != expected_name)
    ):
        raise RuntimeError("manifest contains an unsafe artifact path")
    path = _resolved_regular_file(root, relative)
    if path.stat().st_size != record.get("size_bytes"):
        raise RuntimeError("manifest artifact size mismatch")
    if file_sha256(path) != record.get("sha256"):
        raise RuntimeError("manifest artifact hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("manifest JSON artifact must contain an object")
    return value


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _cursor_progress(cursor: Dict[str, Any]) -> str:
    fields = []
    for key in _PROGRESS_CURSOR_KEYS:
        value = cursor.get(key)
        if isinstance(value, (int, bool)) and not (
            isinstance(value, bool) and key != "complete"
        ):
            fields.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    return "" if not fields else " " + " ".join(fields)


def _load_pinned_tokenizer(config: Dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        trust_remote_code=config["trust_remote_code"],
    )


def _validate_loaded_tokenizer(tokenizer: Any) -> None:
    actual = {
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    expected = {
        "vocab_size": 49152,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 49109,
        "unk_token_id": 0,
    }
    if actual != expected:
        raise RuntimeError("the pinned tokenizer contract changed")


def _resolved_regular_file(root: Path, relative: Path) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError("preparation artifact must not use symlinks")
    resolved = candidate.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise RuntimeError("preparation artifact is missing or escapes its directory")
    return resolved


def _references_sha256(references: Any) -> str:
    digest = hashlib.sha256()
    for reference in references:
        digest.update(reference.dataset_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(int(reference.row_index).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _assert_no_absolute_path_strings(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_absolute_path_strings(nested)
    elif isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise RuntimeError("preparation metadata contains an absolute path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare bounded PT-BR datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="prepare one configured dataset")
    run.add_argument("--dataset", choices=DATASET_IDS, required=True)
    run.add_argument(
        "--profile",
        choices=("smoke", "mvp", "real", PAIRED_PREPARATION_PROFILE),
        required=True,
    )
    run.add_argument("--config-root", type=Path)
    capacity = subparsers.add_parser(
        "capacity", help="measure unique capacity from the full configured source"
    )
    capacity.add_argument("--dataset", choices=DATASET_IDS, required=True)
    capacity.add_argument("--candidate-documents", type=int, required=True)
    capacity.add_argument(
        "--eval-sequences", type=int, default=REAL_EVAL_SEQUENCES_PER_DATASET
    )
    capacity.add_argument("--config-root", type=Path)
    allocate = subparsers.add_parser(
        "allocate-real", help="allocate the fixed real budget without replacement"
    )
    allocate.add_argument("--report", type=Path, action="append", required=True)
    allocate.add_argument("--output", type=Path)
    paired_allocate = subparsers.add_parser(
        "allocate-paired-real",
        help="allocate matched general and forum_tech budgets without replacement",
    )
    paired_allocate.add_argument(
        "--report", type=Path, action="append", required=True
    )
    paired_allocate.add_argument("--output", type=Path)
    paired_materialize = subparsers.add_parser(
        "materialize-paired-real",
        help="materialize paired dataset and training configs from an allocation",
    )
    paired_materialize.add_argument("--allocation", type=Path, required=True)
    paired_materialize.add_argument(
        "--output-config-root", type=Path, required=True
    )
    paired_materialize.add_argument("--base-config-root", type=Path)
    paired_verify = subparsers.add_parser(
        "verify-paired-real",
        help="verify matched training schedules and shared evaluation manifests",
    )
    paired_verify.add_argument("--general-config", type=Path, required=True)
    paired_verify.add_argument("--forum-tech-config", type=Path, required=True)
    paired_verify.add_argument("--output-root", type=Path)
    paired_verify.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate", help="validate prepared shards")
    validate.add_argument("--path", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            run_preparation(args.dataset, args.profile, args.config_root)
        elif args.command == "capacity":
            run_capacity_audit(
                args.dataset,
                candidate_documents=args.candidate_documents,
                eval_sequences=args.eval_sequences,
                config_root=args.config_root,
            )
        elif args.command == "allocate-real":
            run_real_allocation(args.report, output=args.output)
        elif args.command == "allocate-paired-real":
            run_paired_real_allocation(args.report, output=args.output)
        elif args.command == "materialize-paired-real":
            run_materialize_paired_real_configs(
                args.allocation,
                output_config_root=args.output_config_root,
                base_config_root=args.base_config_root,
            )
        elif args.command == "verify-paired-real":
            run_verify_paired_real(
                args.general_config,
                args.forum_tech_config,
                output_root=args.output_root,
                output=args.output,
            )
        else:
            manifest = validate_preparation(args.path)
            print(
                json.dumps(
                    {
                        "dataset_id": manifest["dataset_id"],
                        "preparation_id": manifest["preparation_id"],
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
    except ReviewRequired as exc:
        print(json.dumps({"status": "review_required", "message": str(exc)}))
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
