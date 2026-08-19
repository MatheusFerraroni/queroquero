from __future__ import annotations

import argparse
import importlib
import json
import re
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
    resolve_project_path,
    scan_config_sha256,
    sha256_bytes,
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
    pack_for_budgets,
    tokenizer_fingerprint,
)
from .storage import WorkStore, validate_shard, write_split


class ReviewRequired(RuntimeError):
    exit_code = 20


def load_adapter(name: str) -> Any:
    module = importlib.import_module(f"queroquero.datasets.{name}")
    try:
        return module.ADAPTER
    except AttributeError as exc:
        raise RuntimeError(f"dataset adapter {name!r} does not expose ADAPTER") from exc


def run_preparation(dataset_id: str, profile: str, config_root: Path | None = None) -> Path:
    resolved, resolved_sha256 = load_resolved_config(dataset_id, profile, config_root)
    output_root = resolve_project_path(resolved["preparation"]["output_root"])
    work = WorkStore(output_root, dataset_id, scan_config_sha256(resolved))
    resume_cursor, resume_documents = work.load()
    adapter = load_adapter(resolved["dataset"]["adapter"])
    scan = adapter.scan(
        resolved,
        resume_cursor=resume_cursor,
        resume_documents=resume_documents,
        checkpoint=work.checkpoint,
    )
    work.checkpoint(scan.resume_cursor or scan.cursor, scan.documents)

    tokenizer_config = resolved["preparation"]["tokenizer"]
    tokenizer_identity_sha256 = sha256_bytes(canonical_json_bytes(tokenizer_config))
    run_id = preparation_id(
        resolved_sha256, tokenizer_identity_sha256, scan.source_fingerprint
    )
    output_dir = output_root / dataset_id / run_id
    manifest_path = output_dir / "dataset_manifest.json"
    if manifest_path.exists():
        validate_preparation(output_dir)
        work.cleanup()
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

    tokenizer = _load_pinned_tokenizer(tokenizer_config)
    _validate_loaded_tokenizer(tokenizer)
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer)

    min_characters = int(resolved["dataset"]["filters"].get("min_characters", 1))
    tokenized, tokenization_metrics = clean_deduplicate_and_tokenize(
        scan.documents,
        tokenizer,
        dataset_id=dataset_id,
        seed=resolved["preparation"]["seed"],
        min_characters=min_characters,
    )
    packed = pack_for_budgets(
        tokenized,
        dataset_id=dataset_id,
        seed=resolved["preparation"]["seed"],
        sequence_length=resolved["preparation"]["sequence_length"],
        train_sequences=resolved["profile"]["train_sequences"],
        eval_sequences=resolved["profile"]["eval_sequences"],
    )
    shard_size = resolved["preparation"]["storage"]["sequences_per_shard"]
    train_shards = write_split(output_dir, "train", packed.train, shard_size)
    eval_shards = write_split(output_dir, "eval", packed.evaluation, shard_size)

    metrics: Dict[str, Any] = {
        "schema_version": METRICS_SCHEMA,
        "dataset_id": dataset_id,
        "profile": profile,
        "adapter": scan.metrics,
        "tokenization": tokenization_metrics,
        "packing": packed.metrics,
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
        "train_sequences": len(packed.train),
        "eval_sequences": len(packed.evaluation),
        "train_tokens": len(packed.train) * 1024,
        "eval_tokens": len(packed.evaluation) * 1024,
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
            "train": packed.metrics["train_discarded_tail_tokens"],
            "eval": packed.metrics["eval_discarded_tail_tokens"],
        },
        "tokens_not_selected_by_sequence_budget": {
            "train": packed.metrics[
                "train_tokens_not_selected_by_sequence_budget"
            ],
            "eval": packed.metrics["eval_tokens_not_selected_by_sequence_budget"],
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
    validate_preparation(output_dir)
    work.cleanup()
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "manifest": _project_relative(manifest_path),
                "preparation_id": run_id,
                "train_sequences": len(packed.train),
                "eval_sequences": len(packed.evaluation),
            },
            sort_keys=True,
        )
    )
    return manifest_path


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
    if manifest.get("profile") not in expected_budgets:
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
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


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
    run.add_argument("--profile", choices=("smoke", "mvp"), required=True)
    run.add_argument("--config-root", type=Path)
    validate = subparsers.add_parser("validate", help="validate prepared shards")
    validate.add_argument("--path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            run_preparation(args.dataset, args.profile, args.config_root)
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
