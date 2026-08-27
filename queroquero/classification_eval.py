from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from .classification_embeddings import (
    broadcast_payload,
    close_gloo,
    generate_embeddings,
    initialize_gloo,
    load_private_texts,
    run_model_probe,
    validate_embedding_manifests,
)
from .classification_eval_common import (
    MODEL_NAMES,
    PREFLIGHT_SCHEMA,
    assert_safe_metadata,
    load_evaluation_config,
    locate_existing_evaluation,
    resolve_evaluation_inputs,
    unit_by_index,
    write_resolved_evaluation,
)
from .classification_probe import (
    build_report,
    evaluate_unit,
    select_hyperparameters,
    tune_unit,
    validate_report,
)
from .manifest import write_json_atomic


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/classification/evaluation-v1.json")


def run_preflight(config_path: Path) -> Dict[str, Any] | None:
    import torch

    rank, world_size, local_rank = initialize_gloo()
    try:
        payload = None
        if rank == 0:
            resolved, runtime = resolve_evaluation_inputs(config_path)
            if world_size != resolved["embedding"]["world_size"]:
                raise RuntimeError("classification preflight world size changed")
            write_resolved_evaluation(runtime["evaluation_dir"], resolved)
            payload = _portable_runtime(resolved, runtime)
            payload["dependencies"] = _dependency_versions()
        payload = broadcast_payload(payload, rank)
        resolved = payload["resolved"]
        runtime = _runtime_from_portable(payload)
        rank_ids = runtime["sample_ids"][rank::world_size][:8]
        private = load_private_texts(
            runtime["dataset_path"] / "examples.parquet", rank_ids
        )
        private_texts = [private[value] for value in rank_ids]
        model_results = []
        for model_name in MODEL_NAMES:
            LOGGER.info(
                "stage=preflight status=model_started model=%s rank=%d",
                model_name,
                rank,
            )
            model_results.append(
                run_model_probe(
                    runtime["config"],
                    model_name,
                    runtime["model_manifests"],
                    private_texts,
                    local_rank,
                )
            )
        device = torch.device("cuda", local_rank)
        if (
            torch.cuda.get_device_name(device) != "NVIDIA L40S"
            or torch.cuda.get_device_capability(device) != (8, 9)
            or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("classification preflight requires an L40S with BF16")
        rank_result = {
            "rank": rank,
            "local_rank": local_rank,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "models": model_results,
        }
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        if torch.distributed.is_initialized():
            torch.distributed.gather_object(rank_result, gathered, dst=0)
        else:
            gathered = [rank_result]
        if rank != 0:
            return None
        result = {
            "schema_version": PREFLIGHT_SCHEMA,
            "evaluation_id": resolved["evaluation_id"],
            "world_size": world_size,
            "sample_count": resolved["sample_count"],
            "dependencies": payload["dependencies"],
            "ranks": gathered,
            "status": "ok",
        }
        assert_safe_metadata(result)
        target = runtime["evaluation_dir"] / "preflight.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if (
                existing.get("schema_version") != PREFLIGHT_SCHEMA
                or existing.get("evaluation_id") != resolved["evaluation_id"]
                or existing.get("world_size") != world_size
                or existing.get("sample_count") != resolved["sample_count"]
                or existing.get("dependencies") != payload["dependencies"]
                or existing.get("status") != "ok"
            ):
                raise RuntimeError("classification preflight result changed")
            result = existing
        else:
            write_json_atomic(target, result)
            target.chmod(0o600)
        return result
    finally:
        close_gloo()


def run_embed(config_path: Path, model_name: str) -> Dict[str, Any] | None:
    rank, world_size, local_rank = initialize_gloo()
    try:
        payload = None
        if rank == 0:
            existing_resolved, existing_dir = locate_existing_evaluation(config_path)
            resolved, runtime = resolve_evaluation_inputs(config_path)
            if resolved != existing_resolved or runtime["evaluation_dir"] != existing_dir:
                raise RuntimeError("classification evaluation identity changed after preflight")
            payload = _portable_runtime(resolved, runtime)
        payload = broadcast_payload(payload, rank)
        runtime = _runtime_from_portable(payload)
        return generate_embeddings(
            runtime["config"],
            payload["resolved"],
            runtime,
            model_name,
            rank,
            world_size,
            local_rank,
        )
    finally:
        close_gloo()


def run_cpu_command(args: argparse.Namespace) -> Dict[str, Any]:
    config, _ = load_evaluation_config(args.config)
    resolved, evaluation_dir = locate_existing_evaluation(args.config)
    if args.command == "validate-embeddings":
        return validate_embedding_manifests(config, resolved, evaluation_dir)
    if args.command == "tune-unit":
        return tune_unit(config, resolved, evaluation_dir, args.unit_index)
    if args.command == "select-hyperparameters":
        return select_hyperparameters(config, resolved, evaluation_dir)
    if args.command == "evaluate-unit":
        return evaluate_unit(config, resolved, evaluation_dir, args.unit_index)
    if args.command == "report":
        report = build_report(config, resolved, evaluation_dir)
        return {
            "evaluation_id": resolved["evaluation_id"],
            "final_evaluations": report["test_policy"]["final_evaluations"],
            "report_sha256": report["report_sha256"],
            "status": "complete",
        }
    if args.command == "validate-report":
        validate_embedding_manifests(config, resolved, evaluation_dir)
        return validate_report(config, resolved, evaluation_dir)
    raise RuntimeError("unsupported classification evaluation command")


def _portable_runtime(
    resolved: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "resolved": dict(resolved),
        "config": runtime["config"],
        "dataset_path": str(runtime["dataset_path"]),
        "evaluation_dir": str(runtime["evaluation_dir"]),
        "model_manifests": runtime["model_manifests"],
        "sample_ids": runtime["sample_ids"],
    }


def _runtime_from_portable(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "config": payload["config"],
        "dataset_path": Path(payload["dataset_path"]),
        "evaluation_dir": Path(payload["evaluation_dir"]),
        "model_manifests": payload["model_manifests"],
        "sample_ids": payload["sample_ids"],
    }


def _dependency_versions() -> Dict[str, str]:
    names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "torch": "torch",
        "transformers": "transformers",
    }
    versions = {
        key: importlib.metadata.version(distribution)
        for key, distribution in names.items()
    }
    if versions["scikit_learn"] != "1.9.0":
        raise RuntimeError("classification scikit-learn version changed")
    return versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired embedding evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight",
        "validate-embeddings",
        "select-hyperparameters",
        "report",
        "validate-report",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    embed = subparsers.add_parser("embed")
    embed.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    embed.add_argument("--model", choices=MODEL_NAMES, required=True)
    for command in ("tune-unit", "evaluate-unit"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--unit-index", type=int, required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parser().parse_args()
    if args.command == "preflight":
        result = run_preflight(args.config)
    elif args.command == "embed":
        result = run_embed(args.config, args.model)
    else:
        if args.command in {"tune-unit", "evaluate-unit"}:
            unit_by_index(load_evaluation_config(args.config)[0], args.unit_index)
        result = run_cpu_command(args)
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
