from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import MODEL_ID, MODEL_REVISION, PROJECT_ROOT, canonical_json_bytes, sha256_bytes
from .manifest import file_sha256, write_json_atomic
from .model_artifact import export_model_artifact, validate_model_artifact
from .packing import tokenizer_fingerprint
from .training_config import (
    TRAINING_METHOD,
    load_training_config,
    resolve_training_output_roots,
)
from .training_data import (
    ResolvedTrainingInputs,
    TrainingSequence,
    load_evaluation_sequences,
    load_training_sequences,
    resolve_training_inputs,
)
from .training_distributed import (
    DistributedContext,
    global_step_batch,
    rank_step_batch,
)


LOGGER = logging.getLogger("queroquero.train")
RESOLVED_TRAINING_SCHEMA = "queroquero-resolved-training/v2"
RUN_MANIFEST_SCHEMA = "queroquero-training-run/v2"
CHECKPOINT_SCHEMA = "queroquero-training-checkpoint/v2"
LATEST_CHECKPOINT_SCHEMA = "queroquero-latest-checkpoint/v2"
_SIGNAL_REQUESTED = False


class TrainingInterrupted(RuntimeError):
    exit_code = 99


def run_preflight(config_path: str | Path) -> Dict[str, Any] | None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config, config_sha256 = load_training_config(config_path)
    dependencies = _load_training_dependencies(config)
    torch = dependencies["torch"]
    context = DistributedContext.initialize(torch, config)
    try:
        _configure_reproducibility(
            torch, config["training"]["seed"], rank=context.rank
        )
        gpu = _validate_gpu_environment(torch, config, context)
        inputs = resolve_training_inputs(config)
        _verify_inputs_across_ranks(inputs, context)
        base_model, tokenizer = _load_model_and_tokenizer(
            config, inputs, dependencies, model_path=None
        )
        del tokenizer
        base_model.to(context.device)
        model = _wrap_distributed_model(base_model, config, context)
        optimizer = _build_optimizer(base_model, config, dependencies)
        scaler = _build_scaler(torch, config)
        scheduler = dependencies["get_linear_schedule_with_warmup"](
            optimizer,
            num_warmup_steps=0,
            num_training_steps=1,
        )
        train_sequences = load_training_sequences(
            inputs, config["training"]["seed"]
        )
        training = config["training"]
        local_batch = rank_step_batch(
            train_sequences,
            1,
            rank=context.rank,
            world_size=context.world_size,
            micro_batch_size_per_rank=training["micro_batch_size_per_rank"],
            accumulation_steps_per_rank=training[
                "gradient_accumulation_steps_per_rank"
            ],
        )
        torch.cuda.reset_peak_memory_stats(context.device)
        step_metrics = _train_optimizer_step(
            model=model,
            base_model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sequences=local_batch,
            config=config,
            dependencies=dependencies,
            context=context,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(context.device))
        peak_memory = int(context.reduce_max(peak_memory))
        result = {
            "status": "ok",
            "profile": config["profile"],
            "config_sha256": config_sha256,
            "inputs_sha256": inputs.digest(),
            "execution": _execution_metadata(config),
            "gpus": gpu["gpus"],
            "peak_memory_bytes": peak_memory,
            "loss": step_metrics["loss"],
            "loss_finite": True,
            "overflow_retries": step_metrics["overflow_retries"],
            "optimizer": config["training"]["optimizer"],
        }
        if context.is_main:
            LOGGER.info(
                "stage=preflight status=complete world_size=%d "
                "peak_memory_bytes=%d",
                context.world_size,
                peak_memory,
            )
            return result
        return None
    finally:
        context.close()


def cache_pinned_model() -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    return {
        "status": "cached",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }


def run_training(
    config_path: str | Path,
    *,
    resume: bool = False,
    stop_after_checkpoint: bool = False,
) -> Dict[str, Any] | None:
    global _SIGNAL_REQUESTED
    _SIGNAL_REQUESTED = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config, config_sha256 = load_training_config(config_path)
    if stop_after_checkpoint and config["profile"] != "smoke":
        raise RuntimeError("--stop-after-checkpoint is restricted to the smoke profile")
    if resume and stop_after_checkpoint:
        raise RuntimeError("--resume and --stop-after-checkpoint cannot be combined")
    _install_signal_handler()
    dependencies = _load_training_dependencies(config)
    torch = dependencies["torch"]
    context = DistributedContext.initialize(torch, config)
    try:
        if not context.is_main:
            LOGGER.setLevel(logging.WARNING)
        _configure_reproducibility(
            torch, config["training"]["seed"], rank=context.rank
        )
        gpu = _validate_gpu_environment(torch, config, context)
        git_commit = _run_on_main(context, _clean_git_commit)
        inputs = resolve_training_inputs(config)
        _verify_inputs_across_ranks(inputs, context)
        resolved = _resolved_training_metadata(
            config, config_sha256, inputs, git_commit, gpu
        )
        run_id = resolved["run_id"]
        roots = resolve_training_output_roots(config)
        run_dir = roots["runs_root"] / run_id
        checkpoint_root = roots["checkpoints_root"] / run_id
        run_manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics.jsonl"

        if resume:
            def load_resume_state() -> Dict[str, Any]:
                existing_resolved = _load_existing_resolved(run_dir)
                if existing_resolved != resolved:
                    raise RuntimeError(
                        "resolved training metadata changed since the checkpoint"
                    )
                current_manifest = _load_run_manifest(run_manifest_path, run_id)
                checkpoint, checkpoint_manifest = _latest_checkpoint(
                    checkpoint_root, resolved
                )
                if current_manifest.get("baseline_evaluation") is None:
                    raise RuntimeError(
                        "resumable run is missing its baseline evaluation"
                    )
                return {
                    "run_manifest": current_manifest,
                    "checkpoint_path": checkpoint.as_posix(),
                    "start_step": checkpoint_manifest["optimizer_step"],
                }

            resume_state = _run_on_main(context, load_resume_state)
            run_manifest = resume_state["run_manifest"]
            checkpoint_path = Path(resume_state["checkpoint_path"])
            model_path: Path | None = checkpoint_path / "model"
            start_step = resume_state["start_step"]
            if context.is_main:
                LOGGER.info(
                    "stage=resume status=ready run_id=%s optimizer_step=%d",
                    run_id,
                    start_step,
                )
        else:
            run_manifest = {
                "schema_version": RUN_MANIFEST_SCHEMA,
                "run_id": run_id,
                "profile": config["profile"],
                "status": "initializing",
                "config_sha256": config_sha256,
                "inputs_sha256": inputs.digest(),
                "git_commit": git_commit,
                "execution": _execution_metadata(config),
                "baseline_evaluation": None,
                "final_evaluation": None,
                "optimizer_steps_completed": 0,
                "quality_gate_passed": None,
                "promotion_status": "not_evaluated",
                "artifact": None,
                "metrics": None,
            }

            def create_run() -> bool:
                if run_dir.exists() or checkpoint_root.exists():
                    raise RuntimeError(
                        "training run already exists; use --resume only with a "
                        "valid checkpoint"
                    )
                run_dir.mkdir(parents=True, mode=0o700)
                checkpoint_root.mkdir(parents=True, mode=0o700)
                write_json_atomic(run_dir / "resolved_training.json", resolved)
                write_json_atomic(run_manifest_path, run_manifest)
                return True

            _run_on_main(context, create_run)
            checkpoint_path = None
            model_path = None
            start_step = 0

        base_model, tokenizer = _load_model_and_tokenizer(
            config, inputs, dependencies, model_path=model_path
        )
        base_model.to(context.device)
        model = _wrap_distributed_model(base_model, config, context)
        optimizer = _build_optimizer(base_model, config, dependencies)
        scheduler = dependencies["get_linear_schedule_with_warmup"](
            optimizer,
            num_warmup_steps=config["training"]["warmup_steps"],
            num_training_steps=config["training"]["total_optimizer_steps"],
        )
        scaler = _build_scaler(torch, config)
        if resume:
            _restore_checkpoint_state(
                checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                torch=torch,
                expected_step=start_step,
                expected_sequences=start_step
                * config["training"]["global_batch_sequences"],
                rank=context.rank,
                world_size=context.world_size,
                device=context.device,
            )

        evaluation_sequences = load_evaluation_sequences(inputs)
        if not resume:
            if context.is_main:
                LOGGER.info(
                    "stage=evaluate phase=baseline status=started run_id=%s",
                    run_id,
                )
            baseline = _evaluate_model(
                model,
                evaluation_sequences,
                dependencies,
                context=context,
                config=config,
                optimizer_step=0,
            )

            def record_baseline() -> bool:
                _append_metric(
                    metrics_path,
                    {
                        "event": "evaluation",
                        "phase": "baseline",
                        "optimizer_step": 0,
                        **baseline,
                    },
                )
                run_manifest["baseline_evaluation"] = baseline
                run_manifest["status"] = "training"
                write_json_atomic(run_manifest_path, run_manifest)
                LOGGER.info(
                    "stage=evaluate phase=baseline status=complete run_id=%s",
                    run_id,
                )
                return True

            _run_on_main(context, record_baseline)
            run_manifest["baseline_evaluation"] = baseline
            run_manifest["status"] = "training"

        training_sequences = load_training_sequences(
            inputs, config["training"]["seed"]
        )
        training = config["training"]
        global_batch_size = training["global_batch_sequences"]
        total_steps = training["total_optimizer_steps"]
        if len(training_sequences) != total_steps * global_batch_size:
            raise RuntimeError("training schedule length changed")
        torch.cuda.reset_peak_memory_stats(context.device)
        if context.is_main:
            LOGGER.info(
                "stage=train status=started run_id=%s start_step=%d "
                "total_steps=%d world_size=%d",
                run_id,
                start_step,
                total_steps,
                context.world_size,
            )
        for optimizer_step in range(start_step + 1, total_steps + 1):
            global_batch = global_step_batch(
                training_sequences, optimizer_step, global_batch_size
            )
            local_batch = rank_step_batch(
                training_sequences,
                optimizer_step,
                rank=context.rank,
                world_size=context.world_size,
                micro_batch_size_per_rank=training[
                    "micro_batch_size_per_rank"
                ],
                accumulation_steps_per_rank=training[
                    "gradient_accumulation_steps_per_rank"
                ],
            )
            step_metrics = _train_optimizer_step(
                model=model,
                base_model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sequences=local_batch,
                config=config,
                dependencies=dependencies,
                context=context,
            )
            peak_memory = int(
                context.reduce_max(torch.cuda.max_memory_allocated(context.device))
            )
            step_metrics.update(
                {
                    "event": "optimizer_step",
                    "optimizer_step": optimizer_step,
                    "sequences_consumed": optimizer_step * global_batch_size,
                    "dataset_counts": _dataset_counts(global_batch),
                    "peak_memory_bytes": peak_memory,
                }
            )
            run_manifest["optimizer_steps_completed"] = optimizer_step
            run_manifest["status"] = "training"

            def record_optimizer_step() -> bool:
                _append_metric(metrics_path, step_metrics)
                write_json_atomic(run_manifest_path, run_manifest)
                LOGGER.info(
                    "stage=train status=step run_id=%s optimizer_step=%d "
                    "loss=%.6f lr=%.10f",
                    run_id,
                    optimizer_step,
                    step_metrics["loss"],
                    step_metrics["learning_rate"],
                )
                return True

            _run_on_main(context, record_optimizer_step)

            scheduled = optimizer_step in training["checkpoint_steps"]
            interrupted = _interruption_requested(context)
            if scheduled or interrupted:
                checkpoint = _save_checkpoint(
                    checkpoint_root=checkpoint_root,
                    resolved=resolved,
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    torch=torch,
                    optimizer_step=optimizer_step,
                    context=context,
                )

                def record_checkpoint() -> bool:
                    _append_metric(
                        metrics_path,
                        {
                            "event": "checkpoint",
                            "optimizer_step": optimizer_step,
                            "checkpoint_id": checkpoint.name,
                            "reason": "signal" if interrupted else "scheduled",
                        },
                    )
                    return True

                _run_on_main(context, record_checkpoint)
                if stop_after_checkpoint and scheduled:
                    run_manifest["status"] = "checkpoint_ready"

                    def record_checkpoint_ready() -> bool:
                        write_json_atomic(run_manifest_path, run_manifest)
                        return True

                    _run_on_main(context, record_checkpoint_ready)
                    if context.is_main:
                        return {
                            "status": "checkpoint_ready",
                            "run_id": run_id,
                            "optimizer_step": optimizer_step,
                        }
                    return None
                if interrupted:
                    run_manifest["status"] = "interrupted"

                    def record_interruption() -> bool:
                        write_json_atomic(run_manifest_path, run_manifest)
                        _write_interruption_status(run_id, optimizer_step)
                        return True

                    _run_on_main(context, record_interruption)
                    if context.is_main:
                        if context.strategy == "single_process":
                            raise TrainingInterrupted(
                                "training checkpointed after signal at step "
                                f"{optimizer_step}"
                            )
                        return {
                            "status": "interrupted",
                            "run_id": run_id,
                            "optimizer_step": optimizer_step,
                            "exit_code": TrainingInterrupted.exit_code,
                        }
                    return None

        if context.is_main:
            LOGGER.info("stage=train status=complete run_id=%s", run_id)
            LOGGER.info(
                "stage=evaluate phase=final status=started run_id=%s", run_id
            )
        final_evaluation = _evaluate_model(
            model,
            evaluation_sequences,
            dependencies,
            context=context,
            config=config,
            optimizer_step=total_steps,
        )
        quality_gate = (
            final_evaluation["macro"]["loss"]
            <= run_manifest["baseline_evaluation"]["macro"]["loss"]
        )
        run_manifest["final_evaluation"] = final_evaluation
        run_manifest["optimizer_steps_completed"] = total_steps
        run_manifest["quality_gate_passed"] = quality_gate
        run_manifest["promotion_status"] = "eligible" if quality_gate else "blocked"

        def record_final_evaluation() -> bool:
            _append_metric(
                metrics_path,
                {
                    "event": "evaluation",
                    "phase": "final",
                    "optimizer_step": total_steps,
                    **final_evaluation,
                },
            )
            return True

        _run_on_main(context, record_final_evaluation)

        artifact_metadata = None
        if config["profile"] == "mvp":
            training_provenance = {
                "method": TRAINING_METHOD,
                "git_commit": git_commit,
                "run_id": run_id,
                "seed": training["seed"],
                "config_sha256": config_sha256,
                "inputs_sha256": inputs.digest(),
                "optimizer_steps": total_steps,
                "execution": _execution_metadata(config),
                "datasets": [
                    {
                        "dataset_id": dataset.dataset_id,
                        "preparation_id": dataset.manifest["preparation_id"],
                        "dataset_manifest_sha256": dataset.manifest_sha256,
                    }
                    for dataset in inputs.datasets
                ],
            }

            def export_artifact() -> Dict[str, Any]:
                LOGGER.info("stage=export status=started run_id=%s", run_id)
                artifact_path = export_model_artifact(
                    model=base_model,
                    tokenizer=tokenizer,
                    config=config,
                    training=training_provenance,
                    prepared_tokenizer=inputs.tokenizer,
                    artifacts_root=roots["artifacts_root"],
                )
                artifact_manifest = validate_model_artifact(
                    artifact_path, load_model=False
                )
                value = {
                    "artifact_id": artifact_manifest["artifact_id"],
                    "artifact_sha256": artifact_manifest["artifact_sha256"],
                    "path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
                }
                LOGGER.info(
                    "stage=export status=complete artifact_id=%s",
                    artifact_manifest["artifact_id"],
                )
                return value

            artifact_metadata = _run_on_main(context, export_artifact)
            run_manifest["artifact"] = artifact_metadata

        run_manifest["status"] = "complete"

        def complete_run() -> Dict[str, Any]:
            run_manifest["metrics"] = {
                "path": metrics_path.relative_to(run_dir).as_posix(),
                "size_bytes": metrics_path.stat().st_size,
                "sha256": file_sha256(metrics_path),
            }
            write_json_atomic(run_manifest_path, run_manifest)
            return {
                "status": "complete",
                "run_id": run_id,
                "profile": config["profile"],
                "optimizer_steps": total_steps,
                "quality_gate_passed": quality_gate,
                "promotion_status": run_manifest["promotion_status"],
                "artifact": artifact_metadata,
            }

        result = _run_on_main(context, complete_run)
        if context.is_main:
            return result
        return None
    finally:
        context.close()


def _load_training_dependencies(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "common training dependencies are missing; install the requirements "
            "for the selected accelerator"
        ) from exc
    if torch.__version__ != "2.7.1+cu118":
        raise RuntimeError("training requires torch 2.7.1+cu118")
    if torch.version.cuda != config["hardware"]["torch_cuda"]:
        raise RuntimeError("training requires the PyTorch CUDA 11.8 wheel")
    if transformers.__version__ != "5.14.1":
        raise RuntimeError("training requires transformers 5.14.1")
    dependencies = {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }
    if config["training"]["optimizer_implementation"] == "bitsandbytes":
        try:
            import bitsandbytes
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise RuntimeError(
                "P100 training requires requirements-train-p100.txt"
            ) from exc
        if bitsandbytes.__version__ != "0.50.0":
            raise RuntimeError("P100 training requires bitsandbytes 0.50.0")
        dependencies["bitsandbytes"] = bitsandbytes
        dependencies["AdamW8bit"] = AdamW8bit
    return dependencies


def _cuda_binary_arch_is_compatible(
    compiled_arch: str, device_capability: tuple[int, int]
) -> bool:
    match = re.fullmatch(r"sm_(\d+)", compiled_arch)
    if match is None:
        return False
    compiled_capability = divmod(int(match.group(1)), 10)
    return (
        compiled_capability[0] == device_capability[0]
        and compiled_capability[1] <= device_capability[1]
    )


def _validate_gpu_environment(
    torch: Any, config: Dict[str, Any], context: DistributedContext
) -> Dict[str, Any]:
    expected = config["hardware"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != expected["visible_gpus"]:
        raise RuntimeError(
            f"training requires exactly {expected['visible_gpus']} visible GPUs"
        )
    properties = torch.cuda.get_device_properties(context.local_rank)
    if expected["gpu_name_contains"].lower() not in properties.name.lower():
        raise RuntimeError(
            f"training requires {expected['gpu_name_contains']}; found "
            f"{properties.name}"
        )
    capability = tuple(torch.cuda.get_device_capability(context.local_rank))
    if capability != tuple(expected["compute_capability"]):
        raise RuntimeError(
            f"GPU compute capability must be {expected['required_arch']}"
        )
    minimum_bytes = expected["minimum_memory_gib"] * 1024**3
    if int(properties.total_memory) < minimum_bytes:
        raise RuntimeError(
            f"GPU has less than {expected['minimum_memory_gib']} GiB of memory"
        )
    compiled_arches = tuple(torch.cuda.get_arch_list())
    if not any(
        _cuda_binary_arch_is_compatible(arch, capability)
        for arch in compiled_arches
    ):
        raise RuntimeError(
            "installed PyTorch wheel does not include kernels compatible with "
            f"{expected['required_arch']}; compiled architectures: "
            f"{list(compiled_arches)}"
        )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if config["training"]["precision"] == "bf16" and not bf16_supported:
        raise RuntimeError("L40S execution requires native BF16 support")
    if config["training"]["precision"] == "fp16" and bf16_supported:
        raise RuntimeError("P100 execution must not silently switch to BF16")
    nccl_version = None
    if context.backend == "nccl":
        try:
            reported_nccl_version = torch.cuda.nccl.version()
            nccl_version = (
                list(reported_nccl_version)
                if isinstance(reported_nccl_version, tuple)
                else reported_nccl_version
            )
        except Exception as exc:
            raise RuntimeError("NCCL runtime is unavailable") from exc
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "name": properties.name,
        "compute_capability": list(capability),
        "memory_bytes": int(properties.total_memory),
        "bf16_supported": bf16_supported,
    }
    gpus = sorted(context.all_gather_objects(local), key=lambda item: item["rank"])
    if len({item["local_rank"] for item in gpus}) != context.world_size:
        raise RuntimeError("DDP ranks do not map one-to-one to the visible GPUs")
    signatures = {
        (
            item["name"],
            tuple(item["compute_capability"]),
            item["memory_bytes"],
            item["bf16_supported"],
        )
        for item in gpus
    }
    if len(signatures) != 1:
        raise RuntimeError("distributed training requires homogeneous GPUs")
    return {
        "gpus": gpus,
        "torch_cuda": torch.version.cuda,
        "nccl_version": nccl_version,
    }


def _configure_reproducibility(torch: Any, seed: int, *, rank: int = 0) -> None:
    rank_seed = seed + rank
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_model_and_tokenizer(
    config: Dict[str, Any],
    inputs: ResolvedTrainingInputs,
    dependencies: Dict[str, Any],
    *,
    model_path: Path | None,
) -> tuple[Any, Any]:
    torch = dependencies["torch"]
    model_config = config["model"]
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer_fingerprint(tokenizer) != inputs.tokenizer["fingerprint_sha256"]:
        raise RuntimeError("cached tokenizer differs from prepared datasets")
    source: str | Path = model_path if model_path is not None else MODEL_ID
    keyword_arguments: Dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": True,
        "dtype": torch.float32,
        "attn_implementation": "eager",
    }
    if model_path is None:
        keyword_arguments["revision"] = MODEL_REVISION
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        source, **keyword_arguments
    )
    if model.config.model_type != model_config["model_type"]:
        raise RuntimeError("cached model architecture changed")
    if int(model.config.max_position_embeddings) != model_config[
        "native_context_length"
    ]:
        raise RuntimeError("cached model native context changed")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_config["parameter_count"]:
        raise RuntimeError("cached model parameter count changed")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("trainable model parameters must remain FP32")
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count != parameter_count:
        raise RuntimeError("continual pretraining must update every model parameter")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def _wrap_distributed_model(
    model: Any, config: Dict[str, Any], context: DistributedContext
) -> Any:
    if config["execution"]["strategy"] == "single_process":
        return model
    return context.torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        static_graph=False,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def _build_scaler(torch: Any, config: Dict[str, Any]) -> Any | None:
    if not config["training"]["use_grad_scaler"]:
        return None
    return torch.amp.GradScaler("cuda")


def _build_optimizer(
    model: Any, config: Dict[str, Any], dependencies: Dict[str, Any]
) -> Any:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    training = config["training"]
    parameter_groups = [
        {"params": decay, "weight_decay": training["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    common = {
        "lr": training["learning_rate"],
        "betas": tuple(training["betas"]),
        "eps": training["epsilon"],
    }
    if training["optimizer_implementation"] == "bitsandbytes":
        return dependencies["AdamW8bit"](
            parameter_groups,
            **common,
            min_8bit_size=4096,
            percentile_clipping=100,
        )
    try:
        return dependencies["torch"].optim.AdamW(
            parameter_groups,
            **common,
            fused=True,
        )
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError("L40S training requires fused torch.optim.AdamW") from exc


def _train_optimizer_step(
    *,
    model: Any,
    base_model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sequences: list[TrainingSequence],
    config: Dict[str, Any],
    dependencies: Dict[str, Any],
    context: DistributedContext,
) -> Dict[str, Any]:
    torch = dependencies["torch"]
    training = config["training"]
    precision = training["precision"]
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    max_attempts = 8 if precision == "fp16" else 1
    for attempt in range(1, max_attempts + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.monotonic()
        for index, sequence in enumerate(sequences):
            input_ids = torch.tensor(
                sequence.input_ids, dtype=torch.long, device=context.device
            ).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            synchronization = (
                model.no_sync()
                if context.initialized and index < len(sequences) - 1
                else contextlib.nullcontext()
            )
            with synchronization:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    ).loss
                local_loss_finite = bool(torch.isfinite(loss).item())
                if context.any_true(not local_loss_finite):
                    raise RuntimeError("training produced a non-finite loss")
                losses.append(float(loss.detach().cpu()))
                scaled_loss = loss / len(sequences)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
        if not losses:
            raise RuntimeError("optimizer step received no local sequences")
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        if scaler is not None:
            scaler.unscale_(optimizer)
        if training["optimizer_implementation"] == "torch_fused":
            _validate_gradient_dtypes(base_model, torch)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            base_model.parameters(), training["max_grad_norm"]
        )
        grad_norm_value = float(grad_norm.detach().cpu())
        non_finite_gradient = not math.isfinite(grad_norm_value)
        if context.any_true(non_finite_gradient) and scaler is None:
            raise RuntimeError("BF16 training produced a non-finite gradient")
        if scaler is None:
            optimizer.step()
            current_scale = None
        else:
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            current_scale = float(scaler.get_scale())
            overflow = current_scale < previous_scale
            if context.any_true(overflow):
                LOGGER.warning(
                    "stage=train status=overflow retry=%d previous_scale=%s "
                    "new_scale=%s",
                    attempt,
                    previous_scale,
                    current_scale,
                )
                continue
        if training["optimizer_implementation"] == "torch_fused":
            _validate_fused_optimizer_state(optimizer, torch)
        scheduler.step()
        torch.cuda.synchronize(context.device)
        elapsed = context.reduce_max(time.monotonic() - started)
        global_loss_sum = context.reduce_sum(sum(losses))
        global_loss = global_loss_sum / training["global_batch_sequences"]
        global_grad_norm = context.reduce_max(grad_norm_value)
        if not math.isfinite(global_loss):
            raise RuntimeError("training produced a non-finite global loss")
        return {
            "loss": global_loss,
            "learning_rate": learning_rate_used,
            "grad_norm": global_grad_norm,
            "grad_scale": current_scale,
            "overflow_retries": attempt - 1,
            "elapsed_seconds": elapsed,
            "tokens_per_second": training["global_batch_tokens"]
            / max(elapsed, 1e-9),
        }
    raise RuntimeError("FP16 gradient scaling overflowed eight consecutive times")


def _validate_fused_optimizer_state(optimizer: Any, torch: Any) -> None:
    for state in optimizer.state.values():
        for value in state.values():
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and value.dtype != torch.float32
            ):
                raise RuntimeError("fused AdamW optimizer states must remain FP32")


def _validate_gradient_dtypes(model: Any, torch: Any) -> None:
    if any(
        parameter.grad is not None and parameter.grad.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise RuntimeError("L40S gradients must remain FP32")


def _evaluate_model(
    model: Any,
    sequences: Dict[str, list[TrainingSequence]],
    dependencies: Dict[str, Any],
    *,
    context: DistributedContext,
    config: Dict[str, Any],
    optimizer_step: int,
) -> Dict[str, Any]:
    torch = dependencies["torch"]
    autocast_dtype = (
        torch.float16
        if config["training"]["precision"] == "fp16"
        else torch.bfloat16
    )
    model.eval()
    datasets: Dict[str, Any] = {}
    with torch.no_grad():
        for dataset_id in sorted(sequences):
            losses = []
            local_sequences = sequences[dataset_id][context.rank :: context.world_size]
            for sequence in local_sequences:
                input_ids = torch.tensor(
                    sequence.input_ids, dtype=torch.long, device=context.device
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    ).loss
                local_finite = bool(torch.isfinite(loss).item())
                if context.any_true(not local_finite):
                    raise RuntimeError("evaluation produced a non-finite loss")
                losses.append(float(loss.detach().cpu()))
            loss_sum = context.reduce_sum(sum(losses))
            count = int(context.reduce_sum(len(losses)))
            if count != len(sequences[dataset_id]):
                raise RuntimeError("distributed evaluation coverage changed")
            mean_loss = loss_sum / count
            datasets[dataset_id] = {
                "sequences": count,
                "tokens": count * config["training"]["sequence_length"],
                "loss": mean_loss,
                "perplexity": math.exp(min(mean_loss, 80.0)),
            }
    macro_loss = sum(value["loss"] for value in datasets.values()) / len(datasets)
    return {
        "datasets": datasets,
        "macro": {
            "loss": macro_loss,
            "perplexity": math.exp(min(macro_loss, 80.0)),
        },
        "optimizer_step": optimizer_step,
    }


def _resolved_training_metadata(
    config: Dict[str, Any],
    config_sha256: str,
    inputs: ResolvedTrainingInputs,
    git_commit: str,
    gpu: Dict[str, Any],
) -> Dict[str, Any]:
    value = {
        "schema_version": RESOLVED_TRAINING_SCHEMA,
        "config_sha256": config_sha256,
        "inputs_sha256": inputs.digest(),
        "git_commit": git_commit,
        "profile": config["profile"],
        "model": config["model"],
        "training": config["training"],
        "execution": _execution_metadata(config),
        "inputs": inputs.metadata(),
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": importlib.metadata.version("torch"),
            "torch_cuda": gpu["torch_cuda"],
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": (
                importlib.metadata.version("bitsandbytes")
                if config["training"]["optimizer_implementation"] == "bitsandbytes"
                else None
            ),
            "gpus": gpu["gpus"],
            "nccl_version": gpu["nccl_version"],
        },
    }
    run_identity = {
        key: value[key]
        for key in (
            "schema_version",
            "config_sha256",
            "inputs_sha256",
            "git_commit",
            "profile",
            "model",
            "training",
            "execution",
        )
    }
    value["run_id"] = sha256_bytes(canonical_json_bytes(run_identity))[:20]
    return value


def _save_checkpoint(
    *,
    checkpoint_root: Path,
    resolved: Dict[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    torch: Any,
    optimizer_step: int,
    context: DistributedContext,
) -> Path:
    checkpoint_id = f"step-{optimizer_step:06d}"
    final = checkpoint_root / checkpoint_id
    local_rng = {
        "rank": context.rank,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }
    rng_by_rank = context.all_gather_objects(local_rng)

    def write_checkpoint() -> str:
        if final.exists():
            _validate_checkpoint(final, resolved)
            write_json_atomic(
                checkpoint_root / "latest.json",
                {
                    "schema_version": LATEST_CHECKPOINT_SCHEMA,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_manifest_sha256": file_sha256(
                        final / "checkpoint_manifest.json"
                    ),
                },
            )
            return final.as_posix()
        temporary = checkpoint_root / (
            f".{checkpoint_id}.partial-{uuid.uuid4().hex}"
        )
        temporary.mkdir(parents=True, mode=0o700)
        try:
            model_dir = temporary / "model"
            model.config.use_cache = False
            model.save_pretrained(
                model_dir, safe_serialization=True, max_shard_size="5GB"
            )
            sequences_consumed = (
                optimizer_step * resolved["training"]["global_batch_sequences"]
            )
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "rng_by_rank": rng_by_rank,
                    "optimizer_step": optimizer_step,
                    "sequences_consumed": sequences_consumed,
                    "world_size": context.world_size,
                },
                temporary / "training_state.pt",
            )
            records = _recursive_file_records(temporary)
            manifest = {
                "schema_version": CHECKPOINT_SCHEMA,
                "checkpoint_id": checkpoint_id,
                "run_id": resolved["run_id"],
                "optimizer_step": optimizer_step,
                "sequences_consumed": sequences_consumed,
                "world_size": context.world_size,
                "global_batch_sequences": resolved["training"][
                    "global_batch_sequences"
                ],
                "config_sha256": resolved["config_sha256"],
                "inputs_sha256": resolved["inputs_sha256"],
                "git_commit": resolved["git_commit"],
                "files": records,
                "files_sha256": sha256_bytes(canonical_json_bytes(records)),
            }
            write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
            temporary.replace(final)
            write_json_atomic(
                checkpoint_root / "latest.json",
                {
                    "schema_version": LATEST_CHECKPOINT_SCHEMA,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_manifest_sha256": file_sha256(
                        final / "checkpoint_manifest.json"
                    ),
                },
            )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        LOGGER.info(
            "stage=checkpoint status=complete run_id=%s optimizer_step=%d",
            resolved["run_id"],
            optimizer_step,
        )
        return final.as_posix()

    return Path(_run_on_main(context, write_checkpoint))


def _latest_checkpoint(
    checkpoint_root: Path, resolved: Dict[str, Any]
) -> tuple[Path, Dict[str, Any]]:
    latest_path = checkpoint_root / "latest.json"
    if latest_path.is_symlink():
        raise RuntimeError("latest checkpoint marker must not be a symlink")
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("no checkpoint is available for --resume") from exc
    if latest.get("schema_version") != LATEST_CHECKPOINT_SCHEMA:
        raise RuntimeError("latest checkpoint marker schema changed")
    checkpoint_id = latest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not re.fullmatch(
        r"step-[0-9]{6}", checkpoint_id
    ):
        raise RuntimeError("latest checkpoint ID is invalid")
    checkpoint = checkpoint_root / checkpoint_id
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if file_sha256(manifest_path) != latest.get("checkpoint_manifest_sha256"):
        raise RuntimeError("latest checkpoint marker hash changed")
    manifest = _validate_checkpoint(checkpoint, resolved)
    return checkpoint, manifest


def _validate_checkpoint(path: Path, resolved: Dict[str, Any]) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("checkpoint must be a real directory")
    manifest_path = path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint schema changed")
    if manifest.get("checkpoint_id") != path.name:
        raise RuntimeError("checkpoint directory does not match its ID")
    for key in ("run_id", "config_sha256", "inputs_sha256", "git_commit"):
        if manifest.get(key) != resolved.get(key):
            raise RuntimeError(f"checkpoint {key} changed")
    step = manifest.get("optimizer_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise RuntimeError("checkpoint optimizer step is invalid")
    expected_consumed = step * resolved["training"]["global_batch_sequences"]
    if manifest.get("sequences_consumed") != expected_consumed:
        raise RuntimeError("checkpoint data cursor changed")
    if manifest.get("world_size") != resolved["execution"]["world_size"]:
        raise RuntimeError("checkpoint world size changed")
    if (
        manifest.get("global_batch_sequences")
        != resolved["training"]["global_batch_sequences"]
    ):
        raise RuntimeError("checkpoint global batch changed")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("checkpoint file list is missing")
    listed = set()
    for record in records:
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("checkpoint contains an unsafe file path")
        if relative.as_posix() in listed:
            raise RuntimeError("checkpoint lists a file more than once")
        candidate = path / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("checkpoint file is missing or is a symlink")
        if candidate.stat().st_size != record.get("size_bytes"):
            raise RuntimeError("checkpoint file size changed")
        if file_sha256(candidate) != record.get("sha256"):
            raise RuntimeError("checkpoint file hash changed")
        listed.add(relative.as_posix())
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "checkpoint_manifest.json"
    }
    if any(item.is_symlink() for item in path.rglob("*")) or actual != listed:
        raise RuntimeError("checkpoint contains missing or unexpected files")
    if manifest.get("files_sha256") != sha256_bytes(canonical_json_bytes(records)):
        raise RuntimeError("checkpoint aggregate hash changed")
    return manifest


def _restore_checkpoint_state(
    checkpoint: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    torch: Any,
    expected_step: int,
    expected_sequences: int,
    rank: int = 0,
    world_size: int = 1,
    device: Any = "cuda",
) -> None:
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    if state.get("optimizer_step") != expected_step:
        raise RuntimeError("checkpoint state step changed")
    if state.get("sequences_consumed") != expected_sequences:
        raise RuntimeError("checkpoint state data cursor changed")
    if state.get("world_size") != world_size:
        raise RuntimeError("checkpoint state world size changed")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler_state = state.get("scaler")
    if scaler is None:
        if scaler_state is not None:
            raise RuntimeError("BF16 checkpoint unexpectedly contains a scaler")
    else:
        if scaler_state is None:
            raise RuntimeError("FP16 checkpoint is missing its scaler")
        scaler.load_state_dict(scaler_state)
    rng_by_rank = state.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
        raise RuntimeError("checkpoint RNG state does not match world size")
    matching = [
        item
        for item in rng_by_rank
        if isinstance(item, dict) and item.get("rank") == rank
    ]
    if len(matching) != 1:
        raise RuntimeError("checkpoint RNG state does not identify this rank")
    rng = matching[0]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    torch.cuda.set_rng_state(rng["torch_cuda"].cpu(), device=device)


def _recursive_file_records(root: Path) -> list[Dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    ]


def _clean_git_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("training requires a clean Git checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Git HEAD is not a full commit SHA")
    return commit


def _load_existing_resolved(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "resolved_training.json"
    if path.is_symlink():
        raise RuntimeError("resolved training metadata must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("resolved training metadata is missing") from exc
    if not isinstance(value, dict) or value.get("schema_version") != RESOLVED_TRAINING_SCHEMA:
        raise RuntimeError("resolved training metadata schema changed")
    return value


def _load_run_manifest(path: Path, run_id: str) -> Dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError("run manifest must not be a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("run manifest is missing") from exc
    if value.get("schema_version") != RUN_MANIFEST_SCHEMA or value.get("run_id") != run_id:
        raise RuntimeError("run manifest identity changed")
    if value.get("status") == "complete":
        raise RuntimeError("completed training runs cannot be resumed")
    return value


def _append_metric(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _dataset_counts(sequences: Iterable[TrainingSequence]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sequence in sequences:
        counts[sequence.dataset_id] = counts.get(sequence.dataset_id, 0) + 1
    return dict(sorted(counts.items()))


def _execution_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    training = config["training"]
    execution = config["execution"]
    return {
        "strategy": execution["strategy"],
        "backend": execution["backend"],
        "world_size": execution["world_size"],
        "precision": training["precision"],
        "optimizer": training["optimizer"],
        "optimizer_implementation": training["optimizer_implementation"],
        "micro_batch_size_per_rank": training["micro_batch_size_per_rank"],
        "gradient_accumulation_steps_per_rank": training[
            "gradient_accumulation_steps_per_rank"
        ],
        "global_batch_sequences": training["global_batch_sequences"],
        "global_batch_tokens": training["global_batch_tokens"],
    }


def _run_on_main(context: DistributedContext, callback: Any) -> Any:
    payload = None
    if context.is_main:
        try:
            payload = {"ok": True, "value": callback()}
        except Exception as exc:
            payload = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    payload = context.broadcast_object(payload)
    if not payload["ok"]:
        raise RuntimeError(
            f"rank 0 {payload['error_type']}: {payload['message']}"
        )
    return payload["value"]


def _verify_inputs_across_ranks(
    inputs: ResolvedTrainingInputs, context: DistributedContext
) -> None:
    digests = context.all_gather_objects(inputs.digest())
    if len(set(digests)) != 1:
        raise RuntimeError("resolved training inputs differ between ranks")


def _interruption_requested(context: DistributedContext) -> bool:
    requested = None
    if context.is_main:
        marker = os.environ.get("TRAIN_INTERRUPT_FILE")
        marker_exists = False
        if marker:
            marker_path = Path(marker).expanduser().resolve()
            if PROJECT_ROOT not in marker_path.parents:
                raise RuntimeError("TRAIN_INTERRUPT_FILE must stay inside the project")
            marker_exists = marker_path.is_file()
        requested = _SIGNAL_REQUESTED or marker_exists
    return bool(context.broadcast_object(requested))


def _write_interruption_status(run_id: str, optimizer_step: int) -> None:
    requested = os.environ.get("TRAIN_STATUS_FILE")
    if not requested:
        return
    path = Path(requested).expanduser().resolve()
    if PROJECT_ROOT not in path.parents:
        raise RuntimeError("TRAIN_STATUS_FILE must stay inside the project")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "status": "interrupted",
            "exit_code": TrainingInterrupted.exit_code,
            "run_id": run_id,
            "optimizer_step": optimizer_step,
        },
    )


def _install_signal_handler() -> None:
    if not hasattr(signal, "SIGUSR1"):
        return

    def request_checkpoint(signum: int, frame: Any) -> None:
        del signum, frame
        global _SIGNAL_REQUESTED
        _SIGNAL_REQUESTED = True
        LOGGER.warning("stage=signal status=checkpoint_requested")

    signal.signal(signal.SIGUSR1, request_checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-parameter continual pretraining for the pinned Tucano2 model"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "cache-model", help="cache the exact model revision before offline jobs"
    )
    preflight = subparsers.add_parser(
        "preflight", help="validate the selected GPU training stack"
    )
    preflight.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run", help="run or resume continual pretraining")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stop-after-checkpoint", action="store_true")
    validate = subparsers.add_parser("validate", help="validate a model artifact")
    validate.add_argument("--artifact", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    try:
        if args.command == "cache-model":
            result = cache_pinned_model()
        elif args.command == "preflight":
            result = run_preflight(args.config)
        elif args.command == "run":
            result = run_training(
                args.config,
                resume=args.resume,
                stop_after_checkpoint=args.stop_after_checkpoint,
            )
        else:
            manifest = validate_model_artifact(args.artifact, load_model=True)
            result = {
                "status": "valid",
                "artifact_id": manifest["artifact_id"],
                "artifact_sha256": manifest["artifact_sha256"],
            }
        if result is not None:
            print(
                json.dumps(
                    result, ensure_ascii=False, sort_keys=True, allow_nan=False
                )
            )
    except TrainingInterrupted as exc:
        print(json.dumps({"status": "interrupted", "message": str(exc)}))
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
