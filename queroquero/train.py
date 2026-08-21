from __future__ import annotations

import argparse
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


LOGGER = logging.getLogger("queroquero.train")
RESOLVED_TRAINING_SCHEMA = "queroquero-resolved-training/v1"
RUN_MANIFEST_SCHEMA = "queroquero-training-run/v1"
CHECKPOINT_SCHEMA = "queroquero-training-checkpoint/v1"
LATEST_CHECKPOINT_SCHEMA = "queroquero-latest-checkpoint/v1"
_SIGNAL_REQUESTED = False


class TrainingInterrupted(RuntimeError):
    exit_code = 99


def run_preflight(config_path: str | Path) -> Dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config, config_sha256 = load_training_config(config_path)
    dependencies = _load_training_dependencies(config)
    torch = dependencies["torch"]
    _configure_reproducibility(torch, config["training"]["seed"])
    gpu = _validate_gpu_environment(torch, config)
    inputs = resolve_training_inputs(config)
    model, tokenizer = _load_model_and_tokenizer(
        config, inputs, dependencies, model_path=None
    )
    train_sequences = load_training_sequences(inputs, config["training"]["seed"])
    device = torch.device("cuda")
    model.to(device)
    optimizer = _build_optimizer(model, config, dependencies)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = dependencies["get_linear_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=0,
        num_training_steps=1,
    )
    torch.cuda.reset_peak_memory_stats(device)
    sequence = train_sequences[0]
    step_metrics = _train_optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sequences=[sequence],
        config=config,
        dependencies=dependencies,
    )
    result = {
        "status": "ok",
        "profile": config["profile"],
        "config_sha256": config_sha256,
        "inputs_sha256": inputs.digest(),
        "gpu": gpu,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "loss": step_metrics["loss"],
        "loss_finite": True,
        "overflow_retries": step_metrics["overflow_retries"],
        "optimizer": "adamw8bit",
    }
    LOGGER.info(
        "stage=preflight status=complete gpu=%s peak_memory_bytes=%d",
        gpu["name"],
        result["peak_memory_bytes"],
    )
    return result


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
) -> Dict[str, Any]:
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
    _configure_reproducibility(torch, config["training"]["seed"])
    gpu = _validate_gpu_environment(torch, config)
    git_commit = _clean_git_commit()
    inputs = resolve_training_inputs(config)
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
        existing_resolved = _load_existing_resolved(run_dir)
        if existing_resolved != resolved:
            raise RuntimeError("resolved training metadata changed since the checkpoint")
        run_manifest = _load_run_manifest(run_manifest_path, run_id)
        checkpoint_path, checkpoint_manifest = _latest_checkpoint(
            checkpoint_root, resolved
        )
        model_path = checkpoint_path / "model"
        start_step = checkpoint_manifest["optimizer_step"]
        if run_manifest.get("baseline_evaluation") is None:
            raise RuntimeError("resumable run is missing its baseline evaluation")
        LOGGER.info(
            "stage=resume status=ready run_id=%s optimizer_step=%d",
            run_id,
            start_step,
        )
    else:
        if run_dir.exists() or checkpoint_root.exists():
            raise RuntimeError(
                "training run already exists; use --resume only with a valid checkpoint"
            )
        run_dir.mkdir(parents=True, mode=0o700)
        checkpoint_root.mkdir(parents=True, mode=0o700)
        write_json_atomic(run_dir / "resolved_training.json", resolved)
        run_manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": run_id,
            "profile": config["profile"],
            "status": "initializing",
            "config_sha256": config_sha256,
            "inputs_sha256": inputs.digest(),
            "git_commit": git_commit,
            "baseline_evaluation": None,
            "final_evaluation": None,
            "optimizer_steps_completed": 0,
            "quality_gate_passed": None,
            "promotion_status": "not_evaluated",
            "artifact": None,
            "metrics": None,
        }
        write_json_atomic(run_manifest_path, run_manifest)
        model_path = None
        start_step = 0

    model, tokenizer = _load_model_and_tokenizer(
        config, inputs, dependencies, model_path=model_path
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = _build_optimizer(model, config, dependencies)
    scheduler = dependencies["get_linear_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=config["training"]["total_optimizer_steps"],
    )
    scaler = torch.amp.GradScaler("cuda")
    if resume:
        _restore_checkpoint_state(
            checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            torch=torch,
            expected_step=start_step,
            expected_sequences=start_step
            * config["training"]["gradient_accumulation_steps"],
        )

    evaluation_sequences = load_evaluation_sequences(inputs)
    if not resume:
        LOGGER.info("stage=evaluate phase=baseline status=started run_id=%s", run_id)
        baseline = _evaluate_model(
            model, evaluation_sequences, dependencies, optimizer_step=0
        )
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
        LOGGER.info("stage=evaluate phase=baseline status=complete run_id=%s", run_id)

    training_sequences = load_training_sequences(inputs, config["training"]["seed"])
    accumulation = config["training"]["gradient_accumulation_steps"]
    total_steps = config["training"]["total_optimizer_steps"]
    if len(training_sequences) != total_steps * accumulation:
        raise RuntimeError("training schedule length changed")
    torch.cuda.reset_peak_memory_stats(device)
    LOGGER.info(
        "stage=train status=started run_id=%s start_step=%d total_steps=%d",
        run_id,
        start_step,
        total_steps,
    )
    for optimizer_step in range(start_step + 1, total_steps + 1):
        start = (optimizer_step - 1) * accumulation
        batch = training_sequences[start : start + accumulation]
        step_metrics = _train_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sequences=batch,
            config=config,
            dependencies=dependencies,
        )
        step_metrics.update(
            {
                "event": "optimizer_step",
                "optimizer_step": optimizer_step,
                "sequences_consumed": optimizer_step * accumulation,
                "dataset_counts": _dataset_counts(batch),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        _append_metric(metrics_path, step_metrics)
        run_manifest["optimizer_steps_completed"] = optimizer_step
        run_manifest["status"] = "training"
        write_json_atomic(run_manifest_path, run_manifest)
        LOGGER.info(
            "stage=train status=step run_id=%s optimizer_step=%d loss=%.6f lr=%.10f",
            run_id,
            optimizer_step,
            step_metrics["loss"],
            step_metrics["learning_rate"],
        )

        scheduled = optimizer_step in config["training"]["checkpoint_steps"]
        if scheduled or _SIGNAL_REQUESTED:
            checkpoint = _save_checkpoint(
                checkpoint_root=checkpoint_root,
                resolved=resolved,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                torch=torch,
                optimizer_step=optimizer_step,
            )
            _append_metric(
                metrics_path,
                {
                    "event": "checkpoint",
                    "optimizer_step": optimizer_step,
                    "checkpoint_id": checkpoint.name,
                    "reason": "signal" if _SIGNAL_REQUESTED else "scheduled",
                },
            )
            if stop_after_checkpoint and scheduled:
                run_manifest["status"] = "checkpoint_ready"
                write_json_atomic(run_manifest_path, run_manifest)
                return {
                    "status": "checkpoint_ready",
                    "run_id": run_id,
                    "optimizer_step": optimizer_step,
                }
            if _SIGNAL_REQUESTED:
                run_manifest["status"] = "interrupted"
                write_json_atomic(run_manifest_path, run_manifest)
                raise TrainingInterrupted(
                    f"training checkpointed after signal at step {optimizer_step}"
                )

    LOGGER.info("stage=train status=complete run_id=%s", run_id)
    LOGGER.info("stage=evaluate phase=final status=started run_id=%s", run_id)
    final_evaluation = _evaluate_model(
        model, evaluation_sequences, dependencies, optimizer_step=total_steps
    )
    _append_metric(
        metrics_path,
        {
            "event": "evaluation",
            "phase": "final",
            "optimizer_step": total_steps,
            **final_evaluation,
        },
    )
    quality_gate = (
        final_evaluation["macro"]["loss"]
        <= run_manifest["baseline_evaluation"]["macro"]["loss"]
    )
    run_manifest["final_evaluation"] = final_evaluation
    run_manifest["optimizer_steps_completed"] = total_steps
    run_manifest["quality_gate_passed"] = quality_gate
    run_manifest["promotion_status"] = "eligible" if quality_gate else "blocked"

    artifact_metadata = None
    if config["profile"] == "mvp":
        training_provenance = {
            "method": TRAINING_METHOD,
            "git_commit": git_commit,
            "run_id": run_id,
            "seed": config["training"]["seed"],
            "config_sha256": config_sha256,
            "inputs_sha256": inputs.digest(),
            "optimizer_steps": total_steps,
            "datasets": [
                {
                    "dataset_id": dataset.dataset_id,
                    "preparation_id": dataset.manifest["preparation_id"],
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                }
                for dataset in inputs.datasets
            ],
        }
        LOGGER.info("stage=export status=started run_id=%s", run_id)
        artifact_path = export_model_artifact(
            model=model,
            tokenizer=tokenizer,
            config=config,
            training=training_provenance,
            prepared_tokenizer=inputs.tokenizer,
            artifacts_root=roots["artifacts_root"],
        )
        artifact_manifest = validate_model_artifact(artifact_path, load_model=False)
        artifact_metadata = {
            "artifact_id": artifact_manifest["artifact_id"],
            "artifact_sha256": artifact_manifest["artifact_sha256"],
            "path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
        }
        run_manifest["artifact"] = artifact_metadata
        LOGGER.info(
            "stage=export status=complete artifact_id=%s",
            artifact_manifest["artifact_id"],
        )

    run_manifest["status"] = "complete"
    run_manifest["metrics"] = {
        "path": metrics_path.relative_to(run_dir).as_posix(),
        "size_bytes": metrics_path.stat().st_size,
        "sha256": file_sha256(metrics_path),
    }
    write_json_atomic(run_manifest_path, run_manifest)
    result = {
        "status": "complete",
        "run_id": run_id,
        "profile": config["profile"],
        "optimizer_steps": total_steps,
        "quality_gate_passed": quality_gate,
        "promotion_status": run_manifest["promotion_status"],
        "artifact": artifact_metadata,
    }
    return result


def _load_training_dependencies(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import bitsandbytes
        import torch
        import transformers
        from bitsandbytes.optim import AdamW8bit
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install requirements-train.txt"
        ) from exc
    if torch.__version__ != "2.7.1+cu118":
        raise RuntimeError("training requires torch 2.7.1+cu118")
    if torch.version.cuda != config["hardware"]["torch_cuda"]:
        raise RuntimeError("training requires the PyTorch CUDA 11.8 wheel")
    if bitsandbytes.__version__ != "0.50.0":
        raise RuntimeError("training requires bitsandbytes 0.50.0")
    if transformers.__version__ != "5.14.1":
        raise RuntimeError("training requires transformers 5.14.1")
    return {
        "torch": torch,
        "bitsandbytes": bitsandbytes,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "AdamW8bit": AdamW8bit,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def _validate_gpu_environment(torch: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    expected = config["hardware"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != expected["visible_gpus"]:
        raise RuntimeError("training requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    if expected["gpu_name_contains"].lower() not in properties.name.lower():
        raise RuntimeError(f"training requires a P100; found {properties.name}")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != tuple(expected["compute_capability"]):
        raise RuntimeError("P100 compute capability must be sm_60")
    minimum_bytes = expected["minimum_memory_gib"] * 1024**3
    if int(properties.total_memory) < minimum_bytes:
        raise RuntimeError("P100 has less than 11 GiB of visible memory")
    if "sm_60" not in torch.cuda.get_arch_list():
        raise RuntimeError("installed PyTorch wheel does not include sm_60 kernels")
    if torch.cuda.is_bf16_supported():
        raise RuntimeError("P100 execution must not silently switch to BF16")
    return {
        "name": properties.name,
        "compute_capability": list(capability),
        "memory_bytes": int(properties.total_memory),
        "torch_cuda": torch.version.cuda,
    }


def _configure_reproducibility(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    return dependencies["AdamW8bit"](
        [
            {"params": decay, "weight_decay": training["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["epsilon"],
        min_8bit_size=4096,
        percentile_clipping=100,
    )


def _train_optimizer_step(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sequences: list[TrainingSequence],
    config: Dict[str, Any],
    dependencies: Dict[str, Any],
) -> Dict[str, Any]:
    torch = dependencies["torch"]
    device = torch.device("cuda")
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.monotonic()
        for sequence in sequences:
            input_ids = torch.tensor(
                sequence.input_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                ).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training produced a non-finite loss")
            losses.append(float(loss.detach().cpu()))
            scaler.scale(loss / len(sequences)).backward()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["max_grad_norm"]
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            LOGGER.warning(
                "stage=train status=overflow retry=%d previous_scale=%s new_scale=%s",
                attempt,
                previous_scale,
                scaler.get_scale(),
            )
            continue
        scheduler.step()
        elapsed = time.monotonic() - started
        return {
            "loss": sum(losses) / len(losses),
            "learning_rate": learning_rate_used,
            "grad_norm": float(grad_norm.detach().cpu()),
            "grad_scale": float(scaler.get_scale()),
            "overflow_retries": attempt - 1,
            "elapsed_seconds": elapsed,
            "tokens_per_second": len(sequences) * 1024 / max(elapsed, 1e-9),
        }
    raise RuntimeError("FP16 gradient scaling overflowed eight consecutive times")


def _evaluate_model(
    model: Any,
    sequences: Dict[str, list[TrainingSequence]],
    dependencies: Dict[str, Any],
    *,
    optimizer_step: int,
) -> Dict[str, Any]:
    torch = dependencies["torch"]
    device = torch.device("cuda")
    model.eval()
    datasets: Dict[str, Any] = {}
    with torch.no_grad():
        for dataset_id in sorted(sequences):
            losses = []
            for sequence in sequences[dataset_id]:
                input_ids = torch.tensor(
                    sequence.input_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    ).loss
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("evaluation produced a non-finite loss")
                losses.append(float(loss.detach().cpu()))
            mean_loss = sum(losses) / len(losses)
            datasets[dataset_id] = {
                "sequences": len(losses),
                "tokens": len(losses) * 1024,
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
        "inputs": inputs.metadata(),
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": importlib.metadata.version("torch"),
            "torch_cuda": gpu["torch_cuda"],
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "gpu": gpu,
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
) -> Path:
    checkpoint_id = f"step-{optimizer_step:06d}"
    final = checkpoint_root / checkpoint_id
    if final.exists():
        _validate_checkpoint(final, resolved)
        return final
    temporary = checkpoint_root / f".{checkpoint_id}.partial-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, mode=0o700)
    try:
        model_dir = temporary / "model"
        model.config.use_cache = False
        model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="5GB")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "optimizer_step": optimizer_step,
                "sequences_consumed": optimizer_step
                * resolved["training"]["gradient_accumulation_steps"],
            },
            temporary / "training_state.pt",
        )
        records = _recursive_file_records(temporary)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA,
            "checkpoint_id": checkpoint_id,
            "run_id": resolved["run_id"],
            "optimizer_step": optimizer_step,
            "sequences_consumed": optimizer_step
            * resolved["training"]["gradient_accumulation_steps"],
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
    return final


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
    expected_consumed = step * resolved["training"]["gradient_accumulation_steps"]
    if manifest.get("sequences_consumed") != expected_consumed:
        raise RuntimeError("checkpoint data cursor changed")
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
) -> None:
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cuda",
        weights_only=False,
    )
    if state.get("optimizer_step") != expected_step:
        raise RuntimeError("checkpoint state step changed")
    if state.get("sequences_consumed") != expected_sequences:
        raise RuntimeError("checkpoint state data cursor changed")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


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
    preflight = subparsers.add_parser("preflight", help="validate the P100 stack")
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
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except TrainingInterrupted as exc:
        print(json.dumps({"status": "interrupted", "message": str(exc)}))
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
