from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

from .config import (
    DATASET_IDS,
    MODEL_ID,
    MODEL_REVISION,
    canonical_json_bytes,
    sha256_bytes,
)
from .manifest import file_sha256, write_json_atomic
from .packing import tokenizer_fingerprint
from .paired_plan import PAIRED_REAL_POLICY, validate_paired_mixture


MODEL_ARTIFACT_SCHEMA = "tucano2-model-artifact/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[0-9a-f]{20}\Z")


def export_model_artifact(
    *,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    training: Dict[str, Any],
    prepared_tokenizer: Dict[str, Any],
    artifacts_root: Path,
) -> Path:
    artifact_id = _expected_artifact_id(training)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    final = artifacts_root / artifact_id
    if final.exists():
        raise RuntimeError(f"model artifact already exists: {artifact_id}")
    temporary = artifacts_root / f".{artifact_id}.partial-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        model.config.use_cache = True
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(temporary)
        if any(path.is_symlink() for path in temporary.rglob("*")):
            raise RuntimeError("exported model must not contain symlinks")

        file_records = _file_records(temporary)
        tokenizer_contract = _tokenizer_contract(
            tokenizer, prepared_tokenizer, file_records
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        manifest = {
            "schema_version": MODEL_ARTIFACT_SCHEMA,
            "artifact_id": artifact_id,
            "format": "transformers_pretrained",
            "parent_model": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "license": "Apache-2.0",
            },
            "architecture": {
                "model_type": model.config.model_type,
                "parameter_count": parameter_count,
                "native_context_length": int(model.config.max_position_embeddings),
                "training_sequence_length": config["training"]["sequence_length"],
                "weights_dtype": str(next(model.parameters()).dtype).removeprefix(
                    "torch."
                ),
            },
            "tokenizer": tokenizer_contract,
            "training": training,
            "environment": _environment_versions(config),
            "files": file_records,
            "artifact_sha256": _aggregate_files_sha256(file_records),
            "redistribution_status": "internal_research_only",
        }
        _assert_no_absolute_path_strings(manifest)
        write_json_atomic(temporary / "model_artifact_manifest.json", manifest)
        temporary.replace(final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_model_artifact(final, load_model=True)
    return final


def validate_model_artifact(
    path: str | Path, *, load_model: bool = True
) -> Dict[str, Any]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise RuntimeError("model artifact path must not be a symlink")
    root = requested.resolve()
    if not root.is_dir():
        raise RuntimeError("model artifact must be a real directory")
    manifest_path = root / "model_artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("model artifact manifest is missing") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("model artifact manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("model artifact manifest must be an object")
    _assert_no_absolute_path_strings(manifest)
    if manifest.get("schema_version") != MODEL_ARTIFACT_SCHEMA:
        raise RuntimeError("unknown model artifact schema")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not _ID_RE.fullmatch(artifact_id):
        raise RuntimeError("invalid model artifact ID")
    if root.name != artifact_id:
        raise RuntimeError("model artifact directory does not match artifact ID")
    if manifest.get("format") != "transformers_pretrained":
        raise RuntimeError("model artifact format must be transformers_pretrained")
    if manifest.get("parent_model") != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
    }:
        raise RuntimeError("model artifact parent changed")
    if manifest.get("redistribution_status") != "internal_research_only":
        raise RuntimeError("model artifact redistribution policy must remain internal")
    training = manifest.get("training")
    _validate_training_provenance(training)
    _validate_environment(manifest.get("environment"), training)
    if artifact_id != _expected_artifact_id(training):
        raise RuntimeError("model artifact ID does not match training provenance")

    architecture = manifest.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("model_type") != "llama":
        raise RuntimeError("model artifact architecture changed")
    if architecture.get("parameter_count") != 670127616:
        raise RuntimeError("model artifact parameter count changed")
    if architecture.get("native_context_length") != 4096:
        raise RuntimeError("model artifact native context changed")
    if architecture.get("training_sequence_length") != 1024:
        raise RuntimeError("model artifact training context changed")
    if architecture.get("weights_dtype") != "float32":
        raise RuntimeError("model artifact must preserve the FP32 trained weights")

    tokenizer = manifest.get("tokenizer")
    _validate_tokenizer_contract(tokenizer)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("model artifact file list is missing")
    listed = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("model artifact file record must be an object")
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise RuntimeError("model artifact contains an unsafe file path")
        if relative.name == "model_artifact_manifest.json":
            raise RuntimeError("model artifact manifest must not list itself")
        if relative.as_posix() in listed:
            raise RuntimeError("model artifact lists a file more than once")
        listed.add(relative.as_posix())
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError("model artifact file is missing or is a symlink")
        if candidate.stat().st_size != record.get("size_bytes"):
            raise RuntimeError("model artifact file size mismatch")
        if file_sha256(candidate) != record.get("sha256"):
            raise RuntimeError("model artifact file hash mismatch")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "model_artifact_manifest.json"
    }
    if any(item.is_symlink() for item in root.rglob("*")):
        raise RuntimeError("model artifact contains a symlink")
    if listed != actual:
        raise RuntimeError("model artifact contains missing or unlisted files")
    if any(name.endswith((".bin", ".pt", ".pth", ".ckpt")) for name in listed):
        raise RuntimeError("model artifact contains an unsafe weight format")
    if any(name.startswith("adapter_") for name in listed):
        raise RuntimeError("model artifact must contain full weights, not adapters")
    if not _has_safetensors_weights(listed):
        raise RuntimeError("model artifact does not contain safetensors weights")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= listed:
        raise RuntimeError("model artifact is missing model or tokenizer configuration")
    if manifest.get("artifact_sha256") != _aggregate_files_sha256(records):
        raise RuntimeError("model artifact aggregate hash mismatch")
    if tokenizer.get("fingerprint_sha256") != _tokenizer_files_fingerprint(
        records, tokenizer
    ):
        raise RuntimeError("model artifact tokenizer file fingerprint mismatch")

    if load_model:
        _validate_offline_loading(root, manifest)
    return manifest


def _validate_offline_loading(root: Path, manifest: Dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded_tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if tokenizer_fingerprint(loaded_tokenizer) != manifest["tokenizer"][
        "prepared_fingerprint_sha256"
    ]:
        raise RuntimeError("offline tokenizer differs from prepared tokenizer")
    loaded_model = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    )
    if loaded_model.config.model_type != "llama":
        raise RuntimeError("offline model architecture changed")
    if int(loaded_model.config.max_position_embeddings) != 4096:
        raise RuntimeError("offline model context changed")
    if sum(parameter.numel() for parameter in loaded_model.parameters()) != 670127616:
        raise RuntimeError("offline model parameter count changed")


def _file_records(root: Path) -> list[Dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        if path.name == "model_artifact_manifest.json":
            continue
        records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _aggregate_files_sha256(records: list[Dict[str, Any]]) -> str:
    values = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in sorted(records, key=lambda item: item["path"])
    ]
    return sha256_bytes(canonical_json_bytes(values))


def _tokenizer_contract(
    tokenizer: Any,
    prepared_tokenizer: Dict[str, Any],
    records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    contract = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prepared_fingerprint_sha256": prepared_tokenizer["fingerprint_sha256"],
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    contract["fingerprint_sha256"] = _tokenizer_files_fingerprint(records, contract)
    return contract


def _tokenizer_files_fingerprint(
    records: list[Dict[str, Any]], tokenizer: Dict[str, Any]
) -> str:
    model_files = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    }
    tokenizer_records = [
        record
        for record in records
        if record["path"] not in model_files
        and not record["path"].endswith(".safetensors")
    ]
    value = {
        "files": sorted(tokenizer_records, key=lambda item: item["path"]),
        "special_token_ids": {
            key: tokenizer[key]
            for key in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
        "vocab_size": tokenizer["vocab_size"],
    }
    return sha256_bytes(canonical_json_bytes(value))


def _validate_tokenizer_contract(tokenizer: Any) -> None:
    if not isinstance(tokenizer, dict):
        raise RuntimeError("model artifact tokenizer metadata is missing")
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "vocab_size": 49152,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 49109,
        "unk_token_id": 0,
    }
    if any(tokenizer.get(key) != value for key, value in expected.items()):
        raise RuntimeError("model artifact tokenizer contract changed")
    for key in ("prepared_fingerprint_sha256", "fingerprint_sha256"):
        if not isinstance(tokenizer.get(key), str) or not _SHA256_RE.fullmatch(
            tokenizer[key]
        ):
            raise RuntimeError("model artifact tokenizer fingerprint is invalid")


def _validate_training_provenance(training: Any) -> None:
    if not isinstance(training, dict):
        raise RuntimeError("model artifact training provenance is missing")
    if training.get("method") != "full_parameter_continual_pretraining":
        raise RuntimeError("model artifact training method changed")
    optimizer_steps = training.get("optimizer_steps")
    if (
        training.get("seed") != 42
        or not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps < 1
    ):
        raise RuntimeError("model artifact training budget changed")
    profile = training.get("profile")
    if profile is None:
        if optimizer_steps != 192:
            raise RuntimeError("legacy model artifact training budget changed")
    elif profile == "real":
        if optimizer_steps != 52_000:
            raise RuntimeError("real model artifact training budget changed")
        mixture = training.get("data_mixture")
        if isinstance(mixture, dict) and mixture.get("policy") == PAIRED_REAL_POLICY:
            try:
                validate_paired_mixture(mixture)
            except RuntimeError as exc:
                raise RuntimeError(
                    "paired model artifact mixture policy changed"
                ) from exc
            experiment = training.get("experiment")
            if (
                not isinstance(experiment, dict)
                or set(experiment)
                != {
                    "experiment_id",
                    "arm",
                    "allocation_sha256",
                    "schedule_template_sha256",
                    "paired_inputs_sha256",
                }
                or experiment.get("experiment_id") != mixture["experiment_id"]
                or experiment.get("arm") != mixture["arm"]
                or experiment.get("allocation_sha256")
                != mixture["allocation_sha256"]
                or experiment.get("schedule_template_sha256")
                != mixture["schedule_template_sha256"]
                or not isinstance(experiment.get("paired_inputs_sha256"), str)
                or not _SHA256_RE.fullmatch(experiment["paired_inputs_sha256"])
            ):
                raise RuntimeError("paired model artifact experiment metadata changed")
        elif (
            not isinstance(mixture, dict)
            or set(mixture)
            != {"policy", "without_replacement", "allocation_sha256"}
            or mixture.get("policy") != "equal_share_without_replacement"
            or mixture.get("without_replacement") is not True
            or not isinstance(mixture.get("allocation_sha256"), str)
            or not _SHA256_RE.fullmatch(mixture["allocation_sha256"])
        ):
            raise RuntimeError("real model artifact mixture policy changed")
    else:
        raise RuntimeError("model artifact training profile is invalid")
    if not isinstance(training.get("run_id"), str) or not _ID_RE.fullmatch(
        training["run_id"]
    ):
        raise RuntimeError("model artifact run ID is invalid")
    if not isinstance(training.get("git_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", training["git_commit"]
    ):
        raise RuntimeError("model artifact Git commit is invalid")
    for key in ("config_sha256", "inputs_sha256"):
        if not isinstance(training.get(key), str) or not _SHA256_RE.fullmatch(
            training[key]
        ):
            raise RuntimeError("model artifact training digest is invalid")
    execution = training.get("execution")
    valid_executions = (
        {
            "strategy": "single_process",
            "backend": "none",
            "world_size": 1,
            "precision": "fp16",
            "optimizer": "adamw8bit",
            "optimizer_implementation": "bitsandbytes",
            "micro_batch_size_per_rank": 1,
            "gradient_accumulation_steps_per_rank": 8,
            "global_batch_sequences": 8,
            "global_batch_tokens": 8192,
        },
        {
            "strategy": "ddp",
            "backend": "nccl",
            "world_size": 2,
            "precision": "bf16",
            "optimizer": "adamw",
            "optimizer_implementation": "torch_fused",
            "micro_batch_size_per_rank": 1,
            "gradient_accumulation_steps_per_rank": 4,
            "global_batch_sequences": 8,
            "global_batch_tokens": 8192,
        },
    )
    if execution not in valid_executions:
        raise RuntimeError("model artifact execution strategy is invalid")
    datasets = training.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != len(DATASET_IDS) or {
        item.get("dataset_id") for item in datasets if isinstance(item, dict)
    } != set(DATASET_IDS):
        raise RuntimeError("model artifact dataset provenance is incomplete")
    for item in datasets:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("preparation_id"), str)
            or not _ID_RE.fullmatch(item["preparation_id"])
            or not isinstance(item.get("dataset_manifest_sha256"), str)
            or not _SHA256_RE.fullmatch(item["dataset_manifest_sha256"])
        ):
            raise RuntimeError("model artifact dataset provenance is invalid")
    if profile == "real":
        paired = training["data_mixture"].get("policy") == PAIRED_REAL_POLICY
        if any(
            not isinstance(item.get("train_sequences"), int)
            or isinstance(item.get("train_sequences"), bool)
            or item["train_sequences"] < (0 if paired else 1)
            or item.get("eval_sequences") != 256
            or (
                paired
                and (
                    not isinstance(item.get("prepared_train_sequences"), int)
                    or isinstance(item.get("prepared_train_sequences"), bool)
                    or item["prepared_train_sequences"] < 1
                    or item["train_sequences"]
                    > item["prepared_train_sequences"]
                )
            )
            for item in datasets
        ):
            raise RuntimeError("real model artifact dataset allocation is invalid")
        allocated = sum(item["train_sequences"] for item in datasets)
        if allocated != optimizer_steps * execution["global_batch_sequences"]:
            raise RuntimeError("real model artifact allocation total changed")
        if paired:
            prepared_expected = {dataset_id: 0 for dataset_id in DATASET_IDS}
            used_expected = {dataset_id: 0 for dataset_id in DATASET_IDS}
            mixture = training["data_mixture"]
            for pool in mixture["pools"]:
                dataset_id = pool["dataset_id"]
                count = pool["train_sequences"]
                prepared_expected[dataset_id] += count
                if mixture["arm"] == "forum_tech":
                    if pool["role"] != "replacement":
                        used_expected[dataset_id] += count
                elif pool["role"] != "domain":
                    used_expected[dataset_id] += count
            actual_prepared = {
                item["dataset_id"]: item["prepared_train_sequences"]
                for item in datasets
            }
            actual_used = {
                item["dataset_id"]: item["train_sequences"] for item in datasets
            }
            if actual_prepared != prepared_expected or actual_used != used_expected:
                raise RuntimeError("paired model artifact dataset pools changed")


def _validate_environment(environment: Any, training: Dict[str, Any]) -> None:
    if not isinstance(environment, dict):
        raise RuntimeError("model artifact environment is missing")
    required = {
        "python",
        "torch",
        "torch_cuda",
        "transformers",
        "tokenizers",
    }
    if training["execution"]["optimizer_implementation"] == "bitsandbytes":
        required.add("bitsandbytes")
    if set(environment) != required or any(
        not isinstance(environment[key], str) or not environment[key]
        for key in required
    ):
        raise RuntimeError("model artifact environment is incomplete")
    if not environment["python"].startswith("3.12."):
        raise RuntimeError("model artifact Python version changed")
    if environment["torch"] != "2.7.1+cu118":
        raise RuntimeError("model artifact PyTorch version changed")
    if environment["torch_cuda"] != "11.8":
        raise RuntimeError("model artifact CUDA runtime changed")
    if environment["transformers"] != "5.14.1":
        raise RuntimeError("model artifact Transformers version changed")
    if "bitsandbytes" in required and environment["bitsandbytes"] != "0.50.0":
        raise RuntimeError("model artifact bitsandbytes version changed")


def _expected_artifact_id(training: Dict[str, Any]) -> str:
    value = {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "parent_model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "run_id": training.get("run_id"),
        "config_sha256": training.get("config_sha256"),
        "inputs_sha256": training.get("inputs_sha256"),
        "optimizer_steps": training.get("optimizer_steps"),
        "execution": training.get("execution"),
    }
    return sha256_bytes(canonical_json_bytes(value))[:20]


def _environment_versions(config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import transformers

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
    }
    if config["training"]["optimizer_implementation"] == "bitsandbytes":
        versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    return versions


def _has_safetensors_weights(files: set[str]) -> bool:
    if "model.safetensors" in files:
        return True
    return "model.safetensors.index.json" in files and any(
        name.startswith("model-") and name.endswith(".safetensors") for name in files
    )


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
        raise RuntimeError("model artifact metadata contains an absolute path")
