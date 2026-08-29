from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .classification_eval_common import (
    EMBEDDING_MANIFEST_SCHEMA,
    EMBEDDING_VALIDATION_SCHEMA,
    INPUT_VARIANTS,
    MODEL_NAMES,
    POOLINGS,
    classification_dataset_path,
    classification_model_path,
    digest_strings,
    read_json,
)
from .config import MODEL_ID, MODEL_REVISION, canonical_json_bytes, sha256_bytes
from .manifest import file_sha256, write_json_atomic
from .packing import tokenizer_fingerprint


LOGGER = logging.getLogger(__name__)


def content_mask(attention_mask: Any, special_tokens_mask: Any) -> Any:
    import torch

    if attention_mask.shape != special_tokens_mask.shape:
        raise RuntimeError("tokenizer masks have different shapes")
    mask = attention_mask.to(dtype=torch.bool) & ~special_tokens_mask.to(
        dtype=torch.bool
    )
    if not bool(mask.any(dim=1).all().item()):
        raise RuntimeError("classification input has no content tokens")
    return mask


def pool_last_hidden_state(hidden: Any, mask: Any) -> tuple[Any, Any]:
    import torch

    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise RuntimeError("classification hidden-state shape is invalid")
    values = hidden.float()
    expanded = mask.unsqueeze(-1)
    counts = expanded.sum(dim=1)
    masked_mean = (values * expanded).sum(dim=1) / counts
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values
    last_content = values[
        torch.arange(values.shape[0], device=values.device), last_positions
    ]
    if not torch.isfinite(masked_mean).all() or not torch.isfinite(last_content).all():
        raise RuntimeError("classification embeddings contain non-finite values")
    return masked_mean, last_content


def initialize_gloo() -> tuple[int, int, int]:
    import torch

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    if torch.distributed.is_initialized():
        return (
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
            int(os.environ.get("LOCAL_RANK", "0")),
        )
    return 0, 1, 0


def close_gloo() -> None:
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def broadcast_payload(value: Any, rank: int) -> Any:
    import torch

    if not torch.distributed.is_initialized():
        return value
    values = [value if rank == 0 else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def load_private_texts(
    parquet_path: Path,
    requested_ids: Sequence[str],
) -> Dict[str, tuple[str, str]]:
    import pyarrow.parquet as pq

    needed = set(requested_ids)
    found: Dict[str, tuple[str, str]] = {}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=8_192, columns=["sample_id", "title", "first_post"]
    ):
        columns = batch.to_pydict()
        for sample_id, title, first_post in zip(
            columns["sample_id"], columns["title"], columns["first_post"]
        ):
            if sample_id in needed:
                if sample_id in found:
                    raise RuntimeError("classification dataset contains duplicate IDs")
                found[sample_id] = (title, first_post)
    if set(found) != needed:
        raise RuntimeError("classification embedding IDs are missing from the dataset")
    return found


def load_embedding_model(
    config: Mapping[str, Any],
    model_name: str,
    model_manifests: Mapping[str, Mapping[str, Any]],
    device: Any,
) -> tuple[Any, Any, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name not in MODEL_NAMES:
        raise ValueError("unknown classification model")
    model_path = classification_model_path(config, model_name)
    source: str | Path = model_path if model_path is not None else MODEL_ID
    arguments: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if model_name == "base":
        arguments["revision"] = MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(source, **arguments)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise RuntimeError("classification tokenizer has no padding token")
    expected_fingerprint = model_manifests["general"]["tokenizer"][
        "prepared_fingerprint_sha256"
    ]
    actual_fingerprint = tokenizer_fingerprint(tokenizer)
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("classification tokenizer fingerprint changed")

    model_arguments = dict(arguments)
    model_arguments.update(dtype=torch.bfloat16, attn_implementation="eager")
    causal_model = AutoModelForCausalLM.from_pretrained(source, **model_arguments)
    if causal_model.config.model_type != "llama":
        raise RuntimeError("classification model architecture changed")
    if int(causal_model.config.max_position_embeddings) != 4096:
        raise RuntimeError("classification model context changed")
    parameter_count = sum(parameter.numel() for parameter in causal_model.parameters())
    if parameter_count != 670_127_616:
        raise RuntimeError("classification model parameter count changed")
    causal_model.eval()
    causal_model.requires_grad_(False)
    causal_model.to(device)
    backbone = causal_model.model
    metadata = {
        "model_name": model_name,
        "model_id": MODEL_ID if model_name == "base" else None,
        "revision": MODEL_REVISION if model_name == "base" else None,
        "artifact_id": (
            None if model_name == "base" else model_manifests[model_name]["artifact_id"]
        ),
        "artifact_sha256": (
            None
            if model_name == "base"
            else model_manifests[model_name]["artifact_sha256"]
        ),
        "tokenizer_fingerprint_sha256": actual_fingerprint,
        "hidden_size": int(causal_model.config.hidden_size),
        "parameter_count": parameter_count,
    }
    return tokenizer, backbone, metadata


def embed_text_batch(
    tokenizer: Any,
    backbone: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    special_tokens_mask = encoded.pop("special_tokens_mask")
    attention_mask = encoded["attention_mask"]
    mask = content_mask(attention_mask, special_tokens_mask).to(device)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        hidden = backbone(**inputs, return_dict=True).last_hidden_state
    return pool_last_hidden_state(hidden, mask)


def run_model_probe(
    config: Mapping[str, Any],
    model_name: str,
    model_manifests: Mapping[str, Mapping[str, Any]],
    texts: Sequence[str],
    local_rank: int,
) -> Dict[str, Any]:
    import torch

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer, backbone, metadata = load_embedding_model(
        config, model_name, model_manifests, device
    )
    for variant in INPUT_VARIANTS:
        values = [
            title if variant == "title" else f"{title}\n\n{post}"
            for title, post in texts
        ]
        masked_mean, last_content = embed_text_batch(
            tokenizer,
            backbone,
            values,
            max_length=config["embedding"]["max_length"],
            device=device,
        )
        if masked_mean.shape != last_content.shape:
            raise RuntimeError("classification pooling dimensions differ")
    torch.cuda.synchronize(device)
    metadata["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    del backbone, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def generate_embeddings(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    runtime: Mapping[str, Any],
    model_name: str,
    rank: int,
    world_size: int,
    local_rank: int,
) -> Dict[str, Any] | None:
    import numpy as np
    import torch

    if world_size != config["embedding"]["world_size"]:
        raise RuntimeError("classification embedding world size changed")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    ordered_ids = runtime["sample_ids"]
    rank_ids = ordered_ids[rank::world_size]
    texts_by_id = load_private_texts(
        runtime["dataset_path"] / "examples.parquet", rank_ids
    )
    tokenizer, backbone, model_metadata = load_embedding_model(
        config, model_name, runtime["model_manifests"], device
    )
    model_root = runtime["evaluation_dir"] / "embeddings" / model_name
    model_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    model_root.chmod(0o700)
    chunk_size = config["embedding"]["chunk_size_per_rank"]
    batch_size = config["embedding"]["batch_size_per_rank"]

    for variant in INPUT_VARIANTS:
        chunk_root = model_root / variant / "chunks"
        chunk_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        chunk_root.chmod(0o700)
        for start in range(0, len(rank_ids), chunk_size):
            chunk_ids = rank_ids[start : start + chunk_size]
            stem = f"rank-{rank:02d}-start-{start:08d}"
            metadata_path = chunk_root / f"{stem}.json"
            if _valid_existing_chunk(metadata_path, chunk_root, chunk_ids):
                continue
            pooled: Dict[str, list[Any]] = {name: [] for name in POOLINGS}
            for offset in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[offset : offset + batch_size]
                private_texts = [texts_by_id[value] for value in batch_ids]
                batch_texts = [
                    title if variant == "title" else f"{title}\n\n{post}"
                    for title, post in private_texts
                ]
                masked_mean, last_content = embed_text_batch(
                    tokenizer,
                    backbone,
                    batch_texts,
                    max_length=config["embedding"]["max_length"],
                    device=device,
                )
                pooled["masked_mean"].append(masked_mean.cpu().numpy())
                pooled["last_content"].append(last_content.cpu().numpy())
            arrays = {
                name: np.concatenate(values, axis=0).astype(np.float32, copy=False)
                for name, values in pooled.items()
            }
            id_array = np.asarray(chunk_ids, dtype="S64")
            files: Dict[str, Dict[str, Any]] = {}
            for name, array in {"ids": id_array, **arrays}.items():
                filename = f"{stem}-{name}.npy"
                target = chunk_root / filename
                _write_numpy_atomic(target, array)
                files[name] = {
                    "path": filename,
                    "size_bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            chunk_metadata = {
                "rank": rank,
                "rank_start": start,
                "count": len(chunk_ids),
                "ids_sha256": digest_strings(chunk_ids),
                "dimension": int(arrays["masked_mean"].shape[1]),
                "files": files,
            }
            write_json_atomic(metadata_path, chunk_metadata)
            metadata_path.chmod(0o600)
            LOGGER.info(
                "stage=embed status=chunk model=%s variant=%s rank=%d count=%d",
                model_name,
                variant,
                rank,
                len(chunk_ids),
            )

    del backbone, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        return None
    return finalize_embedding_manifest(
        config, resolved, runtime["evaluation_dir"], model_name, model_metadata
    )


def finalize_embedding_manifest(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
    model_name: str,
    model_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    expected_count = resolved["sample_count"]
    for variant in INPUT_VARIANTS:
        chunk_root = evaluation_dir / "embeddings" / model_name / variant / "chunks"
        chunks = []
        all_ids: list[str] = []
        dimensions: set[int] = set()
        for metadata_path in sorted(chunk_root.glob("rank-*.json")):
            metadata = read_json(metadata_path, "classification embedding chunk")
            _validate_chunk_files(metadata, chunk_root)
            ids = _load_ids(chunk_root / metadata["files"]["ids"]["path"])
            if metadata["ids_sha256"] != digest_strings(ids):
                raise RuntimeError("classification embedding chunk ID digest changed")
            all_ids.extend(ids)
            dimensions.add(metadata["dimension"])
            chunks.append(
                {
                    "path": metadata_path.name,
                    "sha256": file_sha256(metadata_path),
                    "count": metadata["count"],
                    "rank": metadata["rank"],
                    "rank_start": metadata["rank_start"],
                }
            )
        if len(all_ids) != expected_count or len(set(all_ids)) != expected_count:
            raise RuntimeError("classification embedding coverage is incomplete")
        if digest_strings(sorted(all_ids)) != resolved["sample_ids_sha256"]:
            raise RuntimeError("classification embedding coverage digest changed")
        if len(dimensions) != 1:
            raise RuntimeError("classification embedding dimensions differ")
        variants[variant] = {
            "count": len(all_ids),
            "dimension": dimensions.pop(),
            "ids_sha256": resolved["sample_ids_sha256"],
            "chunks": chunks,
        }
    manifest = {
        "schema_version": EMBEDDING_MANIFEST_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        "model": dict(model_metadata),
        "runtime": {
            "world_size": config["embedding"]["world_size"],
            "batch_size_per_rank": config["embedding"]["batch_size_per_rank"],
            "runtime_dtype": config["embedding"]["runtime_dtype"],
            "output_dtype": config["embedding"]["output_dtype"],
            "max_length": config["embedding"]["max_length"],
        },
        "variants": variants,
        "status": "complete",
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    target = evaluation_dir / "embeddings" / model_name / "embedding_manifest.json"
    write_json_atomic(target, manifest)
    target.chmod(0o600)
    return manifest


def verify_embedding_manifests(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
) -> Dict[str, Any]:
    dimensions: set[int] = set()
    digests: set[str] = set()
    manifests: Dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        root = evaluation_dir / "embeddings" / model_name
        manifest = read_json(root / "embedding_manifest.json", "embedding manifest")
        manifest_digest = manifest.pop("manifest_sha256", None)
        if manifest_digest != sha256_bytes(canonical_json_bytes(manifest)):
            raise RuntimeError("classification embedding manifest hash changed")
        manifest["manifest_sha256"] = manifest_digest
        if (
            manifest.get("schema_version") != EMBEDDING_MANIFEST_SCHEMA
            or manifest.get("evaluation_id") != resolved["evaluation_id"]
            or manifest.get("status") != "complete"
            or manifest.get("model", {}).get("model_name") != model_name
        ):
            raise RuntimeError("classification embedding manifest is invalid")
        expected_model = resolved["models"][model_name]
        if model_name == "base":
            if (
                manifest["model"].get("model_id") != expected_model["model_id"]
                or manifest["model"].get("revision") != expected_model["revision"]
                or manifest["model"].get("artifact_id") is not None
            ):
                raise RuntimeError("classification baseline embedding identity changed")
        elif (
            manifest["model"].get("artifact_id") != expected_model["artifact_id"]
            or manifest["model"].get("artifact_sha256")
            != expected_model["artifact_sha256"]
        ):
            raise RuntimeError("classification artifact embedding identity changed")
        for variant in INPUT_VARIANTS:
            value = manifest["variants"][variant]
            if value["count"] != resolved["sample_count"]:
                raise RuntimeError("classification embedding count changed")
            dimensions.add(value["dimension"])
            digests.add(value["ids_sha256"])
            chunk_root = root / variant / "chunks"
            listed_metadata = {chunk["path"] for chunk in value["chunks"]}
            actual_metadata = {path.name for path in chunk_root.glob("rank-*.json")}
            if listed_metadata != actual_metadata:
                raise RuntimeError("classification embedding chunk set changed")
            all_ids: list[str] = []
            for chunk in value["chunks"]:
                metadata_path = chunk_root / chunk["path"]
                if file_sha256(metadata_path) != chunk["sha256"]:
                    raise RuntimeError("classification embedding chunk metadata changed")
                metadata = read_json(metadata_path, "embedding chunk")
                _validate_chunk_files(metadata, chunk_root)
                all_ids.extend(
                    _load_ids(chunk_root / metadata["files"]["ids"]["path"])
                )
            if (
                len(all_ids) != resolved["sample_count"]
                or len(set(all_ids)) != resolved["sample_count"]
                or digest_strings(sorted(all_ids)) != resolved["sample_ids_sha256"]
            ):
                raise RuntimeError("classification embedding IDs changed")
        manifests[model_name] = manifest
    if len(dimensions) != 1 or digests != {resolved["sample_ids_sha256"]}:
        raise RuntimeError("classification embeddings are not comparable")
    result = {
        "schema_version": EMBEDDING_VALIDATION_SCHEMA,
        "evaluation_id": resolved["evaluation_id"],
        "models": len(manifests),
        "variants": len(INPUT_VARIANTS),
        "dimension": dimensions.pop(),
        "embedding_manifests": {
            model_name: file_sha256(
                evaluation_dir
                / "embeddings"
                / model_name
                / "embedding_manifest.json"
            )
            for model_name in MODEL_NAMES
        },
        "status": "valid",
    }
    return result


def validate_embedding_manifests(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    evaluation_dir: Path,
) -> Dict[str, Any]:
    result = verify_embedding_manifests(config, resolved, evaluation_dir)
    target = evaluation_dir / "embeddings_validation.json"
    write_json_atomic(target, result)
    target.chmod(0o600)
    return result


def require_validated_embeddings(
    resolved: Mapping[str, Any], evaluation_dir: Path
) -> Dict[str, Any]:
    value = read_json(
        evaluation_dir / "embeddings_validation.json",
        "embedding validation",
    )
    if (
        value.get("schema_version") != EMBEDDING_VALIDATION_SCHEMA
        or value.get("evaluation_id") != resolved["evaluation_id"]
        or value.get("models") != len(MODEL_NAMES)
        or value.get("variants") != len(INPUT_VARIANTS)
        or value.get("status") != "valid"
    ):
        raise RuntimeError("classification embedding validation is invalid")
    expected = {
        model_name: file_sha256(
            evaluation_dir
            / "embeddings"
            / model_name
            / "embedding_manifest.json"
        )
        for model_name in MODEL_NAMES
    }
    if value.get("embedding_manifests") != expected:
        raise RuntimeError("classification embedding manifests changed after validation")
    return value


def load_embedding_rows(
    evaluation_dir: Path,
    model_name: str,
    variant: str,
    pooling: str,
    requested_ids: Sequence[str],
) -> Any:
    import numpy as np

    if model_name not in MODEL_NAMES or variant not in INPUT_VARIANTS or pooling not in POOLINGS:
        raise ValueError("unknown embedding selection")
    requested = set(requested_ids)
    found: Dict[str, Any] = {}
    root = evaluation_dir / "embeddings" / model_name / variant / "chunks"
    for metadata_path in sorted(root.glob("rank-*.json")):
        metadata = read_json(metadata_path, "embedding chunk")
        ids = _load_ids(root / metadata["files"]["ids"]["path"])
        selected_positions = [index for index, value in enumerate(ids) if value in requested]
        if not selected_positions:
            continue
        array = np.load(root / metadata["files"][pooling]["path"], allow_pickle=False)
        for position in selected_positions:
            sample_id = ids[position]
            if sample_id in found:
                raise RuntimeError("classification embedding ID is duplicated")
            found[sample_id] = array[position]
    if set(found) != requested:
        raise RuntimeError("classification probe embedding coverage is incomplete")
    result = np.stack([found[value] for value in requested_ids]).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("classification probe embeddings are non-finite")
    return result


def _valid_existing_chunk(
    metadata_path: Path, root: Path, expected_ids: Sequence[str]
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path, "embedding chunk")
        _validate_chunk_files(metadata, root)
        return (
            metadata["count"] == len(expected_ids)
            and metadata["ids_sha256"] == digest_strings(expected_ids)
        )
    except (KeyError, RuntimeError, ValueError):
        return False


def _validate_chunk_files(metadata: Mapping[str, Any], root: Path) -> None:
    import numpy as np

    if set(metadata.get("files", {})) != {"ids", *POOLINGS}:
        raise RuntimeError("classification embedding chunk files are incomplete")
    for name, record in metadata["files"].items():
        path = root / record["path"]
        if path.parent != root or path.is_symlink() or not path.is_file():
            raise RuntimeError("classification embedding chunk path is invalid")
        if path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
            raise RuntimeError("classification embedding chunk file changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape[0] != metadata["count"]:
            raise RuntimeError("classification embedding chunk row count changed")
        if name != "ids" and (
            array.ndim != 2
            or array.shape[1] != metadata["dimension"]
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise RuntimeError("classification embedding chunk array is invalid")


def _load_ids(path: Path) -> list[str]:
    import numpy as np

    array = np.load(path, allow_pickle=False)
    if array.ndim != 1 or array.dtype.kind != "S":
        raise RuntimeError("classification embedding ID array is invalid")
    return [bytes(value).decode("ascii") for value in array]


def _write_numpy_atomic(path: Path, array: Any) -> None:
    import numpy as np

    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        partial.chmod(0o600)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
