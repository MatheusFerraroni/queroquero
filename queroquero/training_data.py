from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pyarrow.parquet as pq

from .config import canonical_json_bytes, load_resolved_config, resolve_output_root, sha256_bytes
from .manifest import file_sha256
from .prepare import validate_preparation


RESOLVED_INPUTS_SCHEMA = "queroquero-resolved-training-inputs/v1"


@dataclass(frozen=True)
class ResolvedDataset:
    dataset_id: str
    root: Path
    manifest: Dict[str, Any]
    manifest_sha256: str
    relative_manifest_path: str


@dataclass(frozen=True)
class TrainingSequence:
    dataset_id: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedTrainingInputs:
    profile: str
    output_root: Path
    datasets: tuple[ResolvedDataset, ...]
    tokenizer: Dict[str, Any]

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": RESOLVED_INPUTS_SCHEMA,
            "profile": self.profile,
            "datasets": [
                {
                    "dataset_id": dataset.dataset_id,
                    "preparation_id": dataset.manifest["preparation_id"],
                    "resolved_config_sha256": dataset.manifest[
                        "resolved_config_sha256"
                    ],
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "manifest_path": dataset.relative_manifest_path,
                    "counts": {
                        key: dataset.manifest["counts"][key]
                        for key in (
                            "train_sequences",
                            "eval_sequences",
                            "train_tokens",
                            "eval_tokens",
                        )
                    },
                }
                for dataset in self.datasets
            ],
            "tokenizer": {
                key: self.tokenizer[key]
                for key in (
                    "model_id",
                    "revision",
                    "fingerprint_sha256",
                    "vocab_size",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "unk_token_id",
                )
            },
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.metadata()))


def resolve_training_inputs(
    config: Dict[str, Any], output_root: Path | None = None
) -> ResolvedTrainingInputs:
    profile = config["profile"]
    requested_root = output_root or resolve_output_root("derived")
    if requested_root.is_symlink():
        raise RuntimeError("prepared dataset output root must not be a symlink")
    root = requested_root.resolve()
    if not root.is_dir():
        raise RuntimeError("prepared dataset output root must be a real directory")

    resolved_datasets = []
    common_tokenizer: Dict[str, Any] | None = None
    for entry in config["datasets"]:
        dataset_id = entry["dataset_id"]
        _, current_config_sha256 = load_resolved_config(dataset_id, profile)
        dataset_root = root / dataset_id
        candidates = []
        if dataset_root.is_dir() and not dataset_root.is_symlink():
            for manifest_path in sorted(dataset_root.glob("*/dataset_manifest.json")):
                if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
                    raise RuntimeError(
                        f"prepared manifest for {dataset_id} must not be a symlink"
                    )
                try:
                    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"prepared manifest JSON is unreadable for {dataset_id}"
                    ) from exc
                if (
                    candidate.get("dataset_id") == dataset_id
                    and candidate.get("profile") == profile
                    and candidate.get("resolved_config_sha256")
                    == current_config_sha256
                ):
                    candidates.append(manifest_path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one current {profile} manifest for {dataset_id}; "
                f"found {len(candidates)}"
            )
        manifest_path = candidates[0]
        manifest = validate_preparation(manifest_path.parent)
        expected_counts = {
            "train_sequences": entry["train_sequences"],
            "eval_sequences": entry["eval_sequences"],
            "train_tokens": entry["train_sequences"] * 1024,
            "eval_tokens": entry["eval_sequences"] * 1024,
        }
        if any(
            manifest["counts"].get(key) != value
            for key, value in expected_counts.items()
        ):
            raise RuntimeError(f"prepared counts changed for {dataset_id}")
        tokenizer = manifest["tokenizer"]
        if common_tokenizer is None:
            common_tokenizer = tokenizer
        elif _tokenizer_contract(tokenizer) != _tokenizer_contract(common_tokenizer):
            raise RuntimeError("prepared datasets do not share one tokenizer contract")
        resolved_datasets.append(
            ResolvedDataset(
                dataset_id=dataset_id,
                root=manifest_path.parent.resolve(),
                manifest=manifest,
                manifest_sha256=file_sha256(manifest_path),
                relative_manifest_path=manifest_path.relative_to(root).as_posix(),
            )
        )
    if common_tokenizer is None:
        raise RuntimeError("training has no prepared datasets")
    return ResolvedTrainingInputs(
        profile=profile,
        output_root=root,
        datasets=tuple(resolved_datasets),
        tokenizer=common_tokenizer,
    )


def load_split(dataset: ResolvedDataset, split: str) -> list[TrainingSequence]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    sequences = []
    for shard_record in dataset.manifest["splits"][split]:
        relative = Path(shard_record["path"])
        shard = dataset.root / relative
        table = pq.read_table(shard, columns=["input_ids"])
        for values in table.column("input_ids").to_pylist():
            input_ids = tuple(int(value) for value in values)
            if len(input_ids) != 1024:
                raise RuntimeError("training input is not exactly 1024 tokens")
            if any(
                value < 0 or value >= dataset.manifest["tokenizer"]["vocab_size"]
                for value in input_ids
            ):
                raise RuntimeError("training input contains a token outside the vocabulary")
            sequences.append(
                TrainingSequence(dataset_id=dataset.dataset_id, input_ids=input_ids)
            )
    expected = dataset.manifest["counts"][f"{split}_sequences"]
    if len(sequences) != expected:
        raise RuntimeError(f"loaded {split} rows do not match the manifest")
    return sequences


def load_training_sequences(
    inputs: ResolvedTrainingInputs, seed: int
) -> list[TrainingSequence]:
    by_dataset = {
        dataset.dataset_id: load_split(dataset, "train")
        for dataset in inputs.datasets
    }
    lengths = {len(values) for values in by_dataset.values()}
    if len(lengths) != 1:
        raise RuntimeError("equal-weight training requires equal dataset budgets")
    dataset_ids = sorted(by_dataset)
    for dataset_id in dataset_ids:
        rng = random.Random(_stable_seed(seed, dataset_id, "rows"))
        rng.shuffle(by_dataset[dataset_id])

    balanced = []
    rows_per_dataset = lengths.pop()
    for row_index in range(rows_per_dataset):
        round_ids = list(dataset_ids)
        random.Random(_stable_seed(seed, "round", row_index)).shuffle(round_ids)
        balanced.extend(by_dataset[dataset_id][row_index] for dataset_id in round_ids)
    return balanced


def load_evaluation_sequences(
    inputs: ResolvedTrainingInputs,
) -> Dict[str, list[TrainingSequence]]:
    return {
        dataset.dataset_id: load_split(dataset, "eval")
        for dataset in inputs.datasets
    }


def _stable_seed(*values: Any) -> int:
    digest = sha256_bytes(canonical_json_bytes(list(values)))
    return int(digest[:16], 16)


def _tokenizer_contract(tokenizer: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: tokenizer.get(key)
        for key in (
            "model_id",
            "revision",
            "fingerprint_sha256",
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
