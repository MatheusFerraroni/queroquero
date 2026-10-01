from __future__ import annotations

import json
import random
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, overload

import pyarrow.parquet as pq

from .config import canonical_json_bytes, load_resolved_config, resolve_output_root, sha256_bytes
from .manifest import file_sha256
from .paired_plan import (
    PAIRED_REAL_POLICY,
    iter_paired_mixture_slots,
    validate_paired_mixture,
)
from .prepare import validate_preparation


RESOLVED_INPUTS_SCHEMA = "queroquero-resolved-training-inputs/v1"
REAL_RESOLVED_INPUTS_SCHEMA = "queroquero-resolved-training-inputs/v2"
PAIRED_RESOLVED_INPUTS_SCHEMA = "queroquero-resolved-training-inputs/v3"


@dataclass(frozen=True)
class ResolvedDataset:
    dataset_id: str
    root: Path
    manifest: Dict[str, Any]
    manifest_sha256: str
    relative_manifest_path: str
    train_sequences_used: int | None = None


@dataclass(frozen=True)
class TrainingSequence:
    dataset_id: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSequenceReference:
    dataset_id: str
    row_index: int


@dataclass(frozen=True)
class ResolvedTrainingInputs:
    profile: str
    output_root: Path
    datasets: tuple[ResolvedDataset, ...]
    tokenizer: Dict[str, Any]
    data_mixture: Dict[str, Any] | None = None
    preparation_profile: str | None = None

    def metadata(self) -> Dict[str, Any]:
        value = {
            "schema_version": self._metadata_schema(),
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
        if self.profile == "real":
            value["data_mixture"] = self.data_mixture
            value["allocated_train_sequences"] = {
                dataset.dataset_id: (
                    dataset.train_sequences_used
                    if dataset.train_sequences_used is not None
                    else dataset.manifest["counts"]["train_sequences"]
                )
                for dataset in self.datasets
            }
            value["allocated_train_tokens"] = sum(
                value["allocated_train_sequences"].values()
            ) * 1024
            if self._is_paired():
                value["prepared_train_sequences"] = {
                    dataset.dataset_id: dataset.manifest["counts"][
                        "train_sequences"
                    ]
                    for dataset in self.datasets
                }
                value["prepared_train_tokens"] = sum(
                    dataset.manifest["counts"]["train_tokens"]
                    for dataset in self.datasets
                )
                value["preparation_profile"] = self.preparation_profile
                value["paired_inputs_sha256"] = self.paired_inputs_sha256()
        return value

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.metadata()))

    def paired_inputs_sha256(self) -> str:
        if not self._is_paired():
            raise RuntimeError("paired inputs digest requires a paired mixture")
        mixture = validate_paired_mixture(self.data_mixture)
        value = {
            "schema_version": "queroquero-paired-resolved-inputs/v1",
            "experiment_id": mixture["experiment_id"],
            "allocation_sha256": mixture["allocation_sha256"],
            "schedule_template_sha256": mixture["schedule_template_sha256"],
            "preparation_profile": self.preparation_profile,
            "datasets": [
                {
                    "dataset_id": dataset.dataset_id,
                    "preparation_id": dataset.manifest["preparation_id"],
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "prepared_train_sequences": dataset.manifest["counts"][
                        "train_sequences"
                    ],
                    "eval_sequences": dataset.manifest["counts"]["eval_sequences"],
                }
                for dataset in self.datasets
            ],
        }
        return sha256_bytes(canonical_json_bytes(value))

    def _is_paired(self) -> bool:
        return (
            isinstance(self.data_mixture, dict)
            and self.data_mixture.get("policy") == PAIRED_REAL_POLICY
        )

    def _metadata_schema(self) -> str:
        if self._is_paired():
            return PAIRED_RESOLVED_INPUTS_SCHEMA
        if self.profile == "real":
            return REAL_RESOLVED_INPUTS_SCHEMA
        return RESOLVED_INPUTS_SCHEMA


def resolve_training_inputs(
    config: Dict[str, Any], output_root: Path | None = None
) -> ResolvedTrainingInputs:
    profile = config["profile"]
    mixture = config.get("data_mixture")
    paired = (
        isinstance(mixture, dict)
        and mixture.get("policy") == PAIRED_REAL_POLICY
    )
    preparation_profile = (
        mixture["preparation_profile"] if paired else profile
    )
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
        _, current_config_sha256 = load_resolved_config(
            dataset_id, preparation_profile
        )
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
                    and candidate.get("profile") == preparation_profile
                    and candidate.get("resolved_config_sha256")
                    == current_config_sha256
                ):
                    candidates.append(manifest_path)
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one current {preparation_profile} manifest for "
                f"{dataset_id}; "
                f"found {len(candidates)}"
            )
        manifest_path = candidates[0]
        manifest = validate_preparation(manifest_path.parent)
        if profile == "real":
            prepared_profile = manifest.get("selection", {}).get("profile", {})
            if (
                prepared_profile.get("allocation_sha256")
                != config["data_mixture"]["allocation_sha256"]
                or prepared_profile.get("allocation_policy")
                != config["data_mixture"]["policy"]
                or prepared_profile.get("without_replacement") is not True
            ):
                raise RuntimeError(
                    f"prepared real allocation changed for {dataset_id}"
                )
            if paired:
                expected_pools = [
                    {
                        key: value
                        for key, value in pool.items()
                        if key != "dataset_id"
                    }
                    for pool in mixture["pools"]
                    if pool["dataset_id"] == dataset_id
                ]
                if prepared_profile.get("pools") != expected_pools:
                    raise RuntimeError(
                        f"prepared paired pools changed for {dataset_id}"
                    )
        expected_train_sequences = (
            entry["prepared_train_sequences"]
            if paired
            else entry["train_sequences"]
        )
        expected_counts = {
            "train_sequences": expected_train_sequences,
            "eval_sequences": entry["eval_sequences"],
            "train_tokens": expected_train_sequences * 1024,
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
                train_sequences_used=entry["train_sequences"],
            )
        )
    if common_tokenizer is None:
        raise RuntimeError("training has no prepared datasets")
    return ResolvedTrainingInputs(
        profile=profile,
        output_root=root,
        datasets=tuple(resolved_datasets),
        tokenizer=common_tokenizer,
        data_mixture=mixture,
        preparation_profile=preparation_profile,
    )


class LazyTrainingSequenceStore:
    """Resolve lightweight row references while caching one shard per dataset."""

    def __init__(self, inputs: ResolvedTrainingInputs) -> None:
        self._datasets = {dataset.dataset_id: dataset for dataset in inputs.datasets}
        self._shards: Dict[str, tuple[list[int], list[Path]]] = {}
        self._cache: Dict[str, tuple[Path, Any]] = {}
        for dataset in inputs.datasets:
            ends = []
            total = 0
            for record in dataset.manifest["splits"]["train"]:
                total += record["rows"]
                ends.append((total, dataset.root / record["path"]))
            expected = dataset.manifest["counts"]["train_sequences"]
            if total != expected:
                raise RuntimeError("training shard row counts changed")
            self._shards[dataset.dataset_id] = (
                [end for end, _ in ends],
                [path for _, path in ends],
            )

    def load(self, reference: TrainingSequenceReference) -> TrainingSequence:
        dataset = self._datasets.get(reference.dataset_id)
        if dataset is None:
            raise RuntimeError("training reference uses an unknown dataset")
        ends, paths = self._shards[reference.dataset_id]
        shard_index = bisect_right(ends, reference.row_index)
        if shard_index >= len(paths) or reference.row_index < 0:
            raise RuntimeError("training reference is outside the prepared split")
        previous_end = ends[shard_index - 1] if shard_index else 0
        shard_path = paths[shard_index]
        cached = self._cache.get(reference.dataset_id)
        if cached is None or cached[0] != shard_path:
            table = pq.read_table(shard_path, columns=["input_ids"])
            self._cache[reference.dataset_id] = (shard_path, table)
        else:
            table = cached[1]
        values = table.column("input_ids")[reference.row_index - previous_end].as_py()
        input_ids = tuple(int(value) for value in values)
        if len(input_ids) != 1024:
            raise RuntimeError("training input is not exactly 1024 tokens")
        vocab_size = dataset.manifest["tokenizer"]["vocab_size"]
        if any(value < 0 or value >= vocab_size for value in input_ids):
            raise RuntimeError("training input contains a token outside the vocabulary")
        return TrainingSequence(dataset_id=reference.dataset_id, input_ids=input_ids)


class LazyTrainingSchedule(Sequence[TrainingSequence]):
    def __init__(
        self,
        references: Sequence[TrainingSequenceReference],
        store: LazyTrainingSequenceStore,
    ) -> None:
        self.references = tuple(references)
        self.store = store

    def __len__(self) -> int:
        return len(self.references)

    @overload
    def __getitem__(self, index: int) -> TrainingSequence: ...

    @overload
    def __getitem__(self, index: slice) -> list[TrainingSequence]: ...

    def __getitem__(
        self, index: int | slice
    ) -> TrainingSequence | list[TrainingSequence]:
        if isinstance(index, slice):
            return [self.store.load(reference) for reference in self.references[index]]
        return self.store.load(self.references[index])


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


def load_training_schedule(
    inputs: ResolvedTrainingInputs, seed: int
) -> Sequence[TrainingSequence]:
    """Use the legacy eager schedule or a no-replacement lazy real schedule."""

    if inputs.profile != "real":
        return load_training_sequences(inputs, seed)
    if (
        isinstance(inputs.data_mixture, dict)
        and inputs.data_mixture.get("policy") == PAIRED_REAL_POLICY
    ):
        references = build_paired_training_references(inputs, seed)
    else:
        references = build_real_training_references(inputs, seed)
    return LazyTrainingSchedule(references, LazyTrainingSequenceStore(inputs))


def build_real_training_references(
    inputs: ResolvedTrainingInputs, seed: int
) -> tuple[TrainingSequenceReference, ...]:
    if inputs.profile != "real":
        raise RuntimeError("real references require real prepared inputs")
    if not isinstance(inputs.data_mixture, dict) or inputs.data_mixture.get(
        "policy"
    ) != "equal_share_without_replacement":
        raise RuntimeError("real training mixture policy changed")
    counts = {
        dataset.dataset_id: dataset.manifest["counts"]["train_sequences"]
        for dataset in inputs.datasets
    }
    shuffled_rows: Dict[str, list[int]] = {}
    for dataset_id, count in counts.items():
        rows = list(range(count))
        random.Random(_stable_seed(seed, dataset_id, "rows")).shuffle(rows)
        shuffled_rows[dataset_id] = rows

    total = sum(counts.values())
    current = {dataset_id: 0 for dataset_id in counts}
    consumed = {dataset_id: 0 for dataset_id in counts}
    tie_order = sorted(
        counts,
        key=lambda dataset_id: _stable_seed(seed, "interleave", dataset_id),
    )
    tie_rank = {dataset_id: index for index, dataset_id in enumerate(tie_order)}
    references = []
    for _ in range(total):
        active = [
            dataset_id
            for dataset_id in counts
            if consumed[dataset_id] < counts[dataset_id]
        ]
        for dataset_id in active:
            current[dataset_id] += counts[dataset_id]
        selected = max(
            active,
            key=lambda dataset_id: (current[dataset_id], -tie_rank[dataset_id]),
        )
        current[selected] -= total
        row_position = consumed[selected]
        references.append(
            TrainingSequenceReference(
                dataset_id=selected,
                row_index=shuffled_rows[selected][row_position],
            )
        )
        consumed[selected] += 1
    if len(set(references)) != total:
        raise RuntimeError("real training schedule contains duplicate references")
    return tuple(references)


def build_paired_training_references(
    inputs: ResolvedTrainingInputs, seed: int
) -> tuple[TrainingSequenceReference, ...]:
    if inputs.profile != "real":
        raise RuntimeError("paired references require real prepared inputs")
    mixture = validate_paired_mixture(inputs.data_mixture)
    pools = {pool["pool_id"]: pool for pool in mixture["pools"]}
    shuffled_rows: Dict[str, list[int]] = {}
    for pool_id, pool in pools.items():
        start = pool["start_row"]
        rows = list(range(start, start + pool["train_sequences"]))
        random.Random(_stable_seed(seed, "paired_pool", pool_id)).shuffle(rows)
        shuffled_rows[pool_id] = rows

    consumed = {pool_id: 0 for pool_id in pools}
    references = []
    for slot_pool_id in iter_paired_mixture_slots(mixture):
        actual_pool_id = slot_pool_id
        if mixture["arm"] == "general" and pools[slot_pool_id]["role"] == "domain":
            actual_pool_id = "brwac_extra"
        pool = pools[actual_pool_id]
        position = consumed[actual_pool_id]
        if position >= len(shuffled_rows[actual_pool_id]):
            raise RuntimeError("paired training pool was exhausted")
        references.append(
            TrainingSequenceReference(
                dataset_id=pool["dataset_id"],
                row_index=shuffled_rows[actual_pool_id][position],
            )
        )
        consumed[actual_pool_id] += 1

    if len(references) != 416_000:
        raise RuntimeError("paired training schedule length changed")
    if len(set(references)) != len(references):
        raise RuntimeError("paired training schedule contains duplicate references")
    for pool_id, pool in pools.items():
        expected = pool["train_sequences"]
        if mixture["arm"] == "general" and pool["role"] == "domain":
            expected = 0
        if mixture["arm"] == "forum_tech" and pool["role"] == "replacement":
            expected = 0
        if consumed[pool_id] != expected:
            raise RuntimeError("paired training schedule did not consume its pools")
    return tuple(references)


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
