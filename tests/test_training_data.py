import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from queroquero.training_config import load_training_config
from queroquero.training_data import (
    LazyTrainingSequenceStore,
    ResolvedDataset,
    ResolvedTrainingInputs,
    TrainingSequence,
    TrainingSequenceReference,
    build_paired_training_references,
    build_real_training_references,
    load_split,
    load_training_sequences,
    resolve_training_inputs,
)
from queroquero.config import DATASET_IDS
from queroquero.paired_plan import (
    allocate_paired_real_training,
    iter_paired_mixture_slots,
    paired_mixture_for_arm,
)
from tests.test_paired_plan import capacity_report
from queroquero.packing import PackedSequence
from queroquero.storage import write_split


class TrainingDataTests(unittest.TestCase):
    def test_paired_schedules_share_positions_and_replace_domain_one_for_one(self) -> None:
        allocation = allocate_paired_real_training(
            capacity_report(dataset_id, 300_000) for dataset_id in DATASET_IDS
        )
        datasets = tuple(
            ResolvedDataset(
                dataset_id=dataset_id,
                root=Path("/synthetic") / dataset_id,
                manifest={
                    "preparation_id": f"{index + 1:020x}",
                    "resolved_config_sha256": "c" * 64,
                    "counts": {
                        "train_sequences": allocation[
                            "prepared_train_sequences"
                        ][dataset_id],
                        "eval_sequences": 256,
                        "train_tokens": allocation[
                            "prepared_train_sequences"
                        ][dataset_id]
                        * 1024,
                        "eval_tokens": 256 * 1024,
                    },
                },
                manifest_sha256=f"{index + 1:064x}",
                relative_manifest_path=f"{dataset_id}/manifest.json",
            )
            for index, dataset_id in enumerate(DATASET_IDS)
        )
        general_mixture = paired_mixture_for_arm(allocation, "general")
        forum_mixture = paired_mixture_for_arm(allocation, "forum_tech")
        general_inputs = ResolvedTrainingInputs(
            profile="real",
            output_root=Path("/synthetic"),
            datasets=datasets,
            tokenizer={},
            data_mixture=general_mixture,
            preparation_profile="paired_real",
        )
        forum_inputs = ResolvedTrainingInputs(
            profile="real",
            output_root=Path("/synthetic"),
            datasets=datasets,
            tokenizer={},
            data_mixture=forum_mixture,
            preparation_profile="paired_real",
        )

        general = build_paired_training_references(general_inputs, 42)
        forum = build_paired_training_references(forum_inputs, 42)
        slots = tuple(iter_paired_mixture_slots(general_mixture))

        self.assertEqual(len(general), 416_000)
        self.assertEqual(len(set(general)), len(general))
        self.assertEqual(len(set(forum)), len(forum))
        self.assertEqual(
            general_inputs.paired_inputs_sha256(),
            forum_inputs.paired_inputs_sha256(),
        )
        for slot, general_ref, forum_ref in zip(slots, general, forum):
            if slot in {"adrenaline_domain", "outerspace_domain"}:
                self.assertEqual(general_ref.dataset_id, "brwac")
                self.assertEqual(
                    forum_ref.dataset_id,
                    slot.removesuffix("_domain"),
                )
            else:
                self.assertEqual(general_ref, forum_ref)

    def test_real_schedule_is_proportional_deterministic_and_without_replacement(self) -> None:
        counts = [2, 3, 4, 5, 6, 7]
        dataset_ids = (
            "adrenaline",
            "brwac",
            "gigaverbo",
            "multiwoz_ptbr",
            "outerspace",
            "wackywacky",
        )
        datasets = tuple(
            ResolvedDataset(
                dataset_id=dataset_id,
                root=Path("/synthetic") / dataset_id,
                manifest={
                    "preparation_id": f"{count:020x}",
                    "resolved_config_sha256": "c" * 64,
                    "counts": {
                        "train_sequences": count,
                        "eval_sequences": 256,
                        "train_tokens": count * 1024,
                        "eval_tokens": 256 * 1024,
                    },
                },
                manifest_sha256="a" * 64,
                relative_manifest_path=f"{dataset_id}/manifest.json",
            )
            for dataset_id, count in zip(dataset_ids, counts)
        )
        inputs = ResolvedTrainingInputs(
            profile="real",
            output_root=Path("/synthetic"),
            datasets=datasets,
            tokenizer={
                "model_id": "model",
                "revision": "revision",
                "fingerprint_sha256": "f" * 64,
                "vocab_size": 49_152,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 49_109,
                "unk_token_id": 0,
            },
            data_mixture={
                "policy": "equal_share_without_replacement",
                "without_replacement": True,
                "allocation_sha256": "b" * 64,
            },
        )

        first = build_real_training_references(inputs, 42)
        second = build_real_training_references(inputs, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), sum(counts))
        self.assertEqual(len(set(first)), len(first))
        for dataset_id, count in zip(dataset_ids, counts):
            self.assertEqual(
                sum(item.dataset_id == dataset_id for item in first), count
            )
        metadata = inputs.metadata()
        self.assertEqual(
            metadata["allocated_train_tokens"], sum(counts) * 1024
        )
        self.assertEqual(
            metadata["data_mixture"]["policy"],
            "equal_share_without_replacement",
        )
        self.assertNotIn("prepared_train_sequences", metadata)
        self.assertNotIn("prepared_train_tokens", metadata)

    def test_lazy_loader_matches_the_eager_parquet_loader(self) -> None:
        records = [
            PackedSequence(
                sequence_id=f"{index + 1:064x}",
                input_ids=(index + 4,) * 1024,
                source_ref_sha256=(f"{index + 100:064x}",),
                source_token_counts=(1024,),
            )
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            shards = write_split(root, "train", records, 2)
            dataset = ResolvedDataset(
                dataset_id="adrenaline",
                root=root,
                manifest={
                    "counts": {"train_sequences": 5},
                    "splits": {"train": shards},
                    "tokenizer": {"vocab_size": 49_152},
                },
                manifest_sha256="a" * 64,
                relative_manifest_path="adrenaline/manifest.json",
            )
            inputs = ResolvedTrainingInputs(
                profile="real",
                output_root=root,
                datasets=(dataset,),
                tokenizer={},
            )
            eager = load_split(dataset, "train")
            store = LazyTrainingSequenceStore(inputs)
            lazy = [
                store.load(TrainingSequenceReference("adrenaline", index))
                for index in range(5)
            ]
            self.assertEqual(lazy, eager)

    def test_balanced_schedule_is_deterministic_and_exact(self) -> None:
        config, _ = load_training_config("configs/training/p100-smoke.json")
        datasets = tuple(
            ResolvedDataset(
                dataset_id=entry["dataset_id"],
                root=Path("/synthetic") / entry["dataset_id"],
                manifest={"counts": {"train_sequences": 8}},
                manifest_sha256="a" * 64,
                relative_manifest_path=f"{entry['dataset_id']}/synthetic/dataset_manifest.json",
            )
            for entry in config["datasets"]
        )
        inputs = ResolvedTrainingInputs(
            profile="smoke",
            output_root=Path("/synthetic"),
            datasets=datasets,
            tokenizer={},
        )

        def fake_load_split(dataset, split):
            self.assertEqual(split, "train")
            return [
                TrainingSequence(
                    dataset_id=dataset.dataset_id,
                    input_ids=(index,) * 1024,
                )
                for index in range(8)
            ]

        with patch("queroquero.training_data.load_split", side_effect=fake_load_split):
            first = load_training_sequences(inputs, 42)
            second = load_training_sequences(inputs, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 48)
        self.assertEqual(first[24:], second[24:])
        expected_ids = {entry["dataset_id"] for entry in config["datasets"]}
        for start in range(0, len(first), 6):
            self.assertEqual(
                {sequence.dataset_id for sequence in first[start : start + 6]},
                expected_ids,
            )

    def test_resolver_requires_one_current_manifest_per_dataset(self) -> None:
        config, _ = load_training_config("configs/training/p100-smoke.json")
        tokenizer = {
            "model_id": "Polygl0t/Tucano2-0.6B-Base",
            "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
            "fingerprint_sha256": "f" * 64,
            "vocab_size": 49152,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 49109,
            "unk_token_id": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifests = {}
            for index, entry in enumerate(config["datasets"]):
                dataset_id = entry["dataset_id"]
                preparation_id = f"{index + 1:020x}"
                directory = root / dataset_id / preparation_id
                directory.mkdir(parents=True)
                manifest = {
                    "dataset_id": dataset_id,
                    "profile": "smoke",
                    "preparation_id": preparation_id,
                    "resolved_config_sha256": "c" * 64,
                    "counts": {
                        "train_sequences": 8,
                        "eval_sequences": 2,
                        "train_tokens": 8192,
                        "eval_tokens": 2048,
                    },
                    "tokenizer": tokenizer,
                }
                path = directory / "dataset_manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                manifests[directory.resolve()] = manifest

            with (
                patch(
                    "queroquero.training_data.load_resolved_config",
                    return_value=({}, "c" * 64),
                ),
                patch(
                    "queroquero.training_data.validate_preparation",
                    side_effect=lambda path: manifests[Path(path).resolve()],
                ),
            ):
                resolved = resolve_training_inputs(config, root)

            self.assertEqual(len(resolved.datasets), 6)
            self.assertEqual(resolved.profile, "smoke")
            self.assertTrue(all(len(item.manifest_sha256) == 64 for item in resolved.datasets))
            self.assertFalse(
                any(
                    Path(item.relative_manifest_path).is_absolute()
                    for item in resolved.datasets
                )
            )

            duplicate = root / config["datasets"][0]["dataset_id"] / ("f" * 20)
            duplicate.mkdir()
            duplicate_manifest = dict(
                manifests[next(iter(manifests))], preparation_id="f" * 20
            )
            (duplicate / "dataset_manifest.json").write_text(
                json.dumps(duplicate_manifest), encoding="utf-8"
            )
            with (
                patch(
                    "queroquero.training_data.load_resolved_config",
                    return_value=({}, "c" * 64),
                ),
                patch("queroquero.training_data.validate_preparation"),
                self.assertRaisesRegex(RuntimeError, "exactly one"),
            ):
                resolve_training_inputs(config, root)

    def test_paired_resolver_distinguishes_prepared_and_consumed_counts(self) -> None:
        allocation = allocate_paired_real_training(
            capacity_report(dataset_id, 300_000) for dataset_id in DATASET_IDS
        )
        mixture = paired_mixture_for_arm(allocation, "general")
        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/training/l40s-mvp.json"
            ).read_text(encoding="utf-8")
        )
        config["profile"] = "real"
        config["data_mixture"] = mixture
        for entry in config["datasets"]:
            dataset_id = entry["dataset_id"]
            entry["prepared_train_sequences"] = allocation[
                "prepared_train_sequences"
            ][dataset_id]
            entry["train_sequences"] = allocation["general_allocations"][dataset_id]
            entry["eval_sequences"] = 256

        tokenizer = {
            "model_id": "Polygl0t/Tucano2-0.6B-Base",
            "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
            "fingerprint_sha256": "f" * 64,
            "vocab_size": 49152,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 49109,
            "unk_token_id": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifests = {}
            for index, entry in enumerate(config["datasets"]):
                dataset_id = entry["dataset_id"]
                preparation_id = f"{index + 1:020x}"
                directory = root / dataset_id / preparation_id
                directory.mkdir(parents=True)
                prepared = entry["prepared_train_sequences"]
                manifest = {
                    "dataset_id": dataset_id,
                    "profile": "paired_real",
                    "preparation_id": preparation_id,
                    "resolved_config_sha256": "c" * 64,
                    "selection": {
                        "profile": {
                            "allocation_sha256": mixture["allocation_sha256"],
                            "allocation_policy": mixture["policy"],
                            "without_replacement": True,
                            "pools": [
                                {
                                    key: value
                                    for key, value in pool.items()
                                    if key != "dataset_id"
                                }
                                for pool in mixture["pools"]
                                if pool["dataset_id"] == dataset_id
                            ],
                        }
                    },
                    "counts": {
                        "train_sequences": prepared,
                        "eval_sequences": 256,
                        "train_tokens": prepared * 1024,
                        "eval_tokens": 256 * 1024,
                    },
                    "tokenizer": tokenizer,
                }
                path = directory / "dataset_manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                manifests[directory.resolve()] = manifest

            with (
                patch(
                    "queroquero.training_data.load_resolved_config",
                    return_value=({}, "c" * 64),
                ),
                patch(
                    "queroquero.training_data.validate_preparation",
                    side_effect=lambda path: manifests[Path(path).resolve()],
                ),
            ):
                resolved = resolve_training_inputs(config, root)

        metadata = resolved.metadata()
        self.assertEqual(metadata["preparation_profile"], "paired_real")
        self.assertEqual(sum(metadata["allocated_train_sequences"].values()), 416_000)
        self.assertGreater(
            metadata["prepared_train_sequences"]["adrenaline"], 0
        )
        self.assertEqual(metadata["allocated_train_sequences"]["adrenaline"], 0)

    def test_resolver_rejects_corrupt_manifest_instead_of_ignoring_it(self) -> None:
        config, _ = load_training_config("configs/training/p100-smoke.json")
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_id = config["datasets"][0]["dataset_id"]
            corrupt = root / dataset_id / "corrupt"
            corrupt.mkdir(parents=True)
            (corrupt / "dataset_manifest.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "JSON is unreadable"):
                resolve_training_inputs(config, root)


if __name__ == "__main__":
    unittest.main()
