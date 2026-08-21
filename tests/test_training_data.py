import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from queroquero.training_config import load_training_config
from queroquero.training_data import (
    ResolvedDataset,
    ResolvedTrainingInputs,
    TrainingSequence,
    load_training_sequences,
    resolve_training_inputs,
)


class TrainingDataTests(unittest.TestCase):
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
