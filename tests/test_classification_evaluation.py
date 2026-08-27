import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from queroquero.classification_eval_common import (
    EVALUATION_CONFIG_SCHEMA,
    EVALUATION_UNIT_SCHEMA,
    EMBEDDING_VALIDATION_SCHEMA,
    MODEL_NAMES,
    POOLINGS,
    SELECTION_SCHEMA,
    TUNING_UNIT_SCHEMA,
    load_evaluation_config,
    unit_by_index,
    validate_evaluation_config,
)
from queroquero.classification_probe import (
    build_report,
    select_hyperparameters,
    validate_report,
)
from queroquero.config import ConfigError
from queroquero.manifest import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/classification/evaluation-v1.json"


class ClassificationEvaluationTests(unittest.TestCase):
    def test_versioned_config_pins_the_full_paired_experiment(self) -> None:
        config, digest = load_evaluation_config(CONFIG_PATH)
        self.assertEqual(config["schema_version"], EVALUATION_CONFIG_SCHEMA)
        self.assertEqual(len(config["splits"]), 10)
        self.assertEqual(config["paired_report"]["report_id"], "8c7ec8d2317571ce4f48")
        self.assertEqual(config["embedding"]["max_length"], 1024)
        self.assertEqual(config["classifier"]["c_grid"], [0.01, 0.1, 1.0, 10.0])
        self.assertEqual(len(digest), 64)

        changed = deepcopy(config)
        changed["embedding"]["max_length"] = 512
        with self.assertRaisesRegex(ConfigError, "embedding policy"):
            validate_evaluation_config(changed)

    def test_unit_index_covers_seed_task_and_input_matrix(self) -> None:
        config, _ = load_evaluation_config(CONFIG_PATH)
        values = [unit_by_index(config, index) for index in range(20)]
        self.assertEqual(values[0], {
            "unit_index": 0,
            "seed": 42,
            "task": "coarse",
            "input_variant": "title",
        })
        self.assertEqual(values[-1], {
            "unit_index": 19,
            "seed": 46,
            "task": "fine",
            "input_variant": "title_first_post",
        })
        self.assertEqual(
            len({(value["seed"], value["task"], value["input_variant"]) for value in values}),
            20,
        )

    def test_shared_selection_uses_all_models_and_prefers_masked_mean_then_small_c(self) -> None:
        config, _ = load_evaluation_config(CONFIG_PATH)
        resolved = {"evaluation_id": "a" * 20}
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_embedding_validation(root, resolved["evaluation_id"])
            tuning = root / "tuning"
            tuning.mkdir()
            for index in range(20):
                unit = unit_by_index(config, index)
                results = [
                    {
                        "model": model,
                        "pooling": pooling,
                        "c": c_value,
                        "validation": {"accuracy": 0.5, "macro_f1": 0.5},
                    }
                    for model in MODEL_NAMES
                    for pooling in POOLINGS
                    for c_value in config["classifier"]["c_grid"]
                ]
                (tuning / f"unit-{index:02d}.json").write_text(
                    json.dumps({
                        "schema_version": TUNING_UNIT_SCHEMA,
                        "evaluation_id": resolved["evaluation_id"],
                        **unit,
                        "counts": {"test_accessed": 0},
                        "results": results,
                        "status": "complete",
                    }),
                    encoding="utf-8",
                )

            selection = select_hyperparameters(config, resolved, root)
            self.assertEqual(len(selection["selected"]), 4)
            self.assertTrue(
                all(value["pooling"] == "masked_mean" for value in selection["selected"])
            )
            self.assertTrue(all(value["c"] == 0.01 for value in selection["selected"]))
            self.assertFalse(selection["test_accessed"])

    def test_report_computes_second_minus_first_paired_deltas(self) -> None:
        config, _ = load_evaluation_config(CONFIG_PATH)
        resolved = {
            "evaluation_id": "b" * 20,
            "git_commit": "c" * 40,
            "config_sha256": "d" * 64,
            "classification_dataset_id": "e" * 20,
            "paired_report_id": "f" * 20,
            "models": {
                "base": {"revision": "0" * 40},
                "general": {"artifact_id": "1" * 20},
                "forum": {"artifact_id": "2" * 20},
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _write_embedding_validation(root, resolved["evaluation_id"])
            selected = [
                {
                    "task": task,
                    "input_variant": input_variant,
                    "pooling": "masked_mean",
                    "c": 0.01,
                }
                for task in ("coarse", "fine")
                for input_variant in ("title", "title_first_post")
            ]
            (root / "selection.json").write_text(
                json.dumps({
                    "schema_version": SELECTION_SCHEMA,
                    "evaluation_id": resolved["evaluation_id"],
                    "selection_scope": config["classifier"]["selection_scope"],
                    "test_accessed": False,
                    "scores": [{
                        "task": "coarse",
                        "input_variant": "title",
                        "pooling": "masked_mean",
                        "c": 0.01,
                        "mean_validation_macro_f1": 0.5,
                        "mean_validation_accuracy": 0.5,
                        "observations": 15,
                    }],
                    "selected": selected,
                    "status": "complete",
                }),
                encoding="utf-8",
            )
            (root / "preflight.json").write_text(
                json.dumps({
                    "dependencies": {
                        "numpy": "2.5.2",
                        "scipy": "1.17.0",
                        "scikit_learn": "1.9.0",
                        "torch": "2.7.1+cu118",
                        "transformers": "5.14.1",
                    }
                }),
                encoding="utf-8",
            )
            units = root / "evaluation_units"
            units.mkdir()
            private_root = root / "private/predictions"
            private_root.mkdir(parents=True)
            model_scores = {"base": 0.5, "general": 0.6, "forum": 0.7}
            for index in range(20):
                unit = unit_by_index(config, index)
                results = [
                    {
                        "model": model,
                        "accuracy": score,
                        "macro_f1": score,
                        "labels": ["1", "2"],
                        "by_class": {
                            "1": {"precision": score, "recall": score, "f1": score, "support": 10},
                            "2": {"precision": score, "recall": score, "f1": score, "support": 10},
                        },
                        "confusion_matrix": [[10, 0], [0, 10]],
                    }
                    for model, score in model_scores.items()
                ]
                private_path = private_root / f"unit-{index:02d}.parquet"
                private_path.write_bytes(f"private-{index}".encode())
                (units / f"unit-{index:02d}.json").write_text(
                    json.dumps({
                        "schema_version": EVALUATION_UNIT_SCHEMA,
                        "evaluation_id": resolved["evaluation_id"],
                        **unit,
                        "results": results,
                        "private_output": {
                            "relative_path": f"private/predictions/unit-{index:02d}.parquet",
                            "sha256": file_sha256(private_path),
                        },
                        "test_accessed": True,
                        "status": "complete",
                    }),
                    encoding="utf-8",
                )
            report = build_report(config, resolved, root, write=False)
            primary = [
                value
                for value in report["paired_contrasts"]
                if value["task"] == "coarse"
                and value["input_variant"] == "title_first_post"
                and value["metric"] == "macro_f1"
                and value["contrast"] == "domain_proximity"
            ]
            self.assertEqual(len(primary), 1)
            self.assertAlmostEqual(primary[0]["mean"], 0.1)
            self.assertEqual(primary[0]["direction"], "second_minus_first")
            self.assertEqual(primary[0]["endpoint"], "primary")
            self.assertEqual(report["test_policy"]["final_evaluations"], 60)
            build_report(config, resolved, root, write=True)
            self.assertEqual(validate_report(config, resolved, root)["status"], "valid")
            with (root / "report/summary.csv").open("a", encoding="utf-8") as handle:
                handle.write("changed\n")
            with self.assertRaisesRegex(RuntimeError, "report file changed"):
                validate_report(config, resolved, root)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_pooling_excludes_padding_and_special_tokens(self) -> None:
        import torch

        from queroquero.classification_embeddings import (
            content_mask,
            pool_last_hidden_state,
        )

        hidden = torch.tensor([[[1.0], [2.0], [6.0], [100.0]]])
        attention = torch.tensor([[1, 1, 1, 0]])
        special = torch.tensor([[1, 0, 0, 0]])
        mask = content_mask(attention, special)
        mean, last = pool_last_hidden_state(hidden, mask)
        self.assertEqual(mean.tolist(), [[4.0]])
        self.assertEqual(last.tolist(), [[6.0]])

    def test_chunked_embedding_loader_restores_requested_order_and_checks_finiteness(self) -> None:
        import numpy as np

        from queroquero.classification_embeddings import load_embedding_rows

        ids = [character * 64 for character in "abcd"]
        with tempfile.TemporaryDirectory() as temporary_dir:
            evaluation_dir = Path(temporary_dir)
            root = evaluation_dir / "embeddings/base/title/chunks"
            root.mkdir(parents=True)
            for rank, rank_ids, values in (
                (0, [ids[0], ids[2]], [[1.0, 2.0], [3.0, 4.0]]),
                (1, [ids[1], ids[3]], [[5.0, 6.0], [7.0, 8.0]]),
            ):
                stem = f"rank-{rank:02d}-start-00000000"
                np.save(root / f"{stem}-ids.npy", np.asarray(rank_ids, dtype="S64"))
                for pooling in POOLINGS:
                    np.save(
                        root / f"{stem}-{pooling}.npy",
                        np.asarray(values, dtype=np.float32),
                    )
                (root / f"{stem}.json").write_text(
                    json.dumps({
                        "files": {
                            "ids": {"path": f"{stem}-ids.npy"},
                            "masked_mean": {"path": f"{stem}-masked_mean.npy"},
                            "last_content": {"path": f"{stem}-last_content.npy"},
                        }
                    }),
                    encoding="utf-8",
                )
            loaded = load_embedding_rows(
                evaluation_dir, "base", "title", "masked_mean", [ids[3], ids[0]]
            )
            self.assertEqual(loaded.tolist(), [[7.0, 8.0], [1.0, 2.0]])

            np.save(
                root / "rank-01-start-00000000-masked_mean.npy",
                np.asarray([[5.0, 6.0], [float("nan"), 8.0]], dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                load_embedding_rows(
                    evaluation_dir,
                    "base",
                    "title",
                    "masked_mean",
                    [ids[3]],
                )

    def test_slurm_resources_are_fixed_and_arrays_are_bounded(self) -> None:
        gpu = (PROJECT_ROOT / "scripts/classification_embeddings.sbatch").read_text()
        cpu = (PROJECT_ROOT / "scripts/classification_probe.sbatch").read_text()
        submit = (PROJECT_ROOT / "scripts/submit_classification_evaluation.sh").read_text()
        self.assertIn("#SBATCH --gres=gpu:L40S:2", gpu)
        self.assertIn("#SBATCH --mem=128G", gpu)
        self.assertIn("#SBATCH --time=1-00:00:00", gpu)
        self.assertNotIn("#SBATCH --gres", cpu)
        self.assertIn("--array=0-19%4", submit)
        self.assertNotIn("NCCL_P2P_DISABLE=1", gpu + cpu + submit)


def _write_embedding_validation(root: Path, evaluation_id: str) -> None:
    hashes = {}
    for model in MODEL_NAMES:
        path = root / "embeddings" / model / "embedding_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(f'{{"model":"{model}"}}', encoding="utf-8")
        hashes[model] = file_sha256(path)
    (root / "embeddings_validation.json").write_text(
        json.dumps({
            "schema_version": EMBEDDING_VALIDATION_SCHEMA,
            "evaluation_id": evaluation_id,
            "models": 3,
            "variants": 2,
            "dimension": 1024,
            "embedding_manifests": hashes,
            "status": "valid",
        }),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
