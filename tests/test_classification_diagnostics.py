import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from queroquero.classification_diagnostics import (
    CATEGORIES,
    CONFIG_SCHEMA,
    LOW_SHOT_BUDGETS,
    SEEDS,
    _bootstrap_state_means,
    _build_report_payload,
    _classification_metrics,
    _encode_nll_batch,
    _expected_cohort_rows,
    _output_subdirectory,
    _perplexity,
    _score_nll_batch,
    _target_counts_for_texts,
    _validate_low_shot_unit,
    first_post_target_mask,
    load_diagnostics_config,
    low_shot_seed,
    nested_low_shot_ids,
    state_by_index,
    terminal_checkpoint_decision,
    validate_checkpoint_model_for_inference,
    validate_diagnostics_config,
)
from queroquero.config import ConfigError
from queroquero.config import canonical_json_bytes, sha256_bytes
from queroquero.manifest import file_sha256
from queroquero.classification_eval_common import digest_strings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/classification/diagnostics-v1.json"


class ClassificationDiagnosticsTests(unittest.TestCase):
    def test_versioned_config_pins_the_diagnostic_experiment_strictly(self) -> None:
        config, digest = load_diagnostics_config(CONFIG_PATH)

        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            config["source_evaluation"]["evaluation_id"],
            "5e9cb26cc8b35bc1b33e",
        )
        self.assertEqual(config["dataset"]["classification_dataset_id"], "dc4b2ce164eab81812a2")
        self.assertEqual(len(config["splits"]), 10)
        self.assertEqual(len(config["states"]), 9)
        self.assertEqual(config["low_shot"]["budgets_per_class"], list(LOW_SHOT_BUDGETS))
        self.assertEqual(config["statistics"]["bootstrap_repetitions"], 10_000)

        unknown = deepcopy(config)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ConfigError, "keys are incomplete or unknown"):
            validate_diagnostics_config(unknown)

        changed_policy = deepcopy(config)
        changed_policy["nll"]["max_length"] = 512
        with self.assertRaisesRegex(ConfigError, "NLL policy changed"):
            validate_diagnostics_config(changed_policy)

        incomplete_matrix = deepcopy(config)
        incomplete_matrix["splits"][-1] = deepcopy(incomplete_matrix["splits"][0])
        with self.assertRaisesRegex(ConfigError, "split matrix is incomplete"):
            validate_diagnostics_config(incomplete_matrix)

        unsafe_state = deepcopy(config)
        unsafe_state["states"][1]["state_name"] = "../../outside"
        with self.assertRaisesRegex(ConfigError, "state schedule changed"):
            validate_diagnostics_config(unsafe_state)

        wrong_checkpoint = deepcopy(config)
        wrong_checkpoint["states"][1]["relative_path"] = (
            "checkpoints/3cfe6f183912c25b859a/step-026000"
        )
        with self.assertRaisesRegex(ConfigError, "state path changed"):
            validate_diagnostics_config(wrong_checkpoint)

    def test_versioned_config_rejects_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            link = Path(temporary_dir) / "diagnostics.json"
            link.symlink_to(CONFIG_PATH)
            with self.assertRaisesRegex(ConfigError, "must not be a symlink"):
                load_diagnostics_config(link)

    def test_output_directories_refuse_symlink_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "run"
            outside = Path(temporary_dir) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "private").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "must not traverse a symlink"):
                _output_subdirectory(root, "private/nll", "private output")

    def test_state_and_low_shot_unit_indices_are_exact_and_bounded(self) -> None:
        config, _ = load_diagnostics_config(CONFIG_PATH)

        states = [state_by_index(config, index) for index in range(9)]
        self.assertEqual([state["state_index"] for state in states], list(range(9)))
        self.assertEqual(states[0]["state_name"], "base-000000")
        self.assertEqual(states[-1]["state_name"], "forum-052000")
        self.assertEqual([low_shot_seed(config, index) for index in range(5)], list(SEEDS))

        for invalid in (-1, 9, True, 0.0):
            with self.subTest(state_index=invalid):
                with self.assertRaisesRegex(ValueError, "state index"):
                    state_by_index(config, invalid)
        for invalid in (-1, 5, False, 1.0):
            with self.subTest(low_shot_index=invalid):
                with self.assertRaisesRegex(ValueError, "unit index"):
                    low_shot_seed(config, invalid)

    def test_first_post_target_mask_excludes_context_special_padding_and_crossing(self) -> None:
        mask = first_post_target_mask(
            offsets=[
                (0, 0),    # BOS/special token
                (0, 5),    # title token
                (8, 12),   # token crossing the title/post boundary
                (10, 13),  # first token wholly inside the post
                (14, 18),  # post token marked special
                (19, 23),  # padded token
                (24, 24),  # empty offset
                (24, 28),  # regular post token
            ],
            special_tokens_mask=[1, 0, 0, 0, 1, 0, 0, 0],
            attention_mask=[1, 1, 1, 1, 1, 0, 1, 1],
            boundary=10,
        )

        self.assertEqual(mask, [False, False, False, True, False, False, False, True])

        with self.assertRaisesRegex(ValueError, "different lengths"):
            first_post_target_mask([(0, 1)], [0, 0], [1], boundary=0)
        with self.assertRaisesRegex(ValueError, "start and end"):
            first_post_target_mask([(0, 1, 2)], [0], [1], boundary=0)

    def test_nested_low_shot_ids_are_exact_balanced_nested_and_deterministic(self) -> None:
        labels = ["3", "8", "19"]
        pairs = [
            (f"{class_index:02x}{item:062x}", label)
            for class_index, label in enumerate(labels, start=1)
            for item in range(12)
        ]
        train_ids = [sample_id for sample_id, _ in pairs]
        train_labels = [label for _, label in pairs]
        label_by_id = dict(pairs)
        budgets = (2, 4, 8)

        selected = nested_low_shot_ids(
            train_ids,
            train_labels,
            seed=42,
            budgets=budgets,
        )
        repeated = nested_low_shot_ids(
            list(reversed(train_ids)),
            list(reversed(train_labels)),
            seed=42,
            budgets=budgets,
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(set(selected), set(budgets))
        for budget in budgets:
            values = selected[budget]
            self.assertEqual(len(values), budget * len(labels))
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(
                {label: sum(label_by_id[value] == label for value in values) for label in labels},
                {label: budget for label in labels},
            )
        self.assertLessEqual(set(selected[2]), set(selected[4]))
        self.assertLessEqual(set(selected[4]), set(selected[8]))
        self.assertNotEqual(
            selected[8],
            nested_low_shot_ids(train_ids, train_labels, seed=43, budgets=budgets)[8],
        )

    def test_nested_low_shot_ids_fail_closed_on_invalid_or_insufficient_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training IDs are invalid"):
            nested_low_shot_ids(["a", "a"], ["3", "3"], seed=42, budgets=(1,))
        with self.assertRaisesRegex(RuntimeError, "training IDs are invalid"):
            nested_low_shot_ids(["a"], [], seed=42, budgets=(1,))
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            nested_low_shot_ids(
                [f"a{index}" for index in range(3)] + [f"b{index}" for index in range(2)],
                ["3"] * 3 + ["8"] * 2,
                seed=42,
                budgets=(1, 3),
            )

    def test_bootstrap_is_deterministic_and_preserves_paired_state_deltas(self) -> None:
        categories = np.repeat(np.asarray(CATEGORIES, dtype=np.int32), 300)
        row_signal = np.linspace(0.0, 1.0, 1_800, dtype=np.float64)
        offsets = np.arange(9, dtype=np.float64) * 0.125
        matrix = row_signal[:, None] + offsets[None, :]

        first = _bootstrap_state_means(
            categories,
            matrix,
            repetitions=37,
            seed=20260829,
        )
        repeated = _bootstrap_state_means(
            categories,
            matrix,
            repetitions=37,
            seed=20260829,
        )
        changed_seed = _bootstrap_state_means(
            categories,
            matrix,
            repetitions=37,
            seed=20260830,
        )

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed_seed))
        np.testing.assert_allclose(first[:, 1] - first[:, 0], 0.125, rtol=0, atol=1e-12)
        np.testing.assert_allclose(first[:, 8] - first[:, 4], 0.5, rtol=0, atol=1e-12)

        invalid_categories = categories.copy()
        invalid_categories[0] = CATEGORIES[1]
        with self.assertRaisesRegex(RuntimeError, "unbalanced"):
            _bootstrap_state_means(
                invalid_categories,
                matrix,
                repetitions=2,
                seed=1,
            )

    def test_cohort_selection_is_exact_balanced_fresh_and_deterministic(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        config, _ = load_diagnostics_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rows = []
            for category in CATEGORIES:
                for index in range(302):
                    rows.append(
                        {
                            "sample_id": f"s-{category}-{index:04d}",
                            "title": f"Título técnico {category} {index}",
                            "first_post": "Conteúdo sintético suficiente para o teste.",
                            "category_id": category,
                            "title_group_id": f"g-{category}-{index:04d}",
                        }
                    )
            rows.append(
                {
                    "sample_id": "conflict-a",
                    "title": "Conflito A",
                    "first_post": "Conteúdo sintético.",
                    "category_id": CATEGORIES[0],
                    "title_group_id": "conflicting-group",
                }
            )
            rows.append(
                {
                    "sample_id": "conflict-b",
                    "title": "Conflito B",
                    "first_post": "Conteúdo sintético.",
                    "category_id": CATEGORIES[1],
                    "title_group_id": "conflicting-group",
                }
            )
            pq.write_table(pa.Table.from_pylist(rows), root / "examples.parquet")
            excluded_group = f"g-{CATEGORIES[0]}-0301"
            fake_tokenizer = object()
            patches = (
                mock.patch(
                    "queroquero.classification_diagnostics._excluded_title_groups",
                    return_value={excluded_group},
                ),
                mock.patch(
                    "queroquero.classification_diagnostics._load_nll_tokenizer",
                    return_value=fake_tokenizer,
                ),
                mock.patch(
                    "queroquero.classification_diagnostics._target_counts_for_texts",
                    side_effect=lambda _tokenizer, texts, max_length: [40] * len(texts),
                ),
                mock.patch(
                    "queroquero.classification_diagnostics.tokenizer_fingerprint",
                    return_value="f" * 64,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                selected, metadata = _expected_cohort_rows(config, root)
                repeated, repeated_metadata = _expected_cohort_rows(config, root)

        self.assertEqual(selected, repeated)
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(len(selected), 1800)
        self.assertEqual(len({row["sample_id"] for row in selected}), 1800)
        self.assertEqual(len({row["title_group_id"] for row in selected}), 1800)
        self.assertNotIn(excluded_group, {row["title_group_id"] for row in selected})
        self.assertNotIn("conflicting-group", {row["title_group_id"] for row in selected})
        self.assertEqual(
            {category: sum(row["category_id"] == category for row in selected) for category in CATEGORIES},
            {category: 300 for category in CATEGORIES},
        )
        self.assertEqual(metadata["tokenizer_fingerprint_sha256"], "f" * 64)

    def test_nll_encoding_matches_individual_and_batch_with_ptbr_offsets(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the local test environment")

        class FakeTokenizer:
            def __call__(self, texts, *, return_tensors=None, max_length, **_kwargs):
                encoded_rows = []
                for text in texts:
                    title, post = text.split("\n\n", 1)
                    boundary = len(title) + 2
                    offsets = [
                        (0, 0),
                        (0, len(title)),
                        (boundary - 1, boundary + 1),
                        (boundary, boundary + min(3, len(post))),
                        (boundary + min(3, len(post)), boundary + min(6, len(post))),
                        (0, 0),
                    ][:max_length]
                    length = len(offsets)
                    encoded_rows.append(
                        {
                            "input_ids": [1, 2, 3, 4, 5, 6][:length],
                            "attention_mask": [1] * length,
                            "offset_mapping": offsets,
                            "special_tokens_mask": [1, 0, 0, 0, 0, 1][:length],
                        }
                    )
                width = max(len(row["input_ids"]) for row in encoded_rows)
                result = {}
                for key in encoded_rows[0]:
                    padding = (0, 0) if key == "offset_mapping" else 0
                    values = [row[key] + [padding] * (width - len(row[key])) for row in encoded_rows]
                    result[key] = (
                        torch.tensor(values, dtype=torch.long)
                        if return_tensors == "pt"
                        else values
                    )
                return result

        class FakeModel:
            def __call__(self, input_ids, **_kwargs):
                batch, length = input_ids.shape
                logits = torch.arange(8, dtype=torch.float32).repeat(batch, length, 1)
                return SimpleNamespace(logits=logits)

        tokenizer = FakeTokenizer()
        texts = [
            ("Ação técnica", "primeiro post extenso"),
            ("Áudio e vídeo", "outro conteúdo detalhado"),
        ]
        counts = _target_counts_for_texts(tokenizer, texts, max_length=6)
        _, _, encoded_counts = _encode_nll_batch(
            tokenizer, texts, max_length=6, device=torch.device("cpu")
        )
        batch_sums, batch_counts = _score_nll_batch(
            tokenizer,
            FakeModel(),
            texts,
            max_length=6,
            device=torch.device("cpu"),
        )
        individual = [
            _score_nll_batch(
                tokenizer,
                FakeModel(),
                [text],
                max_length=6,
                device=torch.device("cpu"),
            )
            for text in texts
        ]
        self.assertEqual(counts, [2, 2])
        self.assertEqual(encoded_counts, counts)
        self.assertEqual(batch_counts, counts)
        np.testing.assert_allclose(batch_sums, [value[0][0] for value in individual])

    def test_report_contains_accumulated_gains_and_exact_perplexity(self) -> None:
        config, _ = load_diagnostics_config(CONFIG_PATH)
        config = deepcopy(config)
        config["statistics"]["bootstrap_repetitions"] = 31
        resolved = {
            "diagnostic_id": "a" * 20,
            "git_commit": "b" * 40,
            "config_sha256": "c" * 64,
            "classification_dataset_id": config["dataset"]["classification_dataset_id"],
            "source_evaluation": {
                "evaluation_id": config["source_evaluation"]["evaluation_id"]
            },
            "models": config["models"],
            "runs": config["training_runs"],
        }
        labels = [str(category) for category in CATEGORIES]
        confusion = [[300 if row == column else 0 for column in range(6)] for row in range(6)]
        low_units = []
        for seed in SEEDS:
            results = []
            for model_index, model in enumerate(("base", "general", "forum")):
                for budget in LOW_SHOT_BUDGETS:
                    score = 0.5 + model_index * 0.01 + budget / 100_000 + seed / 1_000_000
                    results.append(
                        {
                            "model": model,
                            "budget_per_class": budget,
                            "accuracy": score,
                            "macro_f1": score,
                            "labels": labels,
                            "by_class": {
                                label: {
                                    "precision": score,
                                    "recall": score,
                                    "f1": score,
                                    "support": 300,
                                }
                                for label in labels
                            },
                            "confusion_matrix": confusion,
                        }
                    )
            low_units.append({"seed": seed, "results": results})

        categories = np.repeat(np.asarray(CATEGORIES, dtype=np.int32), 300)
        sample_ids = [f"{index:064x}" for index in range(1800)]
        target_tokens = np.repeat(np.arange(1, 7, dtype=np.int64) * 40, 300)
        category_base = np.repeat(np.arange(1, 7, dtype=np.float64), 300)
        offsets = [0.0, -0.10, -0.20, -0.30, -0.40, -0.05, -0.10, -0.15, -0.05]
        score_values = []
        for offset in offsets:
            means = category_base + offset
            score_values.append(
                {
                    "sample_id": sample_ids,
                    "category_id": categories.tolist(),
                    "target_tokens": target_tokens.tolist(),
                    "nll_sum": (means * target_tokens).tolist(),
                    "mean_nll": means.tolist(),
                }
            )
        with mock.patch(
            "queroquero.classification_diagnostics._load_low_shot_units",
            return_value=low_units,
        ), mock.patch(
            "queroquero.classification_diagnostics._load_state_scores",
            side_effect=[({}, values) for values in score_values],
        ):
            report, csvs = _build_report_payload(config, resolved, Path("unused"))

        self.assertEqual(len(report["checkpoint_curve"]["accumulated_gains_vs_base"]), 8)
        self.assertIn("checkpoint_accumulated_gains.csv", csvs)
        self.assertEqual(len(report["checkpoint_curve"]["increments"]), 8)
        decisions = {
            row["arm"]: row["decision"]
            for row in report["checkpoint_curve"]["terminal_decisions"]
        }
        self.assertEqual(decisions["general"], "still_improving_at_52000")
        self.assertEqual(decisions["forum_tech"], "terminal_regression")
        base = report["checkpoint_curve"]["state_summary"][0]
        self.assertAlmostEqual(base["macro_mean_thread_nll"], 3.5)
        self.assertNotAlmostEqual(base["token_weighted_nll"], 3.5)
        self.assertEqual(_perplexity(1.0), math.e)
        with self.assertRaisesRegex(RuntimeError, "overflowed"):
            _perplexity(1000.0)

    def test_low_shot_metrics_are_recomputed_from_private_predictions(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed in the local test environment")

        labels = [str(category) for category in CATEGORIES]
        ids = [f"{index:064x}" for index in range(1800)]
        truth = [label for label in labels for _ in range(300)]
        metrics = _classification_metrics(truth, truth, labels)
        results = [
            {
                "model": model,
                "budget_per_class": budget,
                **metrics,
            }
            for model in ("base", "general", "forum")
            for budget in LOW_SHOT_BUDGETS
        ]
        rows = [
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "predicted_label": true_label,
                "model": model,
                "seed": 42,
                "budget_per_class": budget,
            }
            for model in ("base", "general", "forum")
            for budget in LOW_SHOT_BUDGETS
            for sample_id, true_label in zip(ids, truth)
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            private = output / "private/low_shot/seed-42.parquet"
            private.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), private, compression="zstd")
            value = {
                "schema_version": "queroquero-cpt-low-shot-unit/v1",
                "diagnostic_id": "a" * 20,
                "source_evaluation_id": "b" * 20,
                "unit_index": 0,
                "seed": 42,
                "task": "coarse",
                "input_variant": "title_first_post",
                "pooling": "masked_mean",
                "c": 0.01,
                "selection": [
                    {
                        "budget_per_class": budget,
                        "examples": budget * 6,
                        "selection_sha256": f"{budget:064x}"[-64:],
                    }
                    for budget in LOW_SHOT_BUDGETS
                ],
                "counts": {
                    "validation_accessed": 0,
                    "test_examples": 1800,
                    "test_examples_per_class": 300,
                    "fits": 12,
                },
                "test_set_sha256": digest_strings(ids),
                "results": results,
                "private_output": {
                    "relative_path": "private/low_shot/seed-42.parquet",
                    "rows": 21_600,
                    "size_bytes": private.stat().st_size,
                    "sha256": file_sha256(private),
                },
                "status": "complete",
            }
            resolved = {"diagnostic_id": "a" * 20}
            _validate_low_shot_unit(value, resolved, output, 0, 42)
            tampered = deepcopy(value)
            tampered["results"][0]["accuracy"] = 0.9
            with self.assertRaisesRegex(
                RuntimeError, "metrics do not match private predictions"
            ):
                _validate_low_shot_unit(tampered, resolved, output, 0, 42)

    def test_checkpoint_inference_validation_ignores_optimizer_payload_hash(self) -> None:
        run = {
            "run_id": "a" * 20,
            "config_sha256": "b" * 64,
            "inputs_sha256": "c" * 64,
            "git_commit": "d" * 40,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "step-000003"
            model = root / "model"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            (root / "training_state.pt").write_bytes(b"private-optimizer-state")
            records = [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ]
            manifest = {
                "schema_version": "queroquero-training-checkpoint/v2",
                "checkpoint_id": "step-000003",
                "run_id": run["run_id"],
                "optimizer_step": 3,
                "sequences_consumed": 24,
                "world_size": 2,
                "global_batch_sequences": 8,
                "config_sha256": run["config_sha256"],
                "inputs_sha256": run["inputs_sha256"],
                "git_commit": run["git_commit"],
                "files": records,
                "files_sha256": sha256_bytes(canonical_json_bytes(records)),
            }
            (root / "checkpoint_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = validate_checkpoint_model_for_inference(
                root, run, 3, verify_model_hashes=True
            )
            self.assertEqual(result["model_files"], 2)

            # The optimizer payload is never loaded; only its recorded size is checked.
            optimizer_state_size = next(
                record["size_bytes"]
                for record in records
                if record["path"] == "training_state.pt"
            )
            (root / "training_state.pt").write_bytes(b"x" * optimizer_state_size)
            self.assertEqual(
                (root / "training_state.pt").stat().st_size,
                optimizer_state_size,
            )
            validate_checkpoint_model_for_inference(
                root, run, 3, verify_model_hashes=True
            )

            (model / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "model file hash changed"):
                validate_checkpoint_model_for_inference(
                    root, run, 3, verify_model_hashes=True
                )

    def test_terminal_checkpoint_decision_uses_the_full_interval(self) -> None:
        self.assertEqual(
            terminal_checkpoint_decision([-0.04, -0.01]),
            "still_improving_at_52000",
        )
        self.assertEqual(
            terminal_checkpoint_decision([-0.01, 0.02]),
            "no_clear_additional_improvement",
        )
        self.assertEqual(
            terminal_checkpoint_decision([0.01, 0.03]),
            "terminal_regression",
        )

    def test_slurm_launchers_pin_modes_resources_and_offline_execution(self) -> None:
        submit = (PROJECT_ROOT / "scripts/submit_classification_diagnostics.sh").read_text()
        cpu = (PROJECT_ROOT / "scripts/classification_diagnostics_cpu.sbatch").read_text()
        gpu = (PROJECT_ROOT / "scripts/classification_diagnostics_gpu.sbatch").read_text()
        combined = submit + cpu + gpu

        self.assertIn("--time=1-00:00:00", submit)
        self.assertIn("--array=0-4%4", submit)
        self.assertIn("--array=0-8%2", submit)
        self.assertIn("--cpus-per-task=16", submit)
        self.assertIn("--cpus-per-task=8", submit)
        self.assertIn("--mem=64G", submit)
        self.assertIn("--gres=gpu:L40S:1", submit)
        for mode in (
            "prepare-cohort",
            "validate-cohort",
            "preflight",
            "low-shot-unit",
            "score-unit",
            "validate-scores",
            "report",
            "validate-report",
        ):
            with self.subTest(mode=mode):
                self.assertIn(mode, submit)

        self.assertIn("#SBATCH --cpus-per-task=16", cpu)
        self.assertIn("#SBATCH --mem=64G", cpu)
        self.assertIn("#SBATCH --time=1-00:00:00", cpu)
        self.assertNotIn("#SBATCH --gres", cpu)
        self.assertIn("#SBATCH --cpus-per-task=8", gpu)
        self.assertIn("#SBATCH --mem=64G", gpu)
        self.assertIn("#SBATCH --gres=gpu:L40S:1", gpu)
        self.assertIn("#SBATCH --time=1-00:00:00", gpu)
        self.assertNotIn("torchrun", gpu)
        self.assertNotIn("NCCL_P2P_DISABLE", combined)
        for variable in (
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        ):
            with self.subTest(variable=variable):
                self.assertIn(variable, cpu)
                self.assertIn(variable, gpu)


if __name__ == "__main__":
    unittest.main()
