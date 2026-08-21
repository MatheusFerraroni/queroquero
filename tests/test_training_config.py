import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from queroquero.training_config import (
    TRAINING_CONFIG_SCHEMA,
    TrainingConfigError,
    load_training_config,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_versioned_configs_match_the_fixed_budgets(self) -> None:
        smoke, smoke_digest = load_training_config(
            "configs/training/p100-smoke.json"
        )
        mvp, mvp_digest = load_training_config("configs/training/p100-mvp.json")
        l40s_smoke, l40s_smoke_digest = load_training_config(
            "configs/training/l40s-smoke.json"
        )
        l40s_mvp, l40s_mvp_digest = load_training_config(
            "configs/training/l40s-mvp.json"
        )

        self.assertEqual(smoke["schema_version"], TRAINING_CONFIG_SCHEMA)
        self.assertEqual(smoke["training"]["total_optimizer_steps"], 6)
        self.assertEqual(smoke["training"]["checkpoint_steps"], [3])
        self.assertEqual(mvp["training"]["total_optimizer_steps"], 192)
        self.assertEqual(mvp["training"]["checkpoint_steps"], [96])
        self.assertEqual(l40s_smoke["training"]["total_optimizer_steps"], 6)
        self.assertEqual(l40s_smoke["training"]["checkpoint_steps"], [3])
        self.assertEqual(l40s_mvp["training"]["total_optimizer_steps"], 192)
        self.assertEqual(l40s_mvp["training"]["checkpoint_steps"], [96])
        self.assertEqual(l40s_mvp["execution"]["world_size"], 2)
        self.assertEqual(l40s_mvp["training"]["precision"], "bf16")
        self.assertEqual(
            2
            * l40s_mvp["training"]["micro_batch_size_per_rank"]
            * l40s_mvp["training"]["gradient_accumulation_steps_per_rank"],
            l40s_mvp["training"]["global_batch_sequences"],
        )
        for digest in (
            smoke_digest,
            mvp_digest,
            l40s_smoke_digest,
            l40s_mvp_digest,
        ):
            self.assertEqual(len(digest), 64)

    def test_training_contract_rejects_lora_or_changed_budget(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/training/p100-mvp.json").read_text(
                encoding="utf-8"
            )
        )
        changed_method = deepcopy(config)
        changed_method["training"]["method"] = "lora"
        with self.assertRaisesRegex(TrainingConfigError, "method"):
            validate_training_config(changed_method)

        changed_budget = deepcopy(config)
        changed_budget["datasets"][0]["train_sequences"] = 128
        with self.assertRaisesRegex(TrainingConfigError, "budgets"):
            validate_training_config(changed_budget)

    def test_training_contract_rejects_non_p100_or_bf16(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/training/p100-smoke.json").read_text(
                encoding="utf-8"
            )
        )
        changed_gpu = deepcopy(config)
        changed_gpu["hardware"]["gpu_name_contains"] = "RTX"
        with self.assertRaisesRegex(TrainingConfigError, "P100"):
            validate_training_config(changed_gpu)

        changed_precision = deepcopy(config)
        changed_precision["training"]["precision"] = "bf16"
        with self.assertRaisesRegex(TrainingConfigError, "precision"):
            validate_training_config(changed_precision)

    def test_l40s_contract_rejects_world_size_or_fp16(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/training/l40s-smoke.json").read_text(
                encoding="utf-8"
            )
        )
        changed_world = deepcopy(config)
        changed_world["execution"]["world_size"] = 1
        with self.assertRaisesRegex(TrainingConfigError, "L40S"):
            validate_training_config(changed_world)

        changed_precision = deepcopy(config)
        changed_precision["training"]["precision"] = "fp16"
        with self.assertRaisesRegex(TrainingConfigError, "precision"):
            validate_training_config(changed_precision)

        changed_batch = deepcopy(config)
        changed_batch["training"]["global_batch_sequences"] = 4
        with self.assertRaisesRegex(TrainingConfigError, "global_batch_sequences"):
            validate_training_config(changed_batch)

    def test_training_contract_rejects_unknown_keys_and_boolean_numbers(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/training/p100-smoke.json").read_text(
                encoding="utf-8"
            )
        )
        unknown = deepcopy(config)
        unknown["training"]["automatic_fallback"] = True
        with self.assertRaisesRegex(TrainingConfigError, "unknown"):
            validate_training_config(unknown)

        boolean_weight = deepcopy(config)
        boolean_weight["datasets"][0]["weight"] = True
        with self.assertRaisesRegex(TrainingConfigError, "weights"):
            validate_training_config(boolean_weight)

    def test_training_config_must_stay_inside_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            outside = Path(temporary_dir) / "training.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(TrainingConfigError, "inside the project"):
                load_training_config(outside)


if __name__ == "__main__":
    unittest.main()
