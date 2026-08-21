import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from queroquero.training_config import (
    TrainingConfigError,
    load_training_config,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_versioned_p100_configs_match_the_fixed_budgets(self) -> None:
        smoke, smoke_digest = load_training_config(
            "configs/training/p100-smoke.json"
        )
        mvp, mvp_digest = load_training_config("configs/training/p100-mvp.json")

        self.assertEqual(smoke["training"]["total_optimizer_steps"], 6)
        self.assertEqual(smoke["training"]["checkpoint_steps"], [3])
        self.assertEqual(mvp["training"]["total_optimizer_steps"], 192)
        self.assertEqual(mvp["training"]["checkpoint_steps"], [96])
        self.assertEqual(len(smoke_digest), 64)
        self.assertEqual(len(mvp_digest), 64)

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
