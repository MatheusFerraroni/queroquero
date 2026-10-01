import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from queroquero.config import DATASET_IDS
from queroquero.experiment_report import (
    PAIRED_EXPERIMENT_REPORT_SCHEMA,
    build_paired_experiment_report,
    validate_paired_experiment_report,
)
from queroquero.paired_plan import PAIRED_REAL_POLICY


class PairedExperimentReportTests(unittest.TestCase):
    def test_completed_runs_produce_private_downstream_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dirs = {}
            artifacts = {}
            for index, arm in enumerate(("general", "forum_tech")):
                run_id = f"{index + 1:020x}"
                artifact_id = f"{index + 11:020x}"
                run_dir = root / f"run-{arm}"
                run_dir.mkdir()
                resolved = {
                    "run_id": run_id,
                    "model": {"model_id": "fixed"},
                    "training": {
                        "seed": 42,
                        "total_optimizer_steps": 52_000,
                    },
                    "execution": {"world_size": 2},
                    "inputs": {
                        "data_mixture": {
                            "policy": PAIRED_REAL_POLICY,
                            "arm": arm,
                            "experiment_id": "a" * 20,
                            "allocation_sha256": "b" * 64,
                            "schedule_template_sha256": "c" * 64,
                        },
                        "paired_inputs_sha256": "d" * 64,
                    },
                }
                evaluation = {
                    "macro": {"loss": 2.0, "perplexity": 7.389},
                    "datasets": {
                        dataset_id: {
                            "loss": 2.0,
                            "perplexity": 7.389,
                        }
                        for dataset_id in DATASET_IDS
                    },
                    "optimizer_step": 0,
                }
                manifest = {
                    "run_id": run_id,
                    "status": "complete",
                    "optimizer_steps_completed": 52_000,
                    "baseline_evaluation": evaluation,
                    "final_evaluation": dict(evaluation, optimizer_step=52_000),
                    "quality_gate_passed": True,
                    "promotion_status": "eligible",
                    "artifact": {
                        "artifact_id": artifact_id,
                        "artifact_sha256": f"{index + 21:064x}",
                        "path": f"artifacts/{artifact_id}",
                    },
                    "experiment": {
                        "experiment_id": "a" * 20,
                        "arm": arm,
                        "allocation_sha256": "b" * 64,
                        "schedule_template_sha256": "c" * 64,
                        "paired_inputs_sha256": "d" * 64,
                    },
                }
                (run_dir / "resolved_training.json").write_text(
                    json.dumps(resolved), encoding="utf-8"
                )
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                run_dirs[arm] = run_dir
                artifacts[arm] = {
                    "artifact_id": artifact_id,
                    "artifact_sha256": f"{index + 21:064x}",
                    "training": {
                        "run_id": run_id,
                        "optimizer_steps": 52_000,
                        "experiment": {
                            "experiment_id": "a" * 20,
                            "arm": arm,
                            "allocation_sha256": "b" * 64,
                            "schedule_template_sha256": "c" * 64,
                            "paired_inputs_sha256": "d" * 64,
                        },
                    },
                }

            with patch(
                "queroquero.experiment_report.validate_model_artifact",
                side_effect=[artifacts["general"], artifacts["forum_tech"]],
            ):
                report = build_paired_experiment_report(
                    general_run_dir=run_dirs["general"],
                    forum_tech_run_dir=run_dirs["forum_tech"],
                    general_artifact=root / "general-artifact",
                    forum_tech_artifact=root / "forum-artifact",
                    general_elapsed_seconds=43_200,
                    forum_tech_elapsed_seconds=43_500,
                )

            self.assertEqual(
                report["schema_version"], PAIRED_EXPERIMENT_REPORT_SCHEMA
            )
            self.assertEqual(validate_paired_experiment_report(report), report)
            self.assertEqual(report["arms"]["general"]["elapsed_seconds"], 43_200)
            serialized = json.dumps(report, sort_keys=True)
            self.assertNotIn(temporary_dir, serialized)
            self.assertNotIn("input_ids", serialized)

            changed = deepcopy(report)
            changed["arms"]["general"]["optimizer_steps"] = 51_999
            with self.assertRaisesRegex(RuntimeError, "general result"):
                validate_paired_experiment_report(changed)


if __name__ == "__main__":
    unittest.main()
