import json
import random
import tempfile
import unittest
from pathlib import Path

from queroquero.train import (
    CHECKPOINT_SCHEMA,
    _dataset_counts,
    _restore_checkpoint_state,
    _validate_checkpoint,
    build_parser,
)
from queroquero.training_data import TrainingSequence


class TrainCoreTests(unittest.TestCase):
    def test_training_module_imports_without_optional_torch(self) -> None:
        self.assertTrue(callable(build_parser))

    def test_cli_requires_explicit_subcommand_and_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run", "--config", "configs/training/p100-smoke.json", "--resume"]
        )
        self.assertEqual(args.command, "run")
        self.assertTrue(args.resume)

    def test_dataset_counts_do_not_expose_sequence_content(self) -> None:
        sequences = [
            TrainingSequence(dataset_id="brwac", input_ids=(7,) * 1024),
            TrainingSequence(dataset_id="adrenaline", input_ids=(9,) * 1024),
            TrainingSequence(dataset_id="brwac", input_ids=(11,) * 1024),
        ]
        self.assertEqual(_dataset_counts(sequences), {"adrenaline": 1, "brwac": 2})

    def test_checkpoint_validation_binds_cursor_and_input_hashes(self) -> None:
        resolved = {
            "run_id": "a" * 20,
            "config_sha256": "b" * 64,
            "inputs_sha256": "c" * 64,
            "git_commit": "d" * 40,
            "training": {"gradient_accumulation_steps": 8},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "step-000003"
            (root / "model").mkdir(parents=True)
            (root / "model/model.safetensors").write_bytes(b"weights")
            (root / "training_state.pt").write_bytes(b"state")
            from queroquero.config import canonical_json_bytes, sha256_bytes
            from queroquero.manifest import file_sha256

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
                "schema_version": CHECKPOINT_SCHEMA,
                "checkpoint_id": root.name,
                "run_id": resolved["run_id"],
                "optimizer_step": 3,
                "sequences_consumed": 24,
                "config_sha256": resolved["config_sha256"],
                "inputs_sha256": resolved["inputs_sha256"],
                "git_commit": resolved["git_commit"],
                "files": records,
                "files_sha256": sha256_bytes(canonical_json_bytes(records)),
            }
            (root / "checkpoint_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(_validate_checkpoint(root, resolved)["optimizer_step"], 3)

            manifest["sequences_consumed"] = 23
            (root / "checkpoint_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "cursor"):
                _validate_checkpoint(root, resolved)

    def test_checkpoint_restore_reinstates_optimizer_scheduler_scaler_and_rng(self) -> None:
        original_python_rng = random.getstate()
        self.addCleanup(random.setstate, original_python_rng)
        expected_python_rng = random.Random(123).getstate()

        class Receiver:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, value):
                self.loaded = value

        class FakeTensor:
            def cpu(self):
                return "cpu-rng"

        class FakeCuda:
            def __init__(self):
                self.loaded = None

            def set_rng_state_all(self, value):
                self.loaded = value

        class FakeTorch:
            def __init__(self):
                self.cuda = FakeCuda()
                self.cpu_rng = None

            def load(self, path, map_location, weights_only):
                self.load_call = (path.name, map_location, weights_only)
                return {
                    "optimizer": {"optimizer": 1},
                    "scheduler": {"scheduler": 2},
                    "scaler": {"scaler": 3},
                    "python_rng_state": expected_python_rng,
                    "torch_rng_state": FakeTensor(),
                    "cuda_rng_state_all": ["cuda-rng"],
                    "optimizer_step": 3,
                    "sequences_consumed": 24,
                }

            def set_rng_state(self, value):
                self.cpu_rng = value

        optimizer = Receiver()
        scheduler = Receiver()
        scaler = Receiver()
        torch = FakeTorch()
        _restore_checkpoint_state(
            Path("/synthetic/checkpoint"),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            torch=torch,
            expected_step=3,
            expected_sequences=24,
        )

        self.assertEqual(optimizer.loaded, {"optimizer": 1})
        self.assertEqual(scheduler.loaded, {"scheduler": 2})
        self.assertEqual(scaler.loaded, {"scaler": 3})
        self.assertEqual(random.getstate(), expected_python_rng)
        self.assertEqual(torch.cpu_rng, "cpu-rng")
        self.assertEqual(torch.cuda.loaded, ["cuda-rng"])


if __name__ == "__main__":
    unittest.main()
