import json
import os
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from queroquero.train import (
    CHECKPOINT_SCHEMA,
    _cuda_binary_arch_is_compatible,
    _dataset_counts,
    _interruption_requested,
    _restore_checkpoint_state,
    _run_on_main,
    _validate_gpu_environment,
    _validate_checkpoint,
    _wrap_distributed_model,
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

    def test_rank_zero_failures_are_broadcast_before_a_barrier(self) -> None:
        class MainContext:
            is_main = True

            def broadcast_object(self, value):
                return value

        with self.assertRaisesRegex(RuntimeError, "rank 0 ValueError: write failed"):
            _run_on_main(
                MainContext(),
                lambda: (_ for _ in ()).throw(ValueError("write failed")),
            )

    def test_interrupt_marker_is_checked_only_by_rank_zero_and_broadcast(self) -> None:
        class MainContext:
            is_main = True

            def broadcast_object(self, value):
                self.broadcast = value
                return value

        with tempfile.TemporaryDirectory() as temporary_dir:
            marker = Path(temporary_dir) / "interrupt"
            marker.touch()
            context = MainContext()
            with patch(
                "queroquero.train.PROJECT_ROOT", Path(temporary_dir).resolve()
            ):
                with patch.dict(
                    os.environ, {"TRAIN_INTERRUPT_FILE": marker.as_posix()}
                ):
                    self.assertTrue(_interruption_requested(context))
            self.assertTrue(context.broadcast)

    def test_l40s_runtime_validates_two_homogeneous_bf16_gpus(self) -> None:
        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/training/l40s-smoke.json"
            ).read_text(encoding="utf-8")
        )

        class Properties:
            name = "NVIDIA L40S"
            total_memory = 48 * 1024**3

        class Nccl:
            @staticmethod
            def version():
                return (2, 21, 5)

        class Cuda:
            nccl = Nccl()

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

            @staticmethod
            def get_device_properties(index):
                self.assertEqual(index, 0)
                return Properties()

            @staticmethod
            def get_device_capability(index):
                self.assertEqual(index, 0)
                return (8, 9)

            @staticmethod
            def get_arch_list():
                return ["sm_60", "sm_86", "sm_90"]

            @staticmethod
            def is_bf16_supported():
                return True

        class Context:
            backend = "nccl"
            rank = 0
            local_rank = 0
            world_size = 2

            @staticmethod
            def all_gather_objects(value):
                other = deepcopy(value)
                other["rank"] = 1
                other["local_rank"] = 1
                return [value, other]

        torch = SimpleNamespace(
            cuda=Cuda(), version=SimpleNamespace(cuda="11.8")
        )
        result = _validate_gpu_environment(torch, config, Context())
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["nccl_version"], [2, 21, 5])

    def test_cuda_binary_arch_compatibility_stays_within_one_major(self) -> None:
        self.assertTrue(_cuda_binary_arch_is_compatible("sm_80", (8, 9)))
        self.assertTrue(_cuda_binary_arch_is_compatible("sm_86", (8, 9)))
        self.assertTrue(_cuda_binary_arch_is_compatible("sm_89", (8, 9)))
        self.assertFalse(_cuda_binary_arch_is_compatible("sm_90", (8, 9)))
        self.assertFalse(_cuda_binary_arch_is_compatible("sm_75", (8, 9)))
        self.assertFalse(_cuda_binary_arch_is_compatible("compute_86", (8, 9)))

    def test_ddp_wrapper_disables_static_graph_for_gradient_accumulation(self) -> None:
        base_model = object()
        wrapped_model = object()
        call = {}

        def distributed_data_parallel(model, **kwargs):
            call["model"] = model
            call["kwargs"] = kwargs
            return wrapped_model

        context = SimpleNamespace(
            local_rank=1,
            torch=SimpleNamespace(
                nn=SimpleNamespace(
                    parallel=SimpleNamespace(
                        DistributedDataParallel=distributed_data_parallel
                    )
                )
            ),
        )
        config = {"execution": {"strategy": "ddp"}}

        result = _wrap_distributed_model(base_model, config, context)

        self.assertIs(result, wrapped_model)
        self.assertIs(call["model"], base_model)
        self.assertEqual(
            call["kwargs"],
            {
                "device_ids": [1],
                "output_device": 1,
                "static_graph": False,
                "find_unused_parameters": False,
                "broadcast_buffers": False,
                "gradient_as_bucket_view": True,
            },
        )

    def test_checkpoint_validation_binds_cursor_and_input_hashes(self) -> None:
        resolved = {
            "run_id": "a" * 20,
            "config_sha256": "b" * 64,
            "inputs_sha256": "c" * 64,
            "git_commit": "d" * 40,
            "training": {"global_batch_sequences": 8},
            "execution": {"world_size": 2},
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
                "world_size": 2,
                "global_batch_sequences": 8,
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
            def __init__(self, cpu_value):
                self.cpu_value = cpu_value

            def cpu(self):
                return self.cpu_value

        class FakeCuda:
            def __init__(self):
                self.loaded = None

            def set_rng_state(self, value, device):
                self.loaded = (value, device)

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
                    "rng_by_rank": [
                        {
                            "rank": 0,
                            "python": expected_python_rng,
                            "torch_cpu": FakeTensor("cpu-rng"),
                            "torch_cuda": FakeTensor("cuda-rng"),
                        }
                    ],
                    "optimizer_step": 3,
                    "sequences_consumed": 24,
                    "world_size": 1,
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
        self.assertEqual(torch.cuda.loaded, ("cuda-rng", "cuda"))


if __name__ == "__main__":
    unittest.main()
