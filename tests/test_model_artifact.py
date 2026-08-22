import json
import tempfile
import unittest
from pathlib import Path

from queroquero.config import DATASET_IDS, canonical_json_bytes, sha256_bytes
from queroquero.manifest import file_sha256
from queroquero.model_artifact import (
    MODEL_ARTIFACT_SCHEMA,
    _aggregate_files_sha256,
    _expected_artifact_id,
    _tokenizer_files_fingerprint,
    validate_model_artifact,
)


class ModelArtifactTests(unittest.TestCase):
    def test_structural_validation_accepts_transformers_v5_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(Path(temporary_dir))
            manifest = validate_model_artifact(root, load_model=False)
            self.assertEqual(manifest["schema_version"], MODEL_ARTIFACT_SCHEMA)
            self.assertEqual(manifest["architecture"]["parameter_count"], 670127616)
            self.assertFalse((root / "special_tokens_map.json").exists())

    def test_structural_validation_accepts_legacy_special_tokens_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(
                Path(temporary_dir), include_special_tokens_map=True
            )
            manifest = validate_model_artifact(root, load_model=False)
            listed = {record["path"] for record in manifest["files"]}
            self.assertIn("special_tokens_map.json", listed)

    def test_structural_validation_rejects_missing_tokenizer_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(
                Path(temporary_dir), include_tokenizer_json=False
            )
            with self.assertRaisesRegex(
                RuntimeError, "missing model or tokenizer configuration"
            ):
                validate_model_artifact(root, load_model=False)

    def test_structural_validation_rejects_changed_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(Path(temporary_dir))
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "size mismatch|hash mismatch"):
                validate_model_artifact(root, load_model=False)

    def test_structural_validation_rejects_adapter_or_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(Path(temporary_dir))
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unlisted files"):
                validate_model_artifact(root, load_model=False)

    def test_l40s_artifact_does_not_require_bitsandbytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(Path(temporary_dir), target="l40s")
            manifest = validate_model_artifact(root, load_model=False)
            self.assertEqual(manifest["training"]["execution"]["world_size"], 2)
            self.assertNotIn("bitsandbytes", manifest["environment"])

    def test_real_artifact_accepts_positive_generalized_training_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._write_artifact(
                Path(temporary_dir), target="l40s", real=True
            )
            manifest = validate_model_artifact(root, load_model=False)
            self.assertEqual(manifest["training"]["optimizer_steps"], 52_000)
            self.assertEqual(
                sum(
                    item["train_sequences"]
                    for item in manifest["training"]["datasets"]
                ),
                416_000,
            )

    def _write_artifact(
        self,
        parent: Path,
        *,
        target: str = "p100",
        include_special_tokens_map: bool = False,
        include_tokenizer_json: bool = True,
        real: bool = False,
    ) -> Path:
        executions = {
            "p100": {
                "strategy": "single_process",
                "backend": "none",
                "world_size": 1,
                "precision": "fp16",
                "optimizer": "adamw8bit",
                "optimizer_implementation": "bitsandbytes",
                "micro_batch_size_per_rank": 1,
                "gradient_accumulation_steps_per_rank": 8,
                "global_batch_sequences": 8,
                "global_batch_tokens": 8192,
            },
            "l40s": {
                "strategy": "ddp",
                "backend": "nccl",
                "world_size": 2,
                "precision": "bf16",
                "optimizer": "adamw",
                "optimizer_implementation": "torch_fused",
                "micro_batch_size_per_rank": 1,
                "gradient_accumulation_steps_per_rank": 4,
                "global_batch_sequences": 8,
                "global_batch_tokens": 8192,
            },
        }
        training = {
            "method": "full_parameter_continual_pretraining",
            "git_commit": "b" * 40,
            "run_id": "c" * 20,
            "seed": 42,
            "config_sha256": "d" * 64,
            "inputs_sha256": "e" * 64,
            "optimizer_steps": 192,
            "execution": executions[target],
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "preparation_id": f"{index + 1:020x}",
                    "dataset_manifest_sha256": f"{index + 1:064x}",
                }
                for index, dataset_id in enumerate(DATASET_IDS)
            ],
        }
        if real:
            training["optimizer_steps"] = 52_000
            training["profile"] = "real"
            training["data_mixture"] = {
                "policy": "equal_share_without_replacement",
                "without_replacement": True,
                "allocation_sha256": "a" * 64,
            }
            budgets = [69_334, 69_334, 69_333, 69_333, 69_333, 69_333]
            for item, budget in zip(training["datasets"], budgets):
                item["train_sequences"] = budget
                item["eval_sequences"] = 256
        artifact_id = _expected_artifact_id(training)
        root = parent / artifact_id
        root.mkdir()
        files = {
            "config.json": b"{}",
            "model.safetensors": b"synthetic-safe-weights",
            "tokenizer_config.json": b"{}",
        }
        if include_tokenizer_json:
            files["tokenizer.json"] = b"{}"
        if include_special_tokens_map:
            files["special_tokens_map.json"] = b"{}"
        for name, content in files.items():
            (root / name).write_bytes(content)
        records = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(root.iterdir())
        ]
        tokenizer = {
            "model_id": "Polygl0t/Tucano2-0.6B-Base",
            "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
            "prepared_fingerprint_sha256": "f" * 64,
            "vocab_size": 49152,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 49109,
            "unk_token_id": 0,
        }
        tokenizer["fingerprint_sha256"] = _tokenizer_files_fingerprint(
            records, tokenizer
        )
        manifest = {
            "schema_version": MODEL_ARTIFACT_SCHEMA,
            "artifact_id": artifact_id,
            "format": "transformers_pretrained",
            "parent_model": {
                "model_id": "Polygl0t/Tucano2-0.6B-Base",
                "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
                "license": "Apache-2.0",
            },
            "architecture": {
                "model_type": "llama",
                "parameter_count": 670127616,
                "native_context_length": 4096,
                "training_sequence_length": 1024,
                "weights_dtype": "float32",
            },
            "tokenizer": tokenizer,
            "training": training,
            "environment": {
                "python": "3.12.0",
                "torch": "2.7.1+cu118",
                "torch_cuda": "11.8",
                "transformers": "5.14.1",
                "tokenizers": "0.0.0",
            },
            "files": records,
            "artifact_sha256": _aggregate_files_sha256(records),
            "redistribution_status": "internal_research_only",
        }
        if target == "p100":
            manifest["environment"]["bitsandbytes"] = "0.50.0"
        (root / "model_artifact_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        return root


if __name__ == "__main__":
    unittest.main()
