from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from queroquero.config import canonical_json_bytes, sha256_bytes
from queroquero.datasets.base import Document, ScanResult
from queroquero.prepare import (
    ReviewRequired,
    run_capacity_audit,
    run_preparation,
    validate_preparation,
)


class _Backend:
    def to_str(self) -> str:
        return '{"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"ByteLevel"}}'


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 49109
    unk_token_id = 0
    backend_tokenizer = _Backend()

    def __len__(self) -> int:
        return 49152

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(256)}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [4 + ord(character) % 200 for character in text]}


class SyntheticAdapter:
    def scan(self, config, resume_cursor=None, resume_documents=None, checkpoint=None):
        del config, resume_cursor, resume_documents
        documents = [
            Document(
                text=(
                    f"PRIVATE_TEXT_SENTINEL_{index} "
                    + (f"conteúdo sintético distinto {index}. " * 80)
                ),
                source_ref=f"https://source.invalid/PRIVATE_REF_SENTINEL/{index}",
                source_position={"index": index},
            )
            for index in range(32)
        ]
        if checkpoint is not None:
            checkpoint({"next_index": 16}, documents[:16])
        return ScanResult(
            documents=documents,
            source_fingerprint={
                "kind": "synthetic/v1",
                "sha256": "a" * 64,
                "records": len(documents),
            },
            metrics={"documents_selected": len(documents)},
            cursor={"next_index": len(documents), "complete": True},
        )


class BlockedAdapter:
    def scan(self, config, resume_cursor=None, resume_documents=None, checkpoint=None):
        del config, resume_cursor, resume_documents, checkpoint
        report = {
            "schema_version": "queroquero-boilerplate-report/v2",
            "profile": "mvp",
            "decision": "pending",
            "contains_examples": False,
            "candidate_documents": 0,
            "analysis": {},
            "simulation": {},
            "applied": {},
            "finalization_blocked": True,
        }
        return ScanResult(
            documents=[],
            source_fingerprint={"kind": "synthetic/v1", "sha256": "b" * 64},
            metrics={"documents_selected": 0},
            cursor={"complete": True, "finalization_blocked": True},
            extra_reports={"boilerplate_report": report},
        )


class CapacityAdapter:
    def __init__(self, source_documents: int = 600):
        self.source_documents = source_documents
        self.resumed = False

    def scan(self, config, resume_cursor=None, resume_documents=None, checkpoint=None):
        self.resumed = self.resumed or resume_cursor is not None
        limit = min(config["profile"]["candidate_documents"], self.source_documents)
        documents = list(resume_documents or [])
        for index in range(len(documents), limit):
            documents.append(
                Document(
                    text=(
                        f"PRIVATE_CAPACITY_TEXT_{index} "
                        + (f"conteúdo único {index}. " * 55)
                    ),
                    source_ref=f"private-capacity:{index}",
                    source_position={"index": index},
                )
            )
        cursor = {"documents_selected": len(documents), "complete": True}
        if checkpoint is not None:
            checkpoint(cursor, documents)
        return ScanResult(
            documents=documents,
            source_fingerprint={
                "kind": "synthetic-capacity/v1",
                "sha256": "c" * 64,
                "records": self.source_documents,
            },
            metrics={"documents_selected": len(documents)},
            cursor=cursor,
        )


def resolved_config(dataset_id: str = "brwac", profile: str = "smoke"):
    preparation = {
        "schema_version": "queroquero-preparation/v1",
        "tokenizer": {
            "model_id": "Polygl0t/Tucano2-0.6B-Base",
            "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
            "trust_remote_code": False,
        },
        "sequence_length": 1024,
        "seed": 42,
        "output_root": "derived",
        "storage": {
            "format": "parquet",
            "compression": "zstd",
            "sequences_per_shard": 1024,
        },
        "cleaning": {
            "unicode_normalization": "NFC",
            "strip_html": True,
            "strip_control_characters": True,
        },
    }
    filters = {"min_characters": 1}
    if dataset_id == "wackywacky":
        filters["boilerplate"] = {
            "schema_version": 4,
            "decision_by_profile": {"smoke": "remove_exact", "mvp": "pending"},
            "cross_domain_paragraphs": {
                "minimum_characters": 80,
                "minimum_documents": 5,
                "minimum_domains": 3,
            },
            "within_domain_blocks": {
                "lines_per_block": 3,
                "minimum_characters": 60,
                "minimum_documents": 5,
            },
            "document_filter": {
                "minimum_remaining_characters": 300,
                "maximum_removed_fraction": 0.8,
            },
        }
        filters["page_filter"] = {
            "search_title_markers": [
                "resultados da pesquisa",
                "resultados de pesquisa",
                "resultados da busca",
                "resultados de busca",
                "search results",
            ],
            "search_query_parameters": ["search"],
            "search_query_value_markers": [
                "especial:pesquisar",
                "special:search",
            ],
            "search_path_segments": [
                "search",
                "busca",
                "buscar",
                "pesquisa",
                "pesquisar",
            ],
            "listing_path_segments": [
                "tag",
                "tags",
                "category",
                "categories",
                "categoria",
                "categorias",
                "archive",
                "archives",
                "arquivo",
                "arquivos",
            ],
        }
        filters["line_filter"] = {"minimum_characters": 40}
    result = {
        "schema_version": "queroquero-resolved-preparation/v1",
        "dataset_id": dataset_id,
        "profile_name": profile,
        "preparation": preparation,
        "dataset": {
            "schema_version": "queroquero-dataset-config/v1",
            "dataset_id": dataset_id,
            "adapter": dataset_id,
            "source": {"kind": "synthetic"},
            "filters": filters,
            "redistribution_status": "internal_research_only",
        },
        "profile": {
            "train_sequences": 8 if profile == "smoke" else 256,
            "eval_sequences": 2 if profile == "smoke" else 32,
            "candidate_documents": 128 if profile == "smoke" else 4096,
            "selection": "engineering_prefix" if profile == "smoke" else "representative",
        },
    }
    return result, sha256_bytes(canonical_json_bytes(result))


def file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class PreparationIntegrationTests(unittest.TestCase):
    def test_capacity_audit_is_private_capped_exact_and_resumable(self) -> None:
        resolved, _ = resolved_config(profile="mvp")
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_root = Path(temporary_dir)
            capped_adapter = CapacityAdapter(source_documents=700)
            with (
                patch(
                    "queroquero.prepare.load_resolved_config",
                    return_value=(resolved, "d" * 64),
                ),
                patch(
                    "queroquero.prepare.resolve_output_root",
                    return_value=output_root,
                ),
                patch(
                    "queroquero.prepare.load_adapter",
                    return_value=capped_adapter,
                ),
                patch(
                    "queroquero.prepare._load_pinned_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                redirect_stdout(io.StringIO()),
            ):
                first = run_capacity_audit("brwac", candidate_documents=600)
                second = run_capacity_audit("brwac", candidate_documents=600)
            self.assertEqual(first, second)
            self.assertTrue(capped_adapter.resumed)
            report = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(report["capacity_kind"], "lower_bound")
            serialized = first.read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_CAPACITY_TEXT", serialized)
            self.assertNotIn("private-capacity:", serialized)

            exact_adapter = CapacityAdapter(source_documents=600)
            with (
                patch(
                    "queroquero.prepare.load_resolved_config",
                    return_value=(resolved, "d" * 64),
                ),
                patch(
                    "queroquero.prepare.resolve_output_root",
                    return_value=output_root,
                ),
                patch(
                    "queroquero.prepare.load_adapter",
                    return_value=exact_adapter,
                ),
                patch(
                    "queroquero.prepare._load_pinned_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                redirect_stdout(io.StringIO()),
            ):
                exact_path = run_capacity_audit(
                    "brwac", candidate_documents=700
                )
            exact_report = json.loads(exact_path.read_text(encoding="utf-8"))
            self.assertEqual(exact_report["capacity_kind"], "exact")

    def test_two_isolated_runs_are_byte_identical_private_and_fully_validated(self) -> None:
        resolved, digest = resolved_config()
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir)
            output_roots = [project / "first", project / "second"]
            manifests = []
            for output_root in output_roots:
                with (
                    patch(
                        "queroquero.prepare.load_resolved_config",
                        return_value=(resolved, digest),
                    ),
                    patch(
                        "queroquero.prepare.resolve_output_root",
                        return_value=output_root,
                    ),
                    patch("queroquero.prepare.load_adapter", return_value=SyntheticAdapter()),
                    patch(
                        "queroquero.prepare._load_pinned_tokenizer",
                        return_value=FakeTokenizer(),
                    ),
                    patch("queroquero.prepare.PROJECT_ROOT", project.resolve()),
                    redirect_stdout(io.StringIO()),
                ):
                    manifests.append(run_preparation("brwac", "smoke"))

            first_root = manifests[0].parent
            second_root = manifests[1].parent
            self.assertEqual(first_root.name, second_root.name)
            self.assertEqual(file_bytes(first_root), file_bytes(second_root))
            self.assertEqual(validate_preparation(first_root)["counts"]["train_sequences"], 8)
            self.assertEqual(validate_preparation(second_root)["counts"]["eval_sequences"], 2)

            all_bytes = b"".join(file_bytes(first_root).values())
            self.assertNotIn(b"PRIVATE_TEXT_SENTINEL", all_bytes)
            self.assertNotIn(b"PRIVATE_REF_SENTINEL", all_bytes)
            table = pq.read_table(first_root / "train/shard-00000.parquet")
            self.assertEqual(
                table.column_names,
                [
                    "sequence_id",
                    "input_ids",
                    "source_ref_sha256",
                    "source_token_counts",
                ],
            )

            shard = first_root / "train/shard-00000.parquet"
            with shard.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_preparation(first_root)

            second_manifest = json.loads(manifests[1].read_text(encoding="utf-8"))
            second_manifest["splits"]["train"][0]["rows"] += 1
            manifests[1].write_text(
                json.dumps(second_manifest, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "row or token count"):
                validate_preparation(second_root)

    def test_review_gate_writes_report_without_loading_tokenizer(self) -> None:
        resolved, digest = resolved_config("wackywacky", "mvp")
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir)
            output_root = project / "derived"
            with (
                patch(
                    "queroquero.prepare.load_resolved_config",
                    return_value=(resolved, digest),
                ),
                patch(
                    "queroquero.prepare.resolve_output_root",
                    return_value=output_root,
                ),
                patch("queroquero.prepare.load_adapter", return_value=BlockedAdapter()),
                patch("queroquero.prepare._load_pinned_tokenizer") as tokenizer_loader,
                patch("queroquero.prepare.PROJECT_ROOT", project.resolve()),
            ):
                with self.assertRaises(ReviewRequired):
                    run_preparation("wackywacky", "mvp")

            tokenizer_loader.assert_not_called()
            reports = list(output_root.glob("wackywacky/*/boilerplate_report.json"))
            self.assertEqual(len(reports), 1)
            self.assertFalse(
                json.loads(reports[0].read_text(encoding="utf-8"))["contains_examples"]
            )
            self.assertFalse(list(output_root.glob("wackywacky/*/dataset_manifest.json")))


if __name__ == "__main__":
    unittest.main()
