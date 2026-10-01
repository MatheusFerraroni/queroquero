from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import zstandard as zstd

from queroquero.config import ConfigError
from queroquero.datasets.wackywacky import (
    WackyWackyAdapter,
    _EXPECTED_COLUMNS,
    _sampled_source_guard,
)


def resolved_config(
    *,
    profile_name: str = "smoke",
    candidate_documents: int = 3,
    decision: str | None = None,
    checkpoint_interval: int = 100,
    minimum_line_characters: int = 1,
) -> dict[str, Any]:
    return {
        "profile_name": profile_name,
        "preparation": {"seed": 42},
        "dataset": {
            "source": {
                "root_env": "TEST_PTBR_DATASET_ROOT",
                "path": "wacky/pages.tsv",
                "format": "tsv",
                "encoding": "utf-8",
                "text_encoding": "hex-zstd-utf8",
                "max_decompressed_text_bytes": 1024 * 1024,
                "discard_truncated_zstd_frame_size_bytes": 65535,
                "text_decode_error_policy": "discard",
                "columns": list(_EXPECTED_COLUMNS),
                "max_field_size_bytes": 1024 * 1024,
                "checkpoint_interval_records": checkpoint_interval,
                "fingerprint_sample_bytes": 64,
            },
            "filters": {
                "status": "done",
                "require_text": True,
                "require_text_md5": True,
                "text_md5_policy": "count_mismatch",
                "exclude_same_as": True,
                "same_as_null_values": ["", "NULL"],
                "page_filter": {
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
                },
                "line_filter": {
                    "minimum_characters": minimum_line_characters,
                },
                "boilerplate": {
                    "schema_version": 4,
                    "decision_by_profile": {
                        "smoke": "remove_exact",
                        "mvp": decision or "pending",
                    },
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
                },
            },
        },
        "profile": {
            "candidate_documents": candidate_documents,
            "selection": (
                "engineering_prefix" if profile_name == "smoke" else "representative"
            ),
        },
    }


def record(
    index: int,
    text: str,
    *,
    domain: str = "domain-a",
    status: str = "done",
    text_md5: str | None = None,
    same_as: str = "",
    title: str = "",
    url: str = "",
    url_final: str = "",
) -> dict[str, str]:
    raw_text = text.encode("utf-8")
    encoded_text = (
        zstd.ZstdCompressor(level=1).compress(raw_text).hex().upper()
        if raw_text
        else ""
    )
    default_md5 = hashlib.md5(raw_text, usedforsecurity=False).hexdigest()
    value = {column: "" for column in _EXPECTED_COLUMNS}
    value.update(
        {
            "id": f"private-record-{index}",
            "domain_id": domain,
            "same_as": same_as,
            "title": title,
            "url": url,
            "url_final": url_final,
            "url_md5": f"url-digest-{index}",
            "status": status,
            "text": encoded_text,
            "text_md5": text_md5 if text_md5 is not None else default_md5,
        }
    )
    return value


def write_tsv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(_EXPECTED_COLUMNS) + "\n")
        for item in records:
            values = [item[column] for column in _EXPECTED_COLUMNS]
            if any("\t" in value or "\n" in value or "\r" in value for value in values):
                raise ValueError("synthetic TSV values must fit on one physical line")
            stream.write("\t".join(values) + "\n")


class WackyWackyAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "wacky" / "pages.tsv"
        self.environment = patch.dict(
            os.environ, {"TEST_PTBR_DATASET_ROOT": str(self.root)}
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_smoke_uses_filtered_prefix_and_does_not_expose_record_ids(self) -> None:
        write_tsv(
            self.source,
            [
                record(1, "não elegível", status="queued"),
                record(2, "", text_md5="present"),
                record(3, "sem hash", text_md5=""),
                record(4, "duplicado", same_as="canonical-record"),
                record(5, "Texto elegível A", same_as="NULL"),
                record(6, '"Texto elegível B" seguido'),
                record(7, "Não deve ser lido"),
            ],
        )

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=2)
        )

        self.assertEqual(
            [document.text for document in result.documents],
            ["Texto elegível A", '"Texto elegível B" seguido'],
        )
        self.assertEqual(result.metrics["rows_seen"], 6)
        self.assertEqual(result.metrics["filtered_status"], 1)
        self.assertEqual(result.metrics["filtered_missing_text"], 1)
        self.assertEqual(result.metrics["filtered_missing_text_md5"], 1)
        self.assertEqual(result.metrics["filtered_same_as"], 1)
        self.assertFalse(result.source_fingerprint["complete_source_scan"])
        report = result.extra_reports["boilerplate_report"]
        self.assertEqual(report["schema_version"], "queroquero-boilerplate-report/v2")
        self.assertEqual(report["decision"], "remove_exact")
        self.assertFalse(report["finalization_blocked"])
        for document in result.documents:
            self.assertNotIn("private-record", document.source_ref)
            self.assertNotIn("domain-a", repr(document.metadata))

    def test_discards_text_decode_errors_by_reason_and_continues(self) -> None:
        invalid_hex = record(1, "Texto")
        invalid_hex["text"] = "não-hexadecimal"
        invalid_zstd = record(2, "Texto")
        invalid_zstd["text"] = b"not-zstd".hex()
        invalid_md5 = record(3, "Texto", text_md5="not-an-md5")

        cases = (
            (invalid_hex, "filtered_invalid_text_hex"),
            (invalid_zstd, "filtered_invalid_zstd_frames"),
            (invalid_md5, "filtered_invalid_text_md5"),
        )
        for invalid_record, expected_metric in cases:
            with self.subTest(metric=expected_metric):
                write_tsv(
                    self.source,
                    [invalid_record, record(10, "registro sintético válido")],
                )
                result = WackyWackyAdapter().scan(
                    resolved_config(candidate_documents=1)
                )
                self.assertEqual(
                    [document.text for document in result.documents],
                    ["registro sintético válido"],
                )
                self.assertEqual(result.metrics["rows_seen"], 2)
                self.assertEqual(result.metrics[expected_metric], 1)

        mismatched_md5 = record(4, "Texto")
        mismatched_md5["text_md5"] = "0" * 32
        write_tsv(self.source, [mismatched_md5])
        result = WackyWackyAdapter().scan(resolved_config(candidate_documents=1))
        self.assertEqual([document.text for document in result.documents], ["Texto"])
        self.assertEqual(result.metrics["text_md5_mismatches"], 1)

    def test_discards_and_counts_non_utf8_decompressed_text(self) -> None:
        invalid_utf8_bytes = b"prefixo-sintetico\xffsufixo-sintetico"
        invalid_utf8 = record(1, "conteúdo sintético substituído")
        invalid_utf8["text"] = (
            zstd.ZstdCompressor(level=1)
            .compress(invalid_utf8_bytes)
            .hex()
        )
        invalid_utf8["text_md5"] = hashlib.md5(
            invalid_utf8_bytes, usedforsecurity=False
        ).hexdigest()
        write_tsv(
            self.source,
            [invalid_utf8, record(2, "registro sintético válido")],
        )

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=1)
        )

        self.assertEqual(
            [document.text for document in result.documents],
            ["registro sintético válido"],
        )
        self.assertEqual(result.metrics["rows_seen"], 2)
        self.assertEqual(
            result.metrics["filtered_non_utf8_decompressed_texts"],
            1,
        )
        self.assertEqual(result.metrics["filtered_corrupt_zstd_frames"], 0)
        self.assertEqual(
            result.metrics["filtered_truncated_zstd_frames_65535_bytes"],
            0,
        )

    def test_discards_and_counts_configured_truncated_zstd_frame(self) -> None:
        raw_bytes = random.Random(42).randbytes(100_000)
        compressed = zstd.ZstdCompressor(level=1).compress(raw_bytes)
        self.assertGreater(len(compressed), 65_535)

        truncated = record(1, "conteúdo sintético substituído")
        truncated["text"] = compressed[:65_535].hex()
        truncated["text_md5"] = hashlib.md5(
            raw_bytes, usedforsecurity=False
        ).hexdigest()
        write_tsv(
            self.source,
            [truncated, record(2, "registro sintético válido")],
        )

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=1)
        )

        self.assertEqual(
            [document.text for document in result.documents],
            ["registro sintético válido"],
        )
        self.assertEqual(result.metrics["rows_seen"], 2)
        self.assertEqual(
            result.metrics["filtered_truncated_zstd_frames_65535_bytes"],
            1,
        )
        self.assertEqual(result.metrics["filtered_corrupt_zstd_frames"], 0)

    def test_discards_and_counts_other_corrupt_zstd_frame(self) -> None:
        raw_bytes = b"A" * 1_367
        compressed = zstd.ZstdCompressor(level=1).compress(raw_bytes)
        corrupt = compressed[:-1]
        self.assertNotEqual(len(corrupt), 65_535)
        self.assertEqual(zstd.frame_content_size(corrupt), len(raw_bytes))

        corrupt_record = record(1, "conteúdo sintético substituído")
        corrupt_record["text"] = corrupt.hex()
        corrupt_record["text_md5"] = hashlib.md5(
            raw_bytes, usedforsecurity=False
        ).hexdigest()
        write_tsv(
            self.source,
            [corrupt_record, record(2, "registro sintético válido")],
        )

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=1)
        )

        self.assertEqual(
            [document.text for document in result.documents],
            ["registro sintético válido"],
        )
        self.assertEqual(result.metrics["rows_seen"], 2)
        self.assertEqual(result.metrics["filtered_corrupt_zstd_frames"], 1)
        self.assertEqual(
            result.metrics["filtered_truncated_zstd_frames_65535_bytes"],
            0,
        )

    def test_discards_decompressed_text_above_configured_limit(self) -> None:
        write_tsv(
            self.source,
            [record(1, "A" * 1024), record(2, "registro sintético válido")],
        )
        config = resolved_config(candidate_documents=1)
        config["dataset"]["source"]["max_decompressed_text_bytes"] = 128

        result = WackyWackyAdapter().scan(config)

        self.assertEqual(
            [document.text for document in result.documents],
            ["registro sintético válido"],
        )
        self.assertEqual(result.metrics["filtered_oversized_decompressed_texts"], 1)

    def test_structural_row_errors_and_programming_errors_remain_fatal(self) -> None:
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_bytes(
            ("\t".join(_EXPECTED_COLUMNS) + "\n").encode("utf-8")
            + b"only-two\tfields\n"
        )
        with self.assertRaisesRegex(ValueError, "does not have 19 columns"):
            WackyWackyAdapter().scan(resolved_config(candidate_documents=1))

        write_tsv(self.source, [record(1, "registro sintético válido")])
        with patch(
            "queroquero.datasets.wackywacky._decode_text",
            side_effect=RuntimeError("synthetic programming error"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic programming error"):
                WackyWackyAdapter().scan(resolved_config(candidate_documents=1))

    def test_filters_search_and_listing_pages_without_exposing_page_values(self) -> None:
        rows = [
            record(
                1,
                "conteúdo de busca que não deve ser selecionado",
                title="Resultados da pesquisa sintética",
            ),
            record(
                2,
                "conteúdo de listagem que não deve ser selecionado",
                url="https://example.invalid/category/synthetic-topic",
            ),
            record(
                3,
                "outro conteúdo de busca que não deve ser selecionado",
                url_final=(
                    "https://example.invalid/index.php?search=synthetic"
                    "&title=Especial%3APesquisar"
                ),
            ),
            record(
                4,
                "Texto editorial sintético preservado",
                title="Estudo sobre resultados da pesquisa sintética",
                url="https://example.invalid/articles/synthetic-topic",
            ),
        ]
        write_tsv(self.source, rows)

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=1)
        )

        self.assertEqual(
            [document.text for document in result.documents],
            ["Texto editorial sintético preservado"],
        )
        self.assertEqual(result.metrics["filtered_page_search"], 2)
        self.assertEqual(result.metrics["filtered_page_listing"], 1)
        serialized = json.dumps(
            {
                "metrics": result.metrics,
                "cursor": result.cursor,
                "report": result.extra_reports,
            },
            ensure_ascii=False,
        )
        self.assertNotIn("example.invalid", serialized)
        self.assertNotIn("Resultados da pesquisa sintética", serialized)

    def test_mvp_scans_the_full_source_and_pending_blocks_finalization(self) -> None:
        rows = [record(index, f"Documento sintético {index}") for index in range(12)]
        write_tsv(self.source, rows)
        config = resolved_config(profile_name="mvp", candidate_documents=4)

        first = WackyWackyAdapter().scan(config)
        second = WackyWackyAdapter().scan(config)

        self.assertEqual(first.documents, second.documents)
        self.assertEqual(first.metrics["rows_seen"], len(rows))
        self.assertEqual(first.metrics["selected_documents"], 4)
        self.assertTrue(first.source_fingerprint["complete_source_scan"])
        self.assertEqual(
            first.source_fingerprint["method"], "streamed-full-sha256-v1"
        )
        self.assertTrue(first.cursor["finalization_blocked"])
        report = first.extra_reports["boilerplate_report"]
        self.assertTrue(report["finalization_blocked"])
        self.assertFalse(report["contains_examples"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("Documento sintético", serialized)
        self.assertNotIn("private-record", serialized)

    def test_byte_identical_copies_have_identical_fingerprint_and_cursor(self) -> None:
        rows = [record(index, f"Documento sintético {index}") for index in range(8)]
        write_tsv(self.source, rows)
        config = resolved_config(profile_name="mvp", candidate_documents=4)
        first = WackyWackyAdapter().scan(config)

        copied_root = self.root / "copied-root"
        copied_source = copied_root / "wacky" / "pages.tsv"
        copied_source.parent.mkdir(parents=True)
        shutil.copyfile(self.source, copied_source)
        with patch.dict(
            os.environ, {"TEST_PTBR_DATASET_ROOT": str(copied_root)}
        ):
            copied = WackyWackyAdapter().scan(config)

        self.assertEqual(first.source_fingerprint, copied.source_fingerprint)
        self.assertEqual(first.cursor, copied.cursor)
        self.assertEqual(first.documents, copied.documents)

    def test_boilerplate_keep_and_remove_exact(self) -> None:
        repeated = (
            "Trecho institucional repetido de forma exata e suficientemente "
            "longo para a regra automática de boilerplate."
        )
        rows = [
            record(
                index,
                f"<p>{repeated}</p><p>final-{index}-{'C' * 320}</p>",
                domain=f"domain-{index % 3}",
            )
            for index in range(6)
        ]
        write_tsv(self.source, rows)

        kept = WackyWackyAdapter().scan(
            resolved_config(
                profile_name="mvp",
                candidate_documents=10,
                decision="keep",
            )
        )
        self.assertFalse(kept.cursor["finalization_blocked"])
        self.assertEqual(
            kept.extra_reports["boilerplate_report"]["analysis"][
                "cross_domain_paragraphs_repeated"
            ],
            1,
        )
        self.assertTrue(all(repeated in document.text for document in kept.documents))

        removed = WackyWackyAdapter().scan(
            resolved_config(
                profile_name="mvp",
                candidate_documents=10,
                decision="remove_exact",
            )
        )
        report = removed.extra_reports["boilerplate_report"]
        self.assertFalse(removed.cursor["finalization_blocked"])
        self.assertEqual(report["simulation"]["affected_documents"], 6)
        self.assertEqual(
            report["analysis"]["matching_cross_domain_paragraph_occurrences"], 6
        )
        self.assertEqual(len(removed.documents), 6)
        self.assertTrue(
            all(document.text.startswith("final-") for document in removed.documents)
        )

        resumed = WackyWackyAdapter().scan(
            resolved_config(
                profile_name="mvp",
                candidate_documents=10,
                decision="remove_exact",
            ),
            resume_cursor=removed.cursor,
            resume_documents=removed.documents,
        )
        self.assertEqual(resumed.documents, removed.documents)
        self.assertEqual(resumed.extra_reports, removed.extra_reports)

    def test_smoke_removes_three_line_blocks_repeated_within_one_domain(self) -> None:
        menu_lines = (
            "Navegação principal do portal sintético",
            "Serviços e informações institucionais sintéticas",
            "Atendimento e canais oficiais sintéticos",
        )
        menu = "\n".join(menu_lines)
        rows = [
            record(
                index,
                f"cabeçalho-{index}-{'A' * 320}\n\n{menu}\n\nrodapé-{index}-{'B' * 320}",
                domain="synthetic-domain",
            )
            for index in range(5)
        ]
        write_tsv(self.source, rows)

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=len(rows))
        )

        report = result.extra_reports["boilerplate_report"]
        self.assertEqual(report["profile"], "smoke")
        self.assertEqual(report["decision"], "remove_exact")
        self.assertFalse(report["finalization_blocked"])
        self.assertEqual(report["analysis"]["within_domain_blocks_repeated"], 1)
        self.assertEqual(report["simulation"]["affected_documents"], 5)
        self.assertEqual(report["applied"], report["simulation"])
        self.assertEqual(len(result.documents), 5)
        self.assertTrue(all(menu not in document.text for document in result.documents))
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(menu_lines[0], serialized)
        self.assertNotIn("synthetic-domain", serialized)

    def test_within_domain_block_stays_below_distinct_document_threshold(self) -> None:
        menu = "\n".join(
            (
                "Primeira linha suficientemente longa do menu sintético",
                "Segunda linha suficientemente longa do menu sintético",
                "Terceira linha suficientemente longa do menu sintético",
            )
        )
        rows = [
            record(
                index,
                (
                    f"conteúdo-{index}-{'A' * 320}\n\n{menu}"
                    + (f"\n\n{menu}" if index == 0 else "")
                ),
                domain="synthetic-domain",
            )
            for index in range(4)
        ]
        write_tsv(self.source, rows)

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=len(rows))
        )

        report = result.extra_reports["boilerplate_report"]
        self.assertEqual(report["analysis"]["within_domain_blocks_repeated"], 0)
        self.assertEqual(report["simulation"]["affected_documents"], 0)
        self.assertTrue(all(menu in document.text for document in result.documents))

    def test_overlapping_blocks_are_removed_once_and_order_is_preserved(self) -> None:
        common_lines = tuple(
            f"linha-{index}-{'M' * 60}" for index in range(4)
        )
        common = "\n".join(common_lines)
        rows = [
            record(
                index,
                f"antes-{index}-{'A' * 320}\n\n{common}\n\ndepois-{index}-{'B' * 320}",
                domain="synthetic-domain",
            )
            for index in range(5)
        ]
        write_tsv(self.source, rows)

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=len(rows))
        )

        report = result.extra_reports["boilerplate_report"]
        self.assertEqual(report["analysis"]["within_domain_blocks_repeated"], 2)
        self.assertEqual(
            report["analysis"]["matching_within_domain_block_occurrences"], 10
        )
        self.assertEqual(
            report["applied"]["removed_characters"],
            len(rows) * (len(common) + 2),
        )
        for index, document in enumerate(result.documents):
            self.assertEqual(
                document.text,
                f"antes-{index}-{'A' * 320}\n\ndepois-{index}-{'B' * 320}",
            )

    def test_global_line_filter_removes_each_normalized_line_below_40(self) -> None:
        original_lines = [
            "A" * 39,
            "B" * 40,
            "C" * 41,
            "título sintético curto",
            "D" * 300,
        ]
        write_tsv(self.source, [record(1, "\n".join(original_lines))])

        result = WackyWackyAdapter().scan(
            resolved_config(
                candidate_documents=1,
                minimum_line_characters=40,
            )
        )

        expected_lines = ["B" * 40, "C" * 41, "D" * 300]
        self.assertEqual(
            [document.text for document in result.documents],
            ["\n".join(expected_lines)],
        )
        self.assertTrue(
            all(
                len(line) >= 40
                for line in result.documents[0].text.splitlines()
                if line
            )
        )
        line_filter = result.extra_reports["boilerplate_report"]["line_filter"]
        self.assertEqual(line_filter["minimum_characters"], 40)
        self.assertEqual(line_filter["lines_considered"], 5)
        self.assertEqual(line_filter["lines_removed"], 2)
        self.assertEqual(line_filter["documents_remaining"], 1)

    def test_remove_exact_discards_short_and_over_removed_documents(self) -> None:
        short_block = "\n".join(
            (
                "linha curta repetida com tamanho suficiente um",
                "linha curta repetida com tamanho suficiente dois",
                "linha curta repetida com tamanho suficiente três",
            )
        )
        large_block = "\n".join(
            (
                "X" * 520,
                "Y" * 520,
                "Z" * 520,
            )
        )
        rows = [
            record(
                index,
                f"{short_block}\n\nrestante-{index}-{'A' * 240}",
                domain="short-domain",
            )
            for index in range(5)
        ]
        rows.extend(
            record(
                index + 5,
                f"{large_block}\n\nrestante-{index}-{'B' * 340}",
                domain="fraction-domain",
            )
            for index in range(5)
        )
        write_tsv(self.source, rows)

        result = WackyWackyAdapter().scan(
            resolved_config(candidate_documents=len(rows))
        )

        applied = result.extra_reports["boilerplate_report"]["applied"]
        self.assertEqual(
            applied["documents_discarded_minimum_remaining_characters"], 5
        )
        self.assertEqual(
            applied["documents_discarded_maximum_removed_fraction"], 5
        )
        self.assertEqual(applied["documents_discarded_total"], 10)
        self.assertEqual(applied["documents_remaining"], 0)
        self.assertEqual(result.documents, [])

    def test_pending_candidates_can_be_reused_after_review(self) -> None:
        repeated = "B" * 90
        rows = [
            record(
                index,
                f"<p>{repeated}</p><p>conteúdo-{index}-{'C' * 320}</p>",
                domain=f"domain-{index % 3}",
            )
            for index in range(6)
        ]
        write_tsv(self.source, rows)
        pending = WackyWackyAdapter().scan(
            resolved_config(profile_name="mvp", candidate_documents=10)
        )

        with patch(
            "queroquero.datasets.wackywacky._hash_prefix",
            side_effect=AssertionError("completed scan must not be read again"),
        ):
            reviewed = WackyWackyAdapter().scan(
                resolved_config(
                    profile_name="mvp",
                    candidate_documents=10,
                    decision="remove_exact",
                ),
                resume_cursor=pending.resume_cursor,
                resume_documents=pending.documents,
            )

        self.assertFalse(reviewed.cursor["finalization_blocked"])
        self.assertEqual(
            reviewed.extra_reports["boilerplate_report"]["analysis"][
                "matching_cross_domain_paragraph_occurrences"
            ],
            6,
        )
        self.assertTrue(
            all(document.text.startswith("conteúdo-") for document in reviewed.documents)
        )

    def test_completed_resume_rehashes_when_local_stat_changes(self) -> None:
        rows = [
            record(index, "A" * 300 + f"-{index}-" + "B" * 300)
            for index in range(20)
        ]
        write_tsv(self.source, rows)
        config = resolved_config(profile_name="mvp", candidate_documents=10)
        pending = WackyWackyAdapter().scan(config)
        before_guard = _sampled_source_guard(self.source, 64)

        data = bytearray(self.source.read_bytes())
        maximum_start = max(0, len(data) - 64)
        sample_positions = {
            0,
            min(maximum_start, len(data) // 4),
            min(maximum_start, len(data) // 2),
            min(maximum_start, (len(data) * 3) // 4),
            maximum_start,
        }
        sampled_indexes = {
            index
            for start in sample_positions
            for index in range(start, min(start + 64, len(data)))
        }
        changed_index = next(
            index
            for index, value in enumerate(data)
            if value == ord("A") and index not in sampled_indexes
        )
        data[changed_index] = ord("C")
        self.source.write_bytes(data)
        after_guard = _sampled_source_guard(self.source, 64)
        self.assertEqual(
            before_guard["resume_guard_sha256"],
            after_guard["resume_guard_sha256"],
        )
        self.assertNotEqual(
            before_guard["local_source_stat_sha256"],
            after_guard["local_source_stat_sha256"],
        )

        with self.assertRaisesRegex(ConfigError, "prefix changed"):
            WackyWackyAdapter().scan(
                resolved_config(
                    profile_name="mvp",
                    candidate_documents=10,
                    decision="keep",
                ),
                resume_cursor=pending.resume_cursor,
                resume_documents=pending.documents,
            )

    def test_resume_from_byte_cursor_and_reject_changed_prefix(self) -> None:
        rows = [record(index, f"Texto {index}") for index in range(5)]
        write_tsv(self.source, rows)
        config = resolved_config(candidate_documents=3, checkpoint_interval=1)
        saved: dict[str, Any] = {}

        class Interrupted(Exception):
            pass

        def stop_after_first(cursor: dict[str, Any], documents: list[Any]) -> None:
            saved["cursor"] = cursor
            saved["documents"] = documents
            raise Interrupted

        with self.assertRaises(Interrupted):
            WackyWackyAdapter().scan(config, checkpoint=stop_after_first)

        resumed = WackyWackyAdapter().scan(
            config,
            resume_cursor=saved["cursor"],
            resume_documents=saved["documents"],
        )
        uninterrupted = WackyWackyAdapter().scan(config)
        self.assertEqual(resumed.documents, uninterrupted.documents)
        self.assertEqual(resumed.metrics, uninterrupted.metrics)

        changed = copy.deepcopy(rows)
        changed[0]["text"] = "Texto alterado"
        write_tsv(self.source, changed)
        with self.assertRaisesRegex(ConfigError, "source changed|prefix changed"):
            WackyWackyAdapter().scan(
                config,
                resume_cursor=saved["cursor"],
                resume_documents=saved["documents"],
            )

    def test_rejects_non_19_column_header_without_reading_values(self) -> None:
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_text("first\tsecond\nvalue\tvalue\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "19-column schema"):
            WackyWackyAdapter().scan(resolved_config())


if __name__ == "__main__":
    unittest.main()
