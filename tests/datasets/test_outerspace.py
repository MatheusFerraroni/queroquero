from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from queroquero.datasets._conversation_zip import ConversationZipFormatError
from queroquero.datasets.base import Document
from queroquero.datasets.outerspace import OuterSpaceAdapter


TEST_ROOT_ENV = "QUEROQUERO_TEST_OUTERSPACE_ROOT"


def resolved_config(
    *, candidate_documents: int = 32, checkpoint_every: int = 64
) -> dict[str, Any]:
    return {
        "preparation": {"seed": 42},
        "dataset": {
            "dataset_id": "outerspace",
            "source": {
                "root_env": TEST_ROOT_ENV,
                "archives_by_profile": {
                    "smoke": "outerspace/conversations_min.zip",
                    "mvp": "outerspace/conversations.zip",
                },
                "format": "zip-tsv",
                "member_pattern": "clear_threads/*.tsv",
                "encoding": "utf-8",
                "columns": ["timestamp", "user", "text"],
                "has_header": False,
                "checkpoint_every": checkpoint_every,
            },
            "filters": {
                "strict_three_columns": True,
                "anonymize_participants": True,
                "omit_timestamps": True,
                "strip_html": True,
            },
        },
        "profile_name": "smoke",
        "profile": {
            "candidate_documents": candidate_documents,
            "selection": "representative",
        },
    }


def write_archive(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("clear_threads/", "")
        for name, value in members.items():
            archive.writestr(name, value)
        archive.writestr("ignored/metadata.txt", "synthetic metadata")


class _PauseScan(Exception):
    pass


class OuterSpaceAdapterTests(unittest.TestCase):
    def test_reads_tsv_in_place_and_anonymizes_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "outerspace" / "conversations_min.zip"
            write_archive(
                archive,
                {
                    "clear_threads/thread-a.tsv": (
                        "instant-a\tactor-alpha\t<p>Mensagem sintética um</p>\n"
                        'instant-b\tactor-beta\tMensagem "sintética" dois\n'
                        "instant-c\tactor-alpha\tMensagem sintética três\n"
                    ),
                    "clear_threads/thread-b.tsv": (
                        "instant-d\tactor-gamma\tOutra mensagem sintética\n"
                    ),
                },
            )
            config = resolved_config()
            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                result = OuterSpaceAdapter().scan(config)

            self.assertFalse((root / "clear_threads").exists())

        self.assertEqual(result.metrics["source_members_eligible"], 2)
        self.assertEqual(result.metrics["documents_emitted"], 2)
        combined = "\n".join(document.text for document in result.documents)
        self.assertIn("Participante 1: Mensagem sintética um", combined)
        self.assertIn('Participante 2: Mensagem "sintética" dois', combined)
        self.assertNotIn("actor-", combined)
        self.assertNotIn("instant-", combined)
        self.assertNotIn("<p>", combined)
        self.assertNotIn("actor-", repr(result.source_fingerprint))
        self.assertNotIn("thread-", repr(result.source_fingerprint))
        for document in result.documents:
            self.assertEqual(document.metadata["document_type"], "conversation")
            self.assertNotIn("timestamp", repr(document))

    def test_resumes_selected_members_and_detects_any_archive_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "outerspace" / "conversations_min.zip"
            members = {
                f"clear_threads/synthetic-{index}.tsv": (
                    f"instant-{index}\tactor-{index}\tConteúdo sintético {index}\n"
                )
                for index in range(6)
            }
            write_archive(archive, members)
            config = resolved_config(candidate_documents=4, checkpoint_every=1)
            saved: tuple[dict[str, Any], list[Document]] | None = None

            def stop(cursor: dict[str, Any], documents: list[Document]) -> None:
                nonlocal saved
                if cursor["next_member_index"] == 2:
                    saved = (dict(cursor), list(documents))
                    raise _PauseScan

            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                with self.assertRaises(_PauseScan):
                    OuterSpaceAdapter().scan(config, checkpoint=stop)
                assert saved is not None
                resumed = OuterSpaceAdapter().scan(
                    config,
                    resume_cursor=saved[0],
                    resume_documents=saved[1],
                )
                complete = OuterSpaceAdapter().scan(config)
                self.assertEqual(resumed.documents, complete.documents)
                self.assertEqual(resumed.metrics, complete.metrics)

                members["ignored/changed.txt"] = "aggregate-only-change"
                write_archive(archive, members)
                with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
                    OuterSpaceAdapter().scan(
                        config,
                        resume_cursor=saved[0],
                        resume_documents=saved[1],
                    )

    def test_rejects_rows_that_do_not_have_exactly_three_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_archive(
                root / "outerspace" / "conversations_min.zip",
                {"clear_threads/synthetic.tsv": "instant\tactor\n"},
            )
            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                with self.assertRaises(ConversationZipFormatError) as raised:
                    OuterSpaceAdapter().scan(resolved_config())

        self.assertNotIn("actor", str(raised.exception))
        self.assertNotIn("synthetic.tsv", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
