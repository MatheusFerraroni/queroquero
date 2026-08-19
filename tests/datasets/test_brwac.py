from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from queroquero.config import ConfigError
from queroquero.datasets.base import stable_hash
from queroquero.datasets.brwac import BrwacAdapter


def resolved_config(candidate_documents: int = 3, checkpoint_interval: int = 64) -> dict[str, Any]:
    return {
        "preparation": {"seed": 42},
        "dataset": {
            "source": {
                "root_env": "TEST_PTBR_DATASET_ROOT",
                "path": "brwac/source.zip",
                "format": "zip",
                "member_glob": "data/*.txt",
                "encoding": "utf-8",
                "checkpoint_interval_documents": checkpoint_interval,
            },
            "filters": {"strict_utf8": True},
        },
        "profile": {
            "candidate_documents": candidate_documents,
            "selection": "representative",
        },
    }


def write_archive(path: Path, documents: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, value in documents.items():
            archive.writestr(member, value)


class BrwacAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "brwac" / "source.zip"
        self.environment = patch.dict(
            os.environ, {"TEST_PTBR_DATASET_ROOT": str(self.root)}
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_reads_only_direct_data_txt_members_in_hash_rank_order(self) -> None:
        members = {
            "data/alpha.txt": "Documento alfa".encode(),
            "data/beta.txt": "Documento beta".encode(),
            "data/gamma.txt": "Documento gama".encode(),
            "data/nested/ignored.txt": b"nested",
            "names.tsv": b"metadata",
            "other.txt": b"outside",
        }
        write_archive(self.archive, members)

        result = BrwacAdapter().scan(resolved_config(candidate_documents=2))

        with zipfile.ZipFile(self.archive) as archive:
            infos = {
                info.filename: info
                for info in archive.infolist()
                if info.filename.startswith("data/")
                and info.filename.count("/") == 1
                and info.filename.endswith(".txt")
            }
        expected = sorted(
            infos,
            key=lambda name: (
                stable_hash(42, "brwac", name),
                name,
            ),
        )[:2]

        self.assertEqual(
            [document.source_position["member_path"] for document in result.documents],
            expected,
        )
        self.assertEqual(result.metrics["eligible_members"], 3)
        self.assertEqual(result.metrics["selected_documents"], 2)
        self.assertEqual(
            result.source_fingerprint["method"], "zip-central-directory-v1"
        )
        self.assertEqual(len(result.source_fingerprint["sha256"]), 64)
        self.assertFalse((self.root / "data").exists())
        self.assertNotIn("metadata", [document.text for document in result.documents])
        self.assertNotIn("nested", [document.text for document in result.documents])

    def test_resume_uses_selection_index_and_rejects_changed_archive(self) -> None:
        write_archive(
            self.archive,
            {
                "data/one.txt": b"Primeiro",
                "data/two.txt": b"Segundo",
                "data/three.txt": b"Terceiro",
            },
        )
        saved: dict[str, Any] = {}

        class Interrupted(Exception):
            pass

        def stop_after_first(cursor: dict[str, Any], documents: list[Any]) -> None:
            saved["cursor"] = cursor
            saved["documents"] = documents
            raise Interrupted

        with self.assertRaises(Interrupted):
            BrwacAdapter().scan(
                resolved_config(candidate_documents=3, checkpoint_interval=1),
                checkpoint=stop_after_first,
            )

        resumed = BrwacAdapter().scan(
            resolved_config(candidate_documents=3, checkpoint_interval=1),
            resume_cursor=saved["cursor"],
            resume_documents=saved["documents"],
        )
        uninterrupted = BrwacAdapter().scan(resolved_config(candidate_documents=3))
        self.assertEqual(resumed.documents, uninterrupted.documents)
        self.assertTrue(resumed.cursor["complete"])

        write_archive(
            self.archive,
            {
                "data/one.txt": b"Alterado",
                "data/two.txt": b"Segundo",
                "data/three.txt": b"Terceiro",
            },
        )
        with self.assertRaisesRegex(ConfigError, "source changed"):
            BrwacAdapter().scan(
                resolved_config(candidate_documents=3),
                resume_cursor=saved["cursor"],
                resume_documents=saved["documents"],
            )

    def test_fails_closed_on_invalid_utf8(self) -> None:
        write_archive(self.archive, {"data/invalid.txt": b"\xff\xfe"})

        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            BrwacAdapter().scan(resolved_config(candidate_documents=1))


if __name__ == "__main__":
    unittest.main()
