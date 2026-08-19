from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from queroquero.datasets.adrenaline import AdrenalineAdapter


TEST_ROOT_ENV = "QUEROQUERO_TEST_ADRENALINE_ROOT"


def resolved_config(profile_name: str = "smoke") -> dict[str, Any]:
    return {
        "preparation": {"seed": 42},
        "dataset": {
            "dataset_id": "adrenaline",
            "source": {
                "root_env": TEST_ROOT_ENV,
                "archives_by_profile": {
                    "smoke": "adrenaline/conversations_min.zip",
                    "mvp": "adrenaline/conversations.zip",
                },
                "format": "zip-tsv",
                "member_pattern": "clear_threads/*.tsv",
                "encoding": "utf-8",
                "columns": ["timestamp", "user", "text"],
                "has_header": False,
                "checkpoint_every": 1,
            },
            "filters": {
                "strict_three_columns": True,
                "anonymize_participants": True,
                "omit_timestamps": True,
                "strip_html": True,
            },
        },
        "profile_name": profile_name,
        "profile": {"candidate_documents": 8, "selection": "representative"},
    }


def write_archive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "clear_threads/synthetic.tsv",
            f"synthetic-instant\tsynthetic-actor\t{text}\n",
        )


class AdrenalineAdapterTests(unittest.TestCase):
    def test_uses_minimum_archive_for_smoke_and_full_archive_for_mvp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_archive(
                root / "adrenaline" / "conversations_min.zip",
                "Conteúdo sintético smoke",
            )
            write_archive(
                root / "adrenaline" / "conversations.zip",
                "Conteúdo sintético mvp",
            )

            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                smoke = AdrenalineAdapter().scan(resolved_config("smoke"))
                mvp = AdrenalineAdapter().scan(resolved_config("mvp"))

        self.assertEqual(smoke.source_fingerprint["archive"], "adrenaline/conversations_min.zip")
        self.assertEqual(mvp.source_fingerprint["archive"], "adrenaline/conversations.zip")
        self.assertIn("Conteúdo sintético smoke", smoke.documents[0].text)
        self.assertIn("Conteúdo sintético mvp", mvp.documents[0].text)
        self.assertNotIn("synthetic-actor", repr(smoke.documents))
        self.assertNotIn("synthetic-instant", repr(smoke.documents))
        self.assertEqual(smoke.documents[0].text.split(":", 1)[0], "Participante 1")


if __name__ == "__main__":
    unittest.main()
