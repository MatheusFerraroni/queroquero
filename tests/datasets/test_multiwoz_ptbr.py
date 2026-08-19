from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from queroquero.datasets.base import Document, stable_hash
from queroquero.datasets.multiwoz_ptbr import (
    MultiWOZFormatError,
    MultiWOZPTBRAdapter,
)


TEST_ROOT_ENV = "QUEROQUERO_TEST_MULTIWOZ_ROOT"


def resolved_config(
    *, candidate_documents: int = 32, checkpoint_every: int = 128
) -> dict[str, Any]:
    return {
        "preparation": {"seed": 42},
        "dataset": {
            "dataset_id": "multiwoz_ptbr",
            "source": {
                "root_env": TEST_ROOT_ENV,
                "relative_directory": "multiwozptbr",
                "format": "json-array",
                "file_pattern": "dialogues_*.json",
                "expected_files": 17,
                "encoding": "utf-8",
                "checkpoint_every": checkpoint_every,
            },
            "filters": {
                "projection": "utterances_only",
                "speaker_labels": {"USER": "Usuário", "SYSTEM": "Assistente"},
                "exclude_frames_slots_states": True,
            },
        },
        "profile_name": "smoke",
        "profile": {
            "candidate_documents": candidate_documents,
            "selection": "representative",
        },
    }


def write_source(root: Path) -> None:
    source = root / "multiwozptbr"
    source.mkdir()
    for file_number in range(1, 18):
        dialogue = {
            "dialogue_id": f"synthetic-dialogue-{file_number}",
            "services": ["synthetic-service"],
            "turns": [
                {
                    "speaker": "USER",
                    "turn_id": "synthetic-turn-a",
                    "utterance": f"<p>Pergunta sintética {file_number}</p>",
                    "frames": [{"state": "STRUCTURED_ONLY_MARKER"}],
                },
                {
                    "speaker": "SYSTEM",
                    "turn_id": "synthetic-turn-b",
                    "utterance": f"Resposta sintética {file_number}",
                    "frames": [{"slots": ["STRUCTURED_ONLY_MARKER"]}],
                },
            ],
        }
        path = source / f"dialogues_{file_number:03d}.json"
        path.write_text(json.dumps([dialogue], ensure_ascii=False), encoding="utf-8")


class _PauseScan(Exception):
    pass


class MultiWOZPTBRAdapterTests(unittest.TestCase):
    def test_projects_only_utterances_and_hash_ranks_dialogues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root)
            config = resolved_config(candidate_documents=5)
            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                first = MultiWOZPTBRAdapter().scan(config)
                second = MultiWOZPTBRAdapter().scan(config)

        self.assertEqual(len(first.documents), 5)
        self.assertEqual(first.documents, second.documents)
        self.assertEqual(first.source_fingerprint, second.source_fingerprint)
        self.assertEqual(first.source_fingerprint["file_count"], 17)
        self.assertEqual(len(first.source_fingerprint["files"]), 17)
        self.assertEqual(
            [document.source_ref for document in first.documents],
            sorted(
                [
                    f"multiwozptbr/dialogues_{index:03d}.json#dialogue_index=0"
                    for index in range(1, 18)
                ],
                key=lambda source_ref: (
                    stable_hash(42, "multiwoz_ptbr", source_ref),
                    source_ref,
                ),
            )[:5],
        )
        for document in first.documents:
            self.assertIn("Usuário: Pergunta sintética", document.text)
            self.assertIn("Assistente: Resposta sintética", document.text)
            self.assertNotIn("STRUCTURED_ONLY_MARKER", repr(document))
            self.assertNotIn("synthetic-dialogue", repr(document))

    def test_resumes_at_dialogue_boundary_and_rejects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root)
            config = resolved_config(candidate_documents=17, checkpoint_every=1)
            saved: tuple[dict[str, Any], list[Document]] | None = None

            def stop(cursor: dict[str, Any], documents: list[Document]) -> None:
                nonlocal saved
                if cursor["dialogues_seen"] == 4:
                    saved = (dict(cursor), list(documents))
                    raise _PauseScan

            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                with self.assertRaises(_PauseScan):
                    MultiWOZPTBRAdapter().scan(config, checkpoint=stop)
                assert saved is not None
                resumed = MultiWOZPTBRAdapter().scan(
                    config,
                    resume_cursor=saved[0],
                    resume_documents=saved[1],
                )
                complete = MultiWOZPTBRAdapter().scan(config)

                self.assertEqual(resumed.documents, complete.documents)
                self.assertEqual(resumed.metrics, complete.metrics)
                self.assertTrue(resumed.cursor["complete"])

                changed = root / "multiwozptbr" / "dialogues_017.json"
                changed.write_text("[]", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
                    MultiWOZPTBRAdapter().scan(
                        config,
                        resume_cursor=saved[0],
                        resume_documents=saved[1],
                    )

    def test_rejects_unknown_speaker_without_exposing_record_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source(root)
            path = root / "multiwozptbr" / "dialogues_001.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value[0]["turns"][0]["speaker"] = "SYNTHETIC_UNKNOWN"
            path.write_text(json.dumps(value), encoding="utf-8")

            with patch.dict(os.environ, {TEST_ROOT_ENV: str(root)}):
                with self.assertRaises(MultiWOZFormatError) as raised:
                    MultiWOZPTBRAdapter().scan(resolved_config())

        self.assertNotIn("SYNTHETIC_UNKNOWN", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
