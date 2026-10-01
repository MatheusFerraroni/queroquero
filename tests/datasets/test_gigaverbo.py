from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from queroquero.datasets.gigaverbo import (
    REVISION,
    GigaverboAdapter,
    _LocalParquetStream,
)


class FakeStream:
    def __init__(self, records: list[Any]) -> None:
        self.records = records
        self.shuffle_calls: list[dict[str, int]] = []
        self.skipped = 0

    def shuffle(self, *, seed: int, buffer_size: int) -> "FakeStream":
        self.shuffle_calls.append({"seed": seed, "buffer_size": buffer_size})
        return self

    def skip(self, count: int) -> "FakeStream":
        skipped = FakeStream(self.records[count:])
        skipped.shuffle_calls = self.shuffle_calls
        skipped.skipped = count
        return skipped

    def __iter__(self):
        return iter(self.records)


def resolved_config(
    *, candidate_documents: int = 2, max_source_records: int = 20
) -> dict[str, Any]:
    return {
        "profile_name": "smoke",
        "preparation": {"seed": 42},
        "dataset": {
            "source": {
                "provider": "local",
                "root_env": "TEST_PTBR_DATASET_ROOT",
                "path": "gigaverbo-v2",
                "format": "parquet",
                "dataset_id": "Polygl0t/gigaverbo-v2",
                "revision": REVISION,
                "config_name": "default",
                "split": "train",
                "streaming": True,
                "shard_glob": "default/train-*.parquet",
                "expected_shards": 224,
            },
            "filters": {"min_edu_int_score": 4},
            "selection": {
                "checkpoint_interval": 1,
            },
            "license_policy": "internal_research_only",
            "redistribution_status": "internal_research_only",
        },
        "profile": {
            "candidate_documents": candidate_documents,
            "max_source_records": max_source_records,
            "shuffle_buffer_size": 100,
            "selection": "engineering_prefix",
        },
    }


class GigaverboAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "gigaverbo-v2"
        data_directory = self.source / "default"
        tree_directory = (
            self.source / ".cache" / "huggingface" / "trees"
        )
        data_directory.mkdir(parents=True)
        tree_directory.mkdir(parents=True)
        files: dict[str, dict[str, Any]] = {}
        for index in range(224):
            name = f"train-{index:05d}-of-00224.parquet"
            content = f"fixture-{index}".encode("ascii")
            (data_directory / name).write_bytes(content)
            digest = hashlib.sha256(name.encode("ascii")).hexdigest()
            files[f"default/{name}"] = {
                "size": len(content),
                "lfs_size": len(content),
                "lfs_sha256": digest,
            }
        (tree_directory / f"{REVISION}.json").write_text(
            json.dumps({"format_version": 1, "files": files}),
            encoding="utf-8",
        )
        environment = patch.dict(
            os.environ, {"TEST_PTBR_DATASET_ROOT": str(self.root)}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_local_parquet_stream_is_deterministic_and_reads_selected_columns(
        self,
    ) -> None:
        paths = []
        for shard_index in range(2):
            path = self.root / f"fixture-{shard_index}.parquet"
            pq.write_table(
                pa.table(
                    {
                        "text": [f"texto-{shard_index}-a", f"texto-{shard_index}-b"],
                        "edu_int_score": [4, 5],
                        "source": ["source-a", "source-b"],
                        "ignored": ["private-a", "private-b"],
                    }
                ),
                path,
            )
            paths.append(path)

        first = list(_LocalParquetStream(paths).shuffle(seed=42, buffer_size=2))
        second = list(_LocalParquetStream(paths).shuffle(seed=42, buffer_size=2))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotIn("ignored", repr(first))

    def test_streams_local_pinned_source_and_keeps_only_safe_metadata(self) -> None:
        records = [
            {
                "id": "raw-id-that-must-not-be-preserved",
                "text": "nota baixa",
                "edu_int_score": 3,
                "source": "source-a",
                "subset": "subset-a",
            },
            {
                "id": "another-private-id",
                "text": "<p>Texto um</p>",
                "edu_int_score": 4,
                "source": "source-a",
                "subset": "subset-a",
                "toxic_score": 0.1,
            },
            {
                "id": "url-record-id",
                "text": "Texto dois",
                "edu_int_score": 5,
                "source": "https://example.invalid/raw",
                "subset": "subset-b",
            },
            {
                "id": "not-read-after-limit",
                "text": "Texto três",
                "edu_int_score": 5,
            },
        ]
        stream = FakeStream(records)
        load_calls: list[dict[str, Any]] = []

        def load_dataset(**kwargs: Any) -> FakeStream:
            load_calls.append(kwargs)
            return stream

        checkpoints: list[tuple[dict[str, Any], int]] = []
        result = GigaverboAdapter(load_dataset).scan(
            resolved_config(),
            checkpoint=lambda cursor, docs: checkpoints.append((cursor, len(docs))),
        )

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0]["path"], "parquet")
        self.assertEqual(load_calls[0]["split"], "train")
        self.assertIs(load_calls[0]["streaming"], True)
        local_files = load_calls[0]["data_files"]["train"]
        self.assertEqual(len(local_files), 224)
        self.assertTrue(local_files[0].endswith("default/train-00000-of-00224.parquet"))
        self.assertTrue(local_files[-1].endswith("default/train-00223-of-00224.parquet"))
        self.assertEqual(stream.shuffle_calls, [{"seed": 42, "buffer_size": 100}])
        self.assertEqual([doc.text for doc in result.documents], ["Texto um", "Texto dois"])
        self.assertEqual(
            result.documents[0].metadata,
            {"source": "source-a", "subset": "subset-a"},
        )
        self.assertEqual(result.documents[1].metadata, {"subset": "subset-b"})
        self.assertNotIn("raw-id", repr(result.documents))
        self.assertNotIn("private-id", repr(result.documents))
        self.assertEqual(result.metrics["source_records_seen"], 3)
        self.assertEqual(result.metrics["records_filtered_score"], 1)
        self.assertEqual(result.metrics["candidate_limit_reached"], 1)
        self.assertEqual([count for _, count in checkpoints], [1, 2])
        self.assertEqual(result.source_fingerprint["revision"], REVISION)
        self.assertEqual(result.source_fingerprint["shard_count"], 224)
        self.assertEqual(
            result.source_fingerprint["kind"], "local_huggingface_snapshot"
        )
        self.assertNotIn(str(self.root), repr(result.source_fingerprint))

    def test_resume_skips_seen_records_and_preserves_deterministic_positions(self) -> None:
        records = [
            {"text": "primeiro", "edu_int_score": 4},
            {"text": "segundo", "edu_int_score": 4},
            {"text": "terceiro", "edu_int_score": 4},
        ]

        first = GigaverboAdapter(lambda **_: FakeStream(records)).scan(
            resolved_config(candidate_documents=1)
        )
        resumed = GigaverboAdapter(lambda **_: FakeStream(records)).scan(
            resolved_config(candidate_documents=2),
            resume_cursor=first.cursor,
            resume_documents=first.documents,
        )

        self.assertEqual([doc.text for doc in resumed.documents], ["primeiro", "segundo"])
        self.assertEqual(
            [doc.source_position for doc in resumed.documents],
            [{"stream_index": 0}, {"stream_index": 1}],
        )
        self.assertEqual(resumed.cursor["records_seen"], 2)

    def test_source_record_limit_is_finite(self) -> None:
        records = [
            {"text": f"documento-{index}", "edu_int_score": 1}
            for index in range(100)
        ]
        result = GigaverboAdapter(lambda **_: FakeStream(records)).scan(
            resolved_config(candidate_documents=2, max_source_records=3)
        )

        self.assertEqual(result.documents, [])
        self.assertEqual(result.metrics["source_records_seen"], 3)
        self.assertEqual(result.metrics["source_record_limit_reached"], 1)

    def test_resume_rejects_changed_local_download_tree(self) -> None:
        records = [{"text": "primeiro", "edu_int_score": 4}]
        first = GigaverboAdapter(lambda **_: FakeStream(records)).scan(
            resolved_config(candidate_documents=1)
        )
        tree_path = self.source / f".cache/huggingface/trees/{REVISION}.json"
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        tree["files"]["default/train-00000-of-00224.parquet"]["lfs_sha256"] = (
            "d" * 64
        )
        tree_path.write_text(json.dumps(tree), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
            GigaverboAdapter(lambda **_: FakeStream(records)).scan(
                resolved_config(candidate_documents=2),
                resume_cursor=first.cursor,
                resume_documents=first.documents,
            )

    def test_rejects_missing_download_tree(self) -> None:
        tree_path = self.source / f".cache/huggingface/trees/{REVISION}.json"
        tree_path.unlink()
        with self.assertRaisesRegex(ValueError, "download tree metadata is missing"):
            GigaverboAdapter(lambda **_: FakeStream([])).scan(resolved_config())

    def test_rejects_unpinned_revision_before_loading_records(self) -> None:
        config = copy.deepcopy(resolved_config())
        config["dataset"]["source"]["revision"] = "main"
        loaded = False

        def load_dataset(**_: Any) -> FakeStream:
            nonlocal loaded
            loaded = True
            return FakeStream([])

        with self.assertRaisesRegex(ValueError, "source.revision"):
            GigaverboAdapter(load_dataset).scan(config)
        self.assertFalse(loaded)


if __name__ == "__main__":
    unittest.main()
