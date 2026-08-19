from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any

from queroquero.datasets.gigaverbo import (
    DATASET_ID,
    REVISION,
    GigaverboAdapter,
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


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def dataset_info(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            sha=REVISION,
            siblings=[
                SimpleNamespace(
                    rfilename="data/default/train-00001-of-00224.parquet",
                    size=12,
                    lfs=SimpleNamespace(sha256="b" * 64),
                ),
                SimpleNamespace(
                    rfilename="data/excluded/train-00000-of-00023.parquet",
                    size=99,
                    lfs=SimpleNamespace(sha256="c" * 64),
                ),
                SimpleNamespace(
                    rfilename="data/default/train-00000-of-00224.parquet",
                    size=10,
                    lfs=SimpleNamespace(sha256="a" * 64),
                ),
            ],
        )


def resolved_config(
    *, candidate_documents: int = 2, max_source_records: int = 20
) -> dict[str, Any]:
    return {
        "profile_name": "smoke",
        "preparation": {"seed": 42},
        "dataset": {
            "source": {
                "provider": "huggingface",
                "dataset_id": DATASET_ID,
                "revision": REVISION,
                "config_name": "default",
                "split": "train",
                "streaming": True,
                "shard_glob": "data/default/train-*.parquet",
            },
            "filters": {"min_edu_int_score": 4},
            "selection": {
                "shuffle_buffer_size": 100,
                "checkpoint_interval": 1,
            },
            "license_policy": "internal_research_only",
            "redistribution_status": "internal_research_only",
        },
        "profile": {
            "candidate_documents": candidate_documents,
            "max_source_records": max_source_records,
            "selection": "engineering_prefix",
        },
    }


class GigaverboAdapterTests(unittest.TestCase):
    def test_streams_pinned_source_filters_and_keeps_only_safe_metadata(self) -> None:
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
        api = FakeApi()
        load_calls: list[dict[str, Any]] = []

        def load_dataset(**kwargs: Any) -> FakeStream:
            load_calls.append(kwargs)
            return stream

        checkpoints: list[tuple[dict[str, Any], int]] = []
        result = GigaverboAdapter(load_dataset, api).scan(
            resolved_config(),
            checkpoint=lambda cursor, docs: checkpoints.append((cursor, len(docs))),
        )

        self.assertEqual(
            load_calls,
            [
                {
                    "path": DATASET_ID,
                    "name": "default",
                    "split": "train",
                    "revision": REVISION,
                    "streaming": True,
                }
            ],
        )
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

        self.assertEqual(
            api.calls,
            [
                {
                    "repo_id": DATASET_ID,
                    "revision": REVISION,
                    "files_metadata": True,
                }
            ],
        )
        self.assertEqual(result.source_fingerprint["revision"], REVISION)
        self.assertEqual(result.source_fingerprint["license_policy"], "internal_research_only")
        self.assertEqual(
            [item["path"] for item in result.source_fingerprint["shards"]],
            [
                "data/default/train-00000-of-00224.parquet",
                "data/default/train-00001-of-00224.parquet",
            ],
        )
        self.assertNotIn("excluded", repr(result.source_fingerprint))

    def test_resume_skips_seen_records_and_preserves_deterministic_positions(self) -> None:
        records = [
            {"text": "primeiro", "edu_int_score": 4},
            {"text": "segundo", "edu_int_score": 4},
            {"text": "terceiro", "edu_int_score": 4},
        ]
        api = FakeApi()

        first = GigaverboAdapter(lambda **_: FakeStream(records), api).scan(
            resolved_config(candidate_documents=1)
        )
        cursor = first.cursor

        resumed = GigaverboAdapter(lambda **_: FakeStream(records), api).scan(
            resolved_config(candidate_documents=2),
            resume_cursor=cursor,
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
        result = GigaverboAdapter(
            lambda **_: FakeStream(records), FakeApi()
        ).scan(resolved_config(candidate_documents=2, max_source_records=3))

        self.assertEqual(result.documents, [])
        self.assertEqual(result.metrics["source_records_seen"], 3)
        self.assertEqual(result.metrics["source_record_limit_reached"], 1)

    def test_resume_rejects_a_changed_resolved_shard_fingerprint(self) -> None:
        records = [{"text": "primeiro", "edu_int_score": 4}]
        first = GigaverboAdapter(
            lambda **_: FakeStream(records), FakeApi()
        ).scan(resolved_config(candidate_documents=1))

        class ChangedApi(FakeApi):
            def dataset_info(self, **kwargs: Any) -> Any:
                info = super().dataset_info(**kwargs)
                info.siblings[0].lfs.sha256 = "d" * 64
                return info

        with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
            GigaverboAdapter(
                lambda **_: FakeStream(records), ChangedApi()
            ).scan(
                resolved_config(candidate_documents=2),
                resume_cursor=first.cursor,
                resume_documents=first.documents,
            )

    def test_rejects_unpinned_revision_before_loading_records(self) -> None:
        config = resolved_config()
        config = copy.deepcopy(config)
        config["dataset"]["source"]["revision"] = "main"
        loaded = False

        def load_dataset(**_: Any) -> FakeStream:
            nonlocal loaded
            loaded = True
            return FakeStream([])

        with self.assertRaisesRegex(ValueError, "source.revision"):
            GigaverboAdapter(load_dataset, FakeApi()).scan(config)
        self.assertFalse(loaded)


if __name__ == "__main__":
    unittest.main()
