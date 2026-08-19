import json
import tempfile
import unittest
from pathlib import Path

from queroquero.datasets.base import Document
from queroquero.packing import PackedSequence
from queroquero.storage import WorkStore, validate_shard, write_split


def sequence(index):
    return PackedSequence(
        sequence_id=f"{index:064d}",
        input_ids=tuple([index + 1] * 1024),
        source_ref_sha256=(f"{index + 10:064d}",),
        source_token_counts=(1024,),
    )


class StorageTest(unittest.TestCase):
    def test_parquet_is_deterministic_and_valid(self) -> None:
        records = [sequence(index) for index in range(3)]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "first"
            second = root / "second"
            first_records = write_split(first, "train", records, 2)
            second_records = write_split(second, "train", records, 2)
            self.assertEqual(
                [record["sha256"] for record in first_records],
                [record["sha256"] for record in second_records],
            )
            for record in first_records:
                validate_shard(first / record["path"])

    def test_work_store_round_trips_resume_state(self) -> None:
        document = Document(
            text="Texto sintético temporário.",
            source_ref="synthetic:1",
            source_position={"row": 1},
            metadata={"safe": True},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = WorkStore(Path(temporary_dir), "synthetic", "a" * 64)
            store.checkpoint({"row": 1}, [document])
            cursor, documents = store.load()
            self.assertEqual(cursor, {"row": 1})
            self.assertEqual(documents, [document])
            progress = json.loads(store.progress_path.read_text(encoding="utf-8"))
            self.assertNotIn(document.text, json.dumps(progress))
            store.cleanup()
            self.assertFalse(store.path.exists())


if __name__ == "__main__":
    unittest.main()
