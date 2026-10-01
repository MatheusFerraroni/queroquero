from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from queroquero.prepare import run_preparation
from scripts.inspect_preparation import inspect_preparation
from tests.test_prepare import FakeTokenizer, SyntheticAdapter, resolved_config


class InspectPreparationTests(unittest.TestCase):
    def test_inspection_summarizes_valid_shards_without_source_content(self) -> None:
        resolved, digest = resolved_config()
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir).resolve()
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
                patch(
                    "queroquero.prepare.load_adapter",
                    return_value=SyntheticAdapter(),
                ),
                patch(
                    "queroquero.prepare._load_pinned_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                patch("queroquero.prepare.PROJECT_ROOT", project),
                redirect_stdout(io.StringIO()),
            ):
                manifest_path = run_preparation("brwac", "smoke")

            root, manifest, summary = inspect_preparation(manifest_path)
            self.assertEqual(root, manifest_path.parent)
            self.assertEqual(manifest["dataset_id"], "brwac")
            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["splits"]["train"]["sequences"], 8)
            self.assertEqual(summary["splits"]["eval"]["sequences"], 2)
            self.assertEqual(summary["unique_sequence_ids"], 10)
            self.assertEqual(
                {length for shard in summary["shards"] for length in shard["tokens_per_sequence"]},
                {1024},
            )
            self.assertNotIn("PRIVATE_TEXT_SENTINEL", str(summary))
            self.assertNotIn("PRIVATE_REF_SENTINEL", str(summary))


if __name__ == "__main__":
    unittest.main()
