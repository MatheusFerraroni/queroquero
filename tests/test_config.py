import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from queroquero.config import ConfigError, load_resolved_config, scan_config_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_dataset_id_cannot_escape_the_config_or_output_directories(self) -> None:
        with self.assertRaisesRegex(ConfigError, "dataset_id"):
            load_resolved_config("../../outside", "smoke")

    def test_all_versioned_configs_resolve_for_both_profiles(self) -> None:
        for dataset_path in sorted((PROJECT_ROOT / "configs/datasets").glob("*.json")):
            for profile in ("smoke", "mvp"):
                config, digest = load_resolved_config(dataset_path.stem, profile)
                self.assertEqual(config["dataset_id"], dataset_path.stem)
                self.assertEqual(len(digest), 64)

    def test_sequence_length_cannot_change(self) -> None:
        preparation = json.loads(
            (PROJECT_ROOT / "configs/preparation.json").read_text(encoding="utf-8")
        )
        changed = deepcopy(preparation)
        changed["sequence_length"] = 128
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "datasets").mkdir()
            (root / "preparation.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            source = PROJECT_ROOT / "configs/datasets/brwac.json"
            (root / "datasets/brwac.json").write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ConfigError, "exactly 1024"):
                load_resolved_config("brwac", "smoke", root)

    def test_only_wacky_post_scan_decision_reuses_the_scan_cache(self) -> None:
        pending, _ = load_resolved_config("wackywacky", "mvp")
        keep = deepcopy(pending)
        keep["dataset"]["filters"]["boilerplate"]["decision"] = "keep"
        changed_threshold = deepcopy(keep)
        changed_threshold["dataset"]["filters"]["boilerplate"][
            "minimum_documents"
        ] = 6

        self.assertEqual(scan_config_sha256(pending), scan_config_sha256(keep))
        self.assertNotEqual(
            scan_config_sha256(keep), scan_config_sha256(changed_threshold)
        )


if __name__ == "__main__":
    unittest.main()
