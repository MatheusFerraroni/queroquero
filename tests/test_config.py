import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from queroquero.config import (
    ConfigError,
    load_resolved_config,
    resolve_output_root,
    scan_config_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_output_root_uses_default_relative_or_explicit_absolute_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir).resolve()
            project = temporary / "project"
            project.mkdir()
            external = temporary / "external-output"

            with (
                patch("queroquero.config.PROJECT_ROOT", project),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(resolve_output_root("derived"), project / "derived")

            with (
                patch("queroquero.config.PROJECT_ROOT", project),
                patch.dict(
                    os.environ,
                    {"PTBR_OUTPUT_ROOT": "alternate-derived"},
                    clear=True,
                ),
            ):
                self.assertEqual(
                    resolve_output_root("derived"), project / "alternate-derived"
                )

            with (
                patch("queroquero.config.PROJECT_ROOT", project),
                patch.dict(
                    os.environ,
                    {"PTBR_OUTPUT_ROOT": str(external)},
                    clear=True,
                ),
            ):
                self.assertEqual(resolve_output_root("derived"), external)

    def test_output_root_rejects_broad_or_dataset_overlapping_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = (Path(temporary_dir) / "project").resolve()
            dataset_root = project / ".remote-datasets"
            dataset_root.mkdir(parents=True)
            with (
                patch("queroquero.config.PROJECT_ROOT", project),
                patch.dict(
                    os.environ,
                    {
                        "PTBR_DATASET_ROOT": str(dataset_root),
                        "PTBR_OUTPUT_ROOT": str(dataset_root / "derived"),
                    },
                    clear=True,
                ),
                self.assertRaisesRegex(ConfigError, "must not overlap"),
            ):
                resolve_output_root("derived")

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
        keep["dataset"]["filters"]["boilerplate"]["decision_by_profile"][
            "mvp"
        ] = "keep"
        changed_threshold = deepcopy(keep)
        changed_threshold["dataset"]["filters"]["boilerplate"][
            "within_domain_blocks"
        ]["minimum_documents"] = 6

        self.assertEqual(scan_config_sha256(pending), scan_config_sha256(keep))
        self.assertNotEqual(
            scan_config_sha256(keep), scan_config_sha256(changed_threshold)
        )


if __name__ == "__main__":
    unittest.main()
