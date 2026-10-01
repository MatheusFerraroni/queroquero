import json
import io
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from queroquero.config import (
    DATASET_IDS,
    canonical_json_bytes,
    sha256_bytes,
    validate_dataset_config,
)
from queroquero.paired_plan import (
    PAIRED_REAL_ALLOCATION_SCHEMA,
    allocate_paired_real_training,
    iter_paired_schedule_slots,
    paired_mixture_for_arm,
    validate_paired_mixture,
    validate_paired_real_allocation,
)
from queroquero.real_plan import CAPACITY_REPORT_SCHEMA, capacity_report_id
from queroquero.prepare import (
    materialize_paired_real_configs,
    run_verify_paired_real,
)
from queroquero.training_config import validate_training_config
from queroquero.training_data import ResolvedDataset, ResolvedTrainingInputs


def capacity_report(dataset_id: str, capacity: int, *, exact: bool = True):
    fingerprint = {
        "kind": "synthetic/v1",
        "sha256": sha256_bytes(dataset_id.encode("utf-8")),
        "records": capacity,
    }
    report = {
        "schema_version": CAPACITY_REPORT_SCHEMA,
        "dataset_id": dataset_id,
        "source_profile": "mvp",
        "scan_config_sha256": "a" * 64,
        "source_fingerprint": fingerprint,
        "source_fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(fingerprint)
        ),
        "tokenizer_fingerprint_sha256": "b" * 64,
        "candidate_documents": max(capacity, 1),
        "documents_selected": max(capacity, 1),
        "documents_tokenized": max(capacity, 1),
        "documents_exact_duplicates": 0,
        "eval_sequences_requested": 256,
        "eval_sequences_available": 256,
        "train_sequence_capacity": capacity,
        "capacity_kind": "exact" if exact else "lower_bound",
        "redistribution_status": "internal_research_only",
    }
    report["capacity_report_id"] = capacity_report_id(report)
    return report


class PairedRealAllocationTests(unittest.TestCase):
    def test_equal_allocation_builds_matched_disjoint_pools(self) -> None:
        reports = [
            capacity_report(dataset_id, 300_000)
            for dataset_id in DATASET_IDS
        ]
        first = allocate_paired_real_training(reports)
        second = allocate_paired_real_training(reversed(reports))

        self.assertEqual(first, second)
        self.assertEqual(
            first["schema_version"], PAIRED_REAL_ALLOCATION_SCHEMA
        )
        self.assertEqual(sum(first["forum_tech_allocations"].values()), 416_000)
        self.assertEqual(sum(first["general_allocations"].values()), 416_000)
        self.assertEqual(first["general_allocations"]["adrenaline"], 0)
        self.assertEqual(first["general_allocations"]["outerspace"], 0)
        replacement = next(
            pool for pool in first["pools"] if pool["pool_id"] == "brwac_extra"
        )
        common = next(
            pool for pool in first["pools"] if pool["pool_id"] == "brwac_common"
        )
        self.assertEqual(
            replacement["train_sequences"],
            first["forum_tech_allocations"]["adrenaline"]
            + first["forum_tech_allocations"]["outerspace"],
        )
        self.assertEqual(replacement["start_row"], common["train_sequences"])
        self.assertEqual(validate_paired_real_allocation(first), first)

        slots = tuple(iter_paired_schedule_slots(first))
        self.assertEqual(len(slots), 416_000)
        self.assertEqual(
            slots, tuple(iter_paired_schedule_slots(validate_paired_real_allocation(first)))
        )
        self.assertEqual(
            slots.count("adrenaline_domain"),
            first["forum_tech_allocations"]["adrenaline"],
        )

    def test_exact_exhaustion_redistributes_but_lower_bound_requests_expansion(self) -> None:
        exact_reports = [
            capacity_report(
                dataset_id,
                10_000 if dataset_id == "adrenaline" else 300_000,
            )
            for dataset_id in DATASET_IDS
        ]
        allocation = allocate_paired_real_training(exact_reports)
        self.assertEqual(allocation["forum_tech_allocations"]["adrenaline"], 10_000)
        self.assertEqual(sum(allocation["forum_tech_allocations"].values()), 416_000)

        lower_bound = [
            capacity_report(
                dataset_id,
                10_000 if dataset_id == "adrenaline" else 300_000,
                exact=dataset_id != "adrenaline",
            )
            for dataset_id in DATASET_IDS
        ]
        with self.assertRaisesRegex(
            RuntimeError, r"expand lower-bound scans.*adrenaline>=69334"
        ):
            allocate_paired_real_training(lower_bound)

    def test_brwac_shortfall_is_fail_closed(self) -> None:
        exact_reports = [
            capacity_report(
                dataset_id,
                100_000 if dataset_id == "brwac" else 300_000,
            )
            for dataset_id in DATASET_IDS
        ]
        with self.assertRaisesRegex(
            RuntimeError,
            r"unique BrWaC capacity.*maximum_replacement_sequences",
        ):
            allocate_paired_real_training(exact_reports)

        lower_bound = [
            capacity_report(
                dataset_id,
                100_000 if dataset_id == "brwac" else 300_000,
                exact=dataset_id != "brwac",
            )
            for dataset_id in DATASET_IDS
        ]
        with self.assertRaisesRegex(
            RuntimeError, r"BrWaC capacity audit is incomplete"
        ):
            allocate_paired_real_training(lower_bound)

    def test_mixture_contract_is_arm_specific_and_private(self) -> None:
        allocation = allocate_paired_real_training(
            capacity_report(dataset_id, 300_000) for dataset_id in DATASET_IDS
        )
        general = paired_mixture_for_arm(allocation, "general")
        forum = paired_mixture_for_arm(allocation, "forum_tech")
        self.assertEqual(validate_paired_mixture(general), general)
        self.assertEqual(validate_paired_mixture(forum), forum)
        self.assertEqual(general["allocation_sha256"], forum["allocation_sha256"])
        self.assertNotEqual(general["arm"], forum["arm"])
        serialized = json.dumps(allocation, sort_keys=True)
        self.assertNotIn("source_ref", serialized)
        self.assertNotIn("text", serialized)

        changed = deepcopy(general)
        changed["pools"][-1]["start_row"] += 1
        with self.assertRaisesRegex(RuntimeError, "overlap|ranges"):
            validate_paired_mixture(changed)

        changed = deepcopy(general)
        changed["pools"][1]["dataset_id"] = "wackywacky"
        with self.assertRaisesRegex(RuntimeError, "pool contract"):
            validate_paired_mixture(changed)

    def test_materializer_generates_valid_dataset_and_training_configs(self) -> None:
        allocation = allocate_paired_real_training(
            capacity_report(dataset_id, 300_000) for dataset_id in DATASET_IDS
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            allocation_path = root / "allocation.json"
            allocation_path.write_text(
                json.dumps(allocation), encoding="utf-8"
            )
            output = root / "configs"
            result = materialize_paired_real_configs(
                allocation_path,
                output_config_root=output,
            )

            self.assertEqual(result["status"], "materialized")
            self.assertEqual(len(result["files"]), 9)
            for dataset_id in DATASET_IDS:
                config = json.loads(
                    (output / "datasets" / f"{dataset_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
                validate_dataset_config(config, dataset_id)
                profile = config["profiles"]["paired_real"]
                self.assertEqual(
                    profile["train_sequences"],
                    allocation["prepared_train_sequences"][dataset_id],
                )
                self.assertEqual(profile["eval_sequences"], 256)
            for arm in ("general", "forum-tech"):
                config = json.loads(
                    (
                        output / "training" / f"l40s-real-{arm}.json"
                    ).read_text(encoding="utf-8")
                )
                validate_training_config(config)
            self.assertEqual(
                json.loads(
                    (
                        output / "allocations" / "paired-real-allocation.json"
                    ).read_text(encoding="utf-8")
                ),
                allocation,
            )

    def test_verification_report_proves_shared_inputs_without_private_content(self) -> None:
        allocation = allocate_paired_real_training(
            capacity_report(dataset_id, 300_000) for dataset_id in DATASET_IDS
        )
        mixtures = {
            arm: paired_mixture_for_arm(allocation, arm)
            for arm in ("general", "forum_tech")
        }
        configs = {
            arm: {
                "model": {"fixed": True},
                "training": {"seed": 42},
                "execution": {"world_size": 2},
                "hardware": {"gpu": "L40S"},
                "data_mixture": mixtures[arm],
            }
            for arm in mixtures
        }
        datasets = tuple(
            ResolvedDataset(
                dataset_id=dataset_id,
                root=Path("/synthetic") / dataset_id,
                manifest={
                    "preparation_id": f"{index + 1:020x}",
                    "splits": {"eval": [{"sha256": f"{index + 2:064x}"}]},
                    "counts": {
                        "train_sequences": allocation[
                            "prepared_train_sequences"
                        ][dataset_id],
                        "eval_sequences": 256,
                    },
                },
                manifest_sha256=f"{index + 3:064x}",
                relative_manifest_path=f"{dataset_id}/manifest.json",
            )
            for index, dataset_id in enumerate(DATASET_IDS)
        )
        inputs = {
            arm: ResolvedTrainingInputs(
                profile="real",
                output_root=Path("/synthetic"),
                datasets=datasets,
                tokenizer={},
                data_mixture=mixtures[arm],
                preparation_profile="paired_real",
            )
            for arm in mixtures
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "verification.json"
            with (
                patch(
                    "queroquero.training_config.load_training_config",
                    side_effect=[(configs["general"], "a" * 64), (configs["forum_tech"], "b" * 64)],
                ),
                patch(
                    "queroquero.training_data.resolve_training_inputs",
                    side_effect=[inputs["general"], inputs["forum_tech"]],
                ),
                redirect_stdout(io.StringIO()),
            ):
                report = run_verify_paired_real(
                    Path("general.json"),
                    Path("forum.json"),
                    output=output,
                )
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["train_sequences_per_arm"], 416_000)
        self.assertEqual(
            report["shared_positions"] + report["replacement_positions"],
            416_000,
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("source_ref", serialized)
        self.assertNotIn("input_ids", serialized)


if __name__ == "__main__":
    unittest.main()
