import json
import unittest
from copy import deepcopy

from queroquero.config import DATASET_IDS, canonical_json_bytes, sha256_bytes
from queroquero.real_plan import (
    CAPACITY_REPORT_SCHEMA,
    allocate_real_training,
    capacity_report_id,
    validate_capacity_report,
    validate_real_allocation,
)


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


class RealAllocationTests(unittest.TestCase):
    def test_equal_capacity_allocation_is_exact_and_deterministic(self) -> None:
        reports = [capacity_report(dataset_id, 100_000) for dataset_id in DATASET_IDS]
        first = allocate_real_training(reports)
        second = allocate_real_training(reversed(reports))

        self.assertEqual(first, second)
        self.assertEqual(sum(item["train_sequences"] for item in first["datasets"]), 416_000)
        self.assertEqual(
            [item["train_sequences"] for item in first["datasets"]],
            [69_334, 69_334, 69_333, 69_333, 69_333, 69_333],
        )
        self.assertEqual(validate_real_allocation(first), first)

    def test_waterfill_redistributes_one_or_multiple_exhausted_datasets(self) -> None:
        one_small = {
            dataset_id: (10_000 if index == 0 else 100_000)
            for index, dataset_id in enumerate(DATASET_IDS)
        }
        allocation = allocate_real_training(
            [capacity_report(key, value) for key, value in one_small.items()]
        )
        budgets = {
            item["dataset_id"]: item["train_sequences"]
            for item in allocation["datasets"]
        }
        self.assertEqual(budgets[DATASET_IDS[0]], 10_000)
        self.assertEqual(sum(budgets.values()), 416_000)

        multiple_small = dict(one_small)
        multiple_small[DATASET_IDS[1]] = 20_000
        allocation = allocate_real_training(
            [capacity_report(key, value) for key, value in multiple_small.items()]
        )
        budgets = [item["train_sequences"] for item in allocation["datasets"]]
        self.assertEqual(budgets[:2], [10_000, 20_000])
        self.assertEqual(budgets[2:], [96_500, 96_500, 96_500, 96_500])

    def test_global_insufficiency_reports_the_maximum_safe_run(self) -> None:
        reports = [capacity_report(dataset_id, 60_000) for dataset_id in DATASET_IDS]
        with self.assertRaisesRegex(
            RuntimeError,
            r"available_sequences=360000 maximum_optimizer_steps=45000",
        ):
            allocate_real_training(reports)

    def test_report_rejects_paths_and_private_payload_fields(self) -> None:
        report = capacity_report(DATASET_IDS[0], 100_000)
        changed = deepcopy(report)
        changed["source_fingerprint"] = {"path": "/private/source.txt"}
        changed["source_fingerprint_sha256"] = sha256_bytes(
            canonical_json_bytes(changed["source_fingerprint"])
        )
        changed["capacity_report_id"] = capacity_report_id(changed)
        with self.assertRaisesRegex(RuntimeError, "absolute path"):
            validate_capacity_report(changed)

        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("text", serialized)
        self.assertNotIn("source_ref", serialized)


if __name__ == "__main__":
    unittest.main()
