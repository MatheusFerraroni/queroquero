import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from queroquero.classification_data import (
    AUDIT_SCHEMA,
    CONFIG_SCHEMA,
    DATASET_MANIFEST_SCHEMA,
    FINAL_SCHEMA,
    REDISTRIBUTION_STATUS,
    _dataset_identity,
    _file_record,
    _table_row_digest,
    _write_examples_csv,
    _write_label_csv,
    _write_parquet_atomic,
    _zip_fingerprint,
    build_classification_dataset,
    load_classification_config,
    validate_classification_config,
    validate_classification_dataset,
)
from queroquero.classification_split import (
    create_classification_split,
    validate_classification_split,
)
from queroquero.config import ConfigError, canonical_json_bytes, sha256_bytes
from queroquero.datasets.base import safe_source_hash, stable_hash
from queroquero.manifest import file_sha256, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONED_CONFIG = PROJECT_ROOT / "configs/classification/adrenaline-v1.json"


class ClassificationDatasetTests(unittest.TestCase):
    def test_slurm_preparation_is_cpu_only_and_limited_to_24_hours(self) -> None:
        batch = (PROJECT_ROOT / "scripts/prepare_classification.sbatch").read_text()
        submit = (PROJECT_ROOT / "scripts/submit_classification.sh").read_text()
        self.assertIn("#SBATCH --time=1-00:00:00", batch)
        self.assertNotIn("#SBATCH --gres", batch)
        self.assertIn("--time=1-00:00:00", submit)

    def test_versioned_configuration_is_strict_and_resolvable(self) -> None:
        config, digest = load_classification_config(VERSIONED_CONFIG)
        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        self.assertEqual(config["benchmark"]["seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(len(digest), 64)

        changed = deepcopy(config)
        changed["unknown"] = True
        with self.assertRaisesRegex(ConfigError, "keys"):
            validate_classification_config(changed)

    def test_build_is_read_only_resumable_and_excludes_cpt_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fixture = _build_source_fixture(root)
            source_hashes_before = {
                path.name: file_sha256(path)
                for path in fixture["source_files"]
            }
            original = _write_parquet_atomic
            calls = 0

            def interrupt_after_second_part(table, path, schema):
                nonlocal calls
                original(table, path, schema)
                if path.parent.name == "parts":
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("synthetic interruption")

            environment = {
                "PTBR_DATASET_ROOT": str(fixture["dataset_root"]),
                "PTBR_OUTPUT_ROOT": str(fixture["cpt_root"]),
                "PTBR_CLASSIFICATION_ROOT": str(fixture["output_root"]),
            }
            with (
                patch.dict("os.environ", environment, clear=True),
                patch(
                    "queroquero.classification_data._write_parquet_atomic",
                    side_effect=interrupt_after_second_part,
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic interruption"),
            ):
                build_classification_dataset(
                    fixture["config_path"], checkpoint_threads=2
                )

            with patch.dict("os.environ", environment, clear=True):
                result = build_classification_dataset(
                    fixture["config_path"], checkpoint_threads=2
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["examples"], 2)
            dataset_path = (
                fixture["output_root"]
                / "adrenaline"
                / result["classification_dataset_id"]
            )
            manifest = validate_classification_dataset(dataset_path)
            self.assertEqual(manifest["counts"]["examples"], 2)
            self.assertFalse(
                any(field.name == "split" for field in pq.read_schema(dataset_path / "examples.parquet"))
            )
            audit = json.loads((dataset_path / "audit.json").read_text())
            self.assertEqual(
                audit["extraction_counts"]["discarded_cpt_overlap_union"], 2
            )
            self.assertEqual(
                audit["deduplication_counts"]["duplicate_records_same_label"], 1
            )
            self.assertEqual(
                audit["deduplication_counts"]["conflicting_content_records"], 2
            )
            self.assertFalse(
                fixture["output_root"].joinpath(".work").exists()
                and any(fixture["output_root"].joinpath(".work").rglob("progress.json"))
            )
            serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("Título elegível", serialized)
            self.assertNotIn(str(root), serialized)
            source_hashes_after = {
                path.name: file_sha256(path)
                for path in fixture["source_files"]
            }
            self.assertEqual(source_hashes_before, source_hashes_after)

            continuous_root = root / "classification-continuous"
            with patch.dict("os.environ", environment, clear=True):
                continuous = build_classification_dataset(
                    fixture["config_path"],
                    output_root=continuous_root,
                    checkpoint_threads=2,
                )
            continuous_path = (
                continuous_root
                / "adrenaline"
                / continuous["classification_dataset_id"]
            )
            for artifact in sorted(path.name for path in dataset_path.iterdir()):
                self.assertEqual(
                    (dataset_path / artifact).read_bytes(),
                    (continuous_path / artifact).read_bytes(),
                    artifact,
                )

            with patch.dict("os.environ", environment, clear=True):
                repeated = build_classification_dataset(
                    fixture["config_path"], checkpoint_threads=2
                )
            self.assertEqual(repeated["status"], "existing")
            self.assertEqual(repeated["classification_dataset_id"], result["classification_dataset_id"])

    def test_unresolved_cpt_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fixture = _build_source_fixture(root, unresolved_hash=True)
            environment = {
                "PTBR_DATASET_ROOT": str(fixture["dataset_root"]),
                "PTBR_OUTPUT_ROOT": str(fixture["cpt_root"]),
                "PTBR_CLASSIFICATION_ROOT": str(fixture["output_root"]),
            }
            with (
                patch.dict("os.environ", environment, clear=True),
                self.assertRaisesRegex(RuntimeError, "crosswalk is incomplete"),
            ):
                build_classification_dataset(fixture["config_path"])


class ClassificationSplitTests(unittest.TestCase):
    def test_balanced_splits_are_deterministic_disjoint_and_model_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_path = _build_split_dataset(root)
            first_path = root / "runs/seed-42/coarse/split_manifest.json"
            first = create_classification_split(
                VERSIONED_CONFIG,
                dataset_path,
                task="coarse",
                seed=42,
                output=first_path,
            )
            repeated = create_classification_split(
                VERSIONED_CONFIG,
                dataset_path,
                task="coarse",
                seed=42,
                output=first_path,
            )
            self.assertEqual(first, repeated)
            self.assertEqual(first["class_budget"], 1000)
            self.assertEqual(first["counts"], {
                "train": 4200,
                "validation": 900,
                "test": 900,
                "total": 6000,
            })
            self.assertEqual(first["compatible_models"], ["base", "general", "forum"])
            self.assertEqual(first["input_variants"], ["title", "title_first_post"])
            all_sets = [set(first["sample_ids"][name]) for name in ("train", "validation", "test")]
            self.assertFalse(all_sets[0] & all_sets[1])
            self.assertFalse(all_sets[0] & all_sets[2])
            self.assertFalse(all_sets[1] & all_sets[2])
            lookup = {
                row["sample_id"]: row["title_group_id"]
                for row in pq.read_table(
                    dataset_path / "examples.parquet",
                    columns=["sample_id", "title_group_id"],
                ).to_pylist()
            }
            split_groups = [
                {lookup[sample_id] for sample_id in first["sample_ids"][name]}
                for name in ("train", "validation", "test")
            ]
            self.assertFalse(split_groups[0] & split_groups[1])
            self.assertFalse(split_groups[0] & split_groups[2])
            self.assertFalse(split_groups[1] & split_groups[2])
            self.assertEqual(
                validate_classification_split(VERSIONED_CONFIG, dataset_path, first_path),
                first,
            )

            second = create_classification_split(
                VERSIONED_CONFIG,
                dataset_path,
                task="coarse",
                seed=43,
                output=root / "runs/seed-43/coarse/split_manifest.json",
            )
            self.assertNotEqual(first["benchmark_id"], second["benchmark_id"])
            self.assertNotEqual(first["sample_ids"]["test"], second["sample_ids"]["test"])

            fine = create_classification_split(
                VERSIONED_CONFIG,
                dataset_path,
                task="fine",
                seed=42,
                output=root / "runs/seed-42/fine/split_manifest.json",
            )
            self.assertEqual(fine["counts"]["total"], 6000)
            self.assertEqual(len(fine["labels"]), 6)
            self.assertEqual(
                fine["selection"]["conflicting_title_groups_excluded"], 1
            )

    def test_changed_split_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset_path = _build_split_dataset(root)
            path = root / "split_manifest.json"
            manifest = create_classification_split(
                VERSIONED_CONFIG,
                dataset_path,
                task="coarse",
                seed=42,
                output=path,
            )
            manifest["sample_ids"]["test"][0] = "f" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                validate_classification_split(VERSIONED_CONFIG, dataset_path, path)


def _build_source_fixture(root: Path, unresolved_hash: bool = False):
    dataset_root = root / "datasets"
    cpt_root = root / "derived"
    output_root = root / "classification"
    source_dir = dataset_root / "adrenaline"
    source_dir.mkdir(parents=True)
    cpt_dir = cpt_root / "adrenaline/test-preparation"
    cpt_dir.mkdir(parents=True)

    conversations = source_dir / "conversations.zip"
    with ZipFile(conversations, "w") as archive:
        archive.writestr("clear_threads/100.tsv", "")
        archive.writestr("clear_threads/100_1.tsv", "")
        archive.writestr("clear_threads/101.tsv", "")
    pretraining_fingerprint, _ = _zip_fingerprint(conversations)

    train_hash = safe_source_hash(
        "adrenaline/conversations.zip/clear_threads/100_1.tsv"
    )
    if unresolved_hash:
        train_hash = "f" * 64
    eval_hash = safe_source_hash(
        "adrenaline/conversations.zip/clear_threads/101.tsv"
    )
    train_dir = cpt_dir / "train"
    eval_dir = cpt_dir / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    train_shard = train_dir / "shard.parquet"
    eval_shard = eval_dir / "shard.parquet"
    _write_source_hash_shard(train_shard, train_hash)
    _write_source_hash_shard(eval_shard, eval_hash)
    cpt_manifest = {
        "schema_version": "queroquero-dataset-manifest/v1",
        "dataset_id": "adrenaline",
        "profile": "paired_real",
        "preparation_id": "a" * 20,
        "source": {"fingerprint": pretraining_fingerprint},
        "splits": {
            "train": [_source_shard_record(train_shard, cpt_dir)],
            "eval": [_source_shard_record(eval_shard, cpt_dir)],
        },
    }
    cpt_manifest_path = cpt_dir / "dataset_manifest.json"
    write_json_atomic(cpt_manifest_path, cpt_manifest)

    forum = source_dir / "forum.adrenaline.com.br.zip"
    prefix = "forum.adrenaline.com.br"
    categories = [
        {
            "id": 3,
            "title_text": "Tecnologia A",
            "subs": [{"id": 0, "title_text": "Subcategoria A", "complete": False}],
        },
        {
            "id": 8,
            "title_text": "Tecnologia B",
            "subs": [{"id": 4, "title_text": "Subcategoria B", "complete": False}],
        },
    ]
    labels_a = [100, 101, 102, 103, 105, 107, 108, 109]
    labels_b = [106]
    with ZipFile(forum, "w") as archive:
        archive.writestr(f"{prefix}/categories.json", json.dumps(categories))
        archive.writestr(
            f"{prefix}/categories_threads/category_3_subcategory_0.json",
            json.dumps(_mapping(3, 0, labels_a)),
        )
        archive.writestr(
            f"{prefix}/categories_threads/category_8_subcategory_4.json",
            json.dumps(_mapping(8, 4, labels_b)),
        )
        records = {
            100: _thread(100, 3, 0, "Sobreposto treino", "Texto privado A"),
            101: _thread(101, 3, 0, "Sobreposto avaliação", "Texto privado B"),
            102: _thread(102, 3, 0, "<b>Título elegível</b>", "Primeiro<br>post"),
            103: _thread(103, 3, 0, "Sem primeiro post", None),
            104: _thread(104, 3, 0, "Sem rótulo", "Texto"),
            105: _thread(105, 3, 0, "Conflito", "Mesmo conteúdo"),
            106: _thread(106, 8, 4, "Conflito", "Mesmo conteúdo"),
            107: _thread(107, 3, 0, "Duplicado", "Mesmo rótulo"),
            108: _thread(108, 3, 0, "Duplicado", "Mesmo rótulo"),
            109: _thread(109, 3, 0, "<br>", "Texto"),
        }
        for thread_id, record in records.items():
            archive.writestr(
                f"{prefix}/threads/{thread_id}.json", json.dumps(record)
            )
    forum_fingerprint, _ = _zip_fingerprint(forum, forum_prefix=prefix)

    config = json.loads(VERSIONED_CONFIG.read_text(encoding="utf-8"))
    config["source"]["forum_archive"] = "adrenaline/forum.adrenaline.com.br.zip"
    config["source"]["pretraining_archive"] = "adrenaline/conversations.zip"
    config["source"]["cpt_manifest"] = {
        "root_env": "PTBR_OUTPUT_ROOT",
        "relative_path": "adrenaline/test-preparation/dataset_manifest.json",
        "preparation_id": "a" * 20,
        "sha256": file_sha256(cpt_manifest_path),
    }
    config["source"]["expected"] = {
        "forum": {
            **forum_fingerprint,
            "labeled_thread_references": 9,
            "unlabeled_thread_entries": 1,
        },
        "pretraining": pretraining_fingerprint,
    }
    config_path = root / "classification-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    validate_classification_config(config)
    return {
        "config_path": config_path,
        "dataset_root": dataset_root,
        "cpt_root": cpt_root,
        "output_root": output_root,
        "source_files": [conversations, forum, cpt_manifest_path, train_shard, eval_shard],
    }


def _mapping(category_id, subcategory_id, thread_ids):
    return {
        "status": "complete",
        "category": category_id,
        "subcategory": subcategory_id,
        "total_threads": len(thread_ids),
        "threads": [
            {"id": value, "category": category_id, "subcategory": subcategory_id}
            for value in thread_ids
        ],
    }


def _thread(thread_id, category_id, subcategory_id, title, first_post):
    messages = [] if first_post is None else [{"message": first_post}]
    return {
        "id": thread_id,
        "category": category_id,
        "subcategory": subcategory_id,
        "title": title,
        "messages": messages,
    }


def _write_source_hash_shard(path: Path, value: str) -> None:
    table = pa.table(
        {"source_ref_sha256": pa.array([[value]], type=pa.list_(pa.string()))}
    )
    pq.write_table(table, path, compression="zstd")


def _source_shard_record(path: Path, root: Path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _build_split_dataset(root: Path) -> Path:
    config, config_sha256 = load_classification_config(VERSIONED_CONFIG)
    categories = [3, 8, 19, 23, 26, 32]
    subcategories = [0, 4, 9, 20, 24, 27]
    rows = []
    for category_id, subcategory_id in zip(categories, subcategories, strict=True):
        count = 1001 if category_id in {3, 8} else 1000
        for index in range(count):
            title = (
                "Título conflitante"
                if category_id in {3, 8} and index == 0
                else f"Título sintético {category_id} {index}"
            )
            first_post = f"Primeiro post sintético {category_id} {index}"
            rows.append(
                {
                    "sample_id": stable_hash("synthetic-sample", category_id, index),
                    "title": title,
                    "first_post": first_post,
                    "category_id": category_id,
                    "category_name": f"Categoria {category_id}",
                    "subcategory_id": subcategory_id,
                    "subcategory_name": f"Subcategoria {subcategory_id}",
                    "title_group_id": stable_hash(
                        "adrenaline-classification-title/v1", title
                    ),
                    "title_chars": len(title),
                    "first_post_chars": len(first_post),
                }
            )
    table = pa.Table.from_pylist(rows, schema=FINAL_SCHEMA).sort_by("sample_id")
    forum_fingerprint = {"sha256": "a" * 64}
    pretraining_fingerprint = {"sha256": "b" * 64}
    cpt_sha256 = "c" * 64
    dataset_id = _dataset_identity(
        config_sha256, forum_fingerprint, pretraining_fingerprint, cpt_sha256
    )
    dataset_path = root / "classification" / "adrenaline" / dataset_id
    dataset_path.mkdir(parents=True)
    _write_parquet_atomic(table, dataset_path / "examples.parquet", FINAL_SCHEMA)
    _write_examples_csv(table, dataset_path / "examples.csv.gz")
    _write_label_csv(
        dataset_path / "categories.csv",
        ("category_id", "category_name"),
        ((value, f"Categoria {value}") for value in categories),
    )
    _write_label_csv(
        dataset_path / "subcategories.csv",
        ("category_id", "subcategory_id", "subcategory_name"),
        (
            (category, subcategory, f"Subcategoria {subcategory}")
            for category, subcategory in zip(categories, subcategories, strict=True)
        ),
    )
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "classification_dataset_id": dataset_id,
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    write_json_atomic(dataset_path / "audit.json", audit)
    files = [
        _file_record(dataset_path / name, dataset_path)
        for name in (
            "examples.parquet",
            "examples.csv.gz",
            "categories.csv",
            "subcategories.csv",
            "audit.json",
        )
    ]
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "classification_dataset_id": dataset_id,
        "dataset_id": "adrenaline",
        "config_sha256": config_sha256,
        "source": {
            "forum_fingerprint": forum_fingerprint,
            "pretraining_fingerprint": pretraining_fingerprint,
            "cpt_manifest": {"preparation_id": "d" * 20, "sha256": cpt_sha256},
        },
        "counts": {
            "examples": table.num_rows,
            "unique_title_groups": len(set(table.column("title_group_id").to_pylist())),
            "categories": 6,
            "subcategories": 6,
        },
        "examples_row_sha256": _table_row_digest(table),
        "files": files,
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    write_json_atomic(dataset_path / "dataset_manifest.json", manifest)
    validate_classification_dataset(dataset_path)
    return dataset_path


if __name__ == "__main__":
    unittest.main()
