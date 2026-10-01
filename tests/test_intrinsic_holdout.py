import hashlib
import importlib.util
import io
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from queroquero import intrinsic_holdout as h
from queroquero.datasets._conversation_zip import _read_conversation
from queroquero.holdout_zip import (
    IndexedZip,
    central_directory,
    info_record,
    record_info,
    select_members,
    stat_identity,
)


def fingerprint(path):
    digest = hashlib.sha256(path.stat().st_size.to_bytes(16, "big"))
    with ZipFile(path) as archive:
        infos = archive.infolist()
        for info in infos:
            raw = json.dumps(
                [info.filename, info.CRC, info.compress_size, info.file_size],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return {
            "archive_size_bytes": path.stat().st_size,
            "central_directory_entries": len(infos),
            "eligible_member_count": sum(not i.is_dir() for i in infos),
            "sha256": digest.hexdigest(),
        }


def source(name):
    return hashlib.sha256(f"adrenaline/conversations.zip/{name}".encode()).hexdigest()


def measurement(tokens):
    loss = 0.5 + tokens[0] / 100000
    return {"targets": 1023, "nll_sum": loss * 1023, "model_loss": loss}


class SyntheticTokenizer:
    eos_token_id = 2
    pad_token_id = 49109

    def __call__(self, text, **kwargs):
        base = 5000 + int(hashlib.sha256(text.encode()).hexdigest()[:4], 16) % 20000
        return {"input_ids": list(range(base, base + 2300))}


class HoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = (
            h.compile_matcher(Path(cls.temporary.name)) if shutil.which("c++") else None
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_zip_streaming_roundtrip_and_zip64_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversations.zip"
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                with archive.open(
                    "clear_threads/100_0.tsv", "w", force_zip64=True
                ) as out:
                    out.write(b"0\tsynthetic-a\t<p>Example synthetic message.</p>\n")
                archive.writestr(
                    "clear_threads/101_0.tsv",
                    "0\tsynthetic-b\tAnother synthetic message.\n",
                )
            infos = [info for _, _, info in central_directory(path)]
            with ZipFile(path) as normal, IndexedZip(path) as bounded:
                self.assertEqual(len(bounded.infolist()), 0)
                for info in infos:
                    restored = record_info(info_record(info))
                    self.assertEqual(bounded.read(restored), normal.read(info.filename))
                actual, _ = _read_conversation(
                    bounded, infos[0], "adrenaline", path.name, 0
                )
                self.assertEqual(
                    actual.text, "Participante 1: Example synthetic message."
                )

    def test_thread_exclusion_includes_siblings_later_in_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversations.zip"
            with ZipFile(path, "w") as archive:
                for name in ("10_0", "20_0", "30_0", "40_0", "10_9", "20_9"):
                    archive.writestr(
                        f"clear_threads/{name}.tsv", "0\tx\tSynthetic text\n"
                    )
            hashes = {
                source("clear_threads/10_9.tsv"),
                source("clear_threads/20_9.tsv"),
            }
            args = {
                "seed": 17,
                "limit": 10,
                "max_member_bytes": 4096,
                "log": lambda _: None,
            }
            one = select_members(path, hashes, fingerprint(path), **args)
            two = select_members(path, hashes, fingerprint(path), **args)
            self.assertEqual(one, two)
            self.assertEqual(one["excluded_threads"], 2)
            self.assertEqual(
                {r["info"]["filename"] for r in one["members"]},
                {"clear_threads/30_0.tsv", "clear_threads/40_0.tsv"},
            )
            with self.assertRaises(RuntimeError):
                select_members(path, hashes | {"f" * 64}, fingerprint(path), **args)
            changed = dict(fingerprint(path), sha256="a" * 64)
            with self.assertRaises(RuntimeError):
                select_members(path, hashes, changed, **args)

    def test_zip_crc_and_truncation_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("clear_threads/1.tsv", "0\ta\tSynthetic\n")
            infos = [i for _, _, i in central_directory(path)]
            info = infos[0]
            info.CRC ^= 1
            with IndexedZip(path) as bounded, self.assertRaises(BadZipFile):
                bounded.read(info)
            path.write_bytes(path.read_bytes()[:-22])
            with self.assertRaises(BadZipFile):
                list(central_directory(path))

    def test_zip64_central_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zip64.zip"
            # Force the directory's ZIP64 end record with a tiny synthetic fixture.
            with patch("zipfile.ZIP_FILECOUNT_LIMIT", 2), ZipFile(path, "w") as archive:
                for i in range(3):
                    archive.writestr(f"clear_threads/{i}.tsv", "0\tx\tSynthetic\n")
            data = bytearray(path.read_bytes())
            end = data.rfind(b"PK\x05\x06")
            struct.pack_into(
                "<HHLL", data, end + 8, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF
            )
            path.write_bytes(data)
            values = list(central_directory(path))
            self.assertEqual(len(values), 3)
            self.assertEqual(values[-1][1], 3)

    def test_slurm_wrapper_cpu_gpu_and_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            script = root / "scripts/submit_intrinsic_holdout.sh"
            shutil.copyfile(
                h.PROJECT_ROOT / "scripts/submit_intrinsic_holdout.sh", script
            )
            executable = root / "sbatch"
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o700)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}")
            cpu = subprocess.run(
                ["bash", str(script), "prepare"],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            gpu = subprocess.run(
                ["bash", str(script), "evaluate", "123"],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            self.assertNotIn("--gres", cpu)
            self.assertIn("--gres=gpu:L40S:1", gpu)
            self.assertIn("--dependency=afterok:123", gpu)
            self.assertIn(f"--export=ALL,HOLDOUT_PROJECT_DIR={root}", gpu)
            bad = subprocess.run(
                ["bash", str(script), "train"],
                env=env,
                capture_output=True,
                check=False,
            )
            self.assertEqual(bad.returncode, 2)

    def test_matcher_against_bruteforce_randomized(self):
        if self.binary is None:
            self.skipTest("C++ compiler unavailable")
        rng = random.Random(72)
        for _ in range(40):
            queries = [[rng.randrange(8) for _ in range(16)] for _ in range(4)]
            training = [[rng.randrange(8) for _ in range(16)] for _ in range(5)]
            expected = []
            for query in queries:
                values = []
                for end in range(16):
                    best = 0
                    for size in range(1, end + 2):
                        snippet = query[end + 1 - size : end + 1]
                        if any(
                            row[start : start + size] == snippet
                            for row in training
                            for start in range(17 - size)
                        ):
                            best = size
                    values.append(best)
                expected.append(values)
            self.assertEqual(
                h.exact_matches(self.binary, queries, [training]), expected
            )

    def test_matcher_does_not_cross_either_boundary(self):
        if self.binary is None:
            self.skipTest("C++ compiler unavailable")
        query = [[1, 2, 3, 4], [5, 6, 7, 8]]
        train = [[9, 9, 1, 2], [3, 4, 5, 6]]
        self.assertEqual(
            h.exact_matches(self.binary, query, [train]), [[1, 2, 1, 2], [1, 2, 0, 0]]
        )

    def test_matcher_truncated_batch_fails(self):
        if self.binary is None:
            self.skipTest("C++ compiler unavailable")
        with self.assertRaises(ValueError):
            h.exact_matches(self.binary, [[1, 2]], [[[1]]])

    def test_threshold_no_silent_relaxation_and_exact_duplicates(self):
        rows = [{"input_ids": [1, 2]}, {"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
        with self.assertRaises(RuntimeError):
            h.choose_holdout(
                rows, [32, 0, 32], {"reject_match_tokens": 32, "eval_sequences": 2}
            )
        selected, counts = h.choose_holdout(
            rows, [0, 0, 0], {"reject_match_tokens": 32, "eval_sequences": 2}
        )
        self.assertEqual(selected, [rows[0], rows[2]])
        self.assertEqual(counts["eligible_sequences"], 2)

    def test_score_resume_is_byte_identical_and_stdout_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"input_ids": [i] * 1024} for i in range(8)]
            identity = {"model": "synthetic", "cohort": "synthetic"}
            continuous, resumed = Path(tmp) / "one.json", Path(tmp) / "two.json"
            h.score_rows(continuous, identity, rows, measurement)
            calls = 0

            def interrupted(tokens):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("synthetic interruption")
                return measurement(tokens)

            with self.assertRaises(RuntimeError):
                h.score_rows(resumed, identity, rows, interrupted)
            self.assertEqual(len(h.unseal(resumed)["records"]), 3)
            output = io.StringIO()
            with redirect_stdout(output):
                h.score_rows(resumed, identity, rows, measurement)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(continuous.read_bytes(), resumed.read_bytes())
            with self.assertRaises(RuntimeError):
                h.score_rows(
                    resumed, dict(identity, cohort="changed"), rows, measurement
                )

    def test_signal_commits_one_unit_and_rejects_bad_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.json"
            rows = [{"input_ids": [1] * 1024} for _ in range(3)]
            identity = {"model": "synthetic", "cohort": "synthetic"}
            with patch.object(h, "STOP", True), self.assertRaises(InterruptedError):
                h.score_rows(path, identity, rows, measurement)
            self.assertEqual(len(h.unseal(path)["records"]), 1)
            state = h.unseal(path)
            state["records"][0]["targets"] = 1024
            h.seal(path, state)
            with self.assertRaises(RuntimeError):
                h.score_rows(path, identity, rows, measurement)

    def test_checkpoint_corruption_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "state.json"
            h.seal(path, {"count": 1})
            data = json.loads(path.read_text())
            data["data"]["count"] = 2
            path.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                h.unseal(path)
            with self.assertRaises(RuntimeError):
                h.safe_path(root, "../outside")

    def test_only_adrenaline_and_macro_change(self):
        original = {
            "datasets": {
                dataset: {"loss": 2.0, "perplexity": math.exp(2.0)}
                for dataset in h.DATASET_IDS
            },
            "macro": {"loss": 2.0, "perplexity": math.exp(2.0)},
        }
        before = deepcopy(original)
        corrected = h.corrected_metrics(original, {"loss": 1.0, "perplexity": math.e})
        self.assertEqual(original, before)
        for dataset in h.DATASET_IDS:
            if dataset != "adrenaline":
                self.assertEqual(
                    corrected["datasets"][dataset], original["datasets"][dataset]
                )
        self.assertAlmostEqual(corrected["macro"]["loss"], 11 / 6)
        self.assertAlmostEqual(corrected["macro"]["perplexity"], math.exp(11 / 6))

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "PyTorch unavailable locally"
    )
    def test_causal_shift_and_uniform_logits(self):
        import torch

        ids = torch.tensor([[0, 1, 2, 0]])
        logits = torch.zeros((1, 4, 3))
        total, count = h.causal_nll(logits, ids)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(total / count, math.log(3), places=6)
        logits[0, 0, 1] = logits[0, 1, 2] = logits[0, 2, 0] = 20
        good, _ = h.causal_nll(logits, ids)
        self.assertLess(good, 1e-6)
        logits[0, 3, :] = 999  # last logit has no next-token target
        self.assertEqual(h.causal_nll(logits, ids)[0], good)

    def make_runtime(self, parent, name):
        root = parent / name
        root.mkdir()
        archive = parent / "conversations.zip"
        if not archive.exists():
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
                for i in range(1, 15):
                    zipped.writestr(
                        f"clear_threads/{i}_0.tsv",
                        f"0\tsynthetic\tSynthetic conversation number {i}.\n",
                    )
        prepared = parent / "prepared"
        prepared.mkdir(exist_ok=True)
        schema = pa.schema(
            [
                ("input_ids", pa.list_(pa.int32(), 1024)),
                ("source_ref_sha256", pa.list_(pa.string())),
            ]
        )
        records = {}
        for split, thread in (("train", 1), ("eval", 2)):
            path = prepared / f"{split}.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "input_ids": list(range(1024)),
                            "source_ref_sha256": [
                                source(f"clear_threads/{thread}_0.tsv")
                            ],
                        }
                    ],
                    schema=schema,
                ),
                path,
            )
            records[split] = [dict(h.file_record(prepared, path), rows=1)]
        manifest = {
            "schema_version": "queroquero-dataset-manifest/v1",
            "dataset_id": "adrenaline",
            "profile": "paired_real",
            "preparation_id": "0" * 20,
            "counts": {"train_sequences": 1, "eval_sequences": 1},
            "splits": records,
            "source": {"fingerprint": fingerprint(archive)},
        }
        path = prepared / "dataset_manifest.json"
        h.write_json_atomic(path, manifest)
        config = h.load_config(
            h.PROJECT_ROOT / "configs/intrinsic/adrenaline-holdout-v1.json"
        )
        config.update(
            candidate_threads=12,
            candidate_sequences=4,
            eval_sequences=2,
            match_batch_sequences=2,
        )
        runtime = {
            "root": root,
            "config": config,
            "archive": archive,
            "manifests": {"adrenaline": (path, manifest, h.file_sha256(path))},
            "tokenizer": {"vocab_size": 49152, "pad_token_id": 49109},
            "identity": {"source_stat": stat_identity(archive)},
        }
        return runtime

    def test_prepare_validate_resume_and_aggregate_report(self):
        if self.binary is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(Path(tmp), "test")
            with (
                h.locked(runtime),
                patch.object(h, "load_tokenizer", return_value=SyntheticTokenizer()),
            ):
                output = h.prepare(runtime)
                self.assertEqual(output["status"], "prepared")
                before = (runtime["root"] / "holdout.parquet").read_bytes()
                with patch.object(
                    h, "build_candidates", side_effect=AssertionError("must resume")
                ):
                    h.prepare(runtime)
                holdout, rows = h.validate_holdout(runtime)
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    before, (runtime["root"] / "holdout.parquet").read_bytes()
                )
                values = {
                    dataset: {
                        "loss": measurement(list(range(1024)))["model_loss"],
                        "perplexity": math.exp(0.5),
                    }
                    for dataset in h.DATASET_IDS
                }
                original = {
                    "datasets": values,
                    "macro": {"loss": 0.5, "perplexity": math.exp(0.5)},
                }
                runtime["report"] = {
                    "arms": {
                        arm: {
                            "baseline_evaluation": original,
                            "final_evaluation": original,
                        }
                        for arm in ("general", "forum_tech")
                    }
                }
                runtime["config"]["original_loss_tolerance"] = 0.001
                for model in h.MODELS:
                    for cohort, cohort_rows in (
                        ("original", h.original_rows(runtime)),
                        ("thread_external", rows),
                    ):
                        identity = {
                            "evaluation_id": runtime["root"].name,
                            "model": model,
                            "cohort": cohort,
                            "holdout_sha256": holdout["file"]["sha256"],
                            "numeric_environment": {"synthetic": True},
                        }
                        h.score_rows(
                            runtime["root"] / f"scores-{model}-{cohort}.json",
                            identity,
                            cohort_rows,
                            measurement,
                        )
                result = h.report_results(runtime)
                public = Path(result["report"])
                serialized = public.read_text()
                self.assertNotIn("input_ids", serialized)
                self.assertNotIn("Synthetic conversation", serialized)
                self.assertNotIn("thread_sha256", serialized)
                self.assertNotIn("http", serialized)
                self.assertNotIn(str(Path(tmp)), serialized)
                report_before = public.read_bytes()
                h.report_results(runtime)
                self.assertEqual(report_before, public.read_bytes())
                score_path = runtime["root"] / "scores-base-original.json"
                score = h.unseal(score_path)
                score["records"][0].update(nll_sum=1023.0, model_loss=1.0)
                h.seal(score_path, score)
                with self.assertRaisesRegex(RuntimeError, "reproduction gate"):
                    h.report_results(runtime)
                self.assertEqual(report_before, public.read_bytes())

    def test_prepare_interrupted_match_resumes_without_retokenization(self):
        if self.binary is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(Path(tmp), "resumed")
            with (
                h.locked(runtime),
                patch.object(h, "load_tokenizer", return_value=SyntheticTokenizer()),
            ):
                original_match = h.exact_matches
                calls = 0

                def interrupt(binary, queries, batches):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("synthetic matcher interruption")
                    return original_match(binary, queries, batches)

                with (
                    patch.object(h, "exact_matches", side_effect=interrupt),
                    self.assertRaises(RuntimeError),
                ):
                    h.prepare(runtime)
                self.assertTrue((runtime["root"] / "match-000000.json").is_file())
                with patch.object(
                    h,
                    "load_tokenizer",
                    side_effect=AssertionError("must not tokenize again"),
                ):
                    h.prepare(runtime)
                h.validate_holdout(runtime)
            continuous = self.make_runtime(Path(tmp), "continuous")
            with (
                h.locked(continuous),
                patch.object(h, "load_tokenizer", return_value=SyntheticTokenizer()),
            ):
                h.prepare(continuous)
            for relative in (
                "holdout.parquet",
                "candidate_state.json",
                "match-000000.json",
                "match-000002.json",
            ):
                self.assertEqual(
                    (runtime["root"] / relative).read_bytes(),
                    (continuous["root"] / relative).read_bytes(),
                )

    def test_resolve_binds_six_original_manifests_and_models(self):
        config = h.load_config(
            h.PROJECT_ROOT / "configs/intrinsic/adrenaline-holdout-v1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            prepared = project / "prepared"
            raw = project / "raw"
            (raw / "adrenaline").mkdir(parents=True)
            (raw / "adrenaline/conversations.zip").write_bytes(
                b"synthetic-identity-only"
            )
            tokenizer = {"fingerprint_sha256": "a" * 64}
            records, paired_records = [], []
            for i, dataset in enumerate(h.DATASET_IDS):
                preparation_id = f"{i:020x}"
                relative = f"{dataset}/{preparation_id}/dataset_manifest.json"
                manifest = {
                    "dataset_id": dataset,
                    "preparation_id": preparation_id,
                    "profile": "paired_real",
                    "counts": {"train_sequences": 5, "eval_sequences": 2},
                    "tokenizer": tokenizer,
                }
                h.write_json_atomic(prepared / relative, manifest)
                digest = h.file_sha256(prepared / relative)
                records.append(
                    {
                        "dataset_id": dataset,
                        "manifest_path": relative,
                        "dataset_manifest_sha256": digest,
                        "preparation_id": preparation_id,
                    }
                )
                paired_records.append(
                    {
                        "dataset_id": dataset,
                        "preparation_id": preparation_id,
                        "dataset_manifest_sha256": digest,
                        "prepared_train_sequences": 5,
                        "eval_sequences": 2,
                    }
                )
                if dataset == "adrenaline":
                    config["cpt_manifest_sha256"] = digest
            common = {
                "experiment_id": "b" * 20,
                "allocation_sha256": "c" * 64,
                "schedule_template_sha256": "d" * 64,
            }
            paired = {
                "schema_version": "queroquero-paired-resolved-inputs/v1",
                **common,
                "preparation_profile": "paired_real",
                "datasets": paired_records,
            }
            digest = h.sha256_bytes(h.canonical_json_bytes(paired))
            report = {
                "report_id": config["paired_report_id"],
                "paired_inputs_sha256": digest,
                "arms": {},
            }
            models = {}
            for name, arm in (("general", "general"), ("forum", "forum_tech")):
                expected = config["models"][name]
                report["arms"][arm] = expected
                models[expected["artifact_id"]] = {
                    "artifact_sha256": expected["artifact_sha256"],
                    "training": {
                        "run_id": expected["run_id"],
                        "experiment": {"arm": arm, "paired_inputs_sha256": digest},
                    },
                    "tokenizer": {"prepared_fingerprint_sha256": "a" * 64},
                }
                run = {
                    "run_id": expected["run_id"],
                    "inputs": {
                        "data_mixture": dict(common, arm=arm),
                        "paired_inputs_sha256": digest,
                        "preparation_profile": "paired_real",
                        "datasets": records,
                    },
                }
                h.write_json_atomic(
                    project / f"runs/{expected['run_id']}/resolved_training.json", run
                )
            h.write_json_atomic(project / config["paired_report_path"], report)
            with (
                patch.object(h, "PROJECT_ROOT", project),
                patch.dict(
                    os.environ,
                    {"PTBR_OUTPUT_ROOT": str(prepared), "PTBR_DATASET_ROOT": str(raw)},
                ),
                patch.object(
                    h,
                    "validate_paired_experiment_report",
                    side_effect=lambda value: value,
                ),
                patch.object(
                    h,
                    "validate_model_artifact",
                    side_effect=lambda path, **kw: models[path.name],
                ),
            ):
                resolved = h.resolve(config)
                self.assertEqual(set(resolved["manifests"]), set(h.DATASET_IDS))
                self.assertEqual(h.resolve(config)["identity"], resolved["identity"])
                changed = deepcopy(config)
                changed["models"]["forum"]["artifact_sha256"] = "f" * 64
                with self.assertRaises(RuntimeError):
                    h.resolve(changed)
                path = prepared / records[-1]["manifest_path"]
                value = h.load_json(path)
                value["counts"]["train_sequences"] += 1
                h.write_json_atomic(path, value)
                with self.assertRaises(RuntimeError):
                    h.resolve(config)


if __name__ == "__main__":
    unittest.main()
