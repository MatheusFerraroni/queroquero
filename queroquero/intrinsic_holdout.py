"""Thread-external Adrenaline evaluation of saved models; never trains a model.

Private token shards/checkpoints are separate from aggregate reports. Preparation
and scoring are independent, idempotent commands; no Slurm/SSH calls from Python.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .classification_data import _load_cpt_source_hashes
from .config import (
    DATASET_IDS,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ROOT,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
)
from .datasets._conversation_zip import _read_conversation
from .datasets.base import clean_text, stable_hash
from .experiment_report import validate_paired_experiment_report
from .holdout_zip import IndexedZip, record_info, select_members, stat_identity
from .manifest import file_sha256, write_json_atomic
from .model_artifact import validate_model_artifact
from .packing import tokenizer_fingerprint

SCHEMA = "queroquero-intrinsic-holdout/v1"
HERE = Path(__file__).resolve().parent
MODELS = ("base", "general", "forum")
WIDTH = 1024
STOP = False


def log(message):
    print(time.strftime("%H:%M:%S") + " | " + message, file=sys.stderr, flush=True)


def stop_requested(signum, frame):
    global STOP
    STOP = True
    log("Interrupção solicitada; encerrando após a unidade atômica atual")


def check_stop():
    if STOP:
        raise InterruptedError(
            "checkpoint preservado; execute novamente o mesmo comando"
        )


def seal(path, data):
    write_json_atomic(
        path, {"data": data, "sha256": sha256_bytes(canonical_json_bytes(data))}
    )


def unseal(path):
    value = load_json(path)
    if set(value) != {"data", "sha256"} or value["sha256"] != sha256_bytes(
        canonical_json_bytes(value["data"])
    ):
        raise RuntimeError("checkpoint JSON checksum mismatch")
    return value["data"]


def safe_path(root, relative):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise RuntimeError("unsafe relative artifact path")
    path = root / part
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise RuntimeError("artifact path escapes its configured root")
    return path


def file_record(root, path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def checked_file(root, record):
    path = safe_path(root, record["path"])
    before = stat_identity(path)
    if (
        before["size_bytes"] != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError("artifact file checksum mismatch")
    if before != stat_identity(path):
        raise RuntimeError("artifact changed while hashing")
    return path


def environment():
    return {
        "python": platform.python_version(),
        **{
            key: importlib.metadata.version(key)
            for key in ("pyarrow", "transformers", "tokenizers")
        },
    }


def load_config(path):
    config = load_json(path)
    expected = {
        "schema_version",
        "seed",
        "sequence_length",
        "eval_sequences",
        "candidate_threads",
        "candidate_sequences",
        "max_member_bytes",
        "match_batch_sequences",
        "reject_match_tokens",
        "original_loss_tolerance",
        "paired_report_path",
        "paired_report_id",
        "cpt_manifest_sha256",
        "archive",
        "dataset_root_env",
        "prepared_root_env",
        "output_root_env",
        "output_default",
        "models",
    }
    if (
        set(config) != expected
        or config["schema_version"] != "queroquero-intrinsic-holdout-config/v1"
    ):
        raise ValueError("invalid intrinsic holdout config schema")
    for key in (
        "seed",
        "eval_sequences",
        "candidate_threads",
        "candidate_sequences",
        "max_member_bytes",
        "match_batch_sequences",
        "reject_match_tokens",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"invalid positive budget: {key}")
    if (
        config["sequence_length"] != WIDTH
        or not config["eval_sequences"]
        <= config["candidate_sequences"]
        <= config["candidate_threads"]
        or config["candidate_sequences"] > 16384
        or config["candidate_threads"] > 65536
        or config["max_member_bytes"] > 16 * 1024 * 1024
        or config["match_batch_sequences"] * WIDTH > 1000000
        or not 13 <= config["reject_match_tokens"] <= WIDTH
        or not 0 < config["original_loss_tolerance"] <= 0.01
        or set(config["models"]) != {"general", "forum"}
    ):
        raise ValueError("unsupported holdout budgets or model contract")
    return config


def resolve(config):
    """Bind all six prepared pools to the original run/report, never latest paths."""
    report_path = safe_path(PROJECT_ROOT, config["paired_report_path"])
    report = validate_paired_experiment_report(load_json(report_path))
    if report["report_id"] != config["paired_report_id"]:
        raise RuntimeError("original paired report identity changed")
    prepared_root = Path(os.environ[config["prepared_root_env"]]).expanduser().resolve()
    raw_root = Path(os.environ[config["dataset_root_env"]]).expanduser().resolve()
    manifests, models, run_hashes = {}, {}, {}
    for name, arm in (("general", "general"), ("forum", "forum_tech")):
        expected = config["models"][name]
        if any(
            report["arms"][arm][key] != expected[key]
            for key in ("artifact_id", "artifact_sha256", "run_id")
        ):
            raise RuntimeError("original report model identity changed")
        model = validate_model_artifact(
            safe_path(PROJECT_ROOT, "artifacts/" + expected["artifact_id"]),
            load_model=False,
        )
        if (
            model["artifact_sha256"] != expected["artifact_sha256"]
            or model["training"]["run_id"] != expected["run_id"]
            or model["training"]["experiment"]["arm"] != arm
            or model["training"]["experiment"]["paired_inputs_sha256"]
            != report["paired_inputs_sha256"]
        ):
            raise RuntimeError("saved model provenance changed")
        models[name] = model
        run_path = safe_path(
            PROJECT_ROOT, f"runs/{expected['run_id']}/resolved_training.json"
        )
        run = load_json(run_path)
        inputs = run["inputs"]
        mixture = inputs["data_mixture"]
        if (
            run["run_id"] != expected["run_id"]
            or mixture["arm"] != arm
            or inputs["paired_inputs_sha256"] != report["paired_inputs_sha256"]
            or inputs["preparation_profile"] != "paired_real"
        ):
            raise RuntimeError("resolved run identity changed")
        paired = {
            "schema_version": "queroquero-paired-resolved-inputs/v1",
            "experiment_id": mixture["experiment_id"],
            "allocation_sha256": mixture["allocation_sha256"],
            "schedule_template_sha256": mixture["schedule_template_sha256"],
            "preparation_profile": inputs["preparation_profile"],
            "datasets": [],
        }
        records = inputs["datasets"]
        if len(records) != 6 or {r["dataset_id"] for r in records} != set(DATASET_IDS):
            raise RuntimeError("expected six original training pools")
        for record in records:
            dataset = record["dataset_id"]
            manifest_path = safe_path(prepared_root, record["manifest_path"])
            digest = file_sha256(manifest_path)
            manifest = load_json(manifest_path)
            if (
                digest != record["dataset_manifest_sha256"]
                or manifest["dataset_id"] != dataset
                or manifest["preparation_id"] != record["preparation_id"]
                or manifest["profile"] != "paired_real"
            ):
                raise RuntimeError("original prepared pool changed")
            if dataset in manifests and manifests[dataset][1] != manifest:
                raise RuntimeError("models do not share the original prepared pools")
            manifests[dataset] = (manifest_path, manifest, digest)
            paired["datasets"].append(
                {
                    "dataset_id": dataset,
                    "preparation_id": manifest["preparation_id"],
                    "dataset_manifest_sha256": digest,
                    "prepared_train_sequences": manifest["counts"]["train_sequences"],
                    "eval_sequences": manifest["counts"]["eval_sequences"],
                }
            )
        if sha256_bytes(canonical_json_bytes(paired)) != report["paired_inputs_sha256"]:
            raise RuntimeError(
                "resolved training inputs differ from the original experiment"
            )
        run_hashes[name] = file_sha256(run_path)
    adrenaline = manifests["adrenaline"]
    if adrenaline[2] != config["cpt_manifest_sha256"]:
        raise RuntimeError("Adrenaline preparation changed")
    tokenizer = adrenaline[1]["tokenizer"]
    if any(item[1]["tokenizer"] != tokenizer for item in manifests.values()):
        raise RuntimeError("training pool tokenizer mismatch")
    if any(
        model["tokenizer"]["prepared_fingerprint_sha256"]
        != tokenizer["fingerprint_sha256"]
        for model in models.values()
    ):
        raise RuntimeError("saved model tokenizer mismatch")
    archive = safe_path(raw_root, config["archive"])
    identity = {
        "schema_version": SCHEMA,
        "config": config,
        "paired_report_sha256": file_sha256(report_path),
        "run_sha256": run_hashes,
        "prepared_manifest_sha256": {k: v[2] for k, v in manifests.items()},
        "source_stat": stat_identity(archive),
        "environment": environment(),
        "implementation_sha256": {
            name: file_sha256(HERE / name)
            for name in (
                "intrinsic_holdout.py",
                "holdout_zip.py",
                "holdout_match.cpp",
                "packing.py",
                "datasets/base.py",
                "datasets/_conversation_zip.py",
            )
        },
    }
    evaluation_id = sha256_bytes(canonical_json_bytes(identity))[:20]
    output_root = Path(
        os.environ.get(
            config["output_root_env"],
            str(safe_path(PROJECT_ROOT, config["output_default"])),
        )
    )
    root = output_root.expanduser().resolve() / evaluation_id
    return {
        "root": root,
        "identity": identity,
        "config": config,
        "report": report,
        "manifests": manifests,
        "models": models,
        "archive": archive,
        "tokenizer": tokenizer,
    }


@contextlib.contextmanager
def locked(runtime):
    root = runtime["root"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another holdout command is active in this output directory"
            ) from None
        identity_file = root / "identity.json"
        if identity_file.exists() and unseal(identity_file) != runtime["identity"]:
            raise RuntimeError("holdout identity changed")
        if not identity_file.exists():
            seal(identity_file, runtime["identity"])
        log(f"Avaliação: {root.name}; saída privada: {root}")
        yield


def token_schema():
    return pa.schema(
        [
            ("thread_sha256", pa.string()),
            ("source_sha256", pa.string()),
            ("block_index", pa.int64()),
            ("input_ids", pa.list_(pa.int32(), WIDTH)),
        ]
    )


def atomic_tokens(root, relative, rows):
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=token_schema()), partial, compression="zstd"
    )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    if pq.ParquetFile(partial).metadata.num_rows != len(rows):
        raise RuntimeError("token chunk validation failed")
    partial.replace(path)
    return file_record(root, path)


def read_tokens(root, record):
    return pq.read_table(checked_file(root, record)).to_pylist()


def load_tokenizer(runtime):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer_fingerprint(tokenizer) != runtime["tokenizer"]["fingerprint_sha256"]:
        raise RuntimeError("pinned tokenizer fingerprint changed")
    return tokenizer


def build_candidates(runtime, index):
    root, config = runtime["root"], runtime["config"]
    path = root / "candidate_state.json"
    state = (
        unseal(path)
        if path.exists()
        else {"next_member": 0, "chunks": [], "sequences": 0, "short": 0, "pad": 0}
    )
    for record in state["chunks"]:
        checked_file(root, record)
    if state.get("complete"):
        return state
    tokenizer = load_tokenizer(runtime)
    rows = []
    members = index["members"]
    start = time.monotonic()
    with IndexedZip(runtime["archive"]) as archive:
        for i in range(state["next_member"], len(members)):
            item = members[i]
            document, _ = _read_conversation(
                archive, record_info(item["info"]), "adrenaline", "conversations.zip", i
            )
            values = (
                []
                if document is None
                else tokenizer(
                    clean_text(document.text, strip_html=True), add_special_tokens=False
                )["input_ids"]
            )
            values = list(values) + [tokenizer.eos_token_id]
            count = len(values) // WIDTH
            if not count:
                state["short"] += 1
            else:
                block = (
                    int(
                        stable_hash(
                            "intrinsic-holdout-block/v1",
                            config["seed"],
                            item["source_sha256"],
                        ),
                        16,
                    )
                    % count
                )
                tokens = values[block * WIDTH : (block + 1) * WIDTH]
                if tokenizer.pad_token_id in tokens:
                    state["pad"] += 1
                elif any(
                    type(t) is not int
                    or not 0 <= t < runtime["tokenizer"]["vocab_size"]
                    for t in tokens
                ):
                    raise RuntimeError("invalid candidate token IDs")
                else:
                    rows.append(
                        {
                            "thread_sha256": item["thread_sha256"],
                            "source_sha256": item["source_sha256"],
                            "block_index": block,
                            "input_ids": tokens,
                        }
                    )
            state["next_member"] = i + 1
            reached = state["sequences"] + len(rows) >= config["candidate_sequences"]
            if (i + 1) % 32 == 0 or reached or i + 1 == len(members) or STOP:
                if rows:
                    state["chunks"].append(
                        atomic_tokens(
                            root, f"candidates/{len(state['chunks']):06d}.parquet", rows
                        )
                    )
                    state["sequences"] += len(rows)
                    rows = []
                if (
                    stat_identity(runtime["archive"])
                    != runtime["identity"]["source_stat"]
                ):
                    raise RuntimeError("archive changed during tokenization")
                state["complete"] = reached or i + 1 == len(members)
                seal(path, state)
                log(
                    f"Candidatos: {state['sequences']:,}; threads examinadas {i + 1:,}/{len(members):,}; {time.monotonic() - start:.0f}s"
                )
                check_stop()
            if reached:
                break
    return state


def compile_matcher(root):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None or sys.byteorder != "little":
        raise RuntimeError(
            "preparation requires a C++17 compiler and a little-endian host"
        )
    binary = root / "match-tokens"
    partial = root / "match-tokens.partial"
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            str(HERE / "holdout_match.cpp"),
            "-o",
            str(partial),
        ],
        check=True,
    )
    partial.replace(binary)
    return binary


def ids_bytes(rows):
    return array.array("I", (token for row in rows for token in row)).tobytes()


def exact_matches(binary, queries, training_batches):
    """Exact matching; no probabilistic membership or cross-block matches."""
    if not queries or len(queries) * len(queries[0]) > 1000000:
        raise ValueError("matcher query budget exceeded")
    width = len(queries[0])
    if any(len(row) != width for row in queries):
        raise ValueError("inconsistent query block length")
    process = subprocess.Popen(
        [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        process.stdin.write(struct.pack("<II", width, len(queries)))
        process.stdin.write(ids_bytes(queries))
        for rows in training_batches:
            if not rows:
                continue
            if len(rows) > 4096 or any(len(row) != width for row in rows):
                raise ValueError("invalid matcher training batch")
            process.stdin.write(struct.pack("<I", len(rows)))
            process.stdin.write(ids_bytes(rows))
        process.stdin.write(struct.pack("<I", 0))
        process.stdin.close()
        raw = process.stdout.read()
        if process.wait() or len(raw) != len(queries) * width * 4:
            raise RuntimeError("exact matcher failed or returned truncated output")
        values = array.array("I")
        values.frombytes(raw)
        result = [list(values[i : i + width]) for i in range(0, len(values), width)]
        if any(value > pos + 1 for row in result for pos, value in enumerate(row)):
            raise RuntimeError("matcher crossed query boundaries")
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
        process.stdout.close()


def training_batches(runtime):
    for dataset, (path, manifest, _) in sorted(runtime["manifests"].items()):
        total = 0
        start = time.monotonic()
        for record in manifest["splits"]["train"]:
            shard = checked_file(path.parent, record)
            before = stat_identity(shard)
            count = 0
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=1024, columns=["input_ids"]
            ):
                rows = batch.column(0).to_pylist()
                if any(
                    len(row) != WIDTH
                    or any(not 0 <= t < runtime["tokenizer"]["vocab_size"] for t in row)
                    for row in rows
                ):
                    raise RuntimeError("invalid training token block")
                count += len(rows)
                yield rows
                check_stop()
            if count != record["rows"] or before != stat_identity(shard):
                raise RuntimeError(
                    "training shard changed or has inconsistent row count"
                )
            total += count
            if total % 8192 < count:
                log(
                    f"Comparação exata/{dataset}: {total:,}/{manifest['counts']['train_sequences']:,} blocos; {time.monotonic() - start:.0f}s"
                )
        if total != manifest["counts"]["train_sequences"]:
            raise RuntimeError("training pool count mismatch")


def choose_holdout(candidates, maxima, config):
    if len(candidates) != len(maxima):
        raise RuntimeError("candidate/matching coverage mismatch")
    accepted, seen, repeated = [], set(), 0
    for candidate, maximum in zip(candidates, maxima, strict=True):
        if maximum >= config["reject_match_tokens"]:
            repeated += 1
            continue
        key = tuple(candidate["input_ids"])
        if key not in seen:
            accepted.append(candidate)
            seen.add(key)
    if len(accepted) < config["eval_sequences"]:
        raise RuntimeError(
            f"holdout insufficient: {len(accepted)} eligible; required {config['eval_sequences']}; do not relax thresholds automatically"
        )
    return accepted[: config["eval_sequences"]], {
        "candidate_sequences": len(candidates),
        "repeated_against_train": repeated,
        "eligible_sequences": len(accepted),
        "selected_sequences": config["eval_sequences"],
    }


def prepare(runtime):
    root, config = runtime["root"], runtime["config"]
    log("Conferindo checksums dos pools de treino antes de reutilizar checkpoints")
    for path, manifest, _ in runtime["manifests"].values():
        for record in manifest["splits"]["train"]:
            checked_file(path.parent, record)
    if (root / "holdout.json").exists():
        validate_holdout(runtime)
        return {"status": "prepared", "evaluation_id": root.name, "output": str(root)}
    binary = compile_matcher(root)
    path = root / "source_index.json"
    if path.exists():
        index = unseal(path)
        if index["source_stat"] != stat_identity(runtime["archive"]):
            raise RuntimeError("source changed since indexing")
    else:
        manifest_path, manifest, digest = runtime["manifests"]["adrenaline"]
        train, evaluation, expected = _load_cpt_source_hashes(
            manifest_path,
            {"sha256": digest, "preparation_id": manifest["preparation_id"]},
        )
        index = select_members(
            runtime["archive"],
            train | evaluation,
            expected,
            seed=config["seed"],
            limit=config["candidate_threads"],
            max_member_bytes=config["max_member_bytes"],
            log=log,
        )
        seal(path, index)
        check_stop()
    state = build_candidates(runtime, index)
    candidates = [
        row for record in state["chunks"] for row in read_tokens(root, record)
    ]
    maxima = []
    for offset in range(0, len(candidates), config["match_batch_sequences"]):
        subset = candidates[offset : offset + config["match_batch_sequences"]]
        match_path = root / f"match-{offset:06d}.json"
        if match_path.exists():
            result = unseal(match_path)
        else:
            log(
                f"Comparação exata: candidatos {offset + 1}–{offset + len(subset)}/{len(candidates)} contra os seis pools de treino"
            )
            values = exact_matches(
                binary, [row["input_ids"] for row in subset], training_batches(runtime)
            )
            result = {
                "offset": offset,
                "maxima": [max(row) for row in values],
                "candidates_sha256": sha256_bytes(canonical_json_bytes(subset)),
            }
            seal(match_path, result)
        if result["offset"] != offset or result["candidates_sha256"] != sha256_bytes(
            canonical_json_bytes(subset)
        ):
            raise RuntimeError("matching checkpoint candidate identity mismatch")
        maxima.extend(result["maxima"])
        check_stop()
    selected, counts = choose_holdout(candidates, maxima, config)
    record = atomic_tokens(root, "holdout.parquet", selected)
    holdout = {
        "schema_version": SCHEMA,
        "evaluation_id": root.name,
        "counts": counts,
        "file": record,
        "excluded_threads": index["excluded_threads"],
        "unique_threads": len({r["thread_sha256"] for r in selected}),
        "matching_scope": "all prepared train blocks of the six original paired pools",
        "threshold_tokens": config["reject_match_tokens"],
        "targets": len(selected) * (WIDTH - 1),
        "source_index_sha256": file_sha256(root / "source_index.json"),
        "candidate_state_sha256": file_sha256(root / "candidate_state.json"),
        "match_files": [
            file_record(root, root / f"match-{i:06d}.json")
            for i in range(0, len(candidates), config["match_batch_sequences"])
        ],
    }
    seal(root / "holdout.json", holdout)
    validate_holdout(runtime)
    return {
        "status": "prepared",
        "evaluation_id": root.name,
        "output": str(root),
        "counts": counts,
    }


def validate_holdout(runtime):
    root = runtime["root"]
    holdout = unseal(root / "holdout.json")
    if (
        holdout["evaluation_id"] != root.name
        or unseal(root / "identity.json") != runtime["identity"]
    ):
        raise RuntimeError("holdout identity mismatch")
    for filename, key in (
        ("source_index.json", "source_index_sha256"),
        ("candidate_state.json", "candidate_state_sha256"),
    ):
        if file_sha256(root / filename) != holdout[key]:
            raise RuntimeError("holdout preparation checkpoint changed")
    index = unseal(root / "source_index.json")
    state = unseal(root / "candidate_state.json")
    candidates = [row for rec in state["chunks"] for row in read_tokens(root, rec)]
    maxima = []
    offset = 0
    for record in holdout["match_files"]:
        result = unseal(checked_file(root, record))
        subset = candidates[offset : offset + len(result["maxima"])]
        if result["offset"] != offset or result["candidates_sha256"] != sha256_bytes(
            canonical_json_bytes(subset)
        ):
            raise RuntimeError("matching proof mismatch")
        maxima.extend(result["maxima"])
        offset += len(subset)
    expected, counts = choose_holdout(candidates, maxima, runtime["config"])
    selected = read_tokens(root, holdout["file"])
    if selected != expected or counts != holdout["counts"]:
        raise RuntimeError("holdout differs from its deterministic selection")
    threads = {r["thread_sha256"] for r in selected}
    if len(threads) != len(selected) or threads & set(index["excluded_thread_hashes"]):
        raise RuntimeError(
            "holdout threads are repeated or exposed to CPT/original evaluation"
        )
    if any(
        len(r["input_ids"]) != WIDTH
        or runtime["tokenizer"]["pad_token_id"] in r["input_ids"]
        for r in selected
    ):
        raise RuntimeError("holdout block length/padding mismatch")
    if holdout["targets"] != len(selected) * (WIDTH - 1):
        raise RuntimeError("holdout causal denominator mismatch")
    return holdout, selected


def original_rows(runtime):
    path, manifest, _ = runtime["manifests"]["adrenaline"]
    rows = []
    for record in manifest["splits"]["eval"]:
        shard = checked_file(path.parent, record)
        values = pq.read_table(shard, columns=["input_ids"]).column(0).to_pylist()
        if len(values) != record["rows"]:
            raise RuntimeError("original evaluation shard count mismatch")
        rows.extend({"input_ids": value} for value in values)
    if len(rows) != manifest["counts"]["eval_sequences"] or any(
        len(r["input_ids"]) != WIDTH for r in rows
    ):
        raise RuntimeError("original evaluation shape changed")
    return rows


def original_metrics(report, name):
    arm = "forum_tech" if name == "forum" else "general"
    return report["arms"][arm][
        "baseline_evaluation" if name == "base" else "final_evaluation"
    ]


def causal_nll(logits, input_ids):
    """Explicit next-token targets: position 0 is context, not a supervised target."""
    import torch

    if (
        logits.shape[:2] != input_ids.shape
        or input_ids.ndim != 2
        or input_ids.shape[1] < 2
    ):
        raise RuntimeError("invalid causal evaluation dimensions")
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    )
    if not torch.isfinite(losses).all():
        raise RuntimeError("non-finite causal loss")
    return float(losses.double().sum().item()), losses.numel()


def validate_score(state, identity, rows):
    if state["identity"] != identity or len(state["records"]) > len(rows):
        raise RuntimeError("score checkpoint identity/count mismatch")
    for index, record in enumerate(state["records"]):
        if (
            record["index"] != index
            or record["input_sha256"]
            != hashlib.sha256(ids_bytes([rows[index]["input_ids"]])).hexdigest()
            or record["targets"] != WIDTH - 1
            or not math.isfinite(record["nll_sum"])
            or record["nll_sum"] < 0
            or not math.isfinite(record["model_loss"])
            or abs(record["nll_sum"] / record["targets"] - record["model_loss"]) > 1e-5
        ):
            raise RuntimeError("score checksum, target shift or denominator mismatch")


def score_rows(path, identity, rows, measure):
    state = unseal(path) if path.exists() else {"identity": identity, "records": []}
    validate_score(state, identity, rows)
    start = time.monotonic()
    initial = len(state["records"])
    log(
        f"Inferência {identity['model']}/{identity['cohort']}: retomada {initial}/{len(rows)} blocos"
    )
    for index in range(initial, len(rows)):
        record = measure(rows[index]["input_ids"])
        record.update(
            index=index,
            input_sha256=hashlib.sha256(
                ids_bytes([rows[index]["input_ids"]])
            ).hexdigest(),
        )
        state["records"].append(record)
        validate_score(state, identity, rows)
        seal(path, state)
        completed = index + 1 - initial
        if completed % 8 == 0 or index + 1 == len(rows):
            elapsed = time.monotonic() - start
            eta = elapsed / completed * (len(rows) - index - 1)
            log(
                f"Inferência {identity['model']}/{identity['cohort']}: {index + 1}/{len(rows)}; {completed / max(elapsed, 0.001):.2f} blocos/s; ETA {eta:.0f}s"
            )
        check_stop()
    return state


def summarize_scores(state):
    if not state["records"]:
        raise RuntimeError("empty evaluation scores")
    targets = sum(r["targets"] for r in state["records"])
    nll_sum = math.fsum(r["nll_sum"] for r in state["records"])
    loss = nll_sum / targets
    return {
        "sequences": len(state["records"]),
        "tokens": len(state["records"]) * WIDTH,
        "targets": targets,
        "nll_sum": nll_sum,
        "loss": loss,
        "perplexity": math.exp(loss),
    }


def evaluate(runtime):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        torch.__version__ != "2.7.1+cu118"
        or importlib.metadata.version("transformers") != "5.14.1"
    ):
        raise RuntimeError(
            "use the original environment: torch 2.7.1+cu118 and transformers 5.14.1"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("inference requires one CUDA GPU with BF16 support")
    holdout, selected = validate_holdout(runtime)
    cohorts = {"original": original_rows(runtime), "thread_external": selected}
    torch.manual_seed(runtime["config"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    numeric_environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weights": "float32",
        "autocast": "bfloat16",
        "attention": "eager",
        "batch_size": 1,
    }
    root = runtime["root"]
    for name in MODELS:
        source = (
            MODEL_ID
            if name == "base"
            else PROJECT_ROOT / "artifacts" / runtime["models"][name]["artifact_id"]
        )
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if name == "base":
            kwargs["revision"] = MODEL_REVISION
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        if (
            tokenizer_fingerprint(tokenizer)
            != runtime["tokenizer"]["fingerprint_sha256"]
        ):
            raise RuntimeError("evaluation model tokenizer differs from prepared input")
        log(f"Carregando modelo {name} para inferência (sem treinamento)")
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.float32, attn_implementation="eager", **kwargs
        )
        if (
            model.config.model_type != "llama"
            or int(model.config.max_position_embeddings) != 4096
            or sum(p.numel() for p in model.parameters()) != 670127616
        ):
            raise RuntimeError("saved model architecture changed")
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        model.to(device)

        def measure(tokens, current_model=model):
            input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = current_model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        labels=input_ids,
                        use_cache=False,
                    )
                total, targets = causal_nll(output.logits, input_ids)
                return {
                    "nll_sum": total,
                    "targets": targets,
                    "model_loss": float(output.loss.item()),
                }

        for cohort, rows in cohorts.items():
            identity = {
                "evaluation_id": root.name,
                "model": name,
                "cohort": cohort,
                "holdout_sha256": holdout["file"]["sha256"],
                "numeric_environment": numeric_environment,
            }
            state = score_rows(
                root / f"scores-{name}-{cohort}.json", identity, rows, measure
            )
            if cohort == "original":
                observed = summarize_scores(state)["loss"]
                expected = original_metrics(runtime["report"], name)["datasets"][
                    "adrenaline"
                ]["loss"]
                difference = observed - expected
                seal(
                    root / f"control-{name}.json",
                    {
                        "expected_loss": expected,
                        "observed_loss": observed,
                        "difference": difference,
                        "passed": abs(difference)
                        <= runtime["config"]["original_loss_tolerance"],
                    },
                )
                if abs(difference) > runtime["config"]["original_loss_tolerance"]:
                    raise RuntimeError(
                        f"original evaluation reproduction failed for {name}; inspect control-{name}.json before accepting new scores"
                    )
        del measure, model
        torch.cuda.empty_cache()
    return report_results(runtime)


def corrected_metrics(original, corrected):
    result = deepcopy(original)
    if set(result["datasets"]) != set(DATASET_IDS):
        raise RuntimeError("original report does not contain all six corpora")
    result["datasets"]["adrenaline"] = corrected
    loss = math.fsum(
        result["datasets"][dataset]["loss"] for dataset in sorted(DATASET_IDS)
    ) / len(DATASET_IDS)
    result["macro"] = {"loss": loss, "perplexity": math.exp(loss)}
    return result


def report_results(runtime):
    root, config = runtime["root"], runtime["config"]
    holdout, selected = validate_holdout(runtime)
    rows_by_cohort = {"original": original_rows(runtime), "thread_external": selected}
    models = {}
    environments = []
    for name in MODELS:
        values = {}
        for cohort, rows in rows_by_cohort.items():
            state = unseal(root / f"scores-{name}-{cohort}.json")
            identity = state["identity"]
            if (
                identity["evaluation_id"] != root.name
                or identity["model"] != name
                or identity["cohort"] != cohort
                or identity["holdout_sha256"] != holdout["file"]["sha256"]
                or len(state["records"]) != len(rows)
            ):
                raise RuntimeError("incomplete or mixed evaluation scores")
            validate_score(state, identity, rows)
            environments.append(identity["numeric_environment"])
            values[cohort] = summarize_scores(state)
        original = original_metrics(runtime["report"], name)
        difference = (
            values["original"]["loss"] - original["datasets"]["adrenaline"]["loss"]
        )
        if abs(difference) > config["original_loss_tolerance"]:
            raise RuntimeError("original loss reproduction gate failed")
        models[name] = {
            "control": values["original"],
            "control_loss_delta": difference,
            "original": original,
            "corrected": corrected_metrics(original, values["thread_external"]),
        }
    if any(item != environments[0] for item in environments):
        raise RuntimeError("models were scored with different numerical environments")
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "evaluation_id": root.name,
        "original_report_id": config["paired_report_id"],
        "holdout_sha256": holdout["file"]["sha256"],
        "config": config,
        "counts": holdout["counts"],
        "models": models,
        "numeric_environment": environments[0],
        "invariants": {
            "same_holdout_three_models": True,
            "original_loss_reproduced": True,
            "no_cpt_or_original_eval_threads": True,
            "no_exact_train_fragment_at_threshold": True,
            "causal_shift_and_denominator_checked": True,
            "five_other_corpora_unchanged": True,
        },
        "limitations": [
            "new conditioned holdout, not a random sample of all Adrenaline content",
            "one full block per eligible thread; no partial blocks or cross-flow packing",
            "exact matches checked within training blocks; not fuzzy matching",
            "cannot exclude exposure during original foundation-model pretraining",
            "global aggregation mixes corrected Adrenaline with five unchanged evaluations",
            "not an estimate of the loss contribution caused by memorization",
        ],
    }
    result["contrasts"] = {}
    for first, second in (("base", "general"), ("base", "forum"), ("general", "forum")):
        result["contrasts"][f"{second}_vs_{first}"] = {
            "global_ppl_reduction_percent": 100
            * (
                1
                - models[second]["corrected"]["macro"]["perplexity"]
                / models[first]["corrected"]["macro"]["perplexity"]
            )
        }
        result["contrasts"][f"{second}_vs_{first}"][
            "adrenaline_ppl_reduction_percent"
        ] = 100 * (
            1
            - models[second]["corrected"]["datasets"]["adrenaline"]["perplexity"]
            / models[first]["corrected"]["datasets"]["adrenaline"]["perplexity"]
        )
    public = root / "report"
    public.mkdir(exist_ok=True)
    write_json_atomic(public / "report.json", result)
    path = public / "intrinsic-results.csv"
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["revision", "model", "corpus", "loss", "perplexity"])
        for revision in ("original", "corrected"):
            for name in MODELS:
                metrics = models[name][revision]
                for corpus, values in [
                    *sorted(metrics["datasets"].items()),
                    ("global", metrics["macro"]),
                ]:
                    writer.writerow(
                        [revision, name, corpus, values["loss"], values["perplexity"]]
                    )
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)
    write_json_atomic(
        public / "checksums.json",
        {
            name: file_sha256(public / name)
            for name in ("report.json", "intrinsic-results.csv")
        },
    )
    return {
        "status": "complete",
        "evaluation_id": root.name,
        "report": str(public / "report.json"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Corrective Adrenaline evaluation; never trains or overwrites the original report"
    )
    parser.add_argument(
        "command", choices=("prepare", "validate", "evaluate", "report")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/intrinsic/adrenaline-holdout-v1.json",
    )
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    try:
        config = load_config(args.config)
        log("Validando modelos, relatório e identidade dos seis conjuntos originais")
        runtime = resolve(config)
        with locked(runtime):
            if args.command == "prepare":
                result = prepare(runtime)
            elif args.command == "evaluate":
                result = evaluate(runtime)
            elif args.command == "report":
                result = report_results(runtime)
            else:
                holdout, _ = validate_holdout(runtime)
                result = {
                    "status": "valid",
                    "evaluation_id": runtime["root"].name,
                    "counts": holdout["counts"],
                }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except InterruptedError as error:
        log(str(error))
        raise SystemExit(3) from None
    except UnicodeError:
        log("Falha de codificação na fonte; nenhum trecho será impresso")
        raise SystemExit(1) from None
    except (
        RuntimeError,
        ValueError,
        OSError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        # No tracebacks/locals: source records must never leak to batch logs.
        log(f"Falha: {type(error).__name__}: {error}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
