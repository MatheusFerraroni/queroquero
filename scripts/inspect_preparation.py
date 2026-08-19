from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from queroquero.config import MODEL_ID, MODEL_REVISION
from queroquero.prepare import validate_preparation


def inspect_preparation(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(path).expanduser()
    if root.name == "dataset_manifest.json":
        root = root.parent
    root = root.resolve()

    manifest = validate_preparation(root)
    metrics_path = root / manifest["metrics"]["path"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    shard_summaries: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, int]] = {}
    all_sequence_ids: set[str] = set()

    for split in ("train", "eval"):
        split_rows = 0
        split_tokens = 0
        split_source_refs: set[str] = set()

        for shard_record in manifest["splits"][split]:
            relative_path = shard_record["path"]
            shard_path = root / relative_path
            parquet_file = pq.ParquetFile(shard_path)
            table = pq.read_table(
                shard_path,
                columns=[
                    "sequence_id",
                    "input_ids",
                    "source_ref_sha256",
                    "source_token_counts",
                ],
            )

            sequence_ids = table["sequence_id"].to_pylist()
            input_ids = table["input_ids"].to_pylist()
            source_refs = table["source_ref_sha256"].to_pylist()
            source_counts = table["source_token_counts"].to_pylist()
            token_lengths = [len(row) for row in input_ids]
            documents_per_sequence = [len(row) for row in source_counts]
            compressions = sorted(
                {
                    parquet_file.metadata.row_group(row_group_index)
                    .column(column_index)
                    .compression
                    for row_group_index in range(
                        parquet_file.metadata.num_row_groups
                    )
                    for column_index in range(
                        parquet_file.metadata.row_group(row_group_index).num_columns
                    )
                }
            )

            if all_sequence_ids.intersection(sequence_ids):
                raise RuntimeError("duplicate sequence_id found during inspection")
            all_sequence_ids.update(sequence_ids)
            split_source_refs.update(
                source_ref for row in source_refs for source_ref in row
            )
            split_rows += table.num_rows
            split_tokens += sum(token_lengths)

            shard_summaries.append(
                {
                    "split": split,
                    "path": relative_path,
                    "rows": table.num_rows,
                    "tokens": sum(token_lengths),
                    "tokens_per_sequence": sorted(set(token_lengths)),
                    "source_documents_per_sequence": {
                        "minimum": min(documents_per_sequence),
                        "maximum": max(documents_per_sequence),
                    },
                    "compression": compressions,
                    "size_bytes": shard_path.stat().st_size,
                }
            )

        split_summaries[split] = {
            "sequences": split_rows,
            "tokens": split_tokens,
            "unique_source_documents": len(split_source_refs),
        }

    summary = {
        "status": "valid",
        "dataset_id": manifest["dataset_id"],
        "profile": manifest["profile"],
        "preparation_id": manifest["preparation_id"],
        "sequence_length": manifest["sequence_length"],
        "format": manifest["format"],
        "counts": manifest["counts"],
        "discarded_tail_tokens": manifest["discarded_tail_tokens"],
        "tokens_not_selected_by_sequence_budget": manifest[
            "tokens_not_selected_by_sequence_budget"
        ],
        "adapter_metrics": metrics["adapter"],
        "tokenization_metrics": metrics["tokenization"],
        "packing_metrics": metrics["packing"],
        "splits": split_summaries,
        "shards": shard_summaries,
        "unique_sequence_ids": len(all_sequence_ids),
    }
    return root, manifest, summary


def decode_sample(
    root: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    row_index: int,
) -> str:
    remaining = row_index
    token_ids: list[int] | None = None

    for shard_record in manifest["splits"][split]:
        shard_path = root / shard_record["path"]
        table = pq.read_table(shard_path, columns=["input_ids"])
        if remaining < table.num_rows:
            token_ids = [int(value) for value in table["input_ids"][remaining].as_py()]
            break
        remaining -= table.num_rows

    if token_ids is None:
        available = manifest["counts"][f"{split}_sequences"]
        raise ValueError(
            f"row {row_index} does not exist in {split}; available rows: {available}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize one prepared dataset without exposing source text."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="preparation directory or its dataset_manifest.json",
    )
    parser.add_argument(
        "--decode-sample",
        action="store_true",
        help="explicitly decode one sequence and print real corpus content",
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval"),
        default="train",
        help="split used by --decode-sample (default: train)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="zero-based row within the selected split (default: 0)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.row < 0:
        raise SystemExit("Erro: --row must be zero or greater")

    try:
        root, manifest, summary = inspect_preparation(args.path)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if args.decode_sample:
            print(
                "AVISO: o texto abaixo contém conteúdo real do corpus; "
                "não o copie para logs, testes, commits ou documentação.",
                file=sys.stderr,
            )
            print(f"\n--- decoded sample split={args.split} row={args.row} ---")
            print(
                decode_sample(
                    root,
                    manifest,
                    split=args.split,
                    row_index=args.row,
                )
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}") from exc


if __name__ == "__main__":
    main()
