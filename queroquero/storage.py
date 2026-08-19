from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from .datasets.base import Document
from .manifest import file_sha256, write_json_atomic
from .packing import PackedSequence


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("input_ids", pa.list_(pa.int32(), 1024), nullable=False),
        pa.field("source_ref_sha256", pa.list_(pa.string()), nullable=False),
        pa.field("source_token_counts", pa.list_(pa.int32()), nullable=False),
    ]
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def write_split(
    output_dir: Path,
    split: str,
    records: Sequence[PackedSequence],
    sequences_per_shard: int,
) -> List[Dict[str, Any]]:
    if not isinstance(sequences_per_shard, int) or isinstance(
        sequences_per_shard, bool
    ) or sequences_per_shard < 1:
        raise ValueError("sequences_per_shard must be a positive integer")
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(records), sequences_per_shard)):
        chunk = records[start : start + sequences_per_shard]
        filename = f"shard-{shard_index:05d}.parquet"
        final_path = split_dir / filename
        partial_path = split_dir / f".{filename}.partial"
        table = pa.Table.from_pydict(
            {
                "sequence_id": [record.sequence_id for record in chunk],
                "input_ids": [list(record.input_ids) for record in chunk],
                "source_ref_sha256": [
                    list(record.source_ref_sha256) for record in chunk
                ],
                "source_token_counts": [
                    list(record.source_token_counts) for record in chunk
                ],
            },
            schema=PARQUET_SCHEMA,
        )
        try:
            pq.write_table(
                table,
                partial_path,
                compression="zstd",
                version="2.6",
                use_dictionary=False,
                write_statistics=True,
            )
            # The final name becomes visible only after a complete read-back.
            validate_shard(partial_path)
            partial_path.replace(final_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        shards.append(
            {
                "path": final_path.relative_to(output_dir).as_posix(),
                "rows": len(chunk),
                "tokens": len(chunk) * 1024,
                "size_bytes": final_path.stat().st_size,
                "sha256": file_sha256(final_path),
            }
        )
    return shards


def validate_shard(path: Path) -> int:
    if path.is_symlink():
        raise RuntimeError(f"Parquet shard must not be a symlink: {path}")
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata.num_rows < 1:
        raise RuntimeError(f"Parquet shard must contain at least one row: {path}")
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            if row_group.column(column_index).compression != "ZSTD":
                raise RuntimeError(f"Parquet shard is not fully ZSTD-compressed: {path}")
    table = pq.read_table(path)
    if table.schema != PARQUET_SCHEMA:
        raise RuntimeError(f"unexpected Parquet schema: {path}")
    input_ids = table.column("input_ids").to_pylist()
    source_refs = table.column("source_ref_sha256").to_pylist()
    source_counts = table.column("source_token_counts").to_pylist()
    sequence_ids = table.column("sequence_id").to_pylist()
    if len(sequence_ids) != len(set(sequence_ids)):
        raise RuntimeError(f"duplicate sequence_id in {path}")
    for row_index, values in enumerate(input_ids):
        sequence_id = sequence_ids[row_index]
        if not isinstance(sequence_id, str) or not _SHA256_RE.fullmatch(sequence_id):
            raise RuntimeError(f"invalid sequence_id at row {row_index} in {path}")
        if len(values) != 1024:
            raise RuntimeError(f"row {row_index} in {path} is not 1024 tokens")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise RuntimeError(f"invalid token ID at row {row_index} in {path}")
        if len(source_refs[row_index]) != len(source_counts[row_index]):
            raise RuntimeError(f"provenance lengths differ at row {row_index} in {path}")
        if not source_refs[row_index] or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in source_refs[row_index]
        ):
            raise RuntimeError(f"invalid source hash at row {row_index} in {path}")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in source_counts[row_index]
        ):
            raise RuntimeError(f"invalid source token count at row {row_index} in {path}")
        if sum(source_counts[row_index]) != 1024:
            raise RuntimeError(f"provenance token count differs at row {row_index} in {path}")
    return len(sequence_ids)


class WorkStore:
    def __init__(self, output_root: Path, dataset_id: str, config_sha256: str) -> None:
        self.config_sha256 = config_sha256
        self.path = output_root / ".work" / dataset_id / config_sha256
        self.progress_path = self.path / "progress.json"
        self.candidates_path = self.path / "candidates.jsonl"

    def load(self) -> Tuple[Optional[Dict[str, Any]], List[Document]]:
        if not self.progress_path.exists() or not self.candidates_path.exists():
            return None, []
        if self.progress_path.is_symlink() or self.candidates_path.is_symlink():
            raise RuntimeError("preparation work files must not be symlinks")
        progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if progress.get("schema_version") != "queroquero-preparation-progress/v1":
            raise RuntimeError("unknown preparation work schema")
        if progress.get("status") != "scanning":
            raise RuntimeError("preparation work state is not resumable")
        if progress.get("scan_config_sha256") != self.config_sha256:
            raise RuntimeError("preparation work configuration changed")
        if (
            progress.get("candidates_size_bytes") != self.candidates_path.stat().st_size
            or progress.get("candidates_sha256") != file_sha256(self.candidates_path)
        ):
            raise RuntimeError("preparation work candidates are incomplete or changed")
        documents = []
        with self.candidates_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                documents.append(
                    Document(
                        text=record["text"],
                        source_ref=record["source_ref"],
                        source_position=record["source_position"],
                        metadata=record["metadata"],
                    )
                )
        if progress.get("candidate_documents") != len(documents):
            raise RuntimeError("preparation work candidate count is inconsistent")
        return progress.get("cursor"), documents

    def checkpoint(self, cursor: Dict[str, Any], documents: List[Document]) -> None:
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.chmod(0o700)
        partial = self.candidates_path.with_suffix(".jsonl.partial")
        with partial.open("w", encoding="utf-8") as handle:
            for document in documents:
                value = {
                    "text": document.text,
                    "source_ref": document.source_ref,
                    "source_position": document.source_position,
                    "metadata": document.metadata,
                }
                handle.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        partial.chmod(0o600)
        candidates_size_bytes = partial.stat().st_size
        candidates_sha256 = file_sha256(partial)
        partial.replace(self.candidates_path)
        write_json_atomic(
            self.progress_path,
            {
                "schema_version": "queroquero-preparation-progress/v1",
                "status": "scanning",
                "scan_config_sha256": self.config_sha256,
                "cursor": cursor,
                "candidate_documents": len(documents),
                "candidates_size_bytes": candidates_size_bytes,
                "candidates_sha256": candidates_sha256,
            },
        )

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)
