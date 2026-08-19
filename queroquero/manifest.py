from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from .config import canonical_json_bytes, sha256_bytes


DATASET_MANIFEST_SCHEMA = "queroquero-dataset-manifest/v1"
METRICS_SCHEMA = "queroquero-preparation-metrics/v1"
PROGRESS_SCHEMA = "queroquero-preparation-progress/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
    temporary.replace(path)


def preparation_id(
    resolved_config_sha256: str,
    tokenizer_fingerprint: str,
    source_fingerprint: Dict[str, Any],
) -> str:
    value = {
        "config_sha256": resolved_config_sha256,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint,
        "source_fingerprint": source_fingerprint,
    }
    return sha256_bytes(canonical_json_bytes(value))[:20]
