from __future__ import annotations

import hashlib
import heapq
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from queroquero.config import ConfigError, resolve_dataset_root
from queroquero.datasets.base import (
    CheckpointCallback,
    Document,
    ScanResult,
    stable_hash,
)


class BrwacAdapter:
    """Read BrWaC documents directly from the source archive."""

    def scan(
        self,
        config: Dict[str, Any],
        resume_cursor: Optional[Dict[str, Any]] = None,
        resume_documents: Optional[List[Document]] = None,
        checkpoint: Optional[CheckpointCallback] = None,
    ) -> ScanResult:
        source = config["dataset"]["source"]
        filters = config["dataset"]["filters"]
        _validate_config(config, source, filters)
        archive_path = _source_path(config)
        candidate_limit = _positive_int(
            config["profile"].get("candidate_documents"),
            "profile.candidate_documents",
        )
        checkpoint_interval = _positive_int(
            source.get("checkpoint_interval_documents", 64),
            "source.checkpoint_interval_documents",
        )
        seed = config["preparation"]["seed"]

        with zipfile.ZipFile(archive_path, "r") as archive:
            fingerprint = _central_directory_fingerprint(archive_path, archive)
            infos = archive.infolist()
            eligible_members = sum(1 for info in infos if _is_document_member(info))
            ranked = heapq.nsmallest(
                candidate_limit,
                (info for info in infos if _is_document_member(info)),
                key=lambda info: (
                    stable_hash(seed, "brwac", info.filename),
                    info.filename,
                ),
            )

            cursor = dict(resume_cursor or {})
            expected_fingerprint = cursor.get("source_fingerprint_sha256")
            if expected_fingerprint and expected_fingerprint != fingerprint["sha256"]:
                raise ConfigError("BrWaC source changed since the saved checkpoint")

            start_index = cursor.get("next_selection_index", 0)
            if (
                not isinstance(start_index, int)
                or isinstance(start_index, bool)
                or not 0 <= start_index <= len(ranked)
            ):
                raise ConfigError("invalid BrWaC resume cursor")

            documents = list(resume_documents or [])
            if len(documents) != start_index:
                raise ConfigError(
                    "BrWaC checkpoint documents do not match the resume cursor"
                )

            for selection_index in range(start_index, len(ranked)):
                info = ranked[selection_index]
                try:
                    raw = archive.read(info)
                    text = raw.decode(source.get("encoding", "utf-8"), errors="strict")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        "BrWaC contains a selected document that is not strict UTF-8"
                    ) from exc

                member_ref = f"{archive_path.name}:{info.filename}"
                documents.append(
                    Document(
                        text=text,
                        source_ref=f"brwac:{member_ref}",
                        source_position={
                            "member_path": info.filename,
                            "selection_rank": selection_index,
                        },
                        metadata={
                            "archive": archive_path.name,
                            "member_crc32": f"{info.CRC:08x}",
                            "uncompressed_bytes": info.file_size,
                        },
                    )
                )

                next_index = selection_index + 1
                if checkpoint and (
                    next_index % checkpoint_interval == 0
                    or next_index == len(ranked)
                ):
                    checkpoint(
                        _cursor(
                            next_index,
                            fingerprint["sha256"],
                            complete=next_index == len(ranked),
                        ),
                        list(documents),
                    )

        final_cursor = _cursor(len(ranked), fingerprint["sha256"], complete=True)
        return ScanResult(
            documents=documents,
            source_fingerprint=fingerprint,
            metrics={
                "archive_entries": len(infos),
                "eligible_members": eligible_members,
                "ranked_members": len(ranked),
                "selected_documents": len(documents),
            },
            cursor=final_cursor,
        )


def _source_path(config: Dict[str, Any]) -> Path:
    root = resolve_dataset_root(config)
    value = config["dataset"]["source"].get("path")
    if not isinstance(value, str) or not value:
        raise ConfigError("BrWaC source.path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ConfigError("BrWaC source.path must be relative to the dataset root")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ConfigError("BrWaC source.path escapes the dataset root")
    if not path.is_file():
        raise ConfigError("BrWaC source archive was not found")
    return path


def _validate_config(
    config: Dict[str, Any],
    source: Dict[str, Any],
    filters: Dict[str, Any],
) -> None:
    if source.get("format") != "zip":
        raise ConfigError("BrWaC source.format must be 'zip'")
    if source.get("member_glob") != "data/*.txt":
        raise ConfigError("BrWaC source.member_glob must be 'data/*.txt'")
    if source.get("encoding") != "utf-8":
        raise ConfigError("BrWaC source.encoding must be 'utf-8'")
    if filters.get("strict_utf8") is not True:
        raise ConfigError("BrWaC filters.strict_utf8 must be true")
    if config["profile"].get("selection") != "representative":
        raise ConfigError("BrWaC selection must be representative")
    seed = config["preparation"].get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("preparation.seed must be a non-negative integer")


def _is_document_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    member = PurePosixPath(info.filename)
    return (
        len(member.parts) == 2
        and member.parts[0] == "data"
        and member.suffix == ".txt"
    )


def _central_directory_fingerprint(
    archive_path: Path,
    archive: zipfile.ZipFile,
) -> Dict[str, Any]:
    archive_size = archive_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(b"queroquero-zip-central-directory-v1\0")
    digest.update(archive_size.to_bytes(8, byteorder="big", signed=False))
    infos = archive.infolist()
    for info in infos:
        name = info.filename.encode("utf-8", errors="surrogatepass")
        digest.update(len(name).to_bytes(8, byteorder="big", signed=False))
        digest.update(name)
        for value in (info.CRC, info.file_size, info.compress_size):
            digest.update(int(value).to_bytes(8, byteorder="big", signed=False))

    return {
        "method": "zip-central-directory-v1",
        "sha256": digest.hexdigest(),
        "archive_size_bytes": archive_size,
        "central_directory_entries": len(infos),
    }


def _cursor(next_index: int, fingerprint: str, complete: bool) -> Dict[str, Any]:
    return {
        "next_selection_index": next_index,
        "source_fingerprint_sha256": fingerprint,
        "complete": complete,
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


ADAPTER = BrwacAdapter()
