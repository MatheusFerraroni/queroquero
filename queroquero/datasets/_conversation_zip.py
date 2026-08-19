from __future__ import annotations

import hashlib
import heapq
import io
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
from zipfile import BadZipFile, ZipFile, ZipInfo

from queroquero.config import ConfigError, resolve_dataset_root

from .base import CheckpointCallback, Document, ScanResult, clean_text, stable_hash


class ConversationZipFormatError(ValueError):
    """Raised when a selected conversation member violates the fixed TSV schema."""


@dataclass(order=False)
class _RankedMember:
    rank_key: tuple[str, str]
    info: ZipInfo = field(compare=False)

    def __lt__(self, other: "_RankedMember") -> bool:
        # heapq is a min-heap. Reversing comparison leaves the worst retained
        # candidate at the root, so it can be replaced in O(log k).
        return self.rank_key > other.rank_key


class ConversationZipAdapter:
    dataset_id: str

    def scan(
        self,
        config: Dict[str, Any],
        resume_cursor: Optional[Dict[str, Any]] = None,
        resume_documents: Optional[List[Document]] = None,
        checkpoint: Optional[CheckpointCallback] = None,
    ) -> ScanResult:
        dataset = _dataset_config(config, self.dataset_id)
        profile_name, profile = _profile_config(config)
        if profile.get("selection") != "representative":
            raise ConfigError("conversation selection must be representative")
        source = _mapping(dataset, "source")
        _validate_source_contract(source)
        _validate_filter_contract(_mapping(dataset, "filters"))
        archive_relative = _archive_for_profile(source, profile_name)
        archive_path = _safe_source_path(resolve_dataset_root(config), archive_relative)
        if not archive_path.is_file():
            raise ConfigError(f"conversation archive not found: {archive_relative}")

        candidate_limit = _positive_int(profile, "candidate_documents")
        seed = _seed(config)
        checkpoint_every = _positive_int(source, "checkpoint_every", default=1000)
        selection_sha256 = stable_hash(
            "conversation-zip-selection/v1",
            self.dataset_id,
            profile_name,
            archive_relative,
            seed,
            candidate_limit,
        )

        try:
            with ZipFile(archive_path) as archive:
                fingerprint, selected = _fingerprint_and_select(
                    archive=archive,
                    archive_path=archive_path,
                    archive_relative=archive_relative,
                    dataset_id=self.dataset_id,
                    seed=seed,
                    limit=candidate_limit,
                )
                fingerprint_sha256 = fingerprint["sha256"]
                cursor = _validated_cursor(
                    resume_cursor,
                    dataset_id=self.dataset_id,
                    fingerprint_sha256=fingerprint_sha256,
                    selection_sha256=selection_sha256,
                    selected_count=len(selected),
                )
                documents = _validated_resume_documents(resume_documents)
                if cursor["documents_emitted"] != len(documents):
                    raise ConfigError("resume cursor and documents disagree")
                next_index = cursor["next_member_index"]
                conversations_seen = cursor["conversations_seen"]
                messages_seen = cursor["messages_seen"]
                empty_conversations = cursor["empty_conversations"]

                for member_index in range(next_index, len(selected)):
                    try:
                        document, row_count = _read_conversation(
                            archive=archive,
                            info=selected[member_index].info,
                            dataset_id=self.dataset_id,
                            archive_name=Path(archive_relative).name,
                            member_rank=member_index,
                        )
                    except RuntimeError:
                        raise ConversationZipFormatError(
                            "selected conversation member could not be read "
                            f"(selected member {member_index})"
                        ) from None
                    conversations_seen += 1
                    messages_seen += row_count
                    if document is None:
                        empty_conversations += 1
                    else:
                        documents.append(document)

                    next_index = member_index + 1
                    if checkpoint and (
                        next_index % checkpoint_every == 0
                        or next_index == len(selected)
                    ):
                        checkpoint(
                            _cursor(
                                dataset_id=self.dataset_id,
                                fingerprint_sha256=fingerprint_sha256,
                                selection_sha256=selection_sha256,
                                next_member_index=next_index,
                                conversations_seen=conversations_seen,
                                messages_seen=messages_seen,
                                empty_conversations=empty_conversations,
                                documents_emitted=len(documents),
                                complete=next_index == len(selected),
                            ),
                            list(documents),
                        )
        except BadZipFile:
            raise ConversationZipFormatError("invalid conversation ZIP archive") from None

        final_cursor = _cursor(
            dataset_id=self.dataset_id,
            fingerprint_sha256=fingerprint_sha256,
            selection_sha256=selection_sha256,
            next_member_index=len(selected),
            conversations_seen=conversations_seen,
            messages_seen=messages_seen,
            empty_conversations=empty_conversations,
            documents_emitted=len(documents),
            complete=True,
        )
        return ScanResult(
            documents=documents,
            source_fingerprint=fingerprint,
            metrics={
                "source_members_eligible": fingerprint["eligible_member_count"],
                "source_conversations_selected": len(selected),
                "source_conversations_seen": conversations_seen,
                "source_messages_seen": messages_seen,
                "source_conversations_empty": empty_conversations,
                "documents_emitted": len(documents),
            },
            cursor=final_cursor,
        )


def _fingerprint_and_select(
    archive: ZipFile,
    archive_path: Path,
    archive_relative: str,
    dataset_id: str,
    seed: int,
    limit: int,
) -> tuple[Dict[str, Any], List[_RankedMember]]:
    digest = hashlib.sha256()
    digest.update(archive_path.stat().st_size.to_bytes(16, "big"))
    selected: List[_RankedMember] = []
    eligible_count = 0

    for info in archive.infolist():
        central_record = json.dumps(
            [info.filename, info.CRC, info.compress_size, info.file_size],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(central_record).to_bytes(8, "big"))
        digest.update(central_record)

        if not _is_conversation_member(info):
            continue
        eligible_count += 1
        rank_key = (
            stable_hash(seed, dataset_id, info.filename),
            info.filename,
        )
        ranked = _RankedMember(rank_key=rank_key, info=info)
        if len(selected) < limit:
            heapq.heappush(selected, ranked)
        elif rank_key < selected[0].rank_key:
            heapq.heapreplace(selected, ranked)

    selected.sort(key=lambda item: item.rank_key)
    fingerprint = {
        "kind": "zip-central-directory/v1",
        "archive": archive_relative,
        "archive_size_bytes": archive_path.stat().st_size,
        "central_directory_entries": len(archive.infolist()),
        "eligible_member_count": eligible_count,
        "sha256": digest.hexdigest(),
    }
    return fingerprint, selected


def _is_conversation_member(info: ZipInfo) -> bool:
    if info.is_dir():
        return False
    path = PurePosixPath(info.filename)
    return path.parent == PurePosixPath("clear_threads") and path.suffix == ".tsv"


def _read_conversation(
    archive: ZipFile,
    info: ZipInfo,
    dataset_id: str,
    archive_name: str,
    member_rank: int,
) -> tuple[Optional[Document], int]:
    participants: Dict[str, int] = {}
    lines: List[str] = []
    row_count = 0

    with archive.open(info, "r") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline="") as text:
            for row_index, raw_line in enumerate(text):
                row = raw_line.rstrip("\r\n").split("\t")
                if len(row) != 3:
                    raise ConversationZipFormatError(
                        "conversation TSV must have exactly three columns "
                        f"(selected member {member_rank}, row {row_index})"
                    )
                _timestamp, raw_participant, raw_message = row
                row_count += 1
                if raw_participant not in participants:
                    participants[raw_participant] = len(participants) + 1
                message = clean_text(raw_message, strip_html=True)
                if message:
                    participant_number = participants[raw_participant]
                    lines.append(f"Participante {participant_number}: {message}")

    if not lines:
        return None, row_count
    source_ref = f"{dataset_id}/{archive_name}/{info.filename}"
    return (
        Document(
            text="\n".join(lines),
            source_ref=source_ref,
            source_position={"member_rank": member_rank},
            metadata={
                "document_type": "conversation",
                "message_count": len(lines),
                "participant_count": len(participants),
            },
        ),
        row_count,
    )


def _dataset_config(config: Dict[str, Any], expected_id: str) -> Dict[str, Any]:
    dataset = _mapping(config, "dataset")
    if dataset.get("dataset_id") != expected_id:
        raise ConfigError(f"adapter requires dataset_id {expected_id!r}")
    return dataset


def _profile_config(config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    profile_name = config.get("profile_name")
    if profile_name not in {"smoke", "mvp"}:
        raise ConfigError("profile_name must be smoke or mvp")
    return profile_name, _mapping(config, "profile")


def _archive_for_profile(source: Dict[str, Any], profile_name: str) -> str:
    archives = _mapping(source, "archives_by_profile")
    value = archives.get(profile_name)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"source archive missing for profile {profile_name!r}")
    return value


def _validate_source_contract(source: Dict[str, Any]) -> None:
    expected = {
        "format": "zip-tsv",
        "member_pattern": "clear_threads/*.tsv",
        "encoding": "utf-8",
        "columns": ["timestamp", "user", "text"],
        "has_header": False,
    }
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            raise ConfigError(f"conversation source.{key} must be {expected_value!r}")


def _validate_filter_contract(filters: Dict[str, Any]) -> None:
    expected = {
        "strict_three_columns": True,
        "anonymize_participants": True,
        "omit_timestamps": True,
        "strip_html": True,
    }
    for key, expected_value in expected.items():
        if filters.get(key) != expected_value:
            raise ConfigError(f"conversation filters.{key} must be {expected_value!r}")


def _safe_source_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ConfigError("dataset source paths must be relative")
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ConfigError("dataset source path escapes dataset root")
    return path


def _seed(config: Dict[str, Any]) -> int:
    preparation = _mapping(config, "preparation")
    value = preparation.get("seed")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError("preparation.seed must be a non-negative integer")
    return value


def _mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _positive_int(
    config: Dict[str, Any], key: str, default: Optional[int] = None
) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{key} must be a positive integer")
    return value


def _validated_resume_documents(
    resume_documents: Optional[List[Document]],
) -> List[Document]:
    documents = list(resume_documents or [])
    if any(not isinstance(document, Document) for document in documents):
        raise ConfigError("resume_documents must contain Document values")
    return documents


def _validated_cursor(
    resume_cursor: Optional[Dict[str, Any]],
    dataset_id: str,
    fingerprint_sha256: str,
    selection_sha256: str,
    selected_count: int,
) -> Dict[str, Any]:
    if resume_cursor is None:
        return _cursor(
            dataset_id=dataset_id,
            fingerprint_sha256=fingerprint_sha256,
            selection_sha256=selection_sha256,
            next_member_index=0,
            conversations_seen=0,
            messages_seen=0,
            empty_conversations=0,
            documents_emitted=0,
            complete=False,
        )
    if resume_cursor.get("adapter") != dataset_id:
        raise ConfigError("resume cursor belongs to a different adapter")
    if resume_cursor.get("source_fingerprint_sha256") != fingerprint_sha256:
        raise RuntimeError("source fingerprint changed; refusing to resume")
    if resume_cursor.get("selection_sha256") != selection_sha256:
        raise RuntimeError("selection configuration changed; refusing to resume")
    for key in (
        "next_member_index",
        "conversations_seen",
        "messages_seen",
        "empty_conversations",
        "documents_emitted",
    ):
        value = resume_cursor.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"invalid resume cursor field: {key}")
    if resume_cursor["next_member_index"] > selected_count:
        raise ConfigError("resume cursor exceeds selected source members")
    return dict(resume_cursor)


def _cursor(
    dataset_id: str,
    fingerprint_sha256: str,
    selection_sha256: str,
    next_member_index: int,
    conversations_seen: int,
    messages_seen: int,
    empty_conversations: int,
    documents_emitted: int,
    complete: bool,
) -> Dict[str, Any]:
    return {
        "adapter": dataset_id,
        "source_fingerprint_sha256": fingerprint_sha256,
        "selection_sha256": selection_sha256,
        "next_member_index": next_member_index,
        "conversations_seen": conversations_seen,
        "messages_seen": messages_seen,
        "empty_conversations": empty_conversations,
        "documents_emitted": documents_emitted,
        "complete": complete,
    }
