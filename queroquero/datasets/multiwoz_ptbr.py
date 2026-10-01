from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from queroquero.config import (
    ConfigError,
    canonical_json_bytes,
    resolve_dataset_root,
    sha256_bytes,
)
from queroquero.manifest import file_sha256

from .base import CheckpointCallback, Document, ScanResult, clean_text, stable_hash


class MultiWOZFormatError(ValueError):
    """Raised when a source JSON does not match the projected dialogue schema."""


class MultiWOZPTBRAdapter:
    dataset_id = "multiwoz_ptbr"

    def scan(
        self,
        config: Dict[str, Any],
        resume_cursor: Optional[Dict[str, Any]] = None,
        resume_documents: Optional[List[Document]] = None,
        checkpoint: Optional[CheckpointCallback] = None,
    ) -> ScanResult:
        dataset = _mapping(config, "dataset")
        if dataset.get("dataset_id") != self.dataset_id:
            raise ConfigError(f"adapter requires dataset_id {self.dataset_id!r}")
        source = _mapping(dataset, "source")
        _validate_source_contract(source)
        _validate_filter_contract(_mapping(dataset, "filters"))
        profile = _mapping(config, "profile")
        if profile.get("selection") != "representative":
            raise ConfigError("MultiWOZ selection must be representative")
        seed = _seed(config)
        candidate_limit = _positive_int(profile, "candidate_documents")
        checkpoint_every = _positive_int(source, "checkpoint_every", default=128)
        selection_identity = ["multiwoz-selection/v1", seed]
        if "capacity_audit" not in config:
            selection_identity.append(candidate_limit)
        selection_sha256 = stable_hash(*selection_identity)

        source_root = resolve_dataset_root(config)
        source_directory = _safe_source_path(
            source_root, _required_string(source, "relative_directory")
        )
        files = _source_files(source_directory, source)
        fingerprint = _source_fingerprint(
            source_root=source_root,
            source_directory=source_directory,
            files=files,
        )
        fingerprint_sha256 = fingerprint["sha256"]
        cursor = _validated_cursor(
            resume_cursor,
            fingerprint_sha256=fingerprint_sha256,
            selection_sha256=selection_sha256,
            file_count=len(files),
        )
        documents = _validated_resume_documents(resume_documents)
        if cursor["documents_selected"] != len(documents):
            raise ConfigError("resume cursor and documents disagree")
        file_index = cursor["next_file_index"]
        dialogue_index = cursor["next_dialogue_index"]
        dialogues_seen = cursor["dialogues_seen"]
        turns_seen = cursor["turns_seen"]
        empty_dialogues = cursor["empty_dialogues"]

        while file_index < len(files):
            dialogues = _load_dialogues(files[file_index], file_index)
            if dialogue_index > len(dialogues):
                raise ConfigError("resume cursor exceeds source file length")

            while dialogue_index < len(dialogues):
                document, turn_count = _project_dialogue(
                    dialogue=dialogues[dialogue_index],
                    file_index=file_index,
                    dialogue_index=dialogue_index,
                    source_root=source_root,
                    source_file=files[file_index],
                )
                dialogues_seen += 1
                turns_seen += turn_count
                if document is None:
                    empty_dialogues += 1
                else:
                    documents.append(document)

                dialogue_index += 1
                if dialogue_index == len(dialogues):
                    next_file_index = file_index + 1
                    next_dialogue_index = 0
                else:
                    next_file_index = file_index
                    next_dialogue_index = dialogue_index

                if len(documents) > candidate_limit * 2:
                    documents = _ranked_documents(documents, seed, candidate_limit)

                if checkpoint and (
                    dialogues_seen % checkpoint_every == 0
                    or next_file_index == len(files)
                ):
                    documents = _ranked_documents(documents, seed, candidate_limit)
                    checkpoint(
                        _cursor(
                            fingerprint_sha256=fingerprint_sha256,
                            selection_sha256=selection_sha256,
                            next_file_index=next_file_index,
                            next_dialogue_index=next_dialogue_index,
                            dialogues_seen=dialogues_seen,
                            turns_seen=turns_seen,
                            empty_dialogues=empty_dialogues,
                            documents_selected=len(documents),
                            complete=next_file_index == len(files),
                        ),
                        list(documents),
                    )

            file_index += 1
            dialogue_index = 0

        documents = _ranked_documents(documents, seed, candidate_limit)
        final_cursor = _cursor(
            fingerprint_sha256=fingerprint_sha256,
            selection_sha256=selection_sha256,
            next_file_index=len(files),
            next_dialogue_index=0,
            dialogues_seen=dialogues_seen,
            turns_seen=turns_seen,
            empty_dialogues=empty_dialogues,
            documents_selected=len(documents),
            complete=True,
        )
        return ScanResult(
            documents=documents,
            source_fingerprint=fingerprint,
            metrics={
                "source_files": len(files),
                "source_dialogues_seen": dialogues_seen,
                "source_turns_seen": turns_seen,
                "source_dialogues_empty": empty_dialogues,
                "documents_emitted": len(documents),
            },
            cursor=final_cursor,
        )


def _source_files(source_directory: Path, source: Dict[str, Any]) -> List[Path]:
    if not source_directory.is_dir():
        raise ConfigError("MultiWOZ source directory was not found")
    pattern = source.get("file_pattern")
    if pattern != "dialogues_*.json":
        raise ConfigError("MultiWOZ file_pattern must be 'dialogues_*.json'")
    expected_count = source.get("expected_files")
    if expected_count != 17:
        raise ConfigError("MultiWOZ expected_files must be exactly 17")

    expected_names = [f"dialogues_{index:03d}.json" for index in range(1, 18)]
    files = [source_directory / name for name in expected_names]
    if any(not path.is_file() for path in files):
        raise ConfigError("MultiWOZ requires dialogues_001.json through dialogues_017.json")
    actual_names = sorted(path.name for path in source_directory.glob(pattern))
    if actual_names != expected_names:
        raise ConfigError("MultiWOZ source must contain exactly the 17 expected JSON files")
    return files


def _validate_source_contract(source: Dict[str, Any]) -> None:
    expected = {
        "format": "json-array",
        "file_pattern": "dialogues_*.json",
        "expected_files": 17,
        "encoding": "utf-8",
    }
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            raise ConfigError(f"MultiWOZ source.{key} must be {expected_value!r}")


def _validate_filter_contract(filters: Dict[str, Any]) -> None:
    expected = {
        "projection": "utterances_only",
        "speaker_labels": {"USER": "Usuário", "SYSTEM": "Assistente"},
        "exclude_frames_slots_states": True,
    }
    for key, expected_value in expected.items():
        if filters.get(key) != expected_value:
            raise ConfigError(f"MultiWOZ filters.{key} must be {expected_value!r}")


def _source_fingerprint(
    source_root: Path,
    source_directory: Path,
    files: List[Path],
) -> Dict[str, Any]:
    records = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    return {
        "kind": "files-sha256/v1",
        "directory": source_directory.relative_to(source_root).as_posix(),
        "file_count": len(records),
        "files": records,
        "sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def _load_dialogues(path: Path, file_index: int) -> List[Any]:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise MultiWOZFormatError(
            f"invalid MultiWOZ JSON in source file {file_index}"
        ) from exc
    if not isinstance(value, list):
        raise MultiWOZFormatError(
            f"MultiWOZ source file {file_index} must contain a JSON array"
        )
    return value


def _project_dialogue(
    dialogue: Any,
    file_index: int,
    dialogue_index: int,
    source_root: Path,
    source_file: Path,
) -> tuple[Optional[Document], int]:
    if not isinstance(dialogue, dict) or not isinstance(dialogue.get("turns"), list):
        raise MultiWOZFormatError(
            "MultiWOZ dialogue must be an object with a turns array "
            f"(source file {file_index}, dialogue {dialogue_index})"
        )

    lines: List[str] = []
    turns = dialogue["turns"]
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise MultiWOZFormatError(
                "MultiWOZ turn must be an object "
                f"(source file {file_index}, dialogue {dialogue_index}, "
                f"turn {turn_index})"
            )
        speaker = turn.get("speaker")
        if speaker == "USER":
            label = "Usuário"
        elif speaker == "SYSTEM":
            label = "Assistente"
        else:
            raise MultiWOZFormatError(
                "MultiWOZ speaker must be USER or SYSTEM "
                f"(source file {file_index}, dialogue {dialogue_index}, "
                f"turn {turn_index})"
            )
        utterance = turn.get("utterance")
        if not isinstance(utterance, str):
            raise MultiWOZFormatError(
                "MultiWOZ utterance must be a string "
                f"(source file {file_index}, dialogue {dialogue_index}, "
                f"turn {turn_index})"
            )
        cleaned = clean_text(utterance, strip_html=True)
        if cleaned:
            lines.append(f"{label}: {cleaned}")

    if not lines:
        return None, len(turns)
    relative_file = source_file.relative_to(source_root).as_posix()
    source_ref = f"{relative_file}#dialogue_index={dialogue_index}"
    return (
        Document(
            text="\n".join(lines),
            source_ref=source_ref,
            source_position={
                "file_index": file_index,
                "dialogue_index": dialogue_index,
            },
            metadata={
                "document_type": "dialogue",
                "turn_count": len(lines),
            },
        ),
        len(turns),
    )


def _ranked_documents(
    documents: List[Document], seed: int, limit: int
) -> List[Document]:
    return sorted(
        documents,
        key=lambda document: (
            stable_hash(seed, "multiwoz_ptbr", document.source_ref),
            document.source_ref,
        ),
    )[:limit]


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


def _required_string(config: Dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
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
    fingerprint_sha256: str,
    selection_sha256: str,
    file_count: int,
) -> Dict[str, Any]:
    if resume_cursor is None:
        return _cursor(
            fingerprint_sha256=fingerprint_sha256,
            selection_sha256=selection_sha256,
            next_file_index=0,
            next_dialogue_index=0,
            dialogues_seen=0,
            turns_seen=0,
            empty_dialogues=0,
            documents_selected=0,
            complete=False,
        )
    if resume_cursor.get("adapter") != "multiwoz_ptbr":
        raise ConfigError("resume cursor belongs to a different adapter")
    if resume_cursor.get("source_fingerprint_sha256") != fingerprint_sha256:
        raise RuntimeError("source fingerprint changed; refusing to resume")
    if resume_cursor.get("selection_sha256") != selection_sha256:
        raise RuntimeError("selection configuration changed; refusing to resume")
    for key in (
        "next_file_index",
        "next_dialogue_index",
        "dialogues_seen",
        "turns_seen",
        "empty_dialogues",
        "documents_selected",
    ):
        value = resume_cursor.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"invalid resume cursor field: {key}")
    if resume_cursor["next_file_index"] > file_count:
        raise ConfigError("resume cursor exceeds source files")
    return dict(resume_cursor)


def _cursor(
    fingerprint_sha256: str,
    selection_sha256: str,
    next_file_index: int,
    next_dialogue_index: int,
    dialogues_seen: int,
    turns_seen: int,
    empty_dialogues: int,
    documents_selected: int,
    complete: bool,
) -> Dict[str, Any]:
    return {
        "adapter": "multiwoz_ptbr",
        "source_fingerprint_sha256": fingerprint_sha256,
        "selection_sha256": selection_sha256,
        "next_file_index": next_file_index,
        "next_dialogue_index": next_dialogue_index,
        "dialogues_seen": dialogues_seen,
        "turns_seen": turns_seen,
        "empty_dialogues": empty_dialogues,
        "documents_selected": documents_selected,
        "complete": complete,
    }


ADAPTER = MultiWOZPTBRAdapter()
