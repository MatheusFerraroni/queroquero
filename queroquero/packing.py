from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import canonical_json_bytes, sha256_bytes
from .datasets.base import Document, clean_text, safe_source_hash, stable_hash


@dataclass(frozen=True)
class TokenizedDocument:
    input_ids: Tuple[int, ...]
    source_ref_sha256: str
    split_score: str


@dataclass(frozen=True)
class PackedSequence:
    sequence_id: str
    input_ids: Tuple[int, ...]
    source_ref_sha256: Tuple[str, ...]
    source_token_counts: Tuple[int, ...]


@dataclass(frozen=True)
class PackingResult:
    train: Tuple[PackedSequence, ...]
    evaluation: Tuple[PackedSequence, ...]
    metrics: Dict[str, int]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    special_ids = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    vocabulary = sorted(tokenizer.get_vocab().items(), key=lambda item: item[0])
    backend_value: Any = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        serialized = backend.to_str()
        try:
            backend_value = json.loads(serialized)
        except (TypeError, ValueError):
            backend_value = str(serialized)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "backend": backend_value,
                "class": tokenizer.__class__.__name__,
                "special_token_ids": special_ids,
                "vocabulary": vocabulary,
            }
        )
    )


def clean_deduplicate_and_tokenize(
    documents: Iterable[Document],
    tokenizer: Any,
    *,
    dataset_id: str,
    seed: int,
    min_characters: int,
    punctuation_spacing: str = "preserve",
) -> Tuple[List[TokenizedDocument], Dict[str, int]]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("the pinned tokenizer must define eos_token_id")
    seen_text_hashes = set()
    seen_source_hashes = set()
    tokenized: List[TokenizedDocument] = []
    metrics = {
        "documents_received": 0,
        "documents_empty_or_short": 0,
        "documents_exact_duplicates": 0,
        "documents_tokenized": 0,
        "tokens_before_budget": 0,
    }

    for document in documents:
        metrics["documents_received"] += 1
        text = clean_text(
            document.text,
            strip_html=True,
            punctuation_spacing=punctuation_spacing,
        )
        if len(text) < min_characters:
            metrics["documents_empty_or_short"] += 1
            continue
        text_hash = stable_hash(text)
        if text_hash in seen_text_hashes:
            metrics["documents_exact_duplicates"] += 1
            continue
        seen_text_hashes.add(text_hash)
        source_hash = safe_source_hash(document.source_ref)
        if source_hash in seen_source_hashes:
            raise RuntimeError("dataset contains a duplicate source reference")
        seen_source_hashes.add(source_hash)
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = [int(value) for value in values]
        if not input_ids:
            metrics["documents_empty_or_short"] += 1
            continue
        # EOS is a document boundary, even when the last content token happens
        # to have the same numeric ID.
        input_ids.append(int(eos_token_id))
        if any(value < 0 or value > 2_147_483_647 for value in input_ids):
            raise RuntimeError("tokenizer produced an ID outside the int32 range")
        tokenized.append(
            TokenizedDocument(
                input_ids=tuple(input_ids),
                source_ref_sha256=source_hash,
                split_score=stable_hash(seed, dataset_id, "split", source_hash),
            )
        )
        metrics["documents_tokenized"] += 1
        metrics["tokens_before_budget"] += len(input_ids)
    return tokenized, metrics


def pack_for_budgets(
    documents: Sequence[TokenizedDocument],
    *,
    dataset_id: str,
    seed: int,
    sequence_length: int,
    train_sequences: int,
    eval_sequences: int,
) -> PackingResult:
    if sequence_length != 1024:
        raise ValueError("sequence_length must be exactly 1024")
    eval_target_tokens = eval_sequences * sequence_length
    ranked = sorted(documents, key=lambda document: document.split_score)
    evaluation_documents: List[TokenizedDocument] = []
    train_documents: List[TokenizedDocument] = []
    eval_tokens = 0
    for document in ranked:
        if eval_tokens < eval_target_tokens:
            evaluation_documents.append(document)
            eval_tokens += len(document.input_ids)
        else:
            train_documents.append(document)

    train, train_tail, train_budget_excess = _pack_split(
        train_documents,
        dataset_id=dataset_id,
        split="train",
        seed=seed,
        sequence_length=sequence_length,
        limit=train_sequences,
    )
    evaluation, eval_tail, eval_budget_excess = _pack_split(
        evaluation_documents,
        dataset_id=dataset_id,
        split="eval",
        seed=seed,
        sequence_length=sequence_length,
        limit=eval_sequences,
    )
    if len(train) != train_sequences or len(evaluation) != eval_sequences:
        raise RuntimeError(
            "candidate budget did not contain enough tokens for the requested "
            f"splits: train={len(train)}/{train_sequences}, "
            f"eval={len(evaluation)}/{eval_sequences}"
        )
    return PackingResult(
        train=tuple(train),
        evaluation=tuple(evaluation),
        metrics={
            "train_documents": len(train_documents),
            "eval_documents": len(evaluation_documents),
            "train_sequences": len(train),
            "eval_sequences": len(evaluation),
            "train_tokens": len(train) * sequence_length,
            "eval_tokens": len(evaluation) * sequence_length,
            "train_discarded_tail_tokens": train_tail,
            "eval_discarded_tail_tokens": eval_tail,
            "train_tokens_not_selected_by_sequence_budget": train_budget_excess,
            "eval_tokens_not_selected_by_sequence_budget": eval_budget_excess,
        },
    )


def _pack_split(
    documents: Sequence[TokenizedDocument],
    *,
    dataset_id: str,
    split: str,
    seed: int,
    sequence_length: int,
    limit: int,
) -> Tuple[List[PackedSequence], int, int]:
    ordered = sorted(
        documents,
        key=lambda document: stable_hash(
            seed, dataset_id, split, document.source_ref_sha256
        ),
    )
    token_buffer: List[int] = []
    source_buffer: List[str] = []
    result: List[PackedSequence] = []
    for document_index, document in enumerate(ordered):
        token_buffer.extend(document.input_ids)
        source_buffer.extend(
            [document.source_ref_sha256] * len(document.input_ids)
        )
        while len(token_buffer) >= sequence_length and len(result) < limit:
            input_ids = tuple(token_buffer[:sequence_length])
            sources = source_buffer[:sequence_length]
            del token_buffer[:sequence_length]
            del source_buffer[:sequence_length]
            counts: "OrderedDict[str, int]" = OrderedDict()
            for source in sources:
                counts[source] = counts.get(source, 0) + 1
            index = len(result)
            sequence_id = stable_hash(
                dataset_id,
                split,
                index,
                bytes().join(int(token).to_bytes(4, "little") for token in input_ids),
            )
            result.append(
                PackedSequence(
                    sequence_id=sequence_id,
                    input_ids=input_ids,
                    source_ref_sha256=tuple(counts.keys()),
                    source_token_counts=tuple(counts.values()),
                )
            )
        if len(result) >= limit:
            tokens_after_budget = len(token_buffer) + sum(
                len(remaining.input_ids)
                for remaining in ordered[document_index + 1 :]
            )
            return (
                result,
                tokens_after_budget % sequence_length,
                tokens_after_budget,
            )
    return result, len(token_buffer), 0
