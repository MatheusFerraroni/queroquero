import unittest

from queroquero.datasets.base import Document
from queroquero.packing import (
    TokenizedDocument,
    clean_deduplicate_and_tokenize,
    pack_for_budgets,
    plan_incremental_packing,
)


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3

    def __len__(self):
        return 128

    def get_vocab(self):
        return {f"token-{index}": index for index in range(128)}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [4 + (ord(character) % 100) for character in text]}


class PackingTest(unittest.TestCase):
    def test_incremental_packing_matches_the_legacy_packer_byte_for_byte(self) -> None:
        documents = [
            Document(
                text=f"Documento incremental distinto {index}. " * 90,
                source_ref=f"synthetic:{index}",
                source_position={"index": index},
            )
            for index in range(30)
        ]
        tokenizer = FakeTokenizer()
        tokenized, legacy_metrics = clean_deduplicate_and_tokenize(
            documents,
            tokenizer,
            dataset_id="synthetic",
            seed=42,
            min_characters=20,
        )
        legacy = pack_for_budgets(
            tokenized,
            dataset_id="synthetic",
            seed=42,
            sequence_length=1024,
            train_sequences=8,
            eval_sequences=2,
        )
        incremental = plan_incremental_packing(
            documents,
            tokenizer,
            dataset_id="synthetic",
            seed=42,
            sequence_length=1024,
            train_sequences=8,
            eval_sequences=2,
            min_characters=20,
        )

        self.assertEqual(tuple(incremental.train), legacy.train)
        self.assertEqual(tuple(incremental.evaluation), legacy.evaluation)
        self.assertEqual(incremental.tokenization_metrics, legacy_metrics)
        self.assertEqual(
            incremental.train.discarded_tail_tokens,
            legacy.metrics["train_discarded_tail_tokens"],
        )
        self.assertEqual(
            incremental.evaluation.tokens_not_selected_by_sequence_budget,
            legacy.metrics["eval_tokens_not_selected_by_sequence_budget"],
        )

    def test_eos_is_always_appended_as_an_explicit_document_boundary(self) -> None:
        class EndsInEosTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return {"input_ids": [7, self.eos_token_id]}

        tokenized, _ = clean_deduplicate_and_tokenize(
            [Document("conteúdo", "synthetic:eos", {"index": 0})],
            EndsInEosTokenizer(),
            dataset_id="synthetic",
            seed=42,
            min_characters=1,
        )
        self.assertEqual(tokenized[0].input_ids, (7, 2, 2))

    def test_clean_deduplicate_split_and_pack_exact_sequences(self) -> None:
        repeated = "<p>Texto sintético longo para validar limpeza.</p> " * 30
        documents = [
            Document(
                text=repeated if index < 2 else f"Documento sintético {index}. " * 80,
                source_ref=f"synthetic:{index}",
                source_position={"index": index},
            )
            for index in range(30)
        ]
        tokenized, metrics = clean_deduplicate_and_tokenize(
            documents,
            FakeTokenizer(),
            dataset_id="synthetic",
            seed=42,
            min_characters=20,
        )
        packed = pack_for_budgets(
            tokenized,
            dataset_id="synthetic",
            seed=42,
            sequence_length=1024,
            train_sequences=8,
            eval_sequences=2,
        )
        self.assertEqual(metrics["documents_exact_duplicates"], 1)
        self.assertEqual(len(packed.train), 8)
        self.assertEqual(len(packed.evaluation), 2)
        all_sequences = packed.train + packed.evaluation
        self.assertTrue(all(len(record.input_ids) == 1024 for record in all_sequences))
        self.assertTrue(
            all(sum(record.source_token_counts) == 1024 for record in all_sequences)
        )
        train_sources = {
            source for record in packed.train for source in record.source_ref_sha256
        }
        eval_sources = {
            source for record in packed.evaluation for source in record.source_ref_sha256
        }
        self.assertTrue(train_sources.isdisjoint(eval_sources))

    def test_insufficient_candidates_fail_explicitly(self) -> None:
        documents = [
            Document("texto curto suficiente", "synthetic:1", {"index": 1})
        ]
        tokenized, _ = clean_deduplicate_and_tokenize(
            documents,
            FakeTokenizer(),
            dataset_id="synthetic",
            seed=42,
            min_characters=1,
        )
        with self.assertRaisesRegex(RuntimeError, "did not contain enough tokens"):
            pack_for_budgets(
                tokenized,
                dataset_id="synthetic",
                seed=42,
                sequence_length=1024,
                train_sequences=8,
                eval_sequences=2,
            )

    def test_oversized_documents_span_sequences_and_only_tail_is_discarded(self) -> None:
        documents = [
            TokenizedDocument(
                input_ids=tuple([10 + index] * 1100),
                source_ref_sha256=f"{index + 1:064x}",
                split_score="0" if index == 0 else "f",
            )
            for index in range(2)
        ]
        packed = pack_for_budgets(
            documents,
            dataset_id="synthetic",
            seed=42,
            sequence_length=1024,
            train_sequences=1,
            eval_sequences=1,
        )
        self.assertEqual(len(packed.train[0].input_ids), 1024)
        self.assertEqual(len(packed.evaluation[0].input_ids), 1024)
        self.assertEqual(packed.metrics["train_discarded_tail_tokens"], 76)
        self.assertEqual(packed.metrics["eval_discarded_tail_tokens"], 76)
        self.assertEqual(
            packed.metrics["train_tokens_not_selected_by_sequence_budget"], 76
        )
        self.assertEqual(
            packed.metrics["eval_tokens_not_selected_by_sequence_budget"], 76
        )


if __name__ == "__main__":
    unittest.main()
