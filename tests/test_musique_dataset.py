import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from dataset import get_musique_dataset, validate_bpe_tokenizer


def tiny_bpe():
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.train_from_iterator(
        [
            "<eos> [EVIDENCE] [PASSAGE] [QUESTION] [REASONING] [A]",
            "Alpha points to Bridge. Bridge was founded in 1960.",
            "What year was the linked organization founded?",
            "distractor irrelevant context",
        ],
        BpeTrainer(
            vocab_size=128,
            special_tokens=[
                "<unk>",
                "<eos>",
                "<|start-latent|>",
                "<|end-latent|>",
                "<|latent|>",
                "<pad>",
            ],
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
        additional_special_tokens=[
            "<|start-latent|>", "<|end-latent|>", "<|latent|>"
        ],
    )


def sample():
    return {
        "id": "2hop__1_2",
        "answerable": True,
        "question": "What year was the linked organization founded?",
        "answer": "1960",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alpha",
                "paragraph_text": "Alpha points to Bridge. Extra nearby context.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Bridge",
                "paragraph_text": "Bridge was founded in 1960. More context.",
                "is_supporting": True,
            },
            *[
                {
                    "idx": index,
                    "title": f"Distractor {index}",
                    "paragraph_text": "distractor irrelevant context " * 20,
                    "is_supporting": False,
                }
                for index in range(2, 14)
            ],
        ],
        "question_decomposition": [
            {
                "question": "Alpha points to what?",
                "answer": "Bridge",
                "paragraph_support_idx": 0,
            },
            {
                "question": "When was #1 founded?",
                "answer": "1960",
                "paragraph_support_idx": 1,
            },
        ],
    }


class MusiqueDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_bpe()
        self.config = SimpleNamespace(
            require_bpe=True,
            debug=False,
            distractors=12,
            max_paragraph_tokens=24,
            max_prefix_tokens=100,
            max_sequence_tokens=128,
            latent_steps=4,
            musique_interface="question_conditioned",
            show_preprocessing_progress=False,
        )

    def write_split(self, records):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_latent_musique_is_bpe_capped_and_answer_supervised(self):
        dataset = get_musique_dataset(
            self.write_split([sample()]), self.config, self.tokenizer, "latent"
        )
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertLessEqual(len(item["input_ids"]), self.config.max_sequence_tokens)
        supervised = [
            token for token, label in zip(item["input_ids"], item["labels"])
            if label != -100
        ]
        self.assertIn("1960", self.tokenizer.decode(supervised))
        rendered = self.tokenizer.decode(item["input_ids"])
        compact_rendered = rendered.replace(" ", "")
        self.assertIn("Alpha", compact_rendered)
        self.assertIn("Bridge", compact_rendered)
        # The tight prefix budget forces at least one distractor out while
        # keeping both labeled evidence passages.
        self.assertNotIn("Distractor13", compact_rendered)

        start = item["input_ids"].index(
            self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        )
        end = item["input_ids"].index(
            self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        )
        latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        question_ids = self.tokenizer.encode(
            " [Q] What year was the linked organization founded?",
            add_special_tokens=False,
        )
        self.assertEqual(item["input_ids"][start - len(question_ids) : start], question_ids)
        self.assertEqual(item["input_ids"][start + 1 : end].count(latent_id), 5)

    def test_strict_layout_places_question_after_boundary_before_updates(self):
        self.config.musique_interface = "strict_continuation"
        dataset = get_musique_dataset(
            self.write_split([sample()]), self.config, self.tokenizer, "latent"
        )
        ids = dataset[0]["input_ids"]
        start = ids.index(self.tokenizer.convert_tokens_to_ids("<|start-latent|>"))
        end = ids.index(self.tokenizer.convert_tokens_to_ids("<|end-latent|>"))
        latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        question_ids = self.tokenizer.encode(
            " [Q] What year was the linked organization founded?",
            add_special_tokens=False,
        )
        self.assertEqual(ids[start + 1 : start + 1 + len(question_ids)], question_ids)
        self.assertEqual(ids[start + 1 : end].count(latent_id), 4)

    def test_strict_zero_updates_keeps_boundary_and_question(self):
        self.config.musique_interface = "strict_continuation"
        self.config.latent_steps = 0
        dataset = get_musique_dataset(
            self.write_split([sample()]), self.config, self.tokenizer, "latent"
        )
        ids = dataset[0]["input_ids"]
        self.assertNotIn(self.tokenizer.convert_tokens_to_ids("<|latent|>"), ids)
        self.assertIn(self.tokenizer.convert_tokens_to_ids("<|start-latent|>"), ids)
        self.assertIn(self.tokenizer.convert_tokens_to_ids("<|end-latent|>"), ids)

    def test_cot_uses_published_decomposition_answers(self):
        dataset = get_musique_dataset(
            self.write_split([sample()]), self.config, self.tokenizer, "cot"
        )
        supervised_text = self.tokenizer.decode(
            [label for label in dataset[0]["labels"] if label != -100]
        )
        self.assertIn("Bridge", supervised_text)
        self.assertIn("1960", supervised_text)

    def test_ungrounded_bridge_is_filtered(self):
        record = sample()
        record["question_decomposition"][0]["answer"] = "Missing Entity"
        dataset = get_musique_dataset(
            self.write_split([record]), self.config, self.tokenizer, "latent"
        )
        self.assertEqual(dataset, [])

    def test_non_bpe_tokenizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a fast BPE tokenizer"):
            validate_bpe_tokenizer(object())

    def test_parallel_preprocessing_preserves_examples(self):
        path = self.write_split([sample(), sample()])
        self.config.preprocessing_workers = 1
        serial = get_musique_dataset(path, self.config, self.tokenizer, "latent")
        self.config.preprocessing_workers = 4
        parallel = get_musique_dataset(path, self.config, self.tokenizer, "latent")
        self.assertEqual(parallel, serial)


if __name__ == "__main__":
    unittest.main()
