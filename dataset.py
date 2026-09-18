# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from concurrent.futures import ThreadPoolExecutor
import json
import itertools
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import random


def load_json_or_jsonl(dataset_path):
    """Load the JSON arrays used by ProsQA or an official JSONL QA split."""

    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_bpe_tokenizer(tokenizer):
    """Reject a non-BPE tokenizer when an experiment promises BPE accounting."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    model_name = type(model).__name__.lower() if model is not None else ""
    if "bpe" not in model_name:
        raise ValueError(
            "MuSiQue requires a fast BPE tokenizer so paragraph and prefix "
            f"budgets are measured consistently; got {type(tokenizer).__name__} "
            f"with backend model {type(model).__name__}."
        )


def _encode(tokenizer, text):
    # ``verbose=False`` avoids a misleading model_max_length warning while we
    # are still measuring candidates that will be cropped before model input.
    return tokenizer.encode(text, add_special_tokens=False, verbose=False)


def _decode(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _sentence_spans(text):
    # MuSiQue paragraphs are Wikipedia prose. This intentionally avoids an
    # external sentence-tokenizer dependency while retaining punctuation.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def build_finite_continuation_prompt(
    evidence,
    continuation,
    latent_tokens,
    *,
    interface,
    tail="",
):
    """Format a finite-state prompt with one explicit access boundary.

    Dataset loaders supply natural-language/symbolic evidence and continuation;
    the strict adapter interprets START as the point where evidence is sealed.
    """

    if latent_tokens < 0:
        raise ValueError("latent_tokens must be non-negative")
    latents = " <|latent|>" * latent_tokens
    if interface == "strict_continuation":
        return (
            evidence
            + " <|start-latent|>"
            + continuation
            + latents
            + " <|end-latent|>"
            + tail
        )
    if interface == "question_conditioned":
        return (
            evidence
            + continuation
            + " <|start-latent|>"
            + latents
            + " <|end-latent|>"
            + tail
        )
    raise ValueError(
        "interface must be question_conditioned or strict_continuation"
    )


def _crop_musique_paragraph(paragraph, support_answers, tokenizer, max_tokens):
    """BPE-crop a paragraph around its answer-bearing evidence sentence."""

    title = paragraph.get("title", "")
    text = paragraph.get("paragraph_text", "")
    full_rendered = f"{title}: {text}"
    full_ids = _encode(tokenizer, full_rendered)
    if len(full_ids) <= max_tokens:
        return _decode(tokenizer, full_ids).strip()

    sentences = _sentence_spans(text)
    lowered_answers = [str(answer).casefold() for answer in support_answers if answer]
    evidence_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if any(answer in sentence.casefold() for answer in lowered_answers)
        ),
        0,
    )
    order = [evidence_index]
    for distance in range(1, len(sentences)):
        for index in (evidence_index - distance, evidence_index + distance):
            if 0 <= index < len(sentences):
                order.append(index)

    selected = []
    for index in order:
        candidate = sorted(selected + [index])
        rendered = f"{title}: " + " ".join(sentences[i] for i in candidate)
        if len(_encode(tokenizer, rendered)) <= max_tokens:
            selected.append(index)
    if selected:
        rendered = f"{title}: " + " ".join(sentences[i] for i in sorted(selected))
    else:
        rendered = f"{title}: {text}"
    ids = _encode(tokenizer, rendered)[:max_tokens]
    return _decode(tokenizer, ids).strip()


def _musique_parts(sample, tokenizer, configs):
    """Return capped evidence, the post-boundary question, and hop targets."""

    if not sample.get("answerable", True):
        raise ValueError(f"MuSiQue example {sample.get('id')} is not answerable")
    decomposition = sample.get("question_decomposition", [])
    answer = str(sample.get("answer", "")).strip()
    if not decomposition or not answer:
        raise ValueError(f"MuSiQue example {sample.get('id')} lacks labels")

    support_by_idx = {}
    for hop in decomposition:
        support_by_idx.setdefault(hop["paragraph_support_idx"], []).append(hop["answer"])
    paragraphs = {paragraph["idx"]: paragraph for paragraph in sample["paragraphs"]}
    # Evidence lock: every bridge/final hop must occur in its labeled paragraph.
    for support_idx, hop_answers in support_by_idx.items():
        support_text = paragraphs[support_idx]["paragraph_text"].casefold()
        if any(str(hop_answer).casefold() not in support_text for hop_answer in hop_answers):
            raise ValueError(
                f"MuSiQue example {sample.get('id')} has an ungrounded bridge"
            )

    distractor_count = int(getattr(configs, "distractors", 0))
    if distractor_count not in {0, 4, 8, 12}:
        raise ValueError("distractors must be one of 0, 4, 8, or 12")
    gold = [paragraphs[index] for index in support_by_idx]
    distractors = [
        paragraph
        for paragraph in sample["paragraphs"]
        if paragraph["idx"] not in support_by_idx
    ][:distractor_count]
    paragraph_cap = int(getattr(configs, "max_paragraph_tokens", 112))
    gold_text = [
        _crop_musique_paragraph(
            paragraph, support_by_idx[paragraph["idx"]], tokenizer, paragraph_cap
        )
        for paragraph in gold
    ]
    distractor_text = [
        _crop_musique_paragraph(paragraph, [], tokenizer, paragraph_cap)
        for paragraph in distractors
    ]

    prefix_cap = int(getattr(configs, "max_prefix_tokens", 2048))

    def render(passages):
        evidence = " ".join(
            f"[PASSAGE {index}] {passage}" for index, passage in enumerate(passages)
        )
        return f"<eos> [EVIDENCE] {evidence}"

    # Remove distractors before touching gold evidence, as required by the
    # evidence-locked protocol.
    kept_distractors = list(distractor_text)
    prefix = render(gold_text + kept_distractors)
    while kept_distractors and len(_encode(tokenizer, prefix)) > prefix_cap:
        kept_distractors.pop()
        prefix = render(gold_text + kept_distractors)
    prefix_ids = _encode(tokenizer, prefix)
    if len(prefix_ids) > prefix_cap:
        # Gold is never silently removed. Token-level clipping is the final
        # fallback after all distractors have gone.
        prefix_ids = _encode(tokenizer, render(gold_text))[:prefix_cap]
        prefix = _decode(tokenizer, prefix_ids)

    hops = [str(hop["answer"]).strip() for hop in decomposition]
    return prefix, f" [Q] {sample['question']}", hops, answer


def get_musique_dataset(dataset_path, configs, tokenizer, mode, question_only=False):
    """Build finite-CoT examples from the official MuSiQue answerable JSONL."""

    if getattr(configs, "require_bpe", True):
        validate_bpe_tokenizer(tokenizer)
    base_dataset = load_json_or_jsonl(dataset_path)
    if configs.debug:
        base_dataset = base_dataset[: int(getattr(configs, "debug_samples", 128))]
    max_sequence_tokens = int(getattr(configs, "max_sequence_tokens", 2304))
    recurrent_updates = int(getattr(configs, "latent_steps", 4))
    musique_interface = getattr(
        configs,
        "continuation_interface",
        getattr(configs, "musique_interface", "question_conditioned"),
    )
    if musique_interface not in {"question_conditioned", "strict_continuation"}:
        raise ValueError(
            "musique_interface must be question_conditioned or strict_continuation"
        )
    preprocessing_workers = int(getattr(configs, "preprocessing_workers", 1))
    if preprocessing_workers < 1:
        raise ValueError("preprocessing_workers must be positive")
    processed = []
    started = time.monotonic()
    show_progress = bool(getattr(configs, "show_preprocessing_progress", True))

    def process_one(index_and_sample):
        source_index, sample = index_and_sample
        try:
            evidence, question, hops, answer = _musique_parts(
                sample, tokenizer, configs
            )
        except ValueError:
            # Filtering is explicit at dataset construction; preprocessing
            # manifests can record these IDs in a full evidence-lock run.
            return None
        if mode == "cot":
            prompt = evidence + question + " [REASONING]"
            continuation = " " + " -> ".join(hops) + f" [A] {answer} <eos>"
        elif mode == "no_cot":
            prompt = evidence + question + " [A]"
            continuation = f" {answer} <eos>"
        elif mode == "latent":
            if musique_interface == "question_conditioned":
                # Existing sealed-prefix interface: the first latent initializes
                # z_0 from evidence + question; the remaining T latents update it.
                prompt = build_finite_continuation_prompt(
                    evidence,
                    question,
                    recurrent_updates + 1,
                    interface=musique_interface,
                    tail=" [A]",
                )
            else:
                # Strict continuation interface: initialize z_0 from evidence at
                # the boundary, then let exactly T updates read the question.
                prompt = build_finite_continuation_prompt(
                    evidence,
                    question,
                    recurrent_updates,
                    interface=musique_interface,
                    tail=" [A]",
                )
            continuation = f" {answer} <eos>"
        else:
            raise ValueError(f"unknown MuSiQue mode: {mode}")

        prompt_ids = _encode(tokenizer, prompt)
        continuation_ids = [] if question_only else _encode(tokenizer, continuation)
        if len(prompt_ids) + len(continuation_ids) > max_sequence_tokens:
            # Evidence is first and gold passages precede distractors. Preserve
            # the complete question, boundary, latent markers, and answer prompt.
            if mode == "latent":
                evidence_ids = _encode(tokenizer, evidence)
                suffix_ids = prompt_ids[len(evidence_ids) :]
                evidence_budget = (
                    max_sequence_tokens - len(suffix_ids) - len(continuation_ids)
                )
                if evidence_budget < 1:
                    return None
                prompt_ids = evidence_ids[:evidence_budget] + suffix_ids
            else:
                prompt_ids = prompt_ids[: max_sequence_tokens - len(continuation_ids)]
        tokens = prompt_ids + continuation_ids
        item = {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
        if question_only:
            item["idx"] = source_index
        else:
            item["labels"] = [-100] * len(prompt_ids) + continuation_ids
        return item

    indexed_samples = enumerate(base_dataset)
    executor = None
    if preprocessing_workers == 1:
        results = map(process_one, indexed_samples)
    else:
        executor = ThreadPoolExecutor(max_workers=preprocessing_workers)
        results = executor.map(process_one, indexed_samples)
    try:
        for source_index, item in enumerate(results):
            if item is not None:
                processed.append(item)
            if show_progress and (source_index + 1) % 1000 == 0:
                print(
                    f"MuSiQue preprocessing {Path(dataset_path).name}: "
                    f"{source_index + 1}/{len(base_dataset)} source examples "
                    f"({preprocessing_workers} workers)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    if show_progress:
        print(
            f"MuSiQue preprocessing complete for {Path(dataset_path).name}: "
            f"retained {len(processed)}/{len(base_dataset)} examples in "
            f"{time.monotonic() - started:.1f}s with {preprocessing_workers} workers",
            flush=True,
        )
    return processed


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """
        # print(features)
        ## print(features[0]["input_ids"])
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def expand_data(
    data,
    k,
    max_steps,
    neg_sampling=False,
    continuation_interface="legacy",
):
    
    assert k <= max_steps + 1
    # k = 1, 2, 3, 4, 5
    
    symbol_to_idx = {}
    for i, s in enumerate(data['idx_to_symbol']):
        symbol_to_idx[s] = i
    
    def get_prompt(latent_tokens, *, answer_marker=False):
        
        random.shuffle(data['edges'])

        evidence = "<eos> " + "|".join(
            [f" {e[0]} {e[1]} " for e in data['edges']]
        ).strip()

        if random.random() < 0.5:
            candidates = str(data['target']) + " " + str(data['neg_target'])
        else:
            candidates = str(data['neg_target']) + " " + str(data['target'])

        continuation = " [Q] " + candidates + " [R] " + str(data['root'])
        tail = " [A] " if answer_marker else " "
        if continuation_interface == "legacy":
            return evidence + continuation + " <|latent|>" * latent_tokens + tail
        return build_finite_continuation_prompt(
            evidence,
            continuation,
            latent_tokens,
            interface=continuation_interface,
            tail=tail,
        )


    # return_data = None
    if k <= max_steps:
        # for n in data["neighbor_k"][str(k)]:
        if neg_sampling:
            if random.random() < 0.2:
                question = get_prompt(k - 1, answer_marker=True)
                continuation = "<|no-answer|>"
                return_data = (question, continuation)
        
            else:
                n = random.choice(data["neighbor_k"][str(k)])
                question = get_prompt(k - 1)
                continuation = str(n)
                return_data = (question, continuation)
        else:
            n = random.choice(data["neighbor_k"][str(k)])
            question = get_prompt(k - 1)
            continuation = str(n)
            return_data = (question, continuation)


    elif k == max_steps + 1:
        if neg_sampling:
            if random.random() < 0.2:
                question = get_prompt(
                    random.randint(0, max_steps - 1), answer_marker=True
                )
                continuation = "<|no-answer|>"
                return_data = (question, continuation)
            else:
                question = get_prompt(max_steps, answer_marker=True)
                continuation = str(data["target"])
                return_data = (question, continuation)
        else:
            question = get_prompt(max_steps, answer_marker=True)
            continuation = str(data["target"])
            return_data = (question, continuation)
    
    else:
        raise ValueError(f"k is {k}, max_steps is {max_steps}")

    # if random.random() < 0.05: # debug print
    #     print(f"Question: {return_data[0]}")
    #     print(f"Continuation: {return_data[1]}")
    
    return return_data



def get_graph_latent_cot_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
):
    base_dataset = json.load(open(dataset_path))
    
    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.randint(0, min(scheduled_stage, len(sample["steps"])))
        else:
            scheduled_stage_to_train = min(scheduled_stage, len(sample["steps"]))
            # 0, 1, 2, 3, 4

        # this range is [0, ..., len(sample["steps"])]
        # including both ends

        expanded_data = expand_data(
            sample,
            scheduled_stage_to_train + 1,
            len(sample["steps"]),
            continuation_interface=getattr(
                configs, "continuation_interface", "legacy"
            ),
        )
        
        # Process each question-continuation pair
        processed_samples = []
        for question, continuation in [expanded_data]:
            question_tokenized = tokenizer.encode(question, add_special_tokens=False)
            continuation_tokenized = tokenizer.encode(continuation, add_special_tokens=False)
            
            tokens = question_tokenized + continuation_tokenized
            
            processed_sample = {
                "input_ids": tokens,
                "labels": [-100] * len(question_tokenized) + continuation_tokenized,
                "attention_mask": [1] * len(tokens),
                "position_ids": list(range(len(tokens))),
            }
            processed_samples.append(processed_sample)
            
        return processed_samples

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for sample in base_dataset:
                processed_samples = process_dataset(sample)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Process each sample and collect all results
        all_processed_samples = []
        for sample in base_dataset:
            processed_samples = process_dataset(sample)
            all_processed_samples.extend(processed_samples)
        
        processed_dataset = all_processed_samples
        random.shuffle(processed_dataset)
        dataset = processed_dataset
    return dataset

def get_graph_latent_question_dataset(
    dataset_path,
    scheduled_stage,
    configs,
    tokenizer,
):
    base_dataset = json.load(open(dataset_path))
    if configs.debug:
        base_dataset = base_dataset[:10000]
    # similar to get_graph_latent_dataset, but we only keep the question
    # without the continuation
    
    def process_dataset(sample, idx):
        expanded_data = expand_data(
            sample,
            len(sample["steps"]) + 1,
            len(sample["steps"]),
            neg_sampling=False,
            continuation_interface=getattr(
                configs, "continuation_interface", "legacy"
            ),
        )
        processed_samples = []
        for question, continuation in [expanded_data]:
            question_tokenized = tokenizer.encode(question, add_special_tokens=False)
            processed_samples.append({
                "input_ids": question_tokenized,
                "attention_mask": [1] * len(question_tokenized),
                "position_ids": list(range(len(question_tokenized))),
                "idx": idx,
            })
        return processed_samples
    
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for idx, sample in enumerate(base_dataset):
                processed_samples = process_dataset(sample, idx)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        all_processed_samples = []
        for idx, sample in enumerate(base_dataset):
            processed_samples = process_dataset(sample, idx)
            all_processed_samples.extend(processed_samples)
        random.shuffle(all_processed_samples)
        dataset = all_processed_samples

    return dataset

def get_graph_cot_dataset(
    dataset_path,
    configs,
    tokenizer,
):
    """
    Creates a dataset for training graph reasoning with chain of thought.
    Each sample will contain the graph edges, question, and the full reasoning path to the answer.
    """
    base_dataset = json.load(open(dataset_path))
    
    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):
        # Shuffle edges for robustness
        random.shuffle(sample['edges'])
        
        # Construct the question part
        question = "<eos> " + "|".join([f" {e[0]} {e[1]} " for e in sample['edges']]).strip() + " [Q] "
        
        # Randomly order target and neg_target in the question
        if random.random() < 0.5:
            question += f"{sample['target']} {sample['neg_target']}"
        else:
            question += f"{sample['neg_target']} {sample['target']}"
            
        question += f" [R] {sample['root']}"
        
        # Construct the chain of thought (optimal path) and answer
        current_node = sample['root']
        continuation = ""
        for i in range(1, 10):
            if str(i) in sample["neighbor_k"]:
                next_node = random.choice(
                    [n for n in sample["neighbor_k"][str(i)] if [current_node, n] in sample['edges']]
                )
                continuation += f" {next_node}"
                current_node = next_node
        
        continuation += f" [A] {sample['target']} <eos>"
        
        # Tokenize question and continuation
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        continuation_tokenized = tokenizer.encode(continuation, add_special_tokens=False)
        
        tokens = question_tokenized + continuation_tokenized
        
        processed_sample = {
            "input_ids": tokens,
            "labels": [-100] * len(question_tokenized) + continuation_tokenized,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
            
        return [processed_sample]

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for sample in base_dataset:
                processed_samples = process_dataset(sample)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Process each sample and collect all results
        all_processed_samples = []
        for sample in base_dataset:
            processed_samples = process_dataset(sample)
            all_processed_samples.extend(processed_samples)
        
        processed_dataset = all_processed_samples
        random.shuffle(processed_dataset)
        dataset = processed_dataset
        
    return dataset

def get_graph_no_cot_dataset(
    dataset_path,
    configs,
    tokenizer,
):
    """
    Creates a dataset for training graph reasoning without chain of thought.
    Each sample will contain the graph edges, question, and only the final answer
    without intermediate reasoning steps.
    """
    base_dataset = json.load(open(dataset_path))
    
    if configs.debug:
        base_dataset = base_dataset[:10000]

    def process_dataset(sample):
        # Shuffle edges for robustness
        random.shuffle(sample['edges'])
        
        # Construct the question part
        question = "<eos> " + "|".join([f" {e[0]} {e[1]} " for e in sample['edges']]).strip() + " [Q] "
        
        # Randomly order target and neg_target in the question
        if random.random() < 0.5:
            question += f"{sample['target']} {sample['neg_target']}"
        else:
            question += f"{sample['neg_target']} {sample['target']}"
            
        question += f" [R] {sample['root']}"
        
        # Only include the answer without the reasoning path
        continuation = f" [A] {sample['target']} <eos>"
        
        # Tokenize question and continuation
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        continuation_tokenized = tokenizer.encode(continuation, add_special_tokens=False)
        
        tokens = question_tokenized + continuation_tokenized
        
        processed_sample = {
            "input_ids": tokens,
            "labels": [-100] * len(question_tokenized) + continuation_tokenized,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
            
        return [processed_sample]

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for sample in base_dataset:
                processed_samples = process_dataset(sample)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Process each sample and collect all results
        all_processed_samples = []
        for sample in base_dataset:
            processed_samples = process_dataset(sample)
            all_processed_samples.extend(processed_samples)
        
        processed_dataset = all_processed_samples
        random.shuffle(processed_dataset)
        dataset = processed_dataset
        
    return dataset

def get_graph_no_latent_question_dataset(
    dataset_path,
    configs,
    tokenizer,
):
    """
    Creates a dataset containing only the questions from the graph reasoning dataset,
    without any latent tokens. Used for inference to get the input questions.
    """
    base_dataset = json.load(open(dataset_path))
    if configs.debug:
        base_dataset = base_dataset[:10000]
    
    def process_dataset(sample, idx):
        # Construct the question part
        random.shuffle(sample['edges'])
        question = "<eos> " + "|".join([f" {e[0]} {e[1]} " for e in sample['edges']]).strip() + " [Q] "
        
        # Randomly order target and neg_target in the question
        if random.random() < 0.5:
            question += f"{sample['target']} {sample['neg_target']}"
        else:
            question += f"{sample['neg_target']} {sample['target']}"
            
        question += f" [R] {sample['root']}"
        
        # Tokenize the question
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        
        processed_sample = {
            "input_ids": question_tokenized,
            "attention_mask": [1] * len(question_tokenized),
            "position_ids": list(range(len(question_tokenized))),
            "idx": idx,
        }
        return [processed_sample]
    
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            # Process each sample and collect all results
            all_processed_samples = []
            for idx, sample in enumerate(base_dataset):
                processed_samples = process_dataset(sample, idx)
                all_processed_samples.extend(processed_samples)
            
            processed_dataset = all_processed_samples
            random.shuffle(processed_dataset)
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Process each sample and collect all results
        all_processed_samples = []
        for idx, sample in enumerate(base_dataset):
            processed_samples = process_dataset(sample, idx)
            all_processed_samples.extend(processed_samples)
        
        processed_dataset = all_processed_samples
        random.shuffle(processed_dataset)
        dataset = processed_dataset

    return dataset
