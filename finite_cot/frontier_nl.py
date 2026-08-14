"""Experiment E1: learned GPT-2 frontier retrieval through a sealed finite state.

The module keeps the scientific interface deliberately narrow.  Prefix encoding
and post-boundary decoding are separate calls; the decoder receives only the
hard finite code and the independently sampled natural-language query.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rbs_adapter import StrictFiniteStateCoconut
from .theory import fano_error_lower_bound
from .training import resolve_device, set_seed


LATENT_TOKEN = "<|latent|>"
DEFAULT_MODEL_ID = "openai-community/gpt2"
SPLITS = ("train", "dev", "test")


def derive_split_seed(n: int, seed: int, split: str) -> int:
    """Derive a stable data seed with no capacity variable in its inputs."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    payload = f"e1-frontier-nl-v1|n={int(n)}|seed={int(seed)}|split={split}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def render_frontier_prefix(frontier: Sequence[int]) -> str:
    lines = ["Memory record:"]
    lines.extend(
        f"Item {index} is {'present' if int(value) else 'absent'}."
        for index, value in enumerate(frontier)
    )
    lines.append("Remember the status of every item.")
    return "\n".join(lines)


def render_query(query: int) -> str:
    return f"\nQuestion: Is item {int(query)} present?\nAnswer:"


def render_example(frontier: Sequence[int], query: int) -> str:
    """Render the auditable serialized example, including the strict boundary."""

    return f"{render_frontier_prefix(frontier)}\n{LATENT_TOKEN}{render_query(query)}"


def _encode_one(tokenizer, text: str) -> Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("E1 template unexpectedly tokenized to an empty sequence")
    return torch.tensor(ids, dtype=torch.long)


class FrontierNLDataset(Dataset):
    """Fixed, deterministic ``(S,V)`` examples for one E1 split."""

    def __init__(
        self,
        n: int,
        num_samples: int,
        seed: int,
        split: str,
        tokenizer,
    ) -> None:
        if n < 1 or num_samples < 1:
            raise ValueError("n and num_samples must be positive")
        self.n = int(n)
        self.num_samples = int(num_samples)
        self.experiment_seed = int(seed)
        self.split = split
        self.split_seed = derive_split_seed(n, seed, split)
        generator = torch.Generator().manual_seed(self.split_seed)
        self.frontiers = torch.randint(
            0, 2, (num_samples, n), generator=generator, dtype=torch.uint8
        )
        self.queries = torch.randint(
            0, n, (num_samples,), generator=generator, dtype=torch.long
        )
        rows = torch.arange(num_samples)
        self.labels = self.frontiers[rows, self.queries].to(torch.long)

        # Tokenization is deterministic and cached once.  It does not depend on
        # d, p, or R, just like the sampled examples themselves.
        self.prefix_input_ids = [
            _encode_one(tokenizer, render_frontier_prefix(frontier.tolist()))
            for frontier in self.frontiers
        ]
        self.query_input_ids = [
            _encode_one(tokenizer, render_query(int(query)))
            for query in self.queries
        ]

        fingerprint = hashlib.sha256()
        fingerprint.update(self.frontiers.numpy().tobytes())
        fingerprint.update(self.queries.numpy().tobytes())
        fingerprint.update(self.labels.numpy().tobytes())
        self.fingerprint = fingerprint.hexdigest()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "prefix_input_ids": self.prefix_input_ids[index],
            "query_input_ids": self.query_input_ids[index],
            "label": self.labels[index],
            "sample_index": torch.tensor(index, dtype=torch.long),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "num_samples": self.num_samples,
            "experiment_seed": self.experiment_seed,
            "split": self.split,
            "split_seed": self.split_seed,
            "seed_derivation": "sha256(e1-frontier-nl-v1|n|seed|split)",
            "fingerprint_sha256": self.fingerprint,
        }


@dataclass
class FrontierNLCollator:
    pad_token_id: int

    def _pad(self, values: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        width = max(value.numel() for value in values)
        ids = torch.full(
            (len(values), width), int(self.pad_token_id), dtype=torch.long
        )
        mask = torch.zeros((len(values), width), dtype=torch.long)
        for row, value in enumerate(values):
            ids[row, : value.numel()] = value
            mask[row, : value.numel()] = 1
        return ids, mask

    def __call__(self, examples: Sequence[Mapping[str, Tensor]]) -> Dict[str, Tensor]:
        prefix_ids, prefix_mask = self._pad(
            [example["prefix_input_ids"] for example in examples]
        )
        query_ids, query_mask = self._pad(
            [example["query_input_ids"] for example in examples]
        )
        return {
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": prefix_mask,
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "labels": torch.stack([example["label"] for example in examples]),
            "sample_indices": torch.stack(
                [example["sample_index"] for example in examples]
            ),
        }


def answer_token_ids(tokenizer) -> Dict[str, int]:
    """Return GPT-2's single-token binary answers, failing loudly otherwise."""

    result: Dict[str, int] = {}
    for label, text in (("no", " no"), ("yes", " yes")):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"E1 answer string {text!r} must map to exactly one token; got {ids}"
            )
        result[label] = int(ids[0])
    if result["no"] == result["yes"]:
        raise ValueError("E1 yes/no answer token IDs must be distinct")
    return result


def validate_e1_config(config: Mapping[str, Any]) -> None:
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(f"E1 model_id must be exactly {DEFAULT_MODEL_ID!r}")
    finite = config.get("finite_state") or {}
    required = {
        "enabled": True,
        "access_mode": "sealed_prefix",
        "bits_per_coordinate": 2,
        "clip_value": 1.0,
        "learnable_clip": False,
    }
    for key, expected in required.items():
        if finite.get(key) != expected:
            raise ValueError(f"E1 finite_state.{key} must be {expected!r}")
    if int(config.get("transcript_length", 0)) != 0:
        raise ValueError("E1 transcript_length must be zero")
    if bool(config.get("train_all_queries", False)):
        raise ValueError("headline E1 does not permit train_all_queries")
    if int(config.get("bits", finite["bits_per_coordinate"])) != 2:
        raise ValueError("E1 fixes p=2 bits per coordinate")


def load_e1_model_and_tokenizer(config: Mapping[str, Any]):
    """Load raw pretrained GPT-2 and add only the latent boundary token."""

    validate_e1_config(config)
    model_id = str(config.get("model_id", DEFAULT_MODEL_ID))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN]}
    )
    if added not in (0, 1):
        raise RuntimeError("E1 must add at most the single latent boundary token")
    boundary_id = int(tokenizer.convert_tokens_to_ids(LATENT_TOKEN))
    tokens = answer_token_ids(tokenizer)

    # This is intentionally from_pretrained, never from_config.
    base = AutoModelForCausalLM.from_pretrained(model_id)
    base.resize_token_embeddings(len(tokenizer))
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    finite = dict(config["finite_state"])
    finite["state_dim"] = int(config["state_dim"])
    model = StrictFiniteStateCoconut(
        base_causallm=base,
        latent_token_id=boundary_id,
        start_latent_id=boundary_id,
        end_latent_id=boundary_id,
        eos_token_id=int(tokenizer.eos_token_id),
        finite_state=finite,
    )
    return model, tokenizer, tokens


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _forward(
    model: StrictFiniteStateCoconut,
    batch: Mapping[str, Tensor],
    tokens: Mapping[str, int],
):
    return model.forward_answer_only(
        prefix_input_ids=batch["prefix_input_ids"],
        prefix_attention_mask=batch["prefix_attention_mask"],
        query_input_ids=batch["query_input_ids"],
        query_attention_mask=batch["query_attention_mask"],
        no_token_id=tokens["no"],
        yes_token_id=tokens["yes"],
        targets=batch["labels"],
    )


@torch.no_grad()
def evaluate_e1(
    model: StrictFiniteStateCoconut,
    loader: DataLoader,
    device: torch.device,
    tokens: Mapping[str, int],
) -> Dict[str, Any]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    code_min: int | None = None
    code_max: int | None = None
    unique_codes: set[tuple[int, ...]] = set()
    code_examples: List[List[int]] = []
    for host_batch in loader:
        batch = _to_device(host_batch, device)
        output = _forward(model, batch, tokens)
        predictions = output.binary_logits.argmax(dim=-1)
        batch_size = batch["labels"].numel()
        correct += int((predictions == batch["labels"]).sum().item())
        total += batch_size
        loss_sum += float(output.loss.item()) * batch_size
        codes = output.state_codes.detach().cpu()
        if codes.numel():
            batch_min, batch_max = int(codes.min()), int(codes.max())
            code_min = batch_min if code_min is None else min(code_min, batch_min)
            code_max = batch_max if code_max is None else max(code_max, batch_max)
        for row in codes.tolist():
            code = tuple(int(value) for value in row)
            unique_codes.add(code)
            if len(code_examples) < 8:
                code_examples.append(list(code))
    if total == 0:
        raise ValueError("cannot evaluate an empty E1 dataset")
    accuracy = correct / total
    return {
        "loss": loss_sum / total,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "number_correct": correct,
        "number_examples": total,
        "finite_state_diagnostics": {
            "min_integer_code": code_min,
            "max_integer_code": code_max,
            "allowed_code_range": [0, model.bottleneck.quantizer.levels - 1],
            "number_unique_final_state_codes": len(unique_codes),
            "state_code_examples": code_examples,
        },
    }


def _linear_schedule(optimizer, max_steps: int, warmup_steps: int):
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max_steps - step
        decay_steps = max(1, max_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _resolve_max_steps(config: Mapping[str, Any], batches_per_epoch: int) -> int:
    if config.get("max_steps") is not None:
        value = int(config["max_steps"])
    elif config.get("epochs") is not None:
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        value = int(config["epochs"]) * math.ceil(batches_per_epoch / accumulation)
    else:
        raise ValueError("E1 training requires max_steps or epochs")
    if value < 1:
        raise ValueError("E1 optimization steps must be positive")
    return value


def run_e1_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train and evaluate one fixed-capacity E1 configuration."""

    validate_e1_config(config)
    n = int(config["n"])
    d = int(config["state_dim"])
    p = 2
    seed = int(config.get("seed", 17))
    if min(n, d) < 1:
        raise ValueError("E1 n and state_dim must be positive")
    retained_bits = d * p
    target_ratio = float(config.get("target_ratio", retained_bits / n))
    actual_ratio = retained_bits / n

    # Seed before model loading so the added embedding row and finite modules are
    # reproducible.  The checkpoint parameters themselves come from raw GPT-2.
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    model, tokenizer, tokens = load_e1_model_and_tokenizer(config)
    model = model.to(device)

    samples = {
        "train": int(config.get("train_samples", 4096)),
        "dev": int(config.get("dev_samples", 1024)),
        "test": int(config.get("test_samples", 2048)),
    }
    datasets = {
        split: FrontierNLDataset(n, samples[split], seed, split, tokenizer)
        for split in SPLITS
    }
    collator = FrontierNLCollator(int(tokenizer.pad_token_id))
    batch_size = int(config.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    train_order_seed = derive_split_seed(n, seed, "train") ^ 0x5EED5EED
    train_generator = torch.Generator().manual_seed(train_order_seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            collate_fn=collator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "train_eval": DataLoader(
            datasets["train"], batch_size=eval_batch_size, shuffle=False,
            collate_fn=collator, num_workers=0,
        ),
        "dev": DataLoader(
            datasets["dev"], batch_size=eval_batch_size, shuffle=False,
            collate_fn=collator, num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=eval_batch_size, shuffle=False,
            collate_fn=collator, num_workers=0,
        ),
    }

    accumulation = int(config.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    max_steps = _resolve_max_steps(config, len(loaders["train"]))
    warmup_steps = int(config.get("warmup_steps", 0))
    if config.get("warmup_ratio") is not None:
        warmup_steps = round(float(config["warmup_ratio"]) * max_steps)
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("warmup must be in [0, max_steps)")
    learning_rate = float(config.get("learning_rate", 5e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = _linear_schedule(optimizer, max_steps, warmup_steps)
    use_bf16 = bool(config.get("bf16", False))
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this CUDA device does not support it")
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iterator: Iterable[Mapping[str, Tensor]] = iter(loaders["train"])
    loss_sum = 0.0
    microbatches = 0
    history: List[Dict[str, float | int]] = []
    log_every = max(1, int(config.get("log_every_steps", 25)))
    training_started = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(accumulation):
            try:
                host_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(loaders["train"])
                host_batch = next(train_iterator)
            batch = _to_device(host_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = _forward(model, batch, tokens)
                loss = output.loss
            (loss / accumulation).backward()
            observed = float(loss.detach().float().item())
            step_loss += observed / accumulation
            loss_sum += observed
            microbatches += 1
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step == max_steps or step % log_every == 0:
            elapsed = time.perf_counter() - training_started
            steps_per_second = step / elapsed if elapsed else 0.0
            progress = {
                "optimization_step": step,
                "train_loss": step_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_per_second,
                "estimated_remaining_seconds": (
                    (max_steps - step) / steps_per_second
                    if steps_per_second else None
                ),
            }
            history.append(progress)
            print(
                json.dumps(
                    {
                        "event": "e1_train_progress",
                        "n": n,
                        "d": d,
                        "seed": seed,
                        **progress,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    train = evaluate_e1(model, loaders["train_eval"], device, tokens)
    dev = evaluate_e1(model, loaders["dev"], device, tokens)
    test = evaluate_e1(model, loaders["test"], device, tokens)
    ledger = model.last_ledger.to_dict() if model.last_ledger else None
    if ledger is None:
        raise RuntimeError("E1 model did not produce a resource ledger")
    expected_ledger = {
        "persistent_bits": retained_bits,
        "transcript_length": 0,
        "input_reads": 1,
        "access_model": "sealed_prefix",
        "recurrent_updates": 0,
        "retains_latent_history": False,
    }
    for key, expected in expected_ledger.items():
        if ledger.get(key) != expected:
            raise RuntimeError(
                f"strict E1 ledger violation: {key}={ledger.get(key)!r}, "
                f"expected {expected!r}"
            )

    fano = fano_error_lower_bound(actual_ratio)
    theory_codec_error = max(0, n - retained_bits) / (2 * n)
    train_loss = loss_sum / microbatches
    return {
        "status": "complete",
        "experiment": "E1",
        "definition": (
            "GPT2-Raw + StrictFiniteState + Frontier Retrieval-NL + "
            "sealed prefix + answer-only + dp-sweep"
        ),
        "model_id": DEFAULT_MODEL_ID,
        "model_initialization": "AutoModelForCausalLM.from_pretrained",
        "tokenizer_initialization": "AutoTokenizer.from_pretrained",
        "pretraining_on_frontier_retrieval": False,
        "latent_boundary_token": LATENT_TOKEN,
        "template_version": "frontier-nl-fixed-v1",
        "answer_token_ids": tokens,
        "finite_state_config": {
            **dict(config["finite_state"]),
            "state_dim": d,
            "transcript_length": 0,
            "latent_slots": 1,
        },
        "n": n,
        "d": d,
        "p": p,
        "R": retained_bits,
        "target_ratio": target_ratio,
        "actual_ratio": actual_ratio,
        "R_over_n": actual_ratio,
        "seed": seed,
        "train_loss": train_loss,
        "train_accuracy": train["accuracy"],
        "train_error": train["error"],
        "dev_error": dev["error"],
        "test_error": test["error"],
        "accuracy": test["accuracy"],
        "error": test["error"],
        "number_correct": test["number_correct"],
        "number_examples": test["number_examples"],
        "train": train,
        "dev": dev,
        "test": test,
        "resource_ledger": ledger,
        "finite_state_diagnostics": test["finite_state_diagnostics"],
        "fano_error_lower_bound": fano,
        "empirical_minus_fano": test["error"] - fano,
        "comparisons": {
            "chance_error": 0.5,
            "test_minus_chance": test["error"] - 0.5,
            "e1_theory_deterministic_codec_error": theory_codec_error,
            "test_minus_e1_theory_codec": test["error"] - theory_codec_error,
            "fano_error_lower_bound": fano,
            "test_minus_fano": test["error"] - fano,
            "exact_retrieval_information_theoretically_possible": (
                retained_bits >= n
            ),
            "learnability_gap_when_possible": (
                test["error"] if retained_bits >= n else None
            ),
            "below_capacity_near_exact_leak_warning": (
                retained_bits < n and test["error"] < 0.01
            ),
        },
        "quantization_audit": {
            "exact_hard_finite_state_enforced": True,
            "straight_through_gradient_only": True,
            "forward_value_is_codebook_decoding": True,
            "bits_per_coordinate": p,
            "alphabet_size": 1 << p,
            "allowed_integer_codes": [0, (1 << p) - 1],
            "persistent_bits": retained_bits,
        },
        "sealed_boundary_audit": {
            "decoder_inputs": ["hard_quantized_state", "post_boundary_query"],
            "post_boundary_prefix_token_access": False,
            "post_boundary_prefix_hidden_state_access": False,
            "post_boundary_kv_cache_access": False,
            "prefix_rereads": 0,
            "past_key_values": False,
            "retained_latent_history": False,
            "transcript_length": 0,
            "recurrent_state_updates": 0,
        },
        "data": {split: datasets[split].metadata() for split in SPLITS},
        "training_budget": {
            "protocol": "fixed_optimization_steps",
            "optimization_steps": max_steps,
            "train_samples": samples["train"],
            "dev_samples": samples["dev"],
            "test_samples": samples["test"],
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "scheduler": "linear_decay",
            "max_grad_norm": max_grad_norm,
            "bf16": use_bf16,
            "device": str(device),
            "seed": seed,
            "train_order_seed": train_order_seed,
            "training_elapsed_seconds": time.perf_counter() - training_started,
        },
        "history": history,
    }


def is_complete_e1_result(
    value: Any, expected: Mapping[str, Any] | None = None
) -> bool:
    if not isinstance(value, dict) or value.get("status") != "complete":
        return False
    required = (
        "n", "d", "p", "R", "seed", "train_error", "dev_error",
        "test_error", "accuracy",
        "number_correct", "number_examples", "resource_ledger",
    )
    if any(key not in value for key in required):
        return False
    if value["R"] != value["d"] * value["p"]:
        return False
    ledger = value["resource_ledger"]
    if not isinstance(ledger, dict):
        return False
    strict_resources = {
        "persistent_bits": value["R"],
        "transcript_length": 0,
        "input_reads": 1,
        "access_model": "sealed_prefix",
        "recurrent_updates": 0,
        "retains_latent_history": False,
    }
    if any(
        ledger.get(key) != expected_value
        for key, expected_value in strict_resources.items()
    ):
        return False
    if expected is not None:
        for key in ("n", "d", "p", "R", "seed"):
            if key in expected and value.get(key) != expected[key]:
                return False
    return True


def result_filename(n: int, d: int, p: int, seed: int) -> str:
    return f"n{n:03d}_R{d * p:04d}_d{d:04d}_p{p}_seed{seed:04d}.json"
