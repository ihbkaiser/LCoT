"""One-step Llama 3.2 finite-state LoRA smoke test.

This intentionally avoids the distributed trainer and ProsQA files. It verifies
that the real Hugging Face checkpoint can be loaded, resized for latent tokens,
wrapped with LoRA + StrictFiniteStateCoconut, and can complete
forward/backward/optimizer.step on one CUDA device.
"""

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `python scripts/smoke_llama32_3b.py` work from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finite_cot.rbs_adapter import StrictFiniteStateCoconut
from run import add_pretrained_lora, create_optimizer, trainable_parameter_counts


LATENT_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B",
        help="Hugging Face causal LM checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--model-bits", type=int, default=2)
    parser.add_argument(
        "--train-task-interfaces",
        action="store_true",
        help="also train the full token embedding/LM-head matrices (high VRAM)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="disable activation checkpointing for comparison/debugging",
    )
    return parser.parse_args()


def nonzero_grad_names(model, needle):
    names = []
    for name, parameter in model.named_parameters():
        if needle in name and parameter.grad is not None:
            if torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0:
                names.append(name)
    return names


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the 3B smoke test")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = add_pretrained_lora(
        model,
        train_task_interfaces=args.train_task_interfaces,
    )
    model = StrictFiniteStateCoconut(
        model,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id,
        finite_state={
            "state_dim": args.state_dim,
            "model_bits": args.model_bits,
            "clip_value": 1.0,
            "learnable_clip": False,
            "access_mode": "readonly_input",
        },
    ).to(device=device, dtype=dtype)
    model.train()

    # Use real tokenizer IDs but explicitly insert recurrent latent slots.
    prefix = tokenizer.encode(
        "A is connected to B. B is connected to C. [Q] A C [R] A",
        add_special_tokens=False,
    )
    answer = tokenizer.encode(" C", add_special_tokens=False)
    if not prefix or not answer:
        raise RuntimeError("unexpected empty tokenization")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    ids = prefix + [latent_id, latent_id] + answer
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(answer) :] = torch.tensor(answer, device=device)

    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite loss: {output.loss.item()}")
    output.loss.backward()

    lora_grads = nonzero_grad_names(model, "lora_")
    projection_grads = (
        nonzero_grad_names(model, "bottleneck.down")
        + nonzero_grad_names(model, "bottleneck.up")
    )
    if not lora_grads:
        raise RuntimeError("no non-zero LoRA gradients found")
    if not projection_grads:
        raise RuntimeError("no non-zero finite-state projection gradients found")

    optimizer.step()
    trainable, total = trainable_parameter_counts(model)
    print(f"SMOKE PASS: loss={output.loss.item():.6f}")
    print(f"trainable={trainable:,} / total={total:,} ({100*trainable/total:.4f}%)")
    print(f"LoRA gradient tensors: {len(lora_grads)}")
    print(f"finite projection gradient tensors: {len(projection_grads)}")
    if device.type == "cuda":
        gib = 1024**3
        print(f"peak CUDA allocated: {torch.cuda.max_memory_allocated(device)/gib:.2f} GiB")
        print(f"peak CUDA reserved:  {torch.cuda.max_memory_reserved(device)/gib:.2f} GiB")


if __name__ == "__main__":
    main()
