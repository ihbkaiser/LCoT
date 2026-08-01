"""Small, dependency-light training loops for the synthetic learned controls."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import FrontierRetrievalDataset, PointerChaseDataset
from .models import (
    FixedHybridFrontierCodec,
    HybridFrontierModel,
    LearnedPointerMachine,
    PrefixRereadControl,
)
from .resources import ResourceLedger
from .theory import approximate_frontier_bits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


@torch.no_grad()
def evaluate_frontier(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    model.eval()
    correct = total = 0
    code_examples = []
    ledger = None
    for batch in loader:
        frontier = batch["frontier"].to(device)
        query = batch["query"].to(device)
        label = batch["label"].to(device)
        output = model(frontier, query)
        correct += (output.logits.argmax(dim=-1) == label).sum().item()
        total += label.numel()
        ledger = output.ledger
        if len(code_examples) < 8:
            for state_code, transcript_code in zip(
                output.state_codes.detach().cpu(),
                output.transcript_codes.detach().cpu(),
            ):
                code_examples.append(
                    {
                        "state": state_code.tolist(),
                        "transcript": transcript_code.tolist(),
                    }
                )
                if len(code_examples) == 8:
                    break
    accuracy = correct / total
    error = 1.0 - accuracy
    theory_bits = approximate_frontier_bits(loader.dataset.frontiers.shape[1], error) if error < 0.5 else 0.0
    return {
        "accuracy": accuracy,
        "error": error,
        "correct": correct,
        "total": total,
        "frontier_information_lower_bound_bits": theory_bits,
        "ledger": ledger.to_dict() if ledger else None,
        "hard_code_examples": code_examples,
    }


def run_frontier_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config.get("seed", 17))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    n = int(config["n"])
    train_data = FrontierRetrievalDataset(
        n, int(config.get("train_samples", 8192)), seed
    )
    dev_data = FrontierRetrievalDataset(
        n, int(config.get("dev_samples", 2048)), seed + 1
    )
    batch_size = int(config.get("batch_size", 128))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    kind = config.get("model", "hybrid")
    if kind in {"prefix_reread", "fixed_hybrid"}:
        model: nn.Module
        if kind == "prefix_reread":
            model = PrefixRereadControl()
        else:
            model = FixedHybridFrontierCodec(
                n=n,
                state_dim=int(config.get("state_dim", 8)),
                bits=int(config.get("bits", 1)),
                transcript_length=int(config.get("transcript_length", 0)),
                transcript_vocab_size=int(config.get("transcript_vocab_size", 2)),
            )
        model = model.to(device)
        epochs_ran = 0
        history = []
    else:
        model = HybridFrontierModel(
            n=n,
            state_dim=int(config.get("state_dim", 8)),
            bits=int(config.get("bits", 1)),
            transcript_length=int(config.get("transcript_length", 0)),
            transcript_vocab_size=int(config.get("transcript_vocab_size", 2)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            transcript_embedding_dim=int(
                config.get("transcript_embedding_dim", 16)
            ),
            quantized=kind != "unquantized",
        ).to(device)
        optimizer = _optimizer(
            model,
            float(config.get("learning_rate", 3e-3)),
            float(config.get("weight_decay", 0.0)),
        )
        history = []
        epochs_ran = int(config.get("epochs", 20))
        train_all_queries = bool(config.get("train_all_queries", True))
        for epoch in range(epochs_ran):
            model.train()
            loss_sum = count = 0
            for batch in train_loader:
                frontier = batch["frontier"].to(device)
                if train_all_queries:
                    output = model.forward_all_queries(frontier)
                    label = frontier.to(torch.long)
                    loss = F.cross_entropy(output.logits.flatten(0, 1), label.flatten())
                else:
                    query = batch["query"].to(device)
                    label = batch["label"].to(device)
                    output = model(frontier, query)
                    loss = F.cross_entropy(output.logits, label)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * label.numel()
                count += label.numel()
            history.append({"epoch": epoch + 1, "train_loss": loss_sum / count})

    metrics = evaluate_frontier(model, dev_loader, device)
    return {
        "task": "frontier_retrieval",
        "model": kind,
        "seed": seed,
        "device": str(device),
        "n": n,
        "epochs_ran": epochs_ran,
        "history": history,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_pointer(
    model: LearnedPointerMachine,
    loader: DataLoader,
    device: torch.device,
    allocated_updates: int,
    target_depth: int,
) -> Dict[str, Any]:
    model.eval()
    executed_updates = min(allocated_updates, target_depth)
    correct = total = total_queries = 0
    code_examples = []
    for batch in loader:
        function = batch["function"].to(device)
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        output = model(function, source, executed_updates, teacher_forcing=False)
        prediction = output.logits.argmax(dim=-1)
        correct += (prediction == target).sum().item()
        total += target.numel()
        total_queries += output.total_queries
        if len(code_examples) < 4:
            trace = torch.stack(output.state_codes, dim=1).detach().cpu()
            code_examples.extend(trace[: 4 - len(code_examples)].tolist())
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "allocated_updates": allocated_updates,
        "executed_updates": executed_updates,
        "batched_oracle_calls": total_queries,
        "hard_state_code_traces": code_examples,
    }


def run_pointer_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config.get("seed", 17))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    nodes = int(config["nodes"])
    depth = int(config["depth"])
    allocated_updates = int(config.get("updates", depth))
    executed_updates = min(allocated_updates, depth)
    train_data = PointerChaseDataset(
        nodes, depth, int(config.get("train_samples", 10000)), seed
    )
    dev_data = PointerChaseDataset(
        nodes, depth, int(config.get("dev_samples", 2000)), seed + 1
    )
    batch_size = int(config.get("batch_size", 128))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    model = LearnedPointerMachine(
        max_nodes=nodes,
        state_dim=int(config.get("state_dim", math.ceil(math.log2(nodes)))),
        bits=int(config.get("bits", 1)),
        hidden_dim=int(config.get("hidden_dim", 64)),
    ).to(device)
    optimizer = _optimizer(
        model,
        float(config.get("learning_rate", 3e-3)),
        float(config.get("weight_decay", 0.0)),
    )
    history = []
    for epoch in range(int(config.get("epochs", 20))):
        model.train()
        loss_sum = count = 0
        for batch in train_loader:
            function = batch["function"].to(device)
            source = batch["source"].to(device)
            path = batch["path"].to(device)
            output = model(
                function,
                source,
                executed_updates,
                path=path,
                teacher_forcing=True,
            )
            if not output.step_logits:
                raise ValueError("learned pointer training requires at least one update")
            losses = [
                F.cross_entropy(logits, path[:, step + 1])
                for step, logits in enumerate(output.step_logits)
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * source.numel()
            count += source.numel()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count})

    metrics = evaluate_pointer(
        model, dev_loader, device, allocated_updates, depth
    )
    ledger = ResourceLedger(
        state_dim=model.state_dim,
        bits_per_coordinate=model.bits,
        recurrent_updates=executed_updates,
        access_model="local_oracle",
        input_reads=executed_updates,
        notes="one batched hard query per recurrent update",
    )
    metrics["ledger"] = ledger.to_dict()
    return {
        "task": "pointer_chase",
        "seed": seed,
        "device": str(device),
        "nodes": nodes,
        "depth": depth,
        "history": history,
        "metrics": metrics,
    }
