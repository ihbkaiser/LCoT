"""Synthetic datasets aligned with the paper's access interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset


class FrontierRetrievalDataset(Dataset):
    """Uniform random frontier followed by an independent membership query."""

    def __init__(self, n: int, num_samples: int, seed: int) -> None:
        if n < 1 or num_samples < 1:
            raise ValueError("n and num_samples must be positive")
        generator = torch.Generator().manual_seed(seed)
        self.frontiers = torch.randint(
            0, 2, (num_samples, n), generator=generator, dtype=torch.float32
        )
        self.queries = torch.randint(
            0, n, (num_samples,), generator=generator, dtype=torch.long
        )
        rows = torch.arange(num_samples)
        self.labels = self.frontiers[rows, self.queries].to(torch.long)

    def __len__(self) -> int:
        return self.frontiers.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "frontier": self.frontiers[index],
            "query": self.queries[index],
            "label": self.labels[index],
        }


class PointerChaseDataset(Dataset):
    """Random functions with exact paths cached for supervision and auditing."""

    def __init__(
        self,
        nodes: int,
        depth: int,
        num_samples: int,
        seed: int,
    ) -> None:
        if nodes < 2 or depth < 0 or num_samples < 1:
            raise ValueError("invalid pointer dataset parameters")
        generator = torch.Generator().manual_seed(seed)
        self.functions = torch.randint(
            0, nodes, (num_samples, nodes), generator=generator, dtype=torch.long
        )
        self.sources = torch.randint(
            0, nodes, (num_samples,), generator=generator, dtype=torch.long
        )
        path = [self.sources]
        rows = torch.arange(num_samples)
        current = self.sources
        for _ in range(depth):
            current = self.functions[rows, current]
            path.append(current)
        self.paths = torch.stack(path, dim=1)
        self.targets = self.paths[:, -1]
        self.nodes = nodes
        self.depth = depth

    def __len__(self) -> int:
        return self.functions.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "function": self.functions[index],
            "source": self.sources[index],
            "path": self.paths[index],
            "target": self.targets[index],
        }


@dataclass
class RareWitnessBatch:
    markers: Tensor
    target: Tensor


def make_rare_witness_batch(
    branches: int,
    batch_size: int,
    seed: int,
    null_probability: float = 0.0,
) -> RareWitnessBatch:
    """Create at-most-one-witness marker vectors.

    ``target == branches`` denotes NULL; otherwise it is the marked index.
    """

    if branches < 2 or batch_size < 1:
        raise ValueError("invalid rare-witness parameters")
    if not 0.0 <= null_probability <= 1.0:
        raise ValueError("null_probability must lie in [0, 1]")
    generator = torch.Generator().manual_seed(seed)
    target = torch.randint(0, branches, (batch_size,), generator=generator)
    is_null = torch.rand(batch_size, generator=generator) < null_probability
    markers = torch.zeros(batch_size, branches)
    rows = torch.arange(batch_size)[~is_null]
    markers[rows, target[~is_null]] = 1.0
    target = target.clone()
    target[is_null] = branches
    return RareWitnessBatch(markers=markers, target=target)
