"""Auditable access interfaces used by the lower-bound experiments."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


class AccessViolation(RuntimeError):
    pass


class SealedPrefixAccess:
    """A read-once prefix that becomes unavailable after summary creation."""

    def __init__(self, prefix: Tensor) -> None:
        self._prefix: Optional[Tensor] = prefix
        self._sealed = False
        self.read_count = 0

    @property
    def sealed(self) -> bool:
        return self._sealed

    def encode_and_seal(self, encoder: Callable[[Tensor], Tensor]) -> Tensor:
        if self._sealed or self._prefix is None:
            raise AccessViolation("prefix has already been sealed")
        self.read_count += 1
        summary = encoder(self._prefix.clone())
        self._prefix = None
        self._sealed = True
        return summary

    def read(self) -> Tensor:
        if self._sealed or self._prefix is None:
            raise AccessViolation("sealed prefix cannot be reread")
        self.read_count += 1
        return self._prefix.clone()


class ReadOnlyInputAccess:
    """Read-only full input for BFS-style recurrent updates."""

    def __init__(self, value: Tensor) -> None:
        self._value = value
        self.read_count = 0

    def read(self) -> Tensor:
        self.read_count += 1
        return self._value.clone()


class LocalOracleAccess:
    """Batched function oracle with at most one query per update."""

    def __init__(self, function_table: Tensor) -> None:
        if function_table.ndim != 2:
            raise ValueError("function_table must have shape [batch, nodes]")
        self._table = function_table.to(torch.long)
        self._active = False
        self.num_queries_this_update = 0
        self.total_queries = 0
        self.completed_updates = 0

    @property
    def batch_size(self) -> int:
        return self._table.shape[0]

    def begin_update(self) -> None:
        if self._active:
            raise AccessViolation("previous oracle update was not closed")
        self._active = True
        self.num_queries_this_update = 0

    def query(self, vertex: Tensor) -> Tensor:
        if not self._active:
            raise AccessViolation("call begin_update before querying the oracle")
        self.num_queries_this_update += 1
        if self.num_queries_this_update > 1:
            raise AccessViolation("local oracle permits at most one query per update")
        vertex = vertex.to(device=self._table.device, dtype=torch.long).view(-1)
        if vertex.shape[0] != self.batch_size:
            raise ValueError("one query vertex is required for each batch item")
        if vertex.numel() and (vertex.min() < 0 or vertex.max() >= self._table.shape[1]):
            raise AccessViolation("oracle query is outside the node range")
        self.total_queries += 1
        rows = torch.arange(self.batch_size, device=self._table.device)
        return self._table[rows, vertex]

    def end_update(self, *, require_query: bool = True) -> None:
        if not self._active:
            raise AccessViolation("no oracle update is active")
        if require_query and self.num_queries_this_update != 1:
            raise AccessViolation("each pointer update must perform exactly one query")
        self._active = False
        self.completed_updates += 1


class SampledInspectionAccess:
    """Expose marker values only at explicitly inspected branches."""

    def __init__(self, markers: Tensor) -> None:
        if markers.ndim != 2:
            raise ValueError("markers must have shape [batch, branches]")
        self._markers = markers
        self.total_inspections = 0

    @property
    def num_branches(self) -> int:
        return self._markers.shape[1]

    def inspect(self, branch: Tensor) -> Tensor:
        branch = branch.to(device=self._markers.device, dtype=torch.long).view(-1)
        if branch.shape[0] != self._markers.shape[0]:
            raise ValueError("one branch index is required for each batch item")
        if branch.numel() and (branch.min() < 0 or branch.max() >= self.num_branches):
            raise AccessViolation("inspection is outside the branch range")
        rows = torch.arange(self._markers.shape[0], device=self._markers.device)
        self.total_inspections += 1
        return self._markers[rows, branch]
