"""Rare-witness full aggregation and sampled-inspection comparators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .access import SampledInspectionAccess


def hypercube_codes(dimension: int, *, device=None, dtype=torch.float32) -> Tensor:
    """Return MSB-first binary identity codes mapped from ``0/1`` to ``-1/+1``."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    branches = 1 << dimension
    indices = torch.arange(branches, device=device, dtype=torch.long)
    shifts = torch.arange(dimension - 1, -1, -1, device=device, dtype=torch.long)
    bits = (indices.unsqueeze(-1) >> shifts) & 1
    return bits.to(dtype) * 2.0 - 1.0


def fixed_point_round(value: Tensor, fractional_bits: int) -> Tensor:
    """Quantize to a fixed-point grid using nearest, ties-to-even rounding."""

    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    scale = float(1 << fractional_bits)
    return torch.round(value * scale) / scale


def decode_hypercube(state: Tensor, *, signal_scale: float = 1.0) -> Tensor:
    """Decode MSB-first hypercube states, using ``dimension``'s branch count as NULL."""

    if state.ndim != 2 or state.shape[1] < 1:
        raise ValueError("state must have shape [batch, dimension]")
    if signal_scale <= 0.0:
        raise ValueError("signal_scale must be positive")
    dimension = state.shape[1]
    branches = 1 << dimension
    is_null = state.abs().amax(dim=-1) < (0.5 * signal_scale)
    signs = (state >= 0).to(torch.long)
    shifts = torch.arange(dimension - 1, -1, -1, device=state.device)
    weights = (1 << shifts).long()
    prediction = (signs * weights).sum(dim=-1)
    return torch.where(is_null, branches, prediction)


@dataclass
class AggregateOutput:
    state: Tensor
    prediction: Tensor
    scalar_additions: int
    marker_accesses: int


@dataclass
class ExhaustiveScanOutput:
    prediction: Tensor
    marker_accesses: int


class HypercubeAggregator:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.codes = hypercube_codes(dimension)

    @property
    def branches(self) -> int:
        return 1 << self.dimension

    def aggregate(
        self,
        markers: Tensor,
        *,
        normalized: bool = False,
        fractional_bits: int | None = None,
    ) -> AggregateOutput:
        if markers.ndim != 2 or markers.shape[1] != self.branches:
            raise ValueError("markers have the wrong branch dimension")
        codes = self.codes.to(device=markers.device, dtype=markers.dtype)
        state = markers @ codes
        if normalized:
            state = state / self.branches
        if fractional_bits is not None:
            state = fixed_point_round(state, fractional_bits)
        prediction = decode_hypercube(
            state, signal_scale=1.0 / self.branches if normalized else 1.0
        )
        return AggregateOutput(
            state=state,
            prediction=prediction,
            scalar_additions=self.branches * self.dimension,
            marker_accesses=self.branches,
        )


def exhaustive_scan(markers: Tensor) -> ExhaustiveScanOutput:
    """Read every marker and return its identity, or ``branches`` for NULL."""

    if markers.ndim != 2 or markers.shape[1] < 2:
        raise ValueError("markers must have shape [batch, branches >= 2]")
    present = markers != 0
    counts = present.sum(dim=-1)
    if bool((counts > 1).any()):
        raise ValueError("exhaustive scan expects at most one witness")
    prediction = present.to(torch.long).argmax(dim=-1)
    prediction = torch.where(counts == 0, markers.shape[1], prediction)
    return ExhaustiveScanOutput(
        prediction=prediction,
        marker_accesses=markers.shape[1],
    )


@dataclass
class SampledOutput:
    success: Tensor
    inspections: int


@dataclass
class SampledCurveOutput:
    budgets: tuple[int, ...]
    success: Tensor
    inspections: int


def sampled_inspection_prefixes(
    markers: Tensor,
    budgets: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
) -> SampledCurveOutput:
    """Evaluate sorted budgets as prefixes of one shared inspection stream."""

    if not budgets or any(budget < 0 for budget in budgets):
        raise ValueError("budgets must be a non-empty sequence of non-negative values")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be strictly increasing")
    access = SampledInspectionAccess(markers)
    success = torch.zeros(markers.shape[0], dtype=torch.bool, device=markers.device)
    snapshots = []
    budget_index = 0
    if budgets[0] == 0:
        snapshots.append(success.clone())
        budget_index = 1
    for inspection in range(1, budgets[-1] + 1):
        branches = torch.randint(
            0,
            markers.shape[1],
            (markers.shape[0],),
            generator=generator,
            device=markers.device,
        )
        success |= access.inspect(branches).to(torch.bool)
        if budget_index < len(budgets) and inspection == budgets[budget_index]:
            snapshots.append(success.clone())
            budget_index += 1
    return SampledCurveOutput(
        budgets=budgets,
        success=torch.stack(snapshots),
        inspections=access.total_inspections,
    )


def sampled_inspection(
    markers: Tensor,
    inspections: int,
    *,
    generator: torch.Generator | None = None,
) -> SampledOutput:
    curve = sampled_inspection_prefixes(
        markers, (inspections,), generator=generator
    )
    return SampledOutput(success=curve.success[0], inspections=curve.inspections)
