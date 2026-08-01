"""Rare-witness full aggregation and sampled-inspection comparators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .access import SampledInspectionAccess


def hypercube_codes(dimension: int, *, device=None, dtype=torch.float32) -> Tensor:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    branches = 1 << dimension
    indices = torch.arange(branches, device=device, dtype=torch.long)
    shifts = torch.arange(dimension, device=device, dtype=torch.long)
    bits = (indices.unsqueeze(-1) >> shifts) & 1
    return bits.to(dtype) * 2.0 - 1.0


def fixed_point_round(value: Tensor, fractional_bits: int) -> Tensor:
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    scale = float(1 << fractional_bits)
    return torch.round(value * scale) / scale


@dataclass
class AggregateOutput:
    state: Tensor
    prediction: Tensor
    scalar_additions: int
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
        is_null = state.abs().amax(dim=-1) < (0.5 / self.branches if normalized else 0.5)
        signs = (state >= 0).to(torch.long)
        weights = (1 << torch.arange(self.dimension, device=state.device)).long()
        prediction = (signs * weights).sum(dim=-1)
        prediction = torch.where(is_null, self.branches, prediction)
        return AggregateOutput(
            state=state,
            prediction=prediction,
            scalar_additions=self.branches * self.dimension,
            marker_accesses=self.branches,
        )


@dataclass
class SampledOutput:
    success: Tensor
    inspections: int


def sampled_inspection(
    markers: Tensor,
    inspections: int,
    *,
    generator: torch.Generator | None = None,
) -> SampledOutput:
    if inspections < 0:
        raise ValueError("inspections must be non-negative")
    access = SampledInspectionAccess(markers)
    success = torch.zeros(markers.shape[0], dtype=torch.bool, device=markers.device)
    for _ in range(inspections):
        branches = torch.randint(
            0,
            markers.shape[1],
            (markers.shape[0],),
            generator=generator,
            device=markers.device,
        )
        success |= access.inspect(branches).to(torch.bool)
    return SampledOutput(success=success, inspections=access.total_inspections)
