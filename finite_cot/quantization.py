"""Finite-alphabet latent-state quantizers.

The scale is global (fixed or learned) and is never estimated per example.  A
hard integer code is used in every forward pass; training uses a straight-
through gradient for the decoded value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class QuantizedTensor:
    """Decoded tensor together with its integer finite-alphabet code."""

    value: Tensor
    codes: Tensor


class FiniteScalarQuantizer(nn.Module):
    """Uniform scalar quantization with exactly ``2**bits`` symbols.

    ``clip_value`` is one global scalar shared by every item and coordinate.
    Making it learnable changes the global codebook during training but does not
    create an input-dependent side channel.
    """

    def __init__(
        self,
        bits: int,
        clip_value: float = 1.0,
        learnable_clip: bool = False,
    ) -> None:
        super().__init__()
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")
        self.bits = int(bits)
        self.levels = 1 << self.bits
        raw_clip = torch.tensor(float(clip_value)).log()
        if learnable_clip:
            self.log_clip = nn.Parameter(raw_clip)
        else:
            self.register_buffer("log_clip", raw_clip)

    @property
    def alphabet_size(self) -> int:
        return self.levels

    @property
    def clip(self) -> Tensor:
        return self.log_clip.exp().clamp_min(torch.finfo(self.log_clip.dtype).eps)

    def encode(self, x: Tensor) -> Tensor:
        """Return integer codes in ``[0, 2**bits - 1]``."""

        clip = self.clip.to(device=x.device, dtype=x.dtype)
        unit = ((x.clamp(-clip, clip) + clip) / (2 * clip)).clamp(0, 1)
        return torch.round(unit * (self.levels - 1)).to(torch.long)

    def decode(self, codes: Tensor, *, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Decode integer codes using the global codebook."""

        if codes.numel() and (codes.min() < 0 or codes.max() >= self.levels):
            raise ValueError("quantizer code is outside its finite alphabet")
        out_dtype = dtype or self.log_clip.dtype
        clip = self.clip.to(device=codes.device, dtype=out_dtype)
        unit = codes.to(out_dtype) / (self.levels - 1)
        return unit * (2 * clip) - clip

    def quantize(self, x: Tensor) -> QuantizedTensor:
        codes = self.encode(x)
        hard = self.decode(codes, dtype=x.dtype)
        # QAT: use the hard quantize/dequantize value in the forward pass and
        # an identity straight-through gradient in the model's floating dtype.
        value = x + (hard - x).detach() if self.training else hard
        return QuantizedTensor(value=value, codes=codes)

    def forward(self, x: Tensor) -> Tensor:
        return self.quantize(x).value


class FiniteStateBottleneck(nn.Module):
    """Project a hidden vector into a ``d``-coordinate finite state and back.

    The expanded hidden vector is a deterministic function of the integer code,
    so its per-example information capacity is still at most ``state_dim * bits``.
    """

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        bits: int,
        clip_value: float = 1.0,
        learnable_clip: bool = False,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.bits = int(bits)
        self.down = nn.Linear(hidden_size, state_dim)
        self.quantizer = FiniteScalarQuantizer(bits, clip_value, learnable_clip)
        self.up = nn.Linear(state_dim, hidden_size)

    @property
    def persistent_bits(self) -> int:
        return self.state_dim * self.bits

    def compress(self, hidden: Tensor) -> QuantizedTensor:
        return self.quantizer.quantize(self.down(hidden))

    def expand(self, state: Tensor) -> Tensor:
        return self.up(state)

    def forward(self, hidden: Tensor, *, return_codes: bool = False):
        quantized = self.compress(hidden)
        expanded = self.expand(quantized.value)
        if return_codes:
            return expanded, quantized.codes
        return expanded
