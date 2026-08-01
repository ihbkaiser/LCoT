"""Machine-readable resource accounting for every experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional


@dataclass
class ResourceLedger:
    state_dim: int
    bits_per_coordinate: int
    recurrent_updates: int = 0
    transcript_length: int = 0
    transcript_vocab_size: int = 2
    access_model: str = "sealed_prefix"
    fractional_bits: Optional[int] = None
    inspections: int = 0
    input_reads: int = 0
    scalar_additions: int = 0
    retains_latent_history: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.state_dim < 0 or self.bits_per_coordinate < 0:
            raise ValueError("state resources must be non-negative")
        if self.transcript_length < 0 or self.transcript_vocab_size < 2:
            raise ValueError("invalid transcript resources")
        if min(self.recurrent_updates, self.inspections, self.input_reads) < 0:
            raise ValueError("resource counts must be non-negative")

    @property
    def state_bits(self) -> int:
        return self.state_dim * self.bits_per_coordinate

    @property
    def transcript_bits(self) -> float:
        return self.transcript_length * math.log2(self.transcript_vocab_size)

    @property
    def retained_summary_bits(self) -> float:
        return self.state_bits + self.transcript_bits

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.update(
            state_bits=self.state_bits,
            transcript_bits=self.transcript_bits,
            retained_summary_bits=self.retained_summary_bits,
        )
        return result
