"""Reference formulas used by experiment reports and tests."""

from __future__ import annotations

import math


def binary_entropy(error: float) -> float:
    if not 0.0 <= error <= 1.0:
        raise ValueError("error must lie in [0, 1]")
    if error in (0.0, 1.0):
        return 0.0
    return -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)


def approximate_frontier_bits(n: int, error: float) -> float:
    """Return ``n * (1 - h2(error))`` for error below one half."""

    if not 0.0 <= error < 0.5:
        raise ValueError("frontier lower bound expects error in [0, 0.5)")
    return n * (1.0 - binary_entropy(error))


def transcript_bits(length: int, vocab_size: int, variable_length: bool = False) -> float:
    if length < 0 or vocab_size < 2:
        raise ValueError("invalid transcript parameters")
    if not variable_length:
        return length * math.log2(vocab_size)
    count = sum(vocab_size**ell for ell in range(length + 1))
    return math.log2(count)


def serialized_state_tokens(
    state_dim: int, bits: int, steps: int, alphabet_size: int
) -> int:
    if min(state_dim, bits, steps) < 0 or alphabet_size < 2:
        raise ValueError("invalid serialization parameters")
    per_state = math.ceil(state_dim * bits / math.log2(alphabet_size))
    return (steps + 1) * per_state


def sampling_success(mass: float, inspections: int) -> float:
    if not 0.0 < mass < 1.0 or inspections < 0:
        raise ValueError("invalid sampling parameters")
    return 1.0 - (1.0 - mass) ** inspections


def required_inspections(mass: float, delta: float) -> int:
    if not 0.0 < mass < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("mass and delta must lie in (0, 1)")
    return math.ceil(math.log(1.0 / delta) / -math.log(1.0 - mass))


def pointer_trace_tokens(nodes: int, depth: int, alphabet_size: int = 2) -> int:
    if nodes < 1 or depth < 0 or alphabet_size < 2:
        raise ValueError("invalid pointer trace parameters")
    width = math.ceil(math.log(nodes, alphabet_size)) if nodes > 1 else 0
    return depth * width
