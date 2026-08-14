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


def h2_inverse(value: float, *, tolerance: float = 1e-12) -> float:
    """Invert binary entropy on ``[0, 1/2]`` by stable bisection."""

    if not 0.0 <= value <= 1.0:
        raise ValueError("binary-entropy target must lie in [0, 1]")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 0.5
    low, high = 0.0, 0.5
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if binary_entropy(middle) < value:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def fano_error_lower_bound(capacity_ratio: float) -> float:
    """Return ``h2^-1(1-rho)`` for ``rho < 1``, otherwise zero."""

    if capacity_ratio < 0.0:
        raise ValueError("capacity ratio must be non-negative")
    if capacity_ratio >= 1.0:
        return 0.0
    return h2_inverse(1.0 - capacity_ratio)


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
