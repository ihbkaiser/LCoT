"""Training-free E1-Theory sanity check for sealed frontier retrieval.

The deterministic codec stores the first ``min(R, n)`` frontier bits in exactly
``d`` base-``2**p`` coordinates.  Queries for retained bits are decoded exactly;
queries for unretained bits receive the fixed guess zero.  This construction is
exact when ``R = d*p >= n``.  When ``R < n``, the state alphabet has fewer than
``2**n`` elements, and an explicit same-code/different-answer pair certifies that
arbitrary-frontier exact retrieval is impossible.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from .frontier_nl import derive_split_seed
from .resources import ResourceLedger
from .theory import fano_error_lower_bound


class DeterministicFrontierCodec(nn.Module):
    """Bit-pack a sealed frontier into a hard finite state without training."""

    def __init__(self, n: int, state_dim: int, bits: int = 2) -> None:
        super().__init__()
        if min(n, state_dim, bits) < 1:
            raise ValueError("n, state_dim, and bits must be positive")
        self.n = int(n)
        self.state_dim = int(state_dim)
        self.bits = int(bits)
        self.persistent_bits = self.state_dim * self.bits
        self.levels = 1 << self.bits
        self.retained_frontier_bits = min(self.n, self.persistent_bits)
        self.ledger = ResourceLedger(
            state_dim=self.state_dim,
            bits_per_coordinate=self.bits,
            recurrent_updates=0,
            transcript_length=0,
            access_model="sealed_prefix",
            input_reads=1,
            retains_latent_history=False,
            notes=(
                "training-free bit packing; decoder receives only integer codes "
                "and the post-boundary query"
            ),
        )

    def encode_prefix(self, frontier: Tensor) -> Tensor:
        """Read the frontier once and return codes in ``[0, 2**p - 1]``."""

        if frontier.ndim != 2 or frontier.shape[1] != self.n:
            raise ValueError(f"frontier must have shape [batch, {self.n}]")
        if frontier.numel() and not torch.all((frontier == 0) | (frontier == 1)):
            raise ValueError("frontier values must be binary")
        batch = frontier.shape[0]
        codes = torch.zeros(
            batch, self.state_dim, dtype=torch.long, device=frontier.device
        )
        retained = frontier[:, : self.retained_frontier_bits].to(torch.long)
        for bit_offset in range(self.bits):
            selected = retained[:, bit_offset :: self.bits]
            if selected.shape[1]:
                codes[:, : selected.shape[1]] |= selected << bit_offset
        return codes

    def decode_query(self, codes: Tensor, query: Tensor) -> Tensor:
        """Decode from only the finite code and an independent query index."""

        if codes.ndim != 2 or codes.shape[1] != self.state_dim:
            raise ValueError(
                f"codes must have shape [batch, {self.state_dim}]"
            )
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.levels):
            raise ValueError("state code lies outside the finite alphabet")
        if query.ndim != 1 or query.shape[0] != codes.shape[0]:
            raise ValueError("query must have shape [batch]")
        if query.numel() and (query.min() < 0 or query.max() >= self.n):
            raise ValueError("query index is outside the frontier")

        answer = torch.zeros_like(query, dtype=torch.long)
        retained = query < self.retained_frontier_bits
        if retained.any():
            retained_query = query[retained]
            coordinate = torch.div(
                retained_query, self.bits, rounding_mode="floor"
            )
            offset = retained_query.remainder(self.bits)
            answer[retained] = (
                codes[retained, coordinate] >> offset
            ).bitwise_and(1)
        return answer

    def forward(self, frontier: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode_prefix(frontier)
        return self.decode_query(codes, query), codes

    def collision_witness(self) -> Dict[str, Any] | None:
        """Return a same-code/different-answer witness exactly when ``R < n``."""

        if self.persistent_bits >= self.n:
            return None
        query = self.persistent_bits
        first = torch.zeros(1, self.n, dtype=torch.uint8)
        second = first.clone()
        second[0, query] = 1
        first_code = self.encode_prefix(first)
        second_code = self.encode_prefix(second)
        if not torch.equal(first_code, second_code):
            raise AssertionError("invalid collision certificate")
        return {
            "query": query,
            "shared_state_code": first_code[0].tolist(),
            "frontier_a_target": 0,
            "frontier_b_target": 1,
            "codes_identical": True,
            "targets_differ": True,
        }


def generate_theory_split(
    n: int, num_samples: int, seed: int, split: str = "test"
) -> tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Generate the same uniform ``S,V,Y=S[V]`` law used by E1-Learned."""

    if min(n, num_samples) < 1:
        raise ValueError("n and num_samples must be positive")
    split_seed = derive_split_seed(n, seed, split)
    generator = torch.Generator().manual_seed(split_seed)
    frontiers = torch.randint(
        0, 2, (num_samples, n), generator=generator, dtype=torch.uint8
    )
    queries = torch.randint(
        0, n, (num_samples,), generator=generator, dtype=torch.long
    )
    rows = torch.arange(num_samples)
    labels = frontiers[rows, queries].to(torch.long)
    fingerprint = hashlib.sha256()
    fingerprint.update(frontiers.numpy().tobytes())
    fingerprint.update(queries.numpy().tobytes())
    fingerprint.update(labels.numpy().tobytes())
    metadata = {
        "n": n,
        "num_samples": num_samples,
        "experiment_seed": seed,
        "split": split,
        "split_seed": split_seed,
        "seed_derivation": "sha256(e1-frontier-nl-v1|n|seed|split)",
        "fingerprint_sha256": fingerprint.hexdigest(),
        "distribution": "S uniform; V independent uniform; Y=S[V]",
        "nl_template": "frontier-nl-fixed-v1",
    }
    return frontiers, queries, labels, metadata


@torch.no_grad()
def run_e1_theory(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one deterministic capacity point and return its proof diagnostics."""

    n = int(config["n"])
    d = int(config["state_dim"])
    p = int(config.get("bits", 2))
    seed = int(config.get("seed", 17))
    num_examples = int(config.get("test_samples", 100_000))
    if p != 2:
        raise ValueError("E1-Theory fixes p=2 to match E1-Learned")
    codec = DeterministicFrontierCodec(n=n, state_dim=d, bits=p)
    frontiers, queries, labels, data = generate_theory_split(
        n, num_examples, seed, "test"
    )
    predictions, codes = codec(frontiers, queries)
    correct = int((predictions == labels).sum())
    accuracy = correct / num_examples
    error = 1.0 - accuracy
    retained_bits = d * p
    ratio = retained_bits / n
    target_ratio = float(config.get("target_ratio", ratio))
    expected_error = max(0, n - retained_bits) / (2 * n)
    fano = fano_error_lower_bound(ratio)
    collision = codec.collision_witness()

    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= 1 << p):
        raise AssertionError("hard code escaped its finite alphabet")
    if retained_bits >= n and correct != num_examples:
        raise AssertionError("constructive exact codec failed above capacity")
    if retained_bits < n and collision is None:
        raise AssertionError("missing below-capacity collision certificate")

    unique_codes = torch.unique(codes, dim=0)
    code_examples: List[List[int]] = codes[:8].tolist()
    ledger = codec.ledger.to_dict()
    return {
        "status": "complete",
        "experiment": "E1-Theory",
        "training_free": True,
        "optimization_steps": 0,
        "model": "deterministic_prefix_bit_packing",
        "task": "Frontier Retrieval-NL (symbolic fixed-template frontier)",
        "access_protocol": "prefix -> hard finite state -> query -> answer",
        "n": n,
        "d": d,
        "p": p,
        "R": retained_bits,
        "target_ratio": target_ratio,
        "actual_ratio": ratio,
        "R_over_n": ratio,
        "seed": seed,
        "accuracy": accuracy,
        "error": error,
        "number_correct": correct,
        "number_examples": num_examples,
        "expected_error_for_codec": expected_error,
        "empirical_minus_expected": error - expected_error,
        "fano_error_lower_bound": fano,
        "empirical_minus_fano": error - fano,
        "exact_retrieval_achievable": retained_bits >= n,
        "exact_retrieval_impossible_for_arbitrary_frontiers": retained_bits < n,
        "capacity_certificate": {
            "number_frontiers": 1 << n,
            "number_finite_states": 1 << retained_bits,
            "state_count_at_least_frontier_count": retained_bits >= n,
            "pigeonhole_impossibility": retained_bits < n,
            "collision_witness": collision,
        },
        "resource_ledger": ledger,
        "finite_state_diagnostics": {
            "min_integer_code": int(codes.min()) if codes.numel() else None,
            "max_integer_code": int(codes.max()) if codes.numel() else None,
            "allowed_code_range": [0, (1 << p) - 1],
            "number_unique_final_state_codes": int(unique_codes.shape[0]),
            "state_code_examples": code_examples,
        },
        "data": data,
    }


def is_complete_theory_result(
    value: Any, expected: Dict[str, Any] | None = None
) -> bool:
    if not isinstance(value, dict) or value.get("status") != "complete":
        return False
    if value.get("experiment") != "E1-Theory" or not value.get("training_free"):
        return False
    required = ("n", "d", "p", "R", "seed", "error", "resource_ledger")
    if any(key not in value for key in required):
        return False
    if value["R"] != value["d"] * value["p"]:
        return False
    ledger = value["resource_ledger"]
    if not isinstance(ledger, dict):
        return False
    if (
        ledger.get("persistent_bits") != value["R"]
        or ledger.get("access_model") != "sealed_prefix"
        or ledger.get("input_reads") != 1
        or ledger.get("recurrent_updates") != 0
        or ledger.get("transcript_length") != 0
        or ledger.get("retains_latent_history") is not False
    ):
        return False
    if expected:
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                return False
    return True


def theory_result_filename(n: int, d: int, p: int, seed: int) -> str:
    return f"n{n:03d}_R{d * p:04d}_d{d:04d}_p{p}_seed{seed:04d}.json"
