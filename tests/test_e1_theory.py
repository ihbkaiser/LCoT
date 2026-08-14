from __future__ import annotations

import itertools

import torch

from finite_cot.frontier_nl import FrontierNLDataset
from finite_cot.frontier_theory import (
    DeterministicFrontierCodec,
    generate_theory_split,
    run_e1_theory,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(text)]


def all_frontiers(n: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product((0, 1), repeat=n)), dtype=torch.uint8
    )


def test_exact_retrieval_for_every_frontier_and_query_when_R_at_least_n():
    n = 4
    codec = DeterministicFrontierCodec(n=n, state_dim=2, bits=2)
    frontiers = all_frontiers(n)
    expanded = frontiers.repeat_interleave(n, dim=0)
    queries = torch.arange(n).repeat(frontiers.shape[0])
    rows = torch.arange(expanded.shape[0])
    labels = expanded[rows, queries].to(torch.long)
    predictions, codes = codec(expanded, queries)
    assert torch.equal(predictions, labels)
    assert int(codes.min()) >= 0 and int(codes.max()) <= 3


def test_below_capacity_has_explicit_same_code_different_answer_collision():
    codec = DeterministicFrontierCodec(n=4, state_dim=1, bits=2)
    witness = codec.collision_witness()
    assert witness is not None
    assert witness["codes_identical"] is True
    assert witness["targets_differ"] is True
    assert witness["query"] == 2
    assert (1 << codec.persistent_bits) < (1 << codec.n)


def test_sealed_decoder_is_independent_of_prefix_after_encoding():
    codec = DeterministicFrontierCodec(n=8, state_dim=3, bits=2)
    prefix = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0]], dtype=torch.uint8)
    codes = codec.encode_prefix(prefix)
    query = torch.tensor([3])
    expected = codec.decode_query(codes, query)
    prefix.fill_(0)
    assert torch.equal(codec.decode_query(codes, query), expected)
    assert codec.decode_query(codes, query).item() == 1


def test_encoding_cannot_depend_on_post_boundary_query():
    codec = DeterministicFrontierCodec(n=8, state_dim=4, bits=2)
    prefix = torch.randint(0, 2, (1, 8), dtype=torch.uint8).repeat(2, 1)
    codes = codec.encode_prefix(prefix)
    assert torch.equal(codes[0], codes[1])
    assert codec.decode_query(codes, torch.tensor([0, 7])).shape == (2,)


def test_theory_and_learned_generators_use_identical_examples():
    tokenizer = TinyTokenizer()
    learned = FrontierNLDataset(16, 1000, 42, "test", tokenizer)
    frontiers, queries, labels, metadata = generate_theory_split(
        16, 1000, 42, "test"
    )
    assert torch.equal(frontiers, learned.frontiers)
    assert torch.equal(queries, learned.queries)
    assert torch.equal(labels, learned.labels)
    assert metadata["fingerprint_sha256"] == learned.fingerprint


def test_theory_result_has_strict_ledger_and_capacity_certificates():
    below = run_e1_theory(
        {"n": 16, "state_dim": 4, "bits": 2, "seed": 17, "test_samples": 4096}
    )
    at = run_e1_theory(
        {"n": 16, "state_dim": 8, "bits": 2, "seed": 17, "test_samples": 4096}
    )
    assert below["training_free"] is True
    assert below["optimization_steps"] == 0
    assert below["exact_retrieval_impossible_for_arbitrary_frontiers"] is True
    assert below["capacity_certificate"]["collision_witness"] is not None
    assert at["exact_retrieval_achievable"] is True
    assert at["error"] == 0.0
    assert at["capacity_certificate"]["collision_witness"] is None
    ledger = at["resource_ledger"]
    assert ledger["persistent_bits"] == 16
    assert ledger["access_model"] == "sealed_prefix"
    assert ledger["input_reads"] == 1
    assert ledger["recurrent_updates"] == 0
    assert ledger["transcript_length"] == 0
    assert ledger["retains_latent_history"] is False
