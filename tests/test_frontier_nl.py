from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from finite_cot.frontier_nl import (
    FrontierNLDataset,
    answer_token_ids,
    derive_split_seed,
    render_example,
)
from finite_cot.quantization import FiniteStateBottleneck
from finite_cot.rbs_adapter import StrictFiniteStateCoconut
from finite_cot.theory import binary_entropy, fano_error_lower_bound, h2_inverse


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == " no":
            return [11]
        if text == " yes":
            return [12]
        # Dataset distribution tests do not need linguistic token fidelity.
        return [1 + (len(text) % 31)]


def tiny_strict_model(state_dim=4, bits=2):
    torch.manual_seed(7)
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
    )
    return StrictFiniteStateCoconut(
        base_causallm=base,
        latent_token_id=3,
        start_latent_id=3,
        end_latent_id=3,
        eos_token_id=2,
        finite_state={
            "state_dim": state_dim,
            "bits_per_coordinate": bits,
            "clip_value": 1.0,
            "learnable_clip": False,
            "access_mode": "sealed_prefix",
        },
    )


def inputs():
    prefix = torch.tensor([[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]])
    prefix_mask = torch.ones_like(prefix)
    query = torch.tensor([[20, 21, 22, 23], [24, 25, 26, 27]])
    query_mask = torch.ones_like(query)
    return prefix, prefix_mask, query, query_mask


def test_hard_finite_alphabet_and_expansion_is_code_determined():
    bottleneck = FiniteStateBottleneck(16, state_dim=7, bits=2)
    bottleneck.train()
    quantized = bottleneck.compress(torch.randn(32, 16))
    assert int(quantized.codes.min()) >= 0
    assert int(quantized.codes.max()) <= 3
    decoded = bottleneck.quantizer.decode(quantized.codes, dtype=quantized.value.dtype)
    # STE affects only backward: the forward value is exactly the hard decoding.
    torch.testing.assert_close(quantized.value, decoded)
    torch.testing.assert_close(
        bottleneck.expand(quantized.value), bottleneck.expand(decoded)
    )


def test_sealed_prefix_state_is_independent_of_post_boundary_query():
    model = tiny_strict_model().eval()
    prefix = torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]])
    prefix_mask = torch.ones_like(prefix)
    query = torch.tensor([[20, 21, 22], [30, 31, 32]])
    query_mask = torch.ones_like(query)
    output = model.forward_answer_only(
        prefix, prefix_mask, query, query_mask, 11, 12, torch.tensor([0, 1])
    )
    assert torch.equal(output.state_codes[0], output.state_codes[1])


def test_target_answer_cannot_change_prediction_logits():
    model = tiny_strict_model().eval()
    prefix, prefix_mask, query, query_mask = inputs()
    first = model.forward_answer_only(
        prefix, prefix_mask, query, query_mask, 11, 12, torch.zeros(2, dtype=torch.long)
    )
    second = model.forward_answer_only(
        prefix, prefix_mask, query, query_mask, 11, 12, torch.ones(2, dtype=torch.long)
    )
    torch.testing.assert_close(first.logits, second.logits)
    torch.testing.assert_close(first.binary_logits, second.binary_logits)


def test_prefix_is_inaccessible_after_hard_code_is_created():
    model = tiny_strict_model().eval()
    prefix, prefix_mask, query, query_mask = inputs()
    state = model.encode_prefix(prefix, prefix_mask)
    baseline = model.decode_codes_query(state.codes, query, query_mask)

    # Destroy the caller's prefix. Decoding is unchanged because its only input
    # from the prefix side is the integer state code.
    prefix.fill_(63)
    prefix_mask.zero_()
    after_mutation = model.decode_codes_query(state.codes, query, query_mask)
    torch.testing.assert_close(baseline, after_mutation)
    assert not hasattr(model, "past_key_values")
    assert not hasattr(model, "prefix_hidden_states")
    assert model.last_state_codes


def test_e1_capacity_ledger_has_no_hidden_resources():
    model = tiny_strict_model(state_dim=5, bits=2).eval()
    prefix, prefix_mask, query, query_mask = inputs()
    model.forward_answer_only(
        prefix, prefix_mask, query, query_mask, 11, 12, torch.tensor([0, 1])
    )
    ledger = model.last_ledger.to_dict()
    assert ledger["persistent_bits"] == 5 * 2
    assert ledger["state_dim"] == 5
    assert ledger["bits_per_coordinate"] == 2
    assert ledger["transcript_length"] == 0
    assert ledger["input_reads"] == 1
    assert ledger["access_model"] == "sealed_prefix"
    assert ledger["recurrent_updates"] == 0
    assert ledger["retains_latent_history"] is False


def test_dataset_labels_uniformity_reproducibility_and_split_separation():
    tokenizer = TinyTokenizer()
    dataset = FrontierNLDataset(8, 20_000, 137, "train", tokenizer)
    rows = torch.arange(len(dataset))
    assert torch.equal(
        dataset.labels, dataset.frontiers[rows, dataset.queries].to(torch.long)
    )
    assert abs(float(dataset.frontiers.float().mean()) - 0.5) < 0.01
    frequencies = torch.bincount(dataset.queries, minlength=8).float() / len(dataset)
    assert torch.all((frequencies - 1 / 8).abs() < 0.01)

    repeated = FrontierNLDataset(8, 20_000, 137, "train", tokenizer)
    dev = FrontierNLDataset(8, 20_000, 137, "dev", tokenizer)
    assert dataset.fingerprint == repeated.fingerprint
    assert dataset.fingerprint != dev.fingerprint
    assert dataset.split_seed == derive_split_seed(8, 137, "train")
    assert dataset.split_seed != dev.split_seed


def test_fixed_template_puts_query_and_answer_only_after_boundary():
    frontier = [1, 0, 1, 0]
    rendered = render_example(frontier, 3)
    prefix, suffix = rendered.split("<|latent|>")
    assert "Question:" not in prefix
    assert "Answer:" not in prefix
    assert "Question: Is item 3 present?" in suffix
    assert not suffix.endswith(" yes") and not suffix.endswith(" no")


def test_answer_strings_are_single_distinct_tokens():
    assert answer_token_ids(TinyTokenizer()) == {"no": 11, "yes": 12}


def test_fano_inverse_and_envelope():
    for error in (0.0, 0.01, 0.1, 0.25, 0.5):
        torch.testing.assert_close(
            torch.tensor(h2_inverse(binary_entropy(error))),
            torch.tensor(error),
            atol=2e-6,
            rtol=0,
        )
    assert fano_error_lower_bound(1.0) == 0.0
    assert fano_error_lower_bound(1.25) == 0.0
    assert math.isclose(fano_error_lower_bound(0.0), 0.5)


def test_experiment_loader_uses_pretrained_not_from_config():
    source = Path("finite_cot/frontier_nl.py").read_text()
    loader = source[
        source.index("def load_e1_model_and_tokenizer"):source.index("def _to_device")
    ]
    assert "AutoModelForCausalLM.from_pretrained" in loader
    assert "AutoModelForCausalLM.from_config" not in loader
