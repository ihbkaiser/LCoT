import os
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

from finite_cot.rbs_adapter import StrictFiniteStateCoconut
from run import (
    LORA_RANK,
    add_pretrained_lora,
    checkpoint_path,
    load_training_checkpoint,
    trainable_state_dict,
)


def tiny_gpt2():
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=40,
            n_positions=16,
            n_ctx=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=38,
            eos_token_id=38,
            pad_token_id=38,
        )
    )


def tiny_qwen3():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            use_sliding_window=False,
        )
    )


def finite_config():
    return {
        "state_dim": 4,
        "model_bits": 2,
        "clip_value": 1.0,
        "learnable_clip": False,
        "access_mode": "readonly_input",
    }


class RunLoraTests(unittest.TestCase):
    def test_best_only_checkpoint_uses_stable_path(self):
        self.assertEqual(
            checkpoint_path("ckpts/run", 7, save_best_only=True),
            os.path.join("ckpts/run", "best_model.pt"),
        )
        self.assertEqual(
            checkpoint_path("ckpts/run", 7),
            os.path.join("ckpts/run", "checkpoint_7"),
        )

    def test_random_base_remains_fully_trainable_without_lora(self):
        model = tiny_gpt2()
        self.assertFalse(any("lora_" in name for name, _ in model.named_parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_lora_freezes_transformer_but_trains_interfaces(self):
        model = add_pretrained_lora(tiny_gpt2())

        lora_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        self.assertTrue(lora_parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in lora_parameters))
        self.assertEqual(model.peft_config["default"].r, LORA_RANK)

        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        self.assertTrue(input_weight.requires_grad)
        self.assertTrue(output_weight.requires_grad)

        frozen_base = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter is not input_weight
        ]
        self.assertTrue(frozen_base)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen_base))

    def test_qwen3_lora_targets_attention_and_mlp_projections(self):
        model = add_pretrained_lora(tiny_qwen3())
        self.assertEqual(
            model.peft_config["default"].target_modules,
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            },
        )

    def test_discrete_state_expands_back_to_transformer_width(self):
        model = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()),
            latent_token_id=33,
            start_latent_id=31,
            end_latent_id=32,
            eos_token_id=38,
            finite_state=finite_config(),
        )
        hidden = torch.randn(2, 16)
        quantized = model.bottleneck.compress(hidden)
        expanded = model.bottleneck.expand(quantized.value)

        self.assertEqual(quantized.codes.shape, (2, 4))
        self.assertEqual(quantized.codes.dtype, torch.long)
        self.assertGreaterEqual(int(quantized.codes.min()), 0)
        self.assertLess(int(quantized.codes.max()), 4)
        self.assertEqual(expanded.shape, hidden.shape)

        # Two latent vocabulary slots collapse to one current state position;
        # ordinary prefix/tail tokens and that state all retain hidden width H.
        ids = torch.tensor([35, 1, 31, 33, 33, 32, 37])
        effective, codes, tail_start, state_position = model._finite_context(ids)
        self.assertEqual(effective.shape, (1, 6, 16))
        self.assertEqual(len(codes), 2)
        self.assertTrue(all(code.shape == (1, 4) for code in codes))
        self.assertEqual(tail_start, 5)
        self.assertEqual(state_position, 3)

    def test_finite_state_forward_batches_examples_by_recurrent_depth(self):
        model = StrictFiniteStateCoconut(
            tiny_gpt2(), 33, 31, 32, 38, finite_config()
        )
        model.eval()
        input_ids = torch.tensor(
            [
                [38, 35, 1, 31, 33, 33, 32, 37, 38],
                [35, 2, 31, 33, 32, 36, 37, 38, 38],
                [38, 38, 35, 3, 4, 5, 6, 36, 37],
            ]
        )
        attention_mask = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        labels = torch.full_like(input_ids, -100)
        labels[0, 7] = 37
        labels[1, 6] = 37
        labels[2, 8] = 37

        expected_losses = []
        with torch.no_grad():
            for index in range(input_ids.shape[0]):
                ids, kept_labels = model._trim_example(
                    input_ids[index], attention_mask[index], labels[index]
                )
                loss, _, _, _ = model._example_forward(
                    ids, kept_labels
                )
                expected_losses.append(loss)

        calls = []
        hook = model.base_causallm.register_forward_hook(
            lambda *unused: calls.append(1)
        )
        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
        finally:
            hook.remove()

        torch.testing.assert_close(output.loss, torch.stack(expected_losses).mean())
        # Two latent depths plus one final prediction pass, independent of the
        # number of examples in the batch.
        self.assertEqual(len(calls), 3)

    def test_batched_finite_state_forward_preserves_gradients(self):
        model = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_qwen3()), 33, 31, 32, 38, finite_config()
        )
        input_ids = torch.tensor(
            [
                [35, 1, 31, 33, 33, 32, 37],
                [35, 2, 31, 33, 32, 36, 37],
            ]
        )
        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = 37
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
        )
        output.loss.backward()

        self.assertIsNotNone(model.bottleneck.down.weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )

    def test_lora_checkpoint_round_trip(self):
        source = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()), 33, 31, 32, 38, finite_config()
        )
        checkpoint = source.state_dict()
        target = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()), 33, 31, 32, 38, finite_config()
        )

        incompatible = load_training_checkpoint(
            target, checkpoint, coconut=True, uses_lora=True
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

        source.eval()
        target.eval()
        ids = torch.tensor([35, 1, 31, 33, 33, 32, 37])
        source_effective, source_codes, _, _ = source._finite_context(ids)
        target_effective, target_codes, _, _ = target._finite_context(ids)
        self.assertTrue(torch.equal(source_effective, target_effective))
        self.assertEqual(len(source_codes), len(target_codes))
        for source_code, target_code in zip(source_codes, target_codes):
            self.assertTrue(torch.equal(source_code, target_code))

    def test_trainable_checkpoint_excludes_frozen_base_weights(self):
        source = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()), 33, 31, 32, 38, finite_config()
        )
        checkpoint = trainable_state_dict(source)

        self.assertTrue(any("lora_" in key for key in checkpoint))
        self.assertTrue(
            any(
                marker in key
                for key in checkpoint
                for marker in ("embedding", "embed_tokens", "wte", "lm_head")
            )
        )
        self.assertTrue(any("bottleneck" in key for key in checkpoint))
        self.assertFalse(
            any(
                "base_layer.weight" in key and "embedding" not in key
                for key in checkpoint
            )
        )
        full_state = source.state_dict()
        self.assertLess(
            sum(value.numel() for value in checkpoint.values()),
            sum(value.numel() for value in full_state.values()),
        )

        target = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()), 33, 31, 32, 38, finite_config()
        )
        load_training_checkpoint(
            target, checkpoint, coconut=True, uses_lora=True
        )
        for key, value in checkpoint.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_pre_lora_checkpoint_is_rejected(self):
        model = StrictFiniteStateCoconut(
            add_pretrained_lora(tiny_gpt2()), 33, 31, 32, 38, finite_config()
        )
        with self.assertRaisesRegex(ValueError, "checkpoint has no LoRA weights"):
            load_training_checkpoint(
                model, tiny_gpt2().state_dict(), coconut=True, uses_lora=True
            )


if __name__ == "__main__":
    unittest.main()
