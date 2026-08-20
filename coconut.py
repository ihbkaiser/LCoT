# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


def _trim_kv_cache(kv_cache, max_length):
    """Trim both modern Transformers Cache objects and legacy KV tuples."""
    if kv_cache is None:
        return None

    if hasattr(kv_cache, "crop"):
        kv_cache.crop(max_length)
        return kv_cache

    return [
        (
            layer[0][:, :, :max_length, :],
            layer[1][:, :, :max_length, :],
        )
        for layer in kv_cache
    ]


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        position_ids=None,
        **kwargs,
    ):

        logits = []

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        # Keep all latent bookkeeping on the GPU.  The previous nested Python
        # loops called ``.item()`` for every latent token and rebuilt the full
        # [batch, sequence, hidden] tensor as a list at every recurrent step.
        latent_mask = input_ids.eq(self.latent_token_id)
        latent_counts = latent_mask.sum(dim=1)
        max_n_latents = int(latent_counts.max().item())
        sequence_positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).expand_as(input_ids)
        latent_positions = sequence_positions.masked_fill(
            ~latent_mask, input_ids.shape[1]
        ).sort(dim=1).values

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (
                0,
                int(latent_positions[:, 0].min().item()),
            )
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = _trim_kv_cache(
                    kv_cache, next_compute_range[0]
                )

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            active_rows = torch.nonzero(
                latent_counts > pass_idx, as_tuple=False
            ).squeeze(1)
            token_indices = latent_positions[active_rows, pass_idx]
            replacement = hidden_states[
                active_rows,
                token_indices - 1 - hidden_states_offset,
                :,
            ]

            # Clone once to avoid mutating a tensor needed by autograd, then use
            # one GPU index operation instead of O(batch * sequence) Python ops.
            updated_embeds = inputs_embeds.clone()
            updated_embeds[active_rows, token_indices, :] = replacement
            inputs_embeds = updated_embeds

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=_trim_kv_cache(kv_cache, next_compute_range[0]),
            output_hidden_states=False,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode=True):
        # Preserve nn.Module semantics so DDP/FSDP and ``eval()`` can recurse
        # through the complete wrapper correctly.
        return super().train(mode)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        batch_size = input_ids.shape[0]
        outputs = self.forward(
            input_ids,
            attention_mask,
            labels=None,
        )
        inputs_embeds = outputs.inputs_embeds
        rows = torch.arange(batch_size, device=input_ids.device)
        last_prompt_positions = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .expand_as(input_ids)
            .masked_fill(attention_mask == 0, -1)
            .max(dim=1)
            .values
        )
        next_tokens = outputs.logits[rows, last_prompt_positions].argmax(dim=-1)
        generated = [next_tokens]
        finished = next_tokens.eq(self.eos_token_id)

        # Right-padded prompts can still be decoded together: padded positions
        # are masked, while generated positions receive their logical position.
        initial_decode_embeds = torch.cat(
            (inputs_embeds, self.embedding(next_tokens).unsqueeze(1)), dim=1
        )
        next_decode_embed = initial_decode_embeds[:, -1:, :]
        all_decode_embeds = initial_decode_embeds if output_embedding else None
        decode_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        prompt_lengths = attention_mask.sum(dim=1)
        decode_positions = torch.cat(
            (
                (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0),
                prompt_lengths.unsqueeze(1),
            ),
            dim=1,
        )
        past_key_values = None

        for token_step in range(1, max_new_tokens):
            if past_key_values is None:
                decoded = self.base_causallm(
                    inputs_embeds=initial_decode_embeds,
                    attention_mask=decode_mask,
                    position_ids=decode_positions,
                    use_cache=True,
                )
            else:
                decoded = self.base_causallm(
                    inputs_embeds=next_decode_embed,
                    attention_mask=decode_mask,
                    position_ids=decode_positions[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            self.gen_forward_cnt += 1
            past_key_values = decoded.past_key_values
            next_tokens = decoded.logits[:, -1, :].argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.eos_token_id),
                next_tokens,
            )
            generated.append(next_tokens)
            finished.logical_or_(next_tokens.eq(self.eos_token_id))

            next_decode_embed = self.embedding(next_tokens).unsqueeze(1)
            if output_embedding:
                all_decode_embeds = torch.cat(
                    (all_decode_embeds, next_decode_embed), dim=1
                )
            decode_mask = torch.cat(
                (decode_mask, torch.ones_like(decode_mask[:, :1])), dim=1
            )
            decode_positions = torch.cat(
                (decode_positions, (prompt_lengths + token_step).unsqueeze(1)), dim=1
            )
            if finished.all() and not synced_gpus:
                break

        generated_tokens = torch.stack(generated, dim=1)
        tokens = torch.cat((input_ids, generated_tokens), dim=1)
        return (tokens, all_decode_embeds) if output_embedding else tokens
