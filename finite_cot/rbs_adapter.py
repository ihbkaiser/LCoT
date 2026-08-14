"""Strict finite-state adapter for the Reasoning-by-Superposition ProsQA model.

Unlike the original Coconut implementation, this reference path never retains a
KV cache or a list of earlier latent vectors.  ``readonly_input`` recomputes an
update from the immutable prefix and the current finite code.  ``sealed_prefix``
uses the prefix only to create the initial code and then applies a state-only
transition.  The implementation favors an auditable contract over throughput.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers.models.gpt2 import GPT2LMHeadModel

from .quantization import FiniteStateBottleneck
from .resources import ResourceLedger


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


class StrictFiniteStateCoconut(nn.Module):
    """Coconut-compatible wrapper with no persistent KV or latent-history leak."""

    def __init__(
        self,
        base_causallm: nn.Module,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        finite_state: Dict,
    ) -> None:
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = int(latent_token_id)
        self.start_latent_id = int(start_latent_id)
        self.end_latent_id = int(end_latent_id)
        self.eos_token_id = int(eos_token_id)
        self.access_mode = finite_state.get("access_mode", "readonly_input")
        if self.access_mode not in {"readonly_input", "sealed_prefix"}:
            raise ValueError("access_mode must be readonly_input or sealed_prefix")
        if isinstance(base_causallm, GPT2LMHeadModel):
            self.embedding = base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = base_causallm.get_input_embeddings()
        hidden_size = getattr(base_causallm.config, "hidden_size", None)
        hidden_size = hidden_size or getattr(base_causallm.config, "n_embd")
        self.bottleneck = FiniteStateBottleneck(
            hidden_size=hidden_size,
            state_dim=int(finite_state["state_dim"]),
            bits=int(
                finite_state.get(
                    "model_bits", finite_state.get("bits_per_coordinate", 2)
                )
            ),
            clip_value=float(finite_state.get("clip_value", 1.0)),
            learnable_clip=bool(finite_state.get("learnable_clip", False)),
        )
        self.sealed_transition = nn.Sequential(
            nn.Linear(self.bottleneck.state_dim, self.bottleneck.state_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck.state_dim, self.bottleneck.state_dim),
        )
        self.last_state_codes: List[List[Tensor]] = []
        self.last_ledger: ResourceLedger | None = None
        self.gen_forward_cnt = 0

    def _base(self, embeds: Tensor, *, hidden: bool = False):
        self.gen_forward_cnt += 1
        return self.base_causallm(
            inputs_embeds=embeds,
            attention_mask=torch.ones(
                embeds.shape[:2], dtype=torch.long, device=embeds.device
            ),
            use_cache=False,
            output_hidden_states=hidden,
        )

    def _trim_example(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None
    ) -> Tuple[Tensor, Tensor | None]:
        keep = attention_mask.to(torch.bool)
        ids = input_ids[keep]
        kept_labels = labels[keep] if labels is not None else None
        return ids, kept_labels

    def _finite_context(
        self, ids: Tensor
    ) -> Tuple[Tensor, List[Tensor], int, int]:
        latent_positions = (ids == self.latent_token_id).nonzero(as_tuple=False).view(-1)
        if latent_positions.numel() == 0:
            return self.embedding(ids).unsqueeze(0), [], -1, 0
        first = int(latent_positions[0])
        last = int(latent_positions[-1])
        prefix_ids = ids[:first]
        tail_ids = ids[last + 1 :]
        if prefix_ids.numel() == 0:
            raise ValueError("a finite-state sequence must have a non-empty prefix")
        prefix_embeds = self.embedding(prefix_ids).unsqueeze(0)

        encoded = self._base(prefix_embeds, hidden=True)
        state_q = self.bottleneck.compress(encoded.hidden_states[-1][:, -1, :])
        state = state_q.value
        codes = [state_q.codes]
        num_latents = int(latent_positions.numel())
        for _ in range(max(0, num_latents - 1)):
            if self.access_mode == "readonly_input":
                update_input = torch.cat(
                    (prefix_embeds, self.bottleneck.expand(state).unsqueeze(1)), dim=1
                )
                updated = self._base(update_input, hidden=True).hidden_states[-1][
                    :, -1, :
                ]
                state_q = self.bottleneck.compress(updated)
            else:
                state_q = self.bottleneck.quantizer.quantize(
                    self.sealed_transition(state)
                )
            state = state_q.value
            codes.append(state_q.codes)

        state_embed = self.bottleneck.expand(state).unsqueeze(1)
        tail_embeds = self.embedding(tail_ids).unsqueeze(0)
        if self.access_mode == "readonly_input":
            effective = torch.cat((prefix_embeds, state_embed, tail_embeds), dim=1)
            state_position = prefix_ids.numel()
            input_reads = num_latents + 1
        else:
            effective = torch.cat((state_embed, tail_embeds), dim=1)
            state_position = 0
            input_reads = 1
        self.last_ledger = ResourceLedger(
            state_dim=self.bottleneck.state_dim,
            bits_per_coordinate=self.bottleneck.bits,
            recurrent_updates=max(0, num_latents - 1),
            access_model=self.access_mode,
            input_reads=input_reads,
            retains_latent_history=False,
            notes="no past_key_values; only the current hard code persists",
        )
        return effective, codes, last + 1, state_position

    def _example_forward(
        self, ids: Tensor, labels: Tensor | None
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        effective, codes, tail_start, state_position = self._finite_context(ids)
        output = self._base(effective, hidden=False)
        logits = output.logits.squeeze(0)
        if labels is None:
            loss = logits.sum() * 0.0
        elif tail_start < 0:
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        else:
            tail_labels = labels[tail_start:]
            prediction_logits = logits[
                state_position : state_position + tail_labels.numel()
            ]
            supervised = tail_labels != -100
            loss = (
                F.cross_entropy(prediction_logits[supervised], tail_labels[supervised])
                if supervised.any()
                else logits.sum() * 0.0
            )
        return loss, effective.squeeze(0), logits, codes

    def forward(
        self, input_ids, attention_mask, labels, position_ids=None, **kwargs
    ) -> Outputs:
        losses = []
        embeds = []
        logits = []
        all_codes: List[List[Tensor]] = []
        for index in range(input_ids.shape[0]):
            ids, kept_labels = self._trim_example(
                input_ids[index], attention_mask[index], labels[index]
            )
            loss, example_embeds, example_logits, codes = self._example_forward(
                ids, kept_labels
            )
            losses.append(loss)
            embeds.append(example_embeds)
            logits.append(example_logits)
            all_codes.append(codes)
        self.last_state_codes = [
            [code.detach().cpu() for code in trace] for trace in all_codes
        ]
        return Outputs(
            loss=torch.stack(losses).mean(),
            inputs_embeds=pad_sequence(embeds, batch_first=True),
            logits=pad_sequence(logits, batch_first=True),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("strict finite-state generation currently uses batch size 1")
        self.gen_forward_cnt = 0
        ids, _ = self._trim_example(input_ids[0], attention_mask[0], None)
        effective, codes, _, _ = self._finite_context(ids)
        generated: List[int] = []
        stopped = False
        for _ in range(max_new_tokens):
            output = self._base(effective, hidden=False)
            next_token = int(output.logits[0, -1].argmax())
            if not stopped and next_token == self.eos_token_id:
                stopped = True
            if not stopped:
                generated.append(next_token)
                token_embed = self.embedding(
                    torch.tensor([next_token], device=ids.device)
                ).unsqueeze(0)
                effective = torch.cat((effective, token_embed), dim=1)
            elif not synced_gpus:
                break
        self.last_state_codes = [[code.detach().cpu() for code in codes]]
        tokens = torch.cat(
            (ids, torch.tensor(generated, device=ids.device, dtype=ids.dtype))
        ).unsqueeze(0)
        return (tokens, effective) if output_embedding else tokens

    def export_state_codes(self):
        return [[code.tolist() for code in trace] for trace in self.last_state_codes]
