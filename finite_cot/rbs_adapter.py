"""Strict finite-state adapter for the Reasoning-by-Superposition ProsQA model.

Unlike the original Coconut implementation, this reference path never retains a
KV cache or a list of earlier latent vectors. ``readonly_input`` recomputes an
update from the immutable prefix and the current finite code. ``sealed_prefix``
uses the prefix only to create the initial code and then applies a state-only
transition. ``strict_read_once`` initializes from evidence before the start
marker, deletes it, and conditions every update on the post-boundary question.
The implementation favors an auditable contract over throughput.
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
        if self.access_mode not in {
            "readonly_input",
            "sealed_prefix",
            "strict_read_once",
        }:
            raise ValueError(
                "access_mode must be readonly_input, sealed_prefix, or "
                "strict_read_once"
            )
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
                    "bits_per_coordinate", finite_state.get("model_bits", 2)
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
        kwargs = {}
        if hidden and getattr(self.base_causallm.config, "model_type", None) == "qwen3":
            # State updates only consume the final hidden state. Avoid projecting
            # every prefix position over Qwen's full vocabulary.
            kwargs["logits_to_keep"] = 1
        return self.base_causallm(
            inputs_embeds=embeds,
            attention_mask=torch.ones(
                embeds.shape[:2], dtype=torch.long, device=embeds.device
            ),
            use_cache=False,
            output_hidden_states=hidden,
            **kwargs,
        )

    def _hidden_backbone(self):
        """Return GPT-2's transformer without its expensive vocabulary head."""

        model = self.base_causallm
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            model = get_base_model()
        if (
            getattr(model.config, "model_type", None) == "gpt2"
            and hasattr(model, "transformer")
        ):
            return model.transformer
        return None

    def _hidden(self, embeds: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Compute hidden states without projecting every token over the vocabulary.

        GPT-2's causal-LM wrapper always materializes full-vocabulary logits. State
        construction only needs transformer activations, so doing that projection
        for an 896-token evidence prefix is pure compute and memory overhead.
        """

        if attention_mask is None:
            attention_mask = torch.ones(
                embeds.shape[:2], dtype=torch.long, device=embeds.device
            )
        backbone = self._hidden_backbone()
        if backbone is not None:
            self.gen_forward_cnt += 1
            return backbone(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        return self._base(embeds, hidden=True).hidden_states[-1]

    def _batched_last_hidden(self, sequences: List[Tensor]) -> Tensor:
        """Run one padded LM batch and select each sequence's last hidden state."""

        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            device=sequences[0].device,
            dtype=torch.long,
        )
        padded = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(padded.shape[1], device=padded.device)
        attention_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        outputs = self._hidden(padded, attention_mask)
        batch_indices = torch.arange(len(sequences), device=padded.device)
        return outputs[batch_indices, lengths - 1]

    def _trim_example(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None
    ) -> Tuple[Tensor, Tensor | None]:
        keep = attention_mask.to(torch.bool)
        ids = input_ids[keep]
        kept_labels = labels[keep] if labels is not None else None
        return ids, kept_labels

    def _strict_context(
        self, ids: Tensor, labels: Tensor | None = None
    ) -> Tuple[Tensor, List[Tensor], Tensor | None]:
        """Build a question-conditioned state after sealing evidence once.

        Layout: evidence, START, question, LATENT * T, END, answer prompt/tail.
        START is the read-once boundary. The structural markers and evidence are
        absent from the effective decoder context.
        """

        start_positions = (ids == self.start_latent_id).nonzero(
            as_tuple=False
        ).view(-1)
        end_positions = (ids == self.end_latent_id).nonzero(
            as_tuple=False
        ).view(-1)
        if start_positions.numel() != 1 or end_positions.numel() != 1:
            raise ValueError(
                "strict_read_once requires exactly one start and one end marker"
            )
        start = int(start_positions[0])
        end = int(end_positions[0])
        if start == 0 or start >= end:
            raise ValueError("strict_read_once requires evidence before the boundary")

        latent_positions = (ids == self.latent_token_id).nonzero(
            as_tuple=False
        ).view(-1)
        if latent_positions.numel() and not bool(
            ((latent_positions > start) & (latent_positions < end)).all()
        ):
            raise ValueError("all strict latent markers must lie inside the boundary")
        first_latent = int(latent_positions[0]) if latent_positions.numel() else end
        continuation_ids = ids[start + 1 : first_latent]
        if continuation_ids.numel() == 0:
            raise ValueError("strict_read_once requires a post-boundary continuation")
        evidence_ids = ids[:start]
        tail_ids = ids[end + 1 :]

        evidence_embeds = self.embedding(evidence_ids).unsqueeze(0)
        initial_hidden = self._hidden(evidence_embeds)[:, -1, :]
        state_q = self.bottleneck.compress(initial_hidden)
        state = state_q.value
        codes = [state_q.codes]
        continuation_embeds = self.embedding(continuation_ids).unsqueeze(0)

        for _ in range(int(latent_positions.numel())):
            update_input = torch.cat(
                (continuation_embeds, self.bottleneck.expand(state).unsqueeze(1)),
                dim=1,
            )
            updated_hidden = self._hidden(update_input)[:, -1, :]
            state_q = self.bottleneck.compress(updated_hidden)
            state = state_q.value
            codes.append(state_q.codes)

        state_embed = self.bottleneck.expand(state).unsqueeze(1)
        tail_embeds = self.embedding(tail_ids).unsqueeze(0)
        effective = torch.cat(
            (continuation_embeds, state_embed, tail_embeds), dim=1
        )
        effective_labels = None
        if labels is not None:
            state_label = labels.new_full((1,), -100)
            effective_labels = torch.cat(
                (labels[start + 1 : first_latent], state_label, labels[end + 1 :])
            )

        self.last_ledger = ResourceLedger(
            state_dim=self.bottleneck.state_dim,
            bits_per_coordinate=self.bottleneck.bits,
            recurrent_updates=int(latent_positions.numel()),
            access_model=self.access_mode,
            input_reads=1,
            retains_latent_history=False,
            notes=(
                "evidence read once; every update sees only the shared "
                "continuation and current hard code; no past_key_values"
            ),
        )
        return effective, codes, effective_labels

    def _strict_forward(self, input_ids, attention_mask, labels) -> Outputs:
        evidence_sequences = []
        continuation_sequences = []
        tail_sequences = []
        mapped_labels = []
        latent_counts = []
        for index in range(input_ids.shape[0]):
            ids, kept_labels = self._trim_example(
                input_ids[index],
                attention_mask[index],
                labels[index] if labels is not None else None,
            )

            start_positions = (ids == self.start_latent_id).nonzero(
                as_tuple=False
            ).view(-1)
            end_positions = (ids == self.end_latent_id).nonzero(
                as_tuple=False
            ).view(-1)
            if start_positions.numel() != 1 or end_positions.numel() != 1:
                raise ValueError(
                    "strict_read_once requires exactly one start and one end marker"
                )
            start = int(start_positions[0])
            end = int(end_positions[0])
            if start == 0 or start >= end:
                raise ValueError(
                    "strict_read_once requires evidence before the boundary"
                )

            latent_positions = (ids == self.latent_token_id).nonzero(
                as_tuple=False
            ).view(-1)
            if latent_positions.numel() and not bool(
                ((latent_positions > start) & (latent_positions < end)).all()
            ):
                raise ValueError(
                    "all strict latent markers must lie inside the boundary"
                )
            first_latent = (
                int(latent_positions[0]) if latent_positions.numel() else end
            )
            continuation_ids = ids[start + 1 : first_latent]
            if continuation_ids.numel() == 0:
                raise ValueError(
                    "strict_read_once requires a post-boundary continuation"
                )

            evidence_sequences.append(self.embedding(ids[:start]))
            continuation_sequences.append(self.embedding(continuation_ids))
            tail_sequences.append(self.embedding(ids[end + 1 :]))
            latent_counts.append(int(latent_positions.numel()))
            if kept_labels is not None:
                mapped_labels.append(
                    torch.cat(
                        (
                            kept_labels[start + 1 : first_latent],
                            kept_labels.new_full((1,), -100),
                            kept_labels[end + 1 :],
                        )
                    )
                )

        # One transformer call handles all evidence prefixes in the batch. The
        # evidence activations are not placed in the decoder context or retained
        # between recurrent updates; only the quantized state survives.
        initial_state = self.bottleneck.compress(
            self._batched_last_hidden(evidence_sequences)
        )
        states: List[Tensor] = [
            initial_state.value[index] for index in range(input_ids.shape[0])
        ]
        all_codes: List[List[Tensor]] = [
            [initial_state.codes[index]] for index in range(input_ids.shape[0])
        ]

        for update_index in range(max(latent_counts, default=0)):
            active = [
                index
                for index, count in enumerate(latent_counts)
                if count > update_index
            ]
            update_hidden = self._batched_last_hidden(
                [
                    torch.cat(
                        (
                            continuation_sequences[index],
                            self.bottleneck.expand(states[index]).unsqueeze(0),
                        ),
                        dim=0,
                    )
                    for index in active
                ]
            )
            updated_state = self.bottleneck.compress(update_hidden)
            for row, index in enumerate(active):
                states[index] = updated_state.value[row]
                all_codes[index].append(updated_state.codes[row])

        effective_sequences = [
            torch.cat(
                (
                    continuation_sequences[index],
                    self.bottleneck.expand(states[index]).unsqueeze(0),
                    tail_sequences[index],
                ),
                dim=0,
            )
            for index in range(input_ids.shape[0])
        ]

        lengths = torch.tensor(
            [sequence.shape[0] for sequence in effective_sequences],
            device=input_ids.device,
            dtype=torch.long,
        )
        effective = pad_sequence(effective_sequences, batch_first=True)
        positions = torch.arange(effective.shape[1], device=input_ids.device)
        effective_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        self.gen_forward_cnt += 1
        outputs = self.base_causallm(
            inputs_embeds=effective,
            attention_mask=effective_mask,
            use_cache=False,
        )

        if labels is None:
            loss = outputs.logits.sum() * 0.0
        else:
            mapped = pad_sequence(
                mapped_labels, batch_first=True, padding_value=-100
            )
            loss = F.cross_entropy(
                outputs.logits[:, :-1].flatten(0, 1),
                mapped[:, 1:].flatten(),
                ignore_index=-100,
            )
        self.last_state_codes = [
            [code.detach() for code in trace] for trace in all_codes
        ]
        self.last_ledger = ResourceLedger(
            state_dim=self.bottleneck.state_dim,
            bits_per_coordinate=self.bottleneck.bits,
            recurrent_updates=max(latent_counts, default=0),
            access_model=self.access_mode,
            input_reads=1,
            retains_latent_history=False,
            notes=(
                "evidence read once; every update sees only the shared "
                "continuation and current hard code; no evidence past_key_values"
            ),
        )
        return Outputs(loss=loss, inputs_embeds=effective, logits=outputs.logits)

    def _finite_context(
        self, ids: Tensor
    ) -> Tuple[Tensor, List[Tensor], int, int]:
        if self.access_mode == "strict_read_once":
            effective, codes, _ = self._strict_context(ids)
            return effective, codes, -1, 0
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

        encoded = self._hidden(prefix_embeds)
        state_q = self.bottleneck.compress(encoded[:, -1, :])
        state = state_q.value
        codes = [state_q.codes]
        num_latents = int(latent_positions.numel())
        for _ in range(max(0, num_latents - 1)):
            if self.access_mode == "readonly_input":
                update_input = torch.cat(
                    (prefix_embeds, self.bottleneck.expand(state).unsqueeze(1)), dim=1
                )
                updated = self._hidden(update_input)[:, -1, :]
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
        if self.access_mode == "strict_read_once":
            return self._strict_forward(input_ids, attention_mask, labels)
        trimmed_ids = []
        trimmed_labels = []
        prefixes: List[Tensor | None] = []
        tails: List[Tensor | None] = []
        valid_mask = attention_mask.to(torch.bool)
        latent_mask = input_ids.eq(self.latent_token_id) & valid_mask
        token_positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        latent_counts_tensor = latent_mask.sum(dim=1)
        first_physical = torch.where(
            latent_mask, token_positions, input_ids.shape[1]
        ).amin(dim=1)
        last_physical = torch.where(latent_mask, token_positions, -1).amax(dim=1)
        valid_start = valid_mask.to(torch.long).argmax(dim=1)
        latent_metadata = torch.stack(
            (
                latent_counts_tensor,
                first_physical - valid_start,
                last_physical - valid_start,
            ),
            dim=1,
        ).tolist()
        latent_counts = [metadata[0] for metadata in latent_metadata]
        all_codes: List[List[Tensor]] = [[] for _ in range(input_ids.shape[0])]

        for index in range(input_ids.shape[0]):
            ids, kept_labels = self._trim_example(
                input_ids[index],
                attention_mask[index],
                labels[index] if labels is not None else None,
            )
            trimmed_ids.append(ids)
            trimmed_labels.append(kept_labels)
            latent_count, first, last = latent_metadata[index]
            if latent_count:
                if first == 0:
                    raise ValueError(
                        "a finite-state sequence must have a non-empty prefix"
                    )
                prefixes.append(self.embedding(ids[:first]))
                tails.append(self.embedding(ids[last + 1 :]))
            else:
                prefixes.append(None)
                tails.append(None)

        states: List[Tensor | None] = [None] * input_ids.shape[0]
        active = [index for index, count in enumerate(latent_counts) if count]
        if active:
            initial_hidden = self._batched_last_hidden(
                [prefixes[index] for index in active]
            )
            initial_state = self.bottleneck.compress(initial_hidden)
            for row, index in enumerate(active):
                states[index] = initial_state.value[row]
                all_codes[index].append(initial_state.codes[row])

        for update_index in range(1, max(latent_counts, default=0)):
            active = [
                index
                for index, count in enumerate(latent_counts)
                if count > update_index
            ]
            if self.access_mode == "readonly_input":
                update_hidden = self._batched_last_hidden(
                    [
                        torch.cat(
                            (
                                prefixes[index],
                                self.bottleneck.expand(states[index]).unsqueeze(0),
                            ),
                            dim=0,
                        )
                        for index in active
                    ]
                )
                updated_state = self.bottleneck.compress(update_hidden)
            else:
                current_states = torch.stack([states[index] for index in active])
                updated_state = self.bottleneck.quantizer.quantize(
                    self.sealed_transition(current_states)
                )
            for row, index in enumerate(active):
                states[index] = updated_state.value[row]
                all_codes[index].append(updated_state.codes[row])

        effective_sequences = []
        state_positions = []
        tail_starts = []
        for index, ids in enumerate(trimmed_ids):
            if latent_counts[index]:
                state_embed = self.bottleneck.expand(states[index]).unsqueeze(0)
                effective_sequences.append(
                    torch.cat((prefixes[index], state_embed, tails[index]), dim=0)
                )
                state_positions.append(prefixes[index].shape[0])
                tail_starts.append(latent_metadata[index][2] + 1)
            else:
                effective_sequences.append(self.embedding(ids))
                state_positions.append(0)
                tail_starts.append(-1)

        effective_lengths = [
            sequence.shape[0] for sequence in effective_sequences
        ]
        lengths = torch.tensor(
            effective_lengths,
            device=input_ids.device,
            dtype=torch.long,
        )
        effective = pad_sequence(effective_sequences, batch_first=True)
        positions = torch.arange(effective.shape[1], device=input_ids.device)
        effective_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        self.gen_forward_cnt += 1
        outputs = self.base_causallm(
            inputs_embeds=effective,
            attention_mask=effective_mask,
            use_cache=False,
        )

        losses = []
        for index, kept_labels in enumerate(trimmed_labels):
            example_logits = outputs.logits[index, : effective_lengths[index]]
            if kept_labels is None:
                loss = example_logits.sum() * 0.0
            elif tail_starts[index] < 0:
                loss = F.cross_entropy(
                    example_logits[:-1], kept_labels[1:], ignore_index=-100
                )
            else:
                tail_labels = kept_labels[tail_starts[index] :]
                prediction_logits = example_logits[
                    state_positions[index] : state_positions[index]
                    + tail_labels.numel()
                ]
                supervised = tail_labels != -100
                loss = (
                    F.cross_entropy(
                        prediction_logits[supervised], tail_labels[supervised]
                    )
                    if supervised.any()
                    else example_logits.sum() * 0.0
                )
            losses.append(loss)

        self.last_state_codes = [
            [code.detach() for code in trace] for trace in all_codes
        ]
        max_latents = max(latent_counts, default=0)
        self.last_ledger = ResourceLedger(
            state_dim=self.bottleneck.state_dim,
            bits_per_coordinate=self.bottleneck.bits,
            recurrent_updates=max(0, max_latents - 1),
            access_model=self.access_mode,
            input_reads=max_latents + 1 if max_latents else 1,
            retains_latent_history=False,
            notes="batched by recurrent depth; no past_key_values",
        )
        return Outputs(
            loss=torch.stack(losses).mean(),
            inputs_embeds=effective,
            logits=outputs.logits,
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
        return [
            [code.detach().cpu().tolist() for code in trace]
            for trace in self.last_state_codes
        ]
