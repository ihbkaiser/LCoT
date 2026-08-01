"""Learned and fixed models for theory-aligned synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from .access import LocalOracleAccess, SealedPrefixAccess
from .quantization import FiniteScalarQuantizer
from .resources import ResourceLedger
from .transcript import DiscreteTranscript


@dataclass
class FrontierOutput:
    logits: Tensor
    state_codes: Tensor
    transcript_codes: Tensor
    ledger: ResourceLedger


class HybridFrontierModel(nn.Module):
    """One-pass frontier encoder with latent and categorical retained memory."""

    def __init__(
        self,
        n: int,
        state_dim: int,
        bits: int,
        transcript_length: int,
        transcript_vocab_size: int,
        hidden_dim: int = 128,
        transcript_embedding_dim: int = 16,
        query_embedding_dim: int = 16,
        clip_value: float = 1.0,
        quantized: bool = True,
    ) -> None:
        super().__init__()
        if state_dim < 0:
            raise ValueError("state_dim must be non-negative")
        if state_dim == 0 and transcript_length == 0:
            raise ValueError("at least one retained channel is required")
        self.n = int(n)
        self.state_dim = int(state_dim)
        self.bits = int(bits)
        self.quantized = bool(quantized)
        self.encoder = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim) if state_dim else None
        self.state_quantizer = (
            FiniteScalarQuantizer(bits, clip_value)
            if state_dim and quantized
            else None
        )
        self.transcript = DiscreteTranscript(
            hidden_dim,
            transcript_length,
            transcript_vocab_size,
            transcript_embedding_dim,
        )
        self.query_embedding = nn.Embedding(n, query_embedding_dim)
        decoder_in = (
            state_dim
            + transcript_length * transcript_embedding_dim
            + query_embedding_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )

    def _encode_summary(self, frontier: Tensor):
        access = SealedPrefixAccess(frontier)
        features = access.encode_and_seal(self.encoder)
        batch = frontier.shape[0]

        if self.state_head is None:
            state = frontier.new_empty(batch, 0)
            state_codes = torch.empty(batch, 0, dtype=torch.long, device=frontier.device)
        else:
            raw_state = self.state_head(features)
            if self.state_quantizer is None:
                state = raw_state
                state_codes = torch.empty(
                    batch, 0, dtype=torch.long, device=frontier.device
                )
            else:
                quantized = self.state_quantizer.quantize(raw_state)
                state, state_codes = quantized.value, quantized.codes

        transcript = self.transcript(features)
        summary = torch.cat((state, transcript.embeddings.flatten(1)), dim=-1)
        ledger = ResourceLedger(
            state_dim=self.state_dim,
            bits_per_coordinate=self.bits if self.quantized else 0,
            transcript_length=self.transcript.length,
            transcript_vocab_size=self.transcript.vocab_size,
            access_model="sealed_prefix",
            input_reads=access.read_count,
            notes=(
                "hard finite state"
                if self.quantized
                else "unquantized control; state_bits is intentionally not claimed"
            ),
        )
        return summary, state_codes, transcript.codes, ledger

    def forward(self, frontier: Tensor, query: Tensor) -> FrontierOutput:
        summary, state_codes, transcript_codes, ledger = self._encode_summary(frontier)
        retained = torch.cat((summary, self.query_embedding(query)), dim=-1)
        logits = self.decoder(retained)
        return FrontierOutput(logits, state_codes, transcript_codes, ledger)

    def forward_all_queries(self, frontier: Tensor) -> FrontierOutput:
        """Train against every legal continuation while encoding the prefix once."""

        summary, state_codes, transcript_codes, ledger = self._encode_summary(frontier)
        batch = frontier.shape[0]
        queries = torch.arange(self.n, device=frontier.device)
        query_embeddings = self.query_embedding(queries).unsqueeze(0).expand(batch, -1, -1)
        expanded_summary = summary.unsqueeze(1).expand(-1, self.n, -1)
        decoder_input = torch.cat((expanded_summary, query_embeddings), dim=-1)
        logits = self.decoder(decoder_input)
        return FrontierOutput(logits, state_codes, transcript_codes, ledger)


class PrefixRereadControl(nn.Module):
    """Deliberately illegal control that answers by rereading the queried bit."""

    def forward(self, frontier: Tensor, query: Tensor) -> FrontierOutput:
        rows = torch.arange(frontier.shape[0], device=frontier.device)
        answer = frontier[rows, query]
        logits = torch.stack((1.0 - answer, answer), dim=-1) * 20.0
        empty = torch.empty(frontier.shape[0], 0, dtype=torch.long, device=frontier.device)
        ledger = ResourceLedger(
            state_dim=0,
            bits_per_coordinate=0,
            access_model="prefix_reread_control",
            input_reads=1,
            notes="outside the continuation lower-bound interface",
        )
        return FrontierOutput(logits, empty, empty, ledger)


class FixedHybridFrontierCodec(nn.Module):
    """Explicit bit packing across latent coordinates and transcript slots.

    This is the matching construction/coverage control. It demonstrates that
    the two retained channels substitute at the code-capacity level, but it is
    not evidence that gradient descent discovers the packing rule.
    """

    def __init__(
        self,
        n: int,
        state_dim: int,
        bits: int,
        transcript_length: int,
        transcript_vocab_size: int,
    ) -> None:
        super().__init__()
        transcript_width = int(round(torch.log2(torch.tensor(float(transcript_vocab_size))).item()))
        if (1 << transcript_width) != transcript_vocab_size:
            raise ValueError("fixed codec requires a power-of-two transcript vocabulary")
        self.n = int(n)
        self.state_dim = int(state_dim)
        self.bits = int(bits)
        self.transcript_length = int(transcript_length)
        self.transcript_vocab_size = int(transcript_vocab_size)
        self.transcript_width = transcript_width

    @property
    def retained_bits(self) -> int:
        return self.state_dim * self.bits + self.transcript_length * self.transcript_width

    @staticmethod
    def _pack(bits: Tensor, slots: int, width: int) -> Tensor:
        if slots == 0:
            return torch.empty(bits.shape[0], 0, dtype=torch.long, device=bits.device)
        grouped = bits.view(bits.shape[0], slots, width).to(torch.long)
        weights = (1 << torch.arange(width, device=bits.device)).to(torch.long)
        return (grouped * weights).sum(dim=-1)

    @staticmethod
    def _unpack(codes: Tensor, width: int) -> Tensor:
        if codes.shape[1] == 0:
            return torch.empty(codes.shape[0], 0, dtype=torch.long, device=codes.device)
        shifts = torch.arange(width, device=codes.device)
        return ((codes.unsqueeze(-1) >> shifts) & 1).flatten(1)

    def _encode(self, frontier: Tensor):
        access = SealedPrefixAccess(frontier)

        def pack(prefix: Tensor):
            capacity = self.retained_bits
            retained = torch.zeros(prefix.shape[0], capacity, device=prefix.device)
            copied = min(self.n, capacity)
            retained[:, :copied] = prefix[:, :copied]
            state_width = self.state_dim * self.bits
            state_codes = self._pack(
                retained[:, :state_width], self.state_dim, self.bits
            )
            transcript_codes = self._pack(
                retained[:, state_width:],
                self.transcript_length,
                self.transcript_width,
            )
            return state_codes, transcript_codes

        state_codes, transcript_codes = access.encode_and_seal(pack)
        decoded = torch.cat(
            (
                self._unpack(state_codes, self.bits),
                self._unpack(transcript_codes, self.transcript_width),
            ),
            dim=-1,
        )
        ledger = ResourceLedger(
            state_dim=self.state_dim,
            bits_per_coordinate=self.bits,
            transcript_length=self.transcript_length,
            transcript_vocab_size=self.transcript_vocab_size,
            access_model="sealed_prefix",
            input_reads=access.read_count,
            notes="fixed bit-packing construction; not learned evidence",
        )
        return decoded, state_codes, transcript_codes, ledger

    def forward(self, frontier: Tensor, query: Tensor) -> FrontierOutput:
        decoded, state_codes, transcript_codes, ledger = self._encode(frontier)
        answer = torch.zeros(frontier.shape[0], device=frontier.device)
        covered = query < min(self.n, self.retained_bits)
        rows = torch.arange(frontier.shape[0], device=frontier.device)[covered]
        answer[covered] = decoded[rows, query[covered]].to(answer.dtype)
        logits = torch.stack((1.0 - answer, answer), dim=-1) * 20.0
        return FrontierOutput(logits, state_codes, transcript_codes, ledger)

    def forward_all_queries(self, frontier: Tensor) -> FrontierOutput:
        decoded, state_codes, transcript_codes, ledger = self._encode(frontier)
        answer = torch.zeros(frontier.shape[0], self.n, device=frontier.device)
        covered = min(self.n, self.retained_bits)
        answer[:, :covered] = decoded[:, :covered].to(answer.dtype)
        logits = torch.stack((1.0 - answer, answer), dim=-1) * 20.0
        return FrontierOutput(logits, state_codes, transcript_codes, ledger)


def boolean_bfs(adjacency: Tensor, source: Tensor, steps: int) -> Tensor:
    """Theorem-3 Boolean frontier recurrence for a batched adjacency matrix."""

    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape [batch, nodes, nodes]")
    batch, nodes, _ = adjacency.shape
    state = torch.zeros(batch, nodes, dtype=torch.bool, device=adjacency.device)
    state[torch.arange(batch, device=adjacency.device), source] = True
    edges = adjacency.to(torch.bool)
    for _ in range(steps):
        outgoing = (state.unsqueeze(-1) & edges).any(dim=1)
        state = state | outgoing
    return state


@dataclass
class PointerOutput:
    logits: Tensor
    step_logits: List[Tensor]
    state_codes: List[Tensor]
    total_queries: int
    completed_updates: int


class LearnedPointerMachine(nn.Module):
    """Trainable recurrent controller with one hard function query per update."""

    def __init__(
        self,
        max_nodes: int,
        state_dim: int,
        bits: int,
        hidden_dim: int = 64,
        clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        if max_nodes < 2 or state_dim < 1:
            raise ValueError("invalid pointer model parameters")
        self.max_nodes = int(max_nodes)
        self.state_dim = int(state_dim)
        self.bits = int(bits)
        self.node_state = nn.Embedding(max_nodes, state_dim)
        self.quantizer = FiniteScalarQuantizer(bits, clip_value)
        self.query_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_nodes),
        )

    def _encode_node(self, node: Tensor) -> Tensor:
        return self.quantizer.quantize(self.node_state(node)).value

    def forward(
        self,
        function_table: Tensor,
        source: Tensor,
        updates: int,
        *,
        path: Optional[Tensor] = None,
        teacher_forcing: bool = False,
    ) -> PointerOutput:
        if updates < 0:
            raise ValueError("updates must be non-negative")
        if teacher_forcing and (path is None or path.shape[1] < updates + 1):
            raise ValueError("teacher forcing requires the complete queried path")
        oracle = LocalOracleAccess(function_table)
        initial = self.quantizer.quantize(self.node_state(source))
        state = initial.value
        codes = [initial.codes]
        current_logits = self.query_head(state)
        step_logits: List[Tensor] = []

        for step in range(updates):
            query = path[:, step] if teacher_forcing else current_logits.argmax(dim=-1)
            oracle.begin_update()
            response = oracle.query(query)
            oracle.end_update()
            # The response is an ephemeral oracle value.  It is immediately
            # encoded and rounded before it can persist into the next update.
            quantized = self.quantizer.quantize(self.node_state(response))
            state = quantized.value
            codes.append(quantized.codes)
            current_logits = self.query_head(state)
            step_logits.append(current_logits)

        return PointerOutput(
            logits=current_logits,
            step_logits=step_logits,
            state_codes=codes,
            total_queries=oracle.total_queries,
            completed_updates=oracle.completed_updates,
        )


@dataclass
class FixedPointerOutput:
    vertex: Tensor
    codes: List[Tensor]
    total_queries: int


class BinaryPointerMachine:
    """Fixed Boolean construction; collisions are explicit when ``d`` is small."""

    def __init__(self, state_dim: int) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)
        self.modulus = 1 << self.state_dim

    def encode(self, vertex: Tensor) -> Tensor:
        reduced = vertex.to(torch.long) % self.modulus
        shifts = torch.arange(self.state_dim, device=vertex.device)
        return ((reduced.unsqueeze(-1) >> shifts) & 1).to(torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        shifts = torch.arange(self.state_dim, device=codes.device)
        weights = (1 << shifts).to(torch.long)
        return (codes.to(torch.long) * weights).sum(dim=-1)

    def run(self, function_table: Tensor, source: Tensor, updates: int) -> FixedPointerOutput:
        oracle = LocalOracleAccess(function_table)
        codes = [self.encode(source)]
        for _ in range(updates):
            current = self.decode(codes[-1])
            oracle.begin_update()
            response = oracle.query(current)
            oracle.end_update()
            codes.append(self.encode(response))
        return FixedPointerOutput(self.decode(codes[-1]), codes, oracle.total_queries)
