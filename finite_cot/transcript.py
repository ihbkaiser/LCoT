"""Fixed-width categorical transcripts with exact capacity accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TranscriptOutput:
    embeddings: Tensor
    codes: Tensor
    one_hot: Tensor


class DiscreteTranscript(nn.Module):
    """Produce ``length`` categorical slots over one fixed alphabet.

    Training uses hard straight-through Gumbel softmax.  Evaluation uses argmax,
    so each retained slot is always one of exactly ``vocab_size`` values.
    """

    def __init__(
        self,
        input_dim: int,
        length: int,
        vocab_size: int,
        embedding_dim: int,
        temperature: float = 1.0,
        stochastic_training: bool = False,
    ) -> None:
        super().__init__()
        if length < 0:
            raise ValueError("length must be non-negative")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.length = int(length)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.temperature = float(temperature)
        self.stochastic_training = bool(stochastic_training)
        self.to_logits = (
            nn.Linear(input_dim, length * vocab_size) if length > 0 else None
        )
        self.codebook = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.codebook, std=embedding_dim**-0.5)

    def forward(self, features: Tensor) -> TranscriptOutput:
        batch = features.shape[0]
        if self.length == 0:
            empty_codes = torch.empty(batch, 0, dtype=torch.long, device=features.device)
            empty_one_hot = features.new_empty(batch, 0, self.vocab_size)
            empty_embeddings = features.new_empty(batch, 0, self.embedding_dim)
            return TranscriptOutput(empty_embeddings, empty_codes, empty_one_hot)

        logits = self.to_logits(features).view(batch, self.length, self.vocab_size)
        if self.training:
            if self.stochastic_training:
                one_hot = F.gumbel_softmax(
                    logits, tau=self.temperature, hard=True, dim=-1
                )
                codes = one_hot.argmax(dim=-1)
            else:
                soft = F.softmax(logits / self.temperature, dim=-1)
                codes = soft.argmax(dim=-1)
                hard = F.one_hot(codes, num_classes=self.vocab_size).to(soft.dtype)
                one_hot = soft + (hard - soft).detach()
        else:
            codes = logits.argmax(dim=-1)
            one_hot = F.one_hot(codes, num_classes=self.vocab_size).to(logits.dtype)
        embeddings = one_hot @ self.codebook.to(logits.dtype)
        return TranscriptOutput(embeddings, codes, one_hot)
