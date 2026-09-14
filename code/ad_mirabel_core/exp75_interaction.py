"""Frozen-encoder, retriever-ID-agnostic interaction model for Exp75.

The primary model consumes only a query embedding and a set of retrieved
document embeddings. Retriever, domain, family, rank, and raw retrieval scores
are deliberately absent from the model interface.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


def pair_interaction(query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
    """Return [q*d, |q-d|, cosine] for a batch of unordered document sets."""
    if query.ndim != 2 or documents.ndim != 3:
        raise ValueError("query must be [batch,dim], documents [batch,k,dim]")
    if query.shape[0] != documents.shape[0] or query.shape[1] != documents.shape[2]:
        raise ValueError("query/document dimensions do not align")
    expanded = query[:, None, :].expand_as(documents)
    cosine = torch.nn.functional.cosine_similarity(expanded, documents, dim=-1)[..., None]
    return torch.cat((expanded * documents, torch.abs(expanded - documents), cosine), dim=-1)


class SharedTopKInteraction(nn.Module):
    """Small shared MLP followed by permutation-invariant mean/max pooling."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.shared = nn.Sequential(
            nn.Linear(2 * self.embedding_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.head = nn.Linear(2 * output_dim, 1)

    def pooled(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        values = self.shared(pair_interaction(query, documents))
        return torch.cat((values.mean(dim=1), values.max(dim=1).values), dim=-1)

    def forward(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(query, documents)).squeeze(-1)


class Top1Interaction(SharedTopKInteraction):
    """Top-1 ablation with the same capacity and no rank input."""

    def pooled(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        values = self.shared(pair_interaction(query, documents[:, :1]))
        first = values[:, 0]
        return torch.cat((first, first), dim=-1)


def threshold_higher(scores: Sequence[float], target_fpr: float = 0.02) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional calibration scores required")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie in (0,1)")
    return float(np.quantile(values, 1 - target_fpr, method="higher"))


def harmonic_adr(member_tpr: float, nonmember_tpr: float) -> float:
    total = float(member_tpr) + float(nonmember_tpr)
    return 0.0 if total == 0 else 2.0 * float(member_tpr) * float(nonmember_tpr) / total


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def stable_bucket(value: object, buckets: int = 100) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16) % int(buckets)


def identifier_hash(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(payload.encode()).hexdigest()

