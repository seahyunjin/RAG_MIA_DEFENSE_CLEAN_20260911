"""Exp52-U universal per-query semantic detector primitives.

The deployed scoring API consumes only a frozen query embedding and the set of
frozen embeddings for documents returned by the live RAG retriever.  It never
accepts domain, retriever, attack-family, membership, turn, or budget metadata.
Document pooling and session aggregation are permutation invariant by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn


FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "attack_family", "attack_method", "attack_template", "family_id",
    "domain", "domain_id", "corpus", "corpus_id", "retriever",
    "retriever_id", "member_label", "membership", "query_budget",
    "remaining_query_count", "total_attack_budget", "turn_index",
})


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(value, float(strength))


class QueryOnlyScorer(nn.Module):
    """S0 shortcut control; it is never eligible as the final detector."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 48) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        if query_embeddings.ndim != 2 or not torch.isfinite(query_embeddings).all():
            raise ValueError("query_embeddings must be a finite [B,D] tensor")
        hidden = self.encoder[:-1](query_embeddings)
        return {"representation": hidden, "logit": self.encoder[-1](hidden).squeeze(-1)}

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


class SemanticInteractionScorer(nn.Module):
    """Small query-conditioned, document-set-invariant interaction model.

    The same projection and scorer parameters are shared for every domain,
    retriever, attack family, and query position.  Adversarial heads are used
    only during training and are not part of the deployed score API.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        projection_dim: int = 48,
        representation_dim: int = 64,
        domain_classes: int = 5,
        family_classes: int = 7,
        retriever_classes: int = 3,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
        )
        self.document_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
        )
        # q, pooled d, q*d, |q-d|, and four invariant scalar summaries.
        fusion_dim = 4 * projection_dim + 4
        self.interaction = nn.Sequential(
            nn.Linear(fusion_dim, representation_dim),
            nn.LayerNorm(representation_dim), nn.GELU(), nn.Dropout(0.10),
        )
        self.attack_head = nn.Linear(representation_dim, 1)
        self.domain_head = nn.Linear(representation_dim, int(domain_classes))
        self.family_head = nn.Linear(representation_dim, int(family_classes))
        self.retriever_head = nn.Linear(representation_dim, int(retriever_classes))

    def parameter_count(self, *, deployed_only: bool = False) -> int:
        excluded: set[int] = set()
        if deployed_only:
            for module in (self.domain_head, self.family_head, self.retriever_head):
                excluded.update(id(p) for p in module.parameters())
        return int(sum(p.numel() for p in self.parameters()
                       if p.requires_grad and id(p) not in excluded))

    def encode(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_embeddings.ndim != 2:
            raise ValueError("query_embeddings must have shape [B,D]")
        if document_embeddings.ndim != 3:
            raise ValueError("document_embeddings must have shape [B,K,D]")
        if query_embeddings.shape[0] != document_embeddings.shape[0]:
            raise ValueError("query/document batch mismatch")
        if document_embeddings.shape[1] == 0:
            raise ValueError("empty retrieval is not a valid detector observation")
        if query_embeddings.shape[-1] != self.embedding_dim or document_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("embedding dimension mismatch")
        if not torch.isfinite(query_embeddings).all() or not torch.isfinite(document_embeddings).all():
            raise ValueError("NaN/Inf detector input")
        if document_mask is None:
            document_mask = torch.ones(document_embeddings.shape[:2], dtype=torch.bool,
                                       device=document_embeddings.device)
        if document_mask.shape != document_embeddings.shape[:2]:
            raise ValueError("document mask shape mismatch")
        if bool((document_mask.sum(dim=1) == 0).any()):
            raise ValueError("every query requires at least one retrieved document")

        query = self.query_projection(query_embeddings)
        documents = self.document_projection(document_embeddings)
        logits = torch.einsum("bd,bkd->bk", query, documents) / math.sqrt(self.projection_dim)
        logits = logits.masked_fill(~document_mask.bool(), -1e4)
        attention = torch.softmax(logits, dim=1) * document_mask.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("bk,bkd->bd", attention, documents)

        valid = document_mask.float()
        count = valid.sum(dim=1).clamp_min(1.0)
        mean_logit = (logits.masked_fill(~document_mask.bool(), 0.0) * valid).sum(dim=1) / count
        centered = (logits - mean_logit[:, None]).masked_fill(~document_mask.bool(), 0.0)
        std_logit = torch.sqrt(centered.square().sum(dim=1) / count).clamp_min(0.0)
        entropy = -(attention.clamp_min(1e-12).log() * attention).sum(dim=1)
        max_attention = attention.max(dim=1).values
        fusion = torch.cat((query, pooled, query * pooled, torch.abs(query - pooled),
                            mean_logit[:, None], std_logit[:, None], entropy[:, None],
                            max_attention[:, None]), dim=1)
        return self.interaction(fusion), attention

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: torch.Tensor | None = None,
        *,
        adversarial_strength: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        representation, attention = self.encode(query_embeddings, document_embeddings, document_mask)
        reversed_representation = gradient_reverse(representation, adversarial_strength)
        return {
            "representation": representation,
            "logit": self.attack_head(representation).squeeze(-1),
            "domain_logits": self.domain_head(reversed_representation),
            "family_logits": self.family_head(reversed_representation),
            "retriever_logits": self.retriever_head(reversed_representation),
            "attention": attention,
        }

    def score(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> float:
        """Metadata-free deployed per-query score."""
        self.eval()
        with torch.inference_mode():
            output = self.forward(query_embedding.reshape(1, -1),
                                  document_embeddings.reshape(1, -1, self.embedding_dim))
        return float(output["logit"].item())


def empirical_upper_p(value: float, normal_reference: Sequence[float]) -> float:
    reference = np.asarray(normal_reference, dtype=np.float64).reshape(-1)
    if reference.size == 0 or not np.isfinite(reference).all() or not math.isfinite(float(value)):
        raise ValueError("finite score and nonempty finite normal reference required")
    return float((1 + np.count_nonzero(reference >= float(value))) / (reference.size + 1))


def cct_p(pvalues: Sequence[float]) -> float:
    p = np.sort(np.clip(np.asarray(pvalues, dtype=np.float64), 1e-12, 1 - 1e-12))
    if p.size == 0 or not np.isfinite(p).all():
        raise ValueError("nonempty finite p-values required")
    statistic = float(np.mean(np.tan(np.pi * (0.5 - p))))
    return float(np.clip(0.5 - np.arctan(statistic) / np.pi, 0.0, 1.0))


def aggregate_pvalues(pvalues: Sequence[float], method: str) -> float:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.size == 0:
        return 1.0
    if method == "MINP":
        return float(np.min(p))
    if method == "CCT":
        return cct_p(p)
    raise ValueError(f"unknown symmetric aggregator: {method}")


def fixed_auxiliary_score(semantic_score: float, auxiliary_score: float,
                          semantic_normal: Sequence[float], auxiliary_normal: Sequence[float]) -> float:
    """Equal fixed-weight standardized fusion; no auxiliary coefficient is fit."""
    sref = np.asarray(semantic_normal, dtype=np.float64)
    aref = np.asarray(auxiliary_normal, dtype=np.float64)
    if min(sref.size, aref.size) == 0 or not np.isfinite([semantic_score, auxiliary_score]).all():
        raise ValueError("finite auxiliary fusion inputs required")
    sz = (float(semantic_score) - float(sref.mean())) / max(float(sref.std()), 1e-6)
    az = (float(auxiliary_score) - float(aref.mean())) / max(float(aref.std()), 1e-6)
    return float((sz + az) / math.sqrt(2.0))


@dataclass
class SymmetricOnlineDetector:
    """Stateful MinP/CCT wrapper with one budget-independent alarm threshold."""

    normal_reference: np.ndarray
    method: str
    threshold: float
    _pvalues: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.normal_reference = np.asarray(self.normal_reference, dtype=np.float64).reshape(-1)
        if self.method not in {"MINP", "CCT"}:
            raise ValueError("method must be MINP or CCT")
        if self.normal_reference.size == 0 or not np.isfinite(self.normal_reference).all():
            raise ValueError("normal_reference must be finite and nonempty")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")

    def reset(self) -> None:
        self._pvalues.clear()

    def update(self, per_query_score: float) -> dict[str, float | int | bool]:
        p = empirical_upper_p(float(per_query_score), self.normal_reference)
        self._pvalues.append(p)
        anomaly = -aggregate_pvalues(self._pvalues, self.method)
        return {"query_count": len(self._pvalues), "p_value": p,
                "aggregate_score": anomaly, "alarm": bool(anomaly > self.threshold)}

    @property
    def final_score(self) -> float:
        return -aggregate_pvalues(self._pvalues, self.method)


__all__ = [
    "FORBIDDEN_INFERENCE_FIELDS", "QueryOnlyScorer", "SemanticInteractionScorer",
    "SymmetricOnlineDetector", "aggregate_pvalues", "cct_p", "empirical_upper_p",
    "fixed_auxiliary_score", "gradient_reverse",
]
