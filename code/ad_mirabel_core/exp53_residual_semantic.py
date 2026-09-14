"""Exp53 metadata-free residual semantic detector primitives.

The deployed detector consumes a frozen query embedding, the unordered set of
native top-k document embeddings, and two normal-only empirical references.
It never accepts attack-family, domain, retriever, membership, query-budget or
turn metadata.  Query-only and document-only nuisance logits are subtracted
with stop-gradient from the joint interaction logit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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


def _validate_query(query: torch.Tensor, embedding_dim: int) -> None:
    if query.ndim != 2 or query.shape[-1] != embedding_dim:
        raise ValueError("query_embeddings must have shape [B,D]")
    if not torch.isfinite(query).all():
        raise ValueError("NaN/Inf detector input")


def _validate_documents(documents: torch.Tensor, embedding_dim: int) -> None:
    if documents.ndim != 3 or documents.shape[-1] != embedding_dim:
        raise ValueError("document_embeddings must have shape [B,K,D]")
    if documents.shape[1] == 0:
        raise ValueError("empty retrieval is not a valid detector observation")
    if not torch.isfinite(documents).all():
        raise ValueError("NaN/Inf detector input")


def _mask(documents: torch.Tensor, document_mask: torch.Tensor | None) -> torch.Tensor:
    if document_mask is None:
        document_mask = torch.ones(documents.shape[:2], dtype=torch.bool,
                                   device=documents.device)
    if document_mask.shape != documents.shape[:2]:
        raise ValueError("document mask shape mismatch")
    if bool((document_mask.sum(dim=1) == 0).any()):
        raise ValueError("every query requires at least one retrieved document")
    return document_mask.bool()


class QueryNuisanceScorer(nn.Module):
    """Query-marginal shortcut branch (R0)."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_query(query_embeddings, self.embedding_dim)
        representation = self.network[:-1](query_embeddings)
        return {"representation": representation,
                "logit": self.network[-1](representation).squeeze(-1)}


class DocumentNuisanceScorer(nn.Module):
    """DeepSets document-marginal shortcut branch (R1)."""

    def __init__(self, embedding_dim: int = 768, projection_dim: int = 24,
                 hidden_dim: int = 32) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.document_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
        )
        self.network = nn.Sequential(
            nn.Linear(2 * projection_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, document_embeddings: torch.Tensor,
                document_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _validate_documents(document_embeddings, self.embedding_dim)
        mask = _mask(document_embeddings, document_mask)
        projected = self.document_projection(document_embeddings)
        weights = mask.float()
        mean = (projected * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        maximum = projected.masked_fill(~mask[..., None], -1e4).max(1).values
        representation = self.network[:-1](torch.cat((mean, maximum), dim=1))
        return {"representation": representation,
                "logit": self.network[-1](representation).squeeze(-1)}


class JointInteractionScorer(nn.Module):
    """Small query-conditioned set scorer (R2).

    Direct query-only and document-only vectors are deliberately omitted from
    the final feature vector.  It contains element-wise interaction terms and
    invariant attention summaries only.
    """

    def __init__(self, embedding_dim: int = 768, projection_dim: int = 32,
                 representation_dim: int = 48) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
        )
        self.document_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
        )
        self.interaction = nn.Sequential(
            nn.Linear(2 * projection_dim + 4, representation_dim),
            nn.LayerNorm(representation_dim), nn.GELU(), nn.Dropout(0.10),
        )
        self.head = nn.Linear(representation_dim, 1)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                document_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _validate_query(query_embeddings, self.embedding_dim)
        _validate_documents(document_embeddings, self.embedding_dim)
        if query_embeddings.shape[0] != document_embeddings.shape[0]:
            raise ValueError("query/document batch mismatch")
        mask = _mask(document_embeddings, document_mask)
        query = self.query_projection(query_embeddings)
        documents = self.document_projection(document_embeddings)
        similarities = torch.einsum("bd,bkd->bk", query, documents) / math.sqrt(self.projection_dim)
        masked = similarities.masked_fill(~mask, -1e4)
        attention = torch.softmax(masked, dim=1) * mask.float()
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("bk,bkd->bd", attention, documents)
        valid = mask.float(); count = valid.sum(1).clamp_min(1)
        mean_similarity = (similarities.masked_fill(~mask, 0) * valid).sum(1) / count
        centered = (similarities - mean_similarity[:, None]).masked_fill(~mask, 0)
        std_similarity = torch.sqrt(centered.square().sum(1) / count).clamp_min(0)
        entropy = -(attention.clamp_min(1e-12).log() * attention).sum(1)
        maximum = attention.max(1).values
        features = torch.cat((query * pooled, torch.abs(query - pooled),
                              mean_similarity[:, None], std_similarity[:, None],
                              entropy[:, None], maximum[:, None]), dim=1)
        representation = self.interaction(features)
        return {"representation": representation,
                "logit": self.head(representation).squeeze(-1), "attention": attention}


def residual_logit(joint_logit: torch.Tensor, query_logit: torch.Tensor,
                   document_logit: torch.Tensor, intercept: torch.Tensor | float) -> torch.Tensor:
    """Residual logit with nuisance branches removed by stop-gradient."""
    return joint_logit - query_logit.detach() - document_logit.detach() + intercept


class ResidualSemanticScorer(nn.Module):
    """Metadata-free deployable R3 scorer containing three shared branches."""

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.query_nuisance = QueryNuisanceScorer(embedding_dim)
        self.document_nuisance = DocumentNuisanceScorer(embedding_dim)
        self.joint = JointInteractionScorer(embedding_dim)
        self.intercept = nn.Parameter(torch.zeros(()))

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                document_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        query = self.query_nuisance(query_embeddings)
        document = self.document_nuisance(document_embeddings, document_mask)
        joint = self.joint(query_embeddings, document_embeddings, document_mask)
        residual = residual_logit(joint["logit"], query["logit"], document["logit"], self.intercept)
        return {"query_logit": query["logit"], "document_logit": document["logit"],
                "joint_logit": joint["logit"], "residual_logit": residual,
                "query_representation": query["representation"],
                "document_representation": document["representation"],
                "joint_representation": joint["representation"]}

    def score(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> float:
        self.eval()
        with torch.inference_mode():
            result = self.forward(query_embedding.reshape(1, -1),
                                  document_embeddings.reshape(1, -1, self.embedding_dim))
        return float(result["residual_logit"].item())

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


def empirical_upper_p(value: float, normal_reference: Sequence[float]) -> float:
    reference = np.asarray(normal_reference, dtype=np.float64).reshape(-1)
    if reference.size == 0 or not np.isfinite(reference).all() or not math.isfinite(float(value)):
        raise ValueError("finite score and nonempty finite normal reference required")
    return float((1 + np.count_nonzero(reference >= float(value))) / (reference.size + 1))


def cct_p(pvalues: Sequence[float]) -> float:
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0 or not np.isfinite(p).all():
        raise ValueError("nonempty finite p-values required")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    statistic = float(np.mean(np.tan(np.pi * (0.5 - p))))
    return float(np.clip(0.5 - np.arctan(statistic) / np.pi, 0.0, 1.0))


def fuse_pvalues(p_b3: float, p_semantic: float, method: str) -> float:
    values = [float(p_b3), float(p_semantic)]
    if method == "CCT":
        return cct_p(values)
    if method == "MINP":
        return float(min(values))
    raise ValueError("fusion method must be CCT or MINP")


def aggregate_pvalues(pvalues: Sequence[float], method: str) -> float:
    values = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1.0
    if method == "MINP":
        return float(np.min(values))
    if method == "CCT":
        return cct_p(values)
    raise ValueError("aggregation method must be MINP or CCT")


def document_fold(document_key: str, folds: int = 5, seed: int = 20260815) -> int:
    """Stable document-level cross-fitting assignment."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    payload = f"{seed}|{document_key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(folds)


def deterministic_matched_indices(features: np.ndarray, source_ids: Sequence[str]) -> np.ndarray:
    """Label-free nearest nuisance match with a different source document.

    The signature intentionally contains no label or family vector.  Ties are
    resolved by the candidate row index, making resume and audit deterministic.
    """
    values = np.asarray(features, dtype=np.float64)
    sources = np.asarray(list(source_ids), dtype=object)
    if values.ndim != 2 or len(values) != len(sources) or not np.isfinite(values).all():
        raise ValueError("finite [N,F] features and N source IDs required")
    scale = np.maximum(values.std(axis=0), 1e-8)
    standardized = (values - values.mean(axis=0)) / scale
    output = np.full(len(values), -1, dtype=np.int64)
    for row in range(len(values)):
        allowed = np.flatnonzero(sources != sources[row])
        if not len(allowed):
            raise ValueError("matched shuffle requires another source document")
        distance = np.square(standardized[allowed] - standardized[row]).sum(axis=1)
        output[row] = int(allowed[np.lexsort((allowed, distance))[0]])
    return output


@dataclass
class SymmetricOnlineDetector:
    """One threshold, query-budget-independent MinP/CCT session detector."""

    method: str
    threshold: float
    _pvalues: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"MINP", "CCT"}:
            raise ValueError("method must be MINP or CCT")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")

    def update_pvalue(self, pvalue: float) -> dict[str, float | int | bool]:
        if not math.isfinite(float(pvalue)) or not 0 <= float(pvalue) <= 1:
            raise ValueError("pvalue must be finite and in [0,1]")
        self._pvalues.append(float(pvalue))
        anomaly = -math.log10(max(aggregate_pvalues(self._pvalues, self.method), 1e-300))
        return {"query_count": len(self._pvalues), "aggregate_score": anomaly,
                "alarm": bool(anomaly > self.threshold)}

    @property
    def final_score(self) -> float:
        return -math.log10(max(aggregate_pvalues(self._pvalues, self.method), 1e-300))


__all__ = [
    "FORBIDDEN_INFERENCE_FIELDS", "QueryNuisanceScorer", "DocumentNuisanceScorer",
    "JointInteractionScorer", "ResidualSemanticScorer", "residual_logit",
    "empirical_upper_p", "cct_p", "fuse_pvalues", "aggregate_pvalues",
    "document_fold", "deterministic_matched_indices", "SymmetricOnlineDetector",
]
