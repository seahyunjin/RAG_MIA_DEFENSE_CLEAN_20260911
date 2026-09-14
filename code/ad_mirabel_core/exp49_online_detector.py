"""Shared online detector primitives for Exp49.

The inference path deliberately has no attack-family, domain, retriever, or
query-budget argument.  Training may use group labels for robust objectives,
but the deployed scorer receives only query text, retrieved text, retrieval
scores, and session-local recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn


FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "attack_family", "attack_method", "member_label", "query_budget",
    "remaining_queries", "domain", "domain_id", "corpus_id", "retriever",
    "retriever_id", "dataset_row_index", "source_experiment_id",
    "target_document_membership", "prompt_template_id",
})


@dataclass(frozen=True)
class DetectorOutput:
    prefix_index: int
    raw_nonconformity: float
    calibrated_score: float
    running_statistic: float
    alarm: bool


class EmpiricalCDF:
    """Finite-sample empirical CDF fitted only from normal calibration scores."""

    def __init__(self, values: Sequence[float], epsilon: float = 1e-4):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(array) or not np.isfinite(array).all():
            raise ValueError("EmpiricalCDF requires nonempty finite normal scores")
        if not 0.0 < float(epsilon) < 0.5:
            raise ValueError("epsilon must be in (0, .5)")
        self.values = np.sort(array)
        self.epsilon = float(epsilon)

    def percentile(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        ranks = np.searchsorted(self.values, array, side="right")
        result = (ranks + 0.5) / (len(self.values) + 1.0)
        return np.clip(result, self.epsilon, 1.0 - self.epsilon)

    def state_dict(self) -> dict[str, object]:
        return {"values": self.values.tolist(), "epsilon": self.epsilon,
                "fit_source": "normal_calibration_only"}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EmpiricalCDF":
        return cls(state["values"], float(state["epsilon"]))


def running_max_threshold(normal_prefix_scores: Sequence[Sequence[float]], alpha: float) -> float:
    """One budget-independent session threshold using each Q30 session maximum."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    maxima = []
    for values in normal_prefix_scores:
        array = np.asarray(values, dtype=float).reshape(-1)
        if not len(array) or not np.isfinite(array).all():
            raise ValueError("every normal session must contain finite prefix scores")
        maxima.append(float(np.max(array)))
    if not maxima:
        raise ValueError("at least one normal session is required")
    return float(np.quantile(np.asarray(maxima), 1.0 - float(alpha), method="higher"))


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(value, float(strength))


class SharedOnlineScorer(nn.Module):
    """Retriever-agnostic query/document fusion followed by one causal GRU."""

    def __init__(
        self,
        embedding_dim: int = 768,
        projection_dim: int = 48,
        hidden_dim: int = 64,
        top_k: int = 5,
        domain_classes: int = 4,
        family_classes: int = 7,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.query_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim), nn.LayerNorm(self.projection_dim), nn.GELU(),
        )
        self.document_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim), nn.LayerNorm(self.projection_dim), nn.GELU(),
        )
        fusion_dim = self.projection_dim * 4 + self.top_k + 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU(),
            nn.Dropout(0.1),
        )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.attack_head = nn.Linear(self.hidden_dim, 1)
        self.domain_head = nn.Linear(self.hidden_dim, int(domain_classes))
        self.family_head = nn.Linear(self.hidden_dim, int(family_classes))
        self.register_buffer("normal_center", torch.zeros(self.hidden_dim), persistent=True)
        self.register_buffer("center_initialized", torch.tensor(False), persistent=True)

    def parameter_count(self, *, inference_only: bool = False) -> int:
        excluded = {id(value) for module in (self.domain_head, self.family_head)
                    for value in module.parameters()} if inference_only else set()
        return int(sum(value.numel() for value in self.parameters()
                       if value.requires_grad and id(value) not in excluded))

    def encode_queries(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        score_percentiles: torch.Tensor,
        document_mask: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused per-query vectors and attention weights.

        Shapes are ``query=[B,T,D]``, ``documents=[B,T,K,D]``, scores/mask
        ``[B,T,K]``.  Document order is not encoded; masked softmax pooling is
        therefore permutation invariant.
        """
        if query_embeddings.ndim != 3 or document_embeddings.ndim != 4:
            raise ValueError("invalid query/document tensor rank")
        if score_percentiles.shape != document_mask.shape:
            raise ValueError("score percentile and document mask shapes differ")
        if document_embeddings.shape[:3] != score_percentiles.shape:
            raise ValueError("document and score shapes differ")
        if document_embeddings.shape[2] != self.top_k:
            raise ValueError(f"expected top_k={self.top_k}")
        query = self.query_projection(query_embeddings)
        documents = self.document_projection(document_embeddings)
        logits = torch.einsum("btd,btkd->btk", query, documents) / math.sqrt(self.projection_dim)
        logits = logits.masked_fill(~document_mask.bool(), -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * document_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("btk,btkd->btd", attention, documents)
        score_values = score_percentiles.masked_fill(~document_mask.bool(), 0.0)
        # Rank coordinates are sorted from the values themselves, so jointly
        # permuting a document, its score, and its mask cannot change z_t.
        sorted_scores = torch.sort(score_values, dim=-1, descending=True).values
        count = document_mask.sum(dim=-1).clamp_min(1).float()
        mean = score_values.sum(dim=-1) / count
        centered = (score_values - mean.unsqueeze(-1)) * document_mask.float()
        std = torch.sqrt(centered.square().sum(dim=-1) / count).clamp_min(0.0)
        entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1)
        fusion = torch.cat((query, pooled, query * pooled, torch.abs(query - pooled),
                            sorted_scores, mean.unsqueeze(-1), std.unsqueeze(-1), entropy.unsqueeze(-1)), dim=-1)
        return self.fusion(fusion), attention

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        score_percentiles: torch.Tensor,
        document_mask: torch.Tensor,
        lengths: torch.Tensor,
        hidden: torch.Tensor | None = None,
        adversarial_strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        fused, attention = self.encode_queries(
            query_embeddings, document_embeddings, score_percentiles, document_mask, lengths,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            fused, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
        )
        encoded, next_hidden = self.gru(packed, hidden)
        states, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=query_embeddings.shape[1],
        )
        reversed_states = gradient_reverse(states, adversarial_strength)
        return {
            "states": states,
            "attack_logits": self.attack_head(states).squeeze(-1),
            "domain_logits": self.domain_head(reversed_states),
            "family_logits": self.family_head(reversed_states),
            "attention": attention,
            "hidden": next_hidden,
        }

    def nonconformity(self, output: dict[str, torch.Tensor], objective: str) -> torch.Tensor:
        if objective == "normality":
            return (output["states"] - self.normal_center.view(1, 1, -1)).square().sum(dim=-1)
        if objective in {"erm", "groupdro", "invariant"}:
            return output["attack_logits"]
        raise ValueError(f"unknown objective: {objective}")


class OnlineRAGMIADetector:
    """Stateful public API with one update rule from Q1 through Q30."""

    def __init__(
        self,
        scorer: SharedOnlineScorer,
        text_encoder: Callable[[Sequence[str]], np.ndarray],
        normal_score_cdf: EmpiricalCDF,
        threshold: float,
        *,
        objective: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.scorer = scorer.eval().to(device)
        self.text_encoder = text_encoder
        self.normal_score_cdf = normal_score_cdf
        self.threshold = float(threshold)
        self.objective = str(objective)
        self.device = torch.device(device)
        self._states: dict[str, dict[str, object]] = {}

    def reset(self, session_id: str) -> None:
        self._states[str(session_id)] = {"hidden": None, "prefix": 0, "running": -math.inf}

    def update(
        self,
        query_text: str,
        retrieved_documents: list[str],
        retrieval_scores: list[float] | None,
        *,
        session_id: str = "default",
        **metadata: object,
    ) -> DetectorOutput:
        forbidden = FORBIDDEN_INFERENCE_FIELDS.intersection(metadata)
        if forbidden:
            raise ValueError(f"forbidden inference metadata: {sorted(forbidden)}")
        key = str(session_id)
        if key not in self._states:
            self.reset(key)
        state = self._states[key]
        documents = list(retrieved_documents[: self.scorer.top_k])
        if not documents:
            documents = [""]
        scores = list(retrieval_scores or [0.0] * len(documents))[: len(documents)]
        if len(scores) != len(documents):
            raise ValueError("retrieval scores and documents must align")
        texts = [str(query_text), *map(str, documents)]
        embeddings = np.asarray(self.text_encoder(texts), dtype=np.float32)
        if embeddings.shape != (len(texts), self.scorer.embedding_dim):
            raise ValueError("text encoder returned an unexpected shape")
        k = self.scorer.top_k
        docs = np.zeros((1, 1, k, self.scorer.embedding_dim), dtype=np.float32)
        mask = np.zeros((1, 1, k), dtype=bool)
        percentiles = np.zeros((1, 1, k), dtype=np.float32)
        docs[0, 0, :len(documents)] = embeddings[1:]
        mask[0, 0, :len(documents)] = True
        percentiles[0, 0, :len(documents)] = self.normal_score_cdf.percentile(scores)
        with torch.inference_mode():
            output = self.scorer(
                torch.from_numpy(embeddings[:1]).view(1, 1, -1).to(self.device),
                torch.from_numpy(docs).to(self.device),
                torch.from_numpy(percentiles).to(self.device),
                torch.from_numpy(mask).to(self.device),
                torch.ones(1, dtype=torch.long, device=self.device),
                hidden=state["hidden"],
            )
            raw = float(self.scorer.nonconformity(output, self.objective)[0, 0].item())
            state["hidden"] = output["hidden"]
        state["prefix"] = int(state["prefix"]) + 1
        state["running"] = max(float(state["running"]), raw)
        # The score CDF is a retrieval-coordinate calibration.  Detector
        # nonconformity remains on its learned scale and is thresholded by the
        # Q30 running-maximum calibration.
        return DetectorOutput(
            prefix_index=int(state["prefix"]), raw_nonconformity=raw,
            calibrated_score=raw, running_statistic=float(state["running"]),
            alarm=bool(float(state["running"]) > self.threshold),
        )


def sequence_mask(lengths: torch.Tensor, maximum: int | None = None) -> torch.Tensor:
    maximum = int(maximum or lengths.max().item())
    return torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def groupdro_loss(losses: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Worst observed group mean, without creating a family-specific head."""
    if losses.ndim != 1 or groups.ndim != 1 or len(losses) != len(groups):
        raise ValueError("losses and groups must be aligned vectors")
    values = [losses[groups == group].mean() for group in torch.unique(groups) if torch.any(groups == group)]
    if not values:
        raise ValueError("at least one group is required")
    return torch.stack(values).max()


def assert_shared_inference_signature() -> None:
    names = set(OnlineRAGMIADetector.update.__annotations__)
    if names.intersection(FORBIDDEN_INFERENCE_FIELDS):
        raise AssertionError("forbidden metadata leaked into update signature")
