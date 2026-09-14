"""Exp50 Relative Retrieval Anomaly Detector (RRAD).

The public detector API deliberately accepts only retrieval observations.  A
caller supplies a full per-query corpus similarity vector and the resulting
document/parent identifiers; query text, embeddings, labels, domain names,
retriever names and budgets are not accepted by any scoring method.

All accumulators are commutative.  Consequently the final score for a fixed
set of observed queries is invariant to query permutation, while first-alert
delay may still depend on arrival order.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from .canonical_mirabel import canonical_mirabel_from_full_scores

EPS = 1e-12


@dataclass(frozen=True)
class TailStatistic:
    value: float
    mad_zero: bool
    statistic: str


@dataclass(frozen=True)
class RRADOutput:
    """Stable online output returned by :meth:`RRADDetector.update`."""

    prefix_index: int
    tail_statistic: float
    tail_p_value: float
    persistence_statistic: float | None
    persistence_p_value: float | None
    aggregate_score: float
    alarm: bool


def robust_mad_tail(scores: Sequence[float], epsilon: float = 1e-6) -> TailStatistic:
    """Robust top-vs-bulk tail statistic, excluding exactly one maximum."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        raise ValueError("scores must be a finite one-dimensional full-corpus vector (n>=3)")
    # Deterministic tie handling: remove the first maximum in stable order.
    top_i = int(np.argmax(x))
    bulk = np.delete(x, top_i)
    med = float(np.median(bulk))
    mad = float(np.median(np.abs(bulk - med)))
    denom = 1.4826 * mad + float(epsilon)
    return TailStatistic(float((x[top_i] - med) / denom), mad <= float(epsilon), "TAIL_ROBUST_MAD")


def gumbel_tail(scores: Sequence[float]) -> TailStatistic:
    """Frozen canonical MIRABEL Gumbel margin on a full corpus vector."""
    x = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    stat = canonical_mirabel_from_full_scores(x)
    return TailStatistic(float(stat.margin), False, "TAIL_GUMBEL")


def empirical_upper_p(value: float, normal_values: Sequence[float]) -> float:
    """Finite-sample upper-tail conformal p with deterministic ties."""
    ref = np.asarray(normal_values, dtype=np.float64)
    if ref.ndim != 1 or ref.size == 0 or not np.isfinite(ref).all():
        raise ValueError("normal reference must be a non-empty finite vector")
    return float(np.clip((1.0 + np.count_nonzero(ref >= float(value))) / (ref.size + 1.0), EPS, 1.0 - EPS))


def cct_p(values: Sequence[float], clip: float = 1e-12) -> float:
    """Equal-weight Cauchy combination of p-values, with stable clipping."""
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("p-values must be a non-empty finite vector")
    q = np.clip(p, float(clip), 1.0 - float(clip))
    tangent = np.tan(np.pi * (0.5 - q))
    t = float(np.mean(tangent))
    return float(np.clip(0.5 - np.arctan(t) / np.pi, 0.0, 1.0))


@dataclass
class RRADDetector:
    """Online RRAD state.

    `normal_tail_reference` and `normal_persistence_reference` are fitted
    solely from normal calibration sessions.  They are immutable numpy arrays
    after construction.  The state stores only query-level retrieval
    statistics and top identifiers, never user text or labels.
    """

    normal_tail_reference: np.ndarray
    statistic: str = "TAIL_ROBUST_MAD"
    aggregator: str = "R2"
    normal_persistence_reference: np.ndarray | None = None
    threshold: float | None = None
    top_k: int = 5
    epsilon: float = 1e-6
    _tail_ps: list[float] = field(default_factory=list, init=False, repr=False)
    _persistence_ps: list[float] = field(default_factory=list, init=False, repr=False)
    _documents: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _parents: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _max_score: float = field(default=-math.inf, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)
    _mad_zero_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        ref = np.asarray(self.normal_tail_reference, dtype=np.float64)
        if ref.ndim != 1 or ref.size == 0 or not np.isfinite(ref).all():
            raise ValueError("normal_tail_reference must be non-empty and finite")
        self.normal_tail_reference = ref.copy()
        if self.normal_persistence_reference is None:
            self.normal_persistence_reference = np.asarray([1.0], dtype=np.float64)
        else:
            p = np.asarray(self.normal_persistence_reference, dtype=np.float64)
            if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all():
                raise ValueError("normal_persistence_reference must be non-empty and finite")
            self.normal_persistence_reference = p.copy()
        if self.statistic not in {"TAIL_ROBUST_MAD", "TAIL_GUMBEL"}:
            raise ValueError("unknown tail statistic")
        if self.aggregator not in {"R0", "R1", "R2", "R3"}:
            raise ValueError("unknown RRAD aggregator")

    def _tail(self, scores: Sequence[float]) -> TailStatistic:
        return robust_mad_tail(scores, self.epsilon) if self.statistic == "TAIL_ROBUST_MAD" else gumbel_tail(scores)

    def reset(self, session_id: str | None = None) -> None:
        """Reset online state; ``session_id`` is accepted only for API parity.

        The identifier is intentionally not retained or used as a feature.
        """
        self._tail_ps.clear(); self._persistence_ps.clear()
        self._documents.clear(); self._parents.clear()
        self._max_score = -math.inf; self._n = 0; self._mad_zero_count = 0

    def update(self, corpus_scores: Sequence[float], top_document_ids: Sequence[str],
               top_parent_document_ids: Sequence[str] | None = None) -> RRADOutput:
        """Consume one query retrieval result and return current prefix score."""
        scores = np.asarray(corpus_scores, dtype=np.float64)
        docs = [str(v) for v in top_document_ids]
        if scores.ndim != 1 or scores.size < 3 or len(docs) < 1:
            raise ValueError("full corpus scores and at least one top document are required")
        if len(docs) > self.top_k:
            docs = docs[: self.top_k]
        parents = [str(v) for v in (top_parent_document_ids if top_parent_document_ids is not None else docs)]
        parents = parents[:len(docs)]
        tail = self._tail(scores)
        p_tail = empirical_upper_p(tail.value, self.normal_tail_reference)
        self._tail_ps.append(p_tail)
        self._documents.update(set(docs)); self._parents.update(set(parents))
        self._n += 1; self._mad_zero_count += int(tail.mad_zero)
        persistence_value = float(max(max(self._documents.values()), max(self._parents.values())))
        p_persist = empirical_upper_p(persistence_value, self.normal_persistence_reference)
        self._persistence_ps.append(p_persist)
        score = self.score
        self._max_score = max(self._max_score, score)
        return RRADOutput(
            prefix_index=self._n,
            tail_statistic=float(tail.value),
            tail_p_value=float(p_tail),
            persistence_statistic=float(persistence_value),
            persistence_p_value=float(p_persist),
            aggregate_score=float(score),
            alarm=bool(self.threshold is not None and score > float(self.threshold)),
        )

    @property
    def score(self) -> float:
        if not self._tail_ps:
            return 0.0
        minp = float(min(self._tail_ps))
        if self.aggregator == "R0":
            return -minp
        # Sort before the floating-point reduction.  CCT is mathematically
        # symmetric, but sorting also makes the final binary result invariant
        # to arrival order at the 1e-12 audit tolerance.
        tail_cct = cct_p(sorted(self._tail_ps))
        if self.aggregator == "R1":
            return -tail_cct
        # A prefix p-value is retained for delay diagnostics, but the final
        # persistence component is computed from the commutative final maximum
        # recurrence count.  Using all prefix p-values would make the final
        # statistic order-dependent even though the underlying set is not.
        final_recurrence = max(max(self._documents.values()), max(self._parents.values()))
        final_persist_p = empirical_upper_p(float(final_recurrence), self.normal_persistence_reference)
        persist_cct = cct_p((final_persist_p,))
        if self.aggregator == "R2":
            return -cct_p((tail_cct, persist_cct))
        # R3 is intentionally a tiny fixed linear control, not a final model.
        return float(-0.5 * tail_cct - 0.3 * persist_cct - 0.2 * minp)

    @property
    def decision(self) -> bool:
        return self.threshold is not None and self.score > float(self.threshold)  # strict >

    @property
    def diagnostics(self) -> dict[str, float | int | str]:
        return {"n_queries": self._n, "mad_zero_count": self._mad_zero_count,
                "unique_documents": len(self._documents), "unique_parents": len(self._parents),
                "score": self.score, "threshold": self.threshold if self.threshold is not None else math.nan,
                "aggregator": self.aggregator, "statistic": self.statistic}


def session_scores(query_scores: Sequence[Sequence[float]], top_ids: Sequence[Sequence[str]],
                   tail_reference: Sequence[float], *, statistic: str, aggregator: str,
                   persistence_reference: Sequence[float] | None = None) -> np.ndarray:
    detector = RRADDetector(np.asarray(tail_reference), statistic=statistic, aggregator=aggregator,
                            normal_persistence_reference=None if persistence_reference is None else np.asarray(persistence_reference))
    return np.asarray([detector.update(scores, ids).aggregate_score for scores, ids in zip(query_scores, top_ids)], dtype=np.float64)


def final_permutation_invariant(query_scores: Sequence[Sequence[float]], top_ids: Sequence[Sequence[str]],
                                tail_reference: Sequence[float], *, statistic: str, aggregator: str,
                                repetitions: int = 100, seed: int = 20260815) -> dict[str, object]:
    base = list(zip(query_scores, top_ids))
    rng = np.random.default_rng(seed)
    values, decisions = [], []
    for _ in range(int(repetitions)):
        perm = rng.permutation(len(base)); det = RRADDetector(np.asarray(tail_reference), statistic=statistic, aggregator=aggregator)
        for i in perm: det.update(base[i][0], base[i][1])
        values.append(det.score); decisions.append(det.decision)
    arr = np.asarray(values, dtype=np.float64)
    return {"repetitions": int(repetitions), "score_std": float(arr.std()), "score_range": float(arr.max() - arr.min()),
            "decision_agreement": float(np.mean(np.asarray(decisions) == decisions[0])) if decisions else 1.0,
            "passed": bool(arr.std() <= 1e-12 and (arr.max() - arr.min()) <= 1e-10 and len(set(decisions)) <= 1)}


__all__ = ["TailStatistic", "RRADOutput", "RRADDetector", "robust_mad_tail", "gumbel_tail", "empirical_upper_p", "cct_p", "session_scores", "final_permutation_invariant"]
