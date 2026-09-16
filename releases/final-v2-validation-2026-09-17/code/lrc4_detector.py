"""Minimal reference implementation of the frozen LRC-4 detector/action.

This module contains no learned classifier and does not load a retriever or
generator.  Pass it the similarities and document IDs returned by a RAG
retriever.  The publication evaluation uses a threshold frozen from benign
calibration queries and a strict ``score > threshold`` decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np


T = TypeVar("T")


def lrc4_score(top_scores: Sequence[float]) -> float:
    """Return s1 - mean(s1..s4) for retrieval-ranked similarities."""
    if len(top_scores) < 4:
        raise ValueError("LRC-4 requires at least four retrieval scores")
    values = np.asarray(top_scores[:4], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("retrieval scores must be finite")
    if np.any(values[:-1] < values[1:]):
        raise ValueError("scores must be supplied in descending retrieval order")
    return float(values[0] - values.mean())


def calibrate_threshold(
    benign_top_scores: Sequence[Sequence[float]], target_fpr: float = 0.025
) -> float:
    """Fit the operating point from benign queries only.

    ``method='higher'`` and the strict comparison below provide a deterministic
    empirical upper-tail threshold without using any attack examples.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between 0 and 1")
    scores = np.asarray([lrc4_score(row) for row in benign_top_scores])
    if scores.size == 0:
        raise ValueError("benign calibration data must not be empty")
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def alarm(top_scores: Sequence[float], threshold: float) -> bool:
    """Apply the frozen strict-greater-than decision rule."""
    return lrc4_score(top_scores) > threshold


@dataclass(frozen=True)
class DefenseDecision:
    score: float
    threshold: float
    alarm: bool
    hidden_document_id: str | None
    context_document_ids: tuple[str, ...]


def defend_retrieval(
    ranked_document_ids: Sequence[str],
    ranked_scores: Sequence[float],
    threshold: float,
    context_size: int = 4,
) -> DefenseDecision:
    """Hide current-turn rank 1 on alarm and backfill deterministically."""
    if len(ranked_document_ids) != len(ranked_scores):
        raise ValueError("document IDs and scores must have equal length")
    if len(ranked_document_ids) < context_size:
        raise ValueError("not enough retrieved documents for the requested context")
    score = lrc4_score(ranked_scores)
    is_alarm = score > threshold
    hidden = ranked_document_ids[0] if is_alarm else None
    start = 1 if is_alarm else 0
    context = tuple(ranked_document_ids[start : start + context_size])
    if len(context) != context_size:
        raise ValueError("rank-1 hide requires one backfill document")
    return DefenseDecision(score, threshold, is_alarm, hidden, context)
