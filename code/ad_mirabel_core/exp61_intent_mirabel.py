"""Minimal, auditable components for Exp61 Intent-Aware Mirabel.

This module deliberately contains no historical detector or session feature.
The only retrieval statistic is the frozen canonical Mirabel implementation;
the proposed policy adds one scalar intent alarm and hide/backfill.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.canonical_mirabel import (
    DEFAULT_CONFIDENCE,
    MIRABEL_FORMULA_VERSION,
    canonical_mirabel_from_full_scores,
)


POLICIES = ("D0_NO_DEFENSE", "D1_ORIGINAL_MIRABEL", "D2_INTENT_MIRABEL", "D3_OR_CONTROL")


def higher_threshold(normal_scores: Sequence[float], target_fpr: float = 0.01) -> float:
    """Lock a strict-``>`` threshold with an empirical upper-tail target.

    ``method='higher'`` is conservative at a finite sample size.  Test labels
    and attack samples are never accepted by this function.
    """

    values = np.asarray(normal_scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite, non-empty normal scores required")
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be strictly between zero and one")
    return float(np.quantile(values, 1.0 - float(target_fpr), method="higher"))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("valid binomial counts required")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def harmonic_member_nonmember(member_tpr: float, nonmember_tpr: float) -> float:
    left, right = float(member_tpr), float(nonmember_tpr)
    if left < 0.0 or right < 0.0 or left > 1.0 or right > 1.0:
        raise ValueError("TPRs must be probabilities")
    return 0.0 if left + right == 0.0 else 2.0 * left * right / (left + right)


@dataclass(frozen=True)
class RetrievalDecision:
    policy: str
    intent_alarm: bool
    mirabel_alarm: bool
    candidate_document: str
    hidden_document: str | None
    before: tuple[str, ...]
    after: tuple[str, ...]
    backfilled: bool
    mirabel_margin: float
    mirabel_threshold: float


def hide_and_backfill(
    ranked_document_ids: Sequence[str],
    *,
    top_k: int,
    hidden_document: str | None,
) -> tuple[tuple[str, ...], bool]:
    """Remove one document before generation and deterministically backfill.

    The caller must provide at least ``k+1`` ranked documents whenever hiding
    is requested.  Silent under-filled contexts are forbidden.
    """

    ids = tuple(str(value) for value in ranked_document_ids)
    if top_k <= 0 or len(ids) < top_k:
        raise ValueError("ranked list must contain at least top_k documents")
    if len(set(ids)) != len(ids):
        raise ValueError("ranked document IDs must be unique")
    before = ids[:top_k]
    if hidden_document is None or hidden_document not in before:
        return before, False
    if len(ids) < top_k + 1:
        raise ValueError("hide requires a k+1 backfill candidate")
    after = tuple(value for value in ids if value != hidden_document)[:top_k]
    if len(after) != top_k or hidden_document in after:
        raise RuntimeError("hide/backfill invariant failed")
    return after, True


def apply_policy(
    sorted_scores: Sequence[float],
    sorted_document_ids: Sequence[str],
    *,
    intent_score: float,
    intent_threshold: float,
    policy: str,
    top_k: int = 3,
    confidence: float = DEFAULT_CONFIDENCE,
) -> RetrievalDecision:
    """Apply exactly one of D0/D1/D2/D3 without weighted fusion or refusal."""

    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    scores = np.asarray(sorted_scores, dtype=np.float64)
    ids = tuple(str(value) for value in sorted_document_ids)
    if scores.ndim != 1 or len(scores) != len(ids) or len(scores) < max(3, top_k + 1):
        raise ValueError("full aligned corpus scores and k+1 IDs are required")
    if np.any(scores[:-1] < scores[1:]):
        raise ValueError("scores must be sorted in nonincreasing order")
    stats = canonical_mirabel_from_full_scores(scores, confidence=confidence)
    candidate = ids[0]  # The exact Mirabel argmax locator, even without an alarm.
    intent_alarm = bool(float(intent_score) > float(intent_threshold))
    mirabel_alarm = bool(stats.margin > 0.0)
    hide = (
        policy == "D1_ORIGINAL_MIRABEL" and mirabel_alarm
        or policy == "D2_INTENT_MIRABEL" and intent_alarm
        or policy == "D3_OR_CONTROL" and (intent_alarm or mirabel_alarm)
    )
    hidden = candidate if hide else None
    after, backfilled = hide_and_backfill(ids, top_k=top_k, hidden_document=hidden)
    return RetrievalDecision(
        policy=policy,
        intent_alarm=intent_alarm,
        mirabel_alarm=mirabel_alarm,
        candidate_document=candidate,
        hidden_document=hidden,
        before=ids[:top_k],
        after=after,
        backfilled=backfilled,
        mirabel_margin=float(stats.margin),
        mirabel_threshold=float(stats.threshold),
    )


def runtime_contract() -> dict[str, object]:
    return {
        "architecture": "shared MPNet + last-two-layer LoRA(r=8) + one linear intent head + canonical Mirabel locator",
        "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
        "mirabel_confidence": DEFAULT_CONFIDENCE,
        "strict_comparator": ">",
        "policies": list(POLICIES),
        "explicit_refusal": False,
        "weighted_fusion": False,
        "session_state": False,
    }
