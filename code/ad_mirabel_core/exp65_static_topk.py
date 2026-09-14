"""Pure, frozen-policy helpers for Exp65 static Top-K Mirabel hiding.

The functions in this module deliberately contain no learned component.  A
caller supplies the already frozen Intent alarm and the original descending
Mirabel/retrieval ordering.  The selected prefix is removed once and the
remaining ordering provides deterministic backfill; the ranking is never
recomputed after removal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticHideResult:
    original_top_k: tuple[str, ...]
    safe_top_k: tuple[str, ...]
    hidden_documents: tuple[str, ...]
    intent_alarm: bool
    hide_k: int
    backfill_count: int


def static_topk_hide(
    sorted_document_ids: Sequence[str],
    *,
    intent_alarm: bool,
    hide_k: int,
    context_k: int = 3,
) -> StaticHideResult:
    """Remove the frozen ranking prefix and deterministically backfill.

    ``sorted_document_ids`` must be the original ordering computed before any
    removal.  At least ``context_k + hide_k`` unique IDs are required whenever
    the alarm is active, making accidental short/no-backfill contexts fail
    closed during the experiment.
    """

    ids = tuple(map(str, sorted_document_ids))
    if context_k < 1 or hide_k < 1:
        raise ValueError("context_k and hide_k must be positive")
    if len(set(ids)) != len(ids):
        raise ValueError("the frozen ranking must not contain duplicate IDs")
    needed = context_k + (hide_k if intent_alarm else 0)
    if len(ids) < needed:
        raise ValueError(f"at least {needed} ranked documents are required")
    original = ids[:context_k]
    hidden = ids[:hide_k] if intent_alarm else ()
    hidden_set = set(hidden)
    safe = tuple(value for value in ids if value not in hidden_set)[:context_k]
    if len(safe) != context_k or hidden_set.intersection(safe):
        raise RuntimeError("deterministic static hide/backfill invariant failed")
    retained_original = sum(value in safe for value in original)
    return StaticHideResult(
        original_top_k=original,
        safe_top_k=safe,
        hidden_documents=hidden,
        intent_alarm=bool(intent_alarm),
        hide_k=int(hide_k),
        backfill_count=context_k - retained_original,
    )


def causal_coverage(
    mirabel_ranks: Sequence[int],
    causal_influences: Sequence[float],
    *,
    ks: Sequence[int] = (1, 2, 3, 5),
) -> list[dict[str, float | int]]:
    """Return Top-K coverage of the best causal item and best causal pair."""

    ranks = np.asarray(mirabel_ranks, dtype=int)
    delta = np.asarray(causal_influences, dtype=float)
    if ranks.ndim != 1 or delta.ndim != 1 or len(ranks) != len(delta) or not len(ranks):
        raise ValueError("aligned non-empty rank and influence vectors required")
    if np.any(ranks < 1) or len(np.unique(ranks)) != len(ranks):
        raise ValueError("Mirabel ranks must be unique positive integers")
    if not np.isfinite(delta).all():
        raise ValueError("causal influences must be finite")
    causal_order = np.argsort(-delta, kind="stable")
    best_rank = int(ranks[causal_order[0]])
    pair_ranks = set(ranks[causal_order[: min(2, len(ranks))]].astype(int).tolist())
    rows = []
    for k in ks:
        if int(k) < 1:
            raise ValueError("coverage K must be positive")
        rows.append(
            {
                "k": int(k),
                "causal_top1_covered": float(best_rank <= int(k)),
                "best_two_set_covered": float(all(rank <= int(k) for rank in pair_ranks)),
            }
        )
    return rows


__all__ = ["StaticHideResult", "causal_coverage", "static_topk_hide"]
