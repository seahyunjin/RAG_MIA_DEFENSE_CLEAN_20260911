"""Frozen CLD/G4 retrieval intervention used by RIF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

import numpy as np

TAU = 0.0901547372341156
T = TypeVar("T")


@dataclass(frozen=True)
class CLDDecision(Generic[T]):
    g4: float
    alarm: bool
    documents: tuple[T, ...]
    removed: T | None


def cld4(scores: Sequence[float]) -> float:
    """Return s1 - mean(s1..s4), using float64 arithmetic."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("CLD4 requires at least four ranked retrieval scores")
    top4 = values[:4]
    return float(top4[0] - top4.mean())


def apply_cld(
    scores: Sequence[float],
    documents: Sequence[T],
    tau: float = TAU,
) -> CLDDecision[T]:
    """Remove rank 1 on a strict frozen-threshold alarm, without refill."""
    if not documents:
        raise ValueError("At least one generator-context document is required")
    risk = cld4(scores)
    alarm = bool(risk > tau)
    kept = tuple(documents[1:] if alarm else documents)
    return CLDDecision(
        g4=risk,
        alarm=alarm,
        documents=kept,
        removed=documents[0] if alarm else None,
    )

