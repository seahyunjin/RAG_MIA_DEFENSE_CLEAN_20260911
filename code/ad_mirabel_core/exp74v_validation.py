"""Validation-only statistical helpers for Exp74V.

This module deliberately contains no fitting or feature extraction.  Exp74V
uses it to keep the final calibration, alarm, and reporting definitions small
and auditable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def threshold_higher(scores: Sequence[float], target_fpr: float) -> float:
    """Benign-only threshold for the frozen strict ``score > threshold`` rule."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty one-dimensional scores required")
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must lie in (0, 1)")
    return float(np.quantile(values, 1.0 - float(target_fpr), method="higher"))


def empirical_upper_tail_p(scores: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    """Conservative empirical upper-tail p values with the +1 correction."""
    values = np.asarray(scores, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    if values.ndim != 1 or ref.ndim != 1 or not len(ref):
        raise ValueError("one-dimensional scores and a non-empty reference are required")
    if not np.isfinite(values).all() or not np.isfinite(ref).all():
        raise ValueError("scores/reference must be finite")
    greater_equal = len(ref) - np.searchsorted(ref, values, side="left")
    return (greater_equal + 1.0) / (len(ref) + 1.0)


def harmonic_adr(member_tpr: float, nonmember_tpr: float) -> float:
    total = float(member_tpr) + float(nonmember_tpr)
    return 0.0 if total == 0.0 else 2.0 * float(member_tpr) * float(nonmember_tpr) / total


def attack_metrics(labels: Iterable[int], alarms: Iterable[bool]) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=int)
    alarm = np.asarray(list(alarms), dtype=bool)
    if y.shape != alarm.shape or set(np.unique(y)) - {0, 1}:
        raise ValueError("aligned binary membership labels and alarms required")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("both member and nonmember groups are required")
    member = float(alarm[y == 1].mean())
    nonmember = float(alarm[y == 0].mean())
    return {
        "member_n": int(np.sum(y == 1)),
        "nonmember_n": int(np.sum(y == 0)),
        "member_tpr": member,
        "nonmember_tpr": nonmember,
        "harmonic_adr": harmonic_adr(member, nonmember),
        "member_nonmember_gap": abs(member - nonmember),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= int(successes) <= int(trials):
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def calibration_selection(rows: Iterable[dict]) -> tuple[int | None, str]:
    """Return the smallest N passing every evaluated development domain."""
    records = list(rows)
    candidates: list[int] = []
    for n in sorted({int(row["n"]) for row in records}):
        group = [row for row in records if int(row["n"]) == n]
        evaluated = [row for row in group if str(row.get("status")) == "EVALUATED"]
        if evaluated and len(evaluated) == len(group) and all(bool(row.get("pass")) for row in evaluated):
            candidates.append(n)
    if not candidates:
        return None, "CALIBRATION_STABILITY_NOT_CONFIRMED"
    return min(candidates), "CALIBRATION_STABILITY_CONFIRMED"


__all__ = [
    "attack_metrics",
    "calibration_selection",
    "empirical_upper_tail_p",
    "harmonic_adr",
    "threshold_higher",
    "wilson_interval",
]
