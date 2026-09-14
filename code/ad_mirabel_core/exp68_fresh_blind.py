"""Pure deterministic helpers for the Exp68 one-time blind campaign."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

from src.exp61_intent_mirabel import wilson_interval


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def harmonic_rate(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else float(2 * left * right / (left + right))


def binary_rate(alarms: Sequence[bool]) -> dict[str, float | int]:
    values = np.asarray(alarms, dtype=bool)
    if values.ndim != 1 or not len(values):
        raise ValueError("non-empty one-dimensional alarms required")
    positives = int(values.sum())
    low, high = wilson_interval(positives, len(values))
    return {"alarms": positives, "n": len(values), "rate": positives / len(values),
            "wilson_low": float(low), "wilson_upper": float(high)}


def strict_empirical_threshold(scores: Sequence[float], target_fpr: float) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional calibration scores required")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0,1)")
    candidates = np.unique(values)
    feasible = []
    for threshold in candidates:
        count = int(np.count_nonzero(values > threshold))
        if count / len(values) <= target_fpr + 1e-15:
            feasible.append((count, float(threshold)))
    if not feasible:
        raise RuntimeError("no strict empirical threshold is feasible")
    count, threshold = max(feasible, key=lambda row: (row[0], -row[1]))
    low, high = wilson_interval(count, len(values))
    return {"threshold": threshold, "false_positives": count, "n": len(values),
            "observed_fpr": count / len(values), "wilson_low": float(low),
            "wilson_upper": float(high), "target_fpr": float(target_fpr)}


def family_detection(labels: Sequence[int], alarms: Sequence[bool]) -> dict[str, float]:
    membership = np.asarray(labels, dtype=int)
    detected = np.asarray(alarms, dtype=bool)
    if membership.shape != detected.shape or set(np.unique(membership)) != {0, 1}:
        raise ValueError("aligned member/nonmember labels required")
    member = float(detected[membership == 1].mean())
    nonmember = float(detected[membership == 0].mean())
    return {"member_tpr": member, "nonmember_tpr": nonmember,
            "adr": harmonic_rate(member, nonmember), "member_nonmember_gap": abs(member - nonmember)}


def identifier_hash(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(map(str, values)))
    return stable_hash(payload)


def deterministic_order(values: Iterable[object], salt: str) -> list[str]:
    unique = set(map(str, values))
    return sorted(unique, key=lambda value: (stable_hash(f"{salt}|{value}"), value))


def effective_auc(auc: float) -> float:
    if not math.isfinite(float(auc)):
        return math.nan
    return float(max(float(auc), 1.0 - float(auc)))


__all__ = [
    "binary_rate", "deterministic_order", "effective_auc", "family_detection",
    "harmonic_rate", "identifier_hash", "stable_hash", "strict_empirical_threshold",
]
