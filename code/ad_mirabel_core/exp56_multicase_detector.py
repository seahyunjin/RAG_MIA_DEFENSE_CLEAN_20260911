"""Pure, privacy-safe primitives for the Exp56 multi-case detector campaign."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from .exp55_domain_adaptive_detector import (
    FORBIDDEN_INFERENCE_FIELDS, cct_p, minp, session_pvalue_curve, strict_alarm,
    validate_inference_payload,
)


def stable_seed(value: object) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def cct_array(columns: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("finite aligned p-value columns required")
    if np.any(matrix <= 0) or np.any(matrix > 1):
        raise ValueError("p-values must lie in (0,1]")
    clipped = np.clip(matrix, 1e-12, 1 - 1e-12)
    statistic = np.tan(np.pi * (.5 - clipped)).mean(axis=1)
    return np.clip(.5 - np.arctan(statistic) / np.pi, 1e-300, 1.)


def minp_array(columns: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("finite aligned p-value columns required")
    if np.any(matrix <= 0) or np.any(matrix > 1):
        raise ValueError("p-values must lie in (0,1]")
    return matrix.min(axis=1)


def aggregate_curve(pvalues: Sequence[float], method: str, q1_threshold: float | None = None) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).reshape(-1)
    if p.size == 0 or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("valid p-values required")
    if method == "MINP":
        return -np.log10(np.maximum(np.minimum.accumulate(p), 1e-300))
    if method == "CCT":
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        running = np.cumsum(np.tan(np.pi * (.5 - clipped))) / np.arange(1, len(p) + 1)
        combined = np.clip(.5 - np.arctan(running) / np.pi, 1e-300, 1.)
        return -np.log10(combined)
    if method == "STICKY_Q1":
        if q1_threshold is None or not math.isfinite(float(q1_threshold)):
            raise ValueError("sticky Q1 requires a threshold")
        score = -np.log10(np.maximum(p, 1e-300))
        return np.maximum.accumulate(score)
    raise ValueError("unknown aggregator")


def deterministic_permutations(length: int, count: int, key: str) -> list[np.ndarray]:
    if length < 1 or count < 1:
        raise ValueError("positive length and count required")
    return [np.random.default_rng(stable_seed(f"{key}|{index}")).permutation(length) for index in range(count)]


def balanced_mixed_indices(
    families: Sequence[str], members: Sequence[int], per_cell: int, key: str,
) -> np.ndarray:
    family = np.asarray(families, dtype=str); member = np.asarray(members, dtype=int)
    if len(family) != len(member) or per_cell < 1:
        raise ValueError("aligned labels and positive per_cell required")
    output: list[int] = []
    for name in sorted(set(family)):
        for label in (0, 1):
            available = np.flatnonzero((family == name) & (member == label))
            if len(available) < per_cell:
                raise ValueError(f"insufficient {name}/{label} observations")
            order = np.random.default_rng(stable_seed(f"{key}|{name}|{label}")).permutation(available)
            output.extend(order[:per_cell].tolist())
    return np.asarray(output, dtype=int)


def benign_interleave(attack: Sequence[float], benign: Sequence[float], benign_between: int) -> np.ndarray:
    a, b = np.asarray(attack, float), np.asarray(benign, float)
    if not len(a) or benign_between < 1 or len(b) < benign_between * max(0, len(a) - 1):
        raise ValueError("insufficient attack/benign observations")
    output, cursor = [], 0
    for index, value in enumerate(a):
        output.append(float(value))
        if index + 1 < len(a):
            output.extend(map(float, b[cursor:cursor + benign_between])); cursor += benign_between
    return np.asarray(output, dtype=float)


def order_by_development_strength(
    pvalues: Sequence[float], families: Sequence[str], development_strength: Mapping[str, float], *, weak_first: bool,
) -> np.ndarray:
    p = np.asarray(pvalues, float); family = np.asarray(families, str)
    if len(p) != len(family) or set(family) - set(development_strength):
        raise ValueError("development-only strength required for every family")
    # Larger anomaly strength is stronger; stable index resolves ties without looking at test p-values.
    order = sorted(range(len(p)), key=lambda i: (float(development_strength[family[i]]), i), reverse=not weak_first)
    return np.asarray(order, dtype=int)


def gate_label(row: Mapping[str, float], family_rates: Mapping[str, float]) -> str:
    common = float(row["normal_fpr"]) <= .02 and float(row["wilson_upper"]) <= .035
    research = (common and float(row["lodo_worst"]) >= .15 and float(row["lofo_median"]) >= .30
                and min(map(float, family_rates.values())) >= .10)
    deployment = (common and float(row["lodo_worst"]) >= .20 and float(row["lofo_median"]) >= .35
                  and sum(float(value) >= .20 for value in family_rates.values()) >= 4
                  and min(map(float, family_rates.values())) >= .10 and float(row["gap"]) <= .20)
    strong = (float(row["normal_fpr"]) <= .02 and float(row["wilson_upper"]) <= .03
              and float(row["lodo_worst"]) >= .40 and float(row["lofo_median"]) >= .50
              and sum(float(value) >= .40 for value in family_rates.values()) >= 4
              and min(map(float, family_rates.values())) >= .20 and float(row["gap"]) <= .15)
    return "STRONG_Q1" if strong else "DEPLOYMENT_CANDIDATE" if deployment else "RESEARCH_PROMISING" if research else "FAILED"


__all__ = [
    "FORBIDDEN_INFERENCE_FIELDS", "aggregate_curve", "balanced_mixed_indices", "benign_interleave",
    "cct_array", "deterministic_permutations", "gate_label", "minp_array",
    "order_by_development_strength", "stable_seed", "strict_alarm", "validate_inference_payload",
]
