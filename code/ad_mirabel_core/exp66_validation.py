"""Pure validation helpers for the frozen Exp66 LoLA campaign.

This module deliberately contains no model fitting or experiment selection.
It only implements deterministic hashing, strict matched-FPR thresholding,
rate summaries, and gate decisions used by the final audit.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.exp61_intent_mirabel import wilson_interval


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> tuple[str, int]:
    """Hash a tree using relative path + content hashes in lexical order."""
    rows = [
        f"{item.relative_to(path)}\0{sha256_file(item)}\n"
        for item in sorted(value for value in path.rglob("*") if value.is_file())
    ]
    return hashlib.sha256("".join(rows).encode()).hexdigest(), len(rows)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def strict_matched_threshold(scores: Sequence[float], target_fpr: float) -> dict[str, float | int]:
    """Return the most permissive observed threshold with strict ``>`` FPR.

    This is for an apples-to-apples ROC operating-point diagnostic.  The
    supplied scores must be benign-only and are the explicit threshold-
    selection cohort; attack observations are never used.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty benign scores required")
    if not 0 < float(target_fpr) < 1:
        raise ValueError("target_fpr must be in (0,1)")
    candidates = np.unique(values)
    feasible: list[tuple[int, float]] = []
    for threshold in candidates:
        positives = int(np.count_nonzero(values > threshold))
        if positives / len(values) <= target_fpr + 1e-15:
            feasible.append((positives, float(threshold)))
    if not feasible:
        raise RuntimeError("no feasible strict threshold")
    positives, threshold = max(feasible, key=lambda item: (item[0], -item[1]))
    low, high = wilson_interval(positives, len(values))
    return {
        "threshold": threshold,
        "false_positives": positives,
        "n": len(values),
        "observed_fpr": positives / len(values),
        "wilson_low": float(low),
        "wilson_upper": float(high),
        "target_fpr": float(target_fpr),
    }


def harmonic_rate(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else float(2.0 * left * right / (left + right))


def member_nonmember_rates(labels: Sequence[int], alarms: Sequence[bool]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(alarms, dtype=bool)
    if y.shape != a.shape or not len(y) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("aligned binary labels and alarms required")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("both member and nonmember observations required")
    member = float(a[y == 1].mean())
    nonmember = float(a[y == 0].mean())
    return {
        "member_tpr": member,
        "nonmember_tpr": nonmember,
        "adr": harmonic_rate(member, nonmember),
        "member_nonmember_gap": abs(member - nonmember),
    }


def stability_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("finite values required")
    mean = float(array.mean())
    return {
        "mean": mean,
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "coefficient_of_variation": float(array.std(ddof=0) / abs(mean)) if not math.isclose(mean, 0.0) else math.inf,
    }


def matched_fpr_verdict(rows: Sequence[dict[str, float | str]]) -> str:
    """Apply the precommitted low-FPR detector comparison rule."""
    low = {0.005, 0.01, 0.02}
    wins = 0
    utility_ok = True
    for target in low:
        lola = next(row for row in rows if float(row["target_fpr"]) == target and row["method"] == "LoLA")
        mirabel = next(row for row in rows if float(row["target_fpr"]) == target and row["method"] == "Original Mirabel")
        wins += float(lola["fm_adr"]) > float(mirabel["fm_adr"])
        utility_ok &= float(lola["retrieval_utility"]) >= 0.95
    if wins >= 2 and utility_ok:
        return "MATCHED_FPR_LOLA_SUPERIOR"
    if wins >= 1 and utility_ok:
        return "MATCHED_FPR_PARTIAL"
    return "MATCHED_FPR_NO_CLEAR_ADVANTAGE"


__all__ = [
    "harmonic_rate",
    "matched_fpr_verdict",
    "member_nonmember_rates",
    "sha256_file",
    "stability_summary",
    "stable_hash",
    "strict_matched_threshold",
    "tree_hash",
]
