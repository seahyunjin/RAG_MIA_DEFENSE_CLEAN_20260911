"""Small, auditable statistical helpers for Exp74-FINAL.

The module deliberately contains no model fitting.  It implements only the
strict comparator, empirical thresholding, Wilson intervals, and Pareto/gate
logic used by the Exp74 orchestration script.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def strict_threshold(reference: Iterable[float], target_fpr: float) -> float:
    """Largest-score alarm threshold with the frozen strict ``score > t`` rule."""
    values = np.sort(np.asarray(list(reference), dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("reference must contain finite values")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie in (0,1)")
    allowed = int(math.floor(target_fpr * len(values)))
    if allowed <= 0:
        return float(values[-1])
    return float(values[max(0, len(values) - allowed - 1)])


def wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float((center + radius) / denominator)


def attack_metrics(labels: Iterable[int], alarms: Iterable[bool]) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=int)
    a = np.asarray(list(alarms), dtype=bool)
    member = a[y == 1]
    nonmember = a[y == 0]
    if not len(member) or not len(nonmember):
        raise ValueError("both member and nonmember groups are required")
    mt, nt = float(member.mean()), float(nonmember.mean())
    return {"member_tpr": mt, "nonmember_tpr": nt, "adr": (mt + nt) / 2,
            "member_nonmember_gap": abs(mt - nt)}


def pareto_relation(lola: Iterable[Mapping[str, float]], mirabel: Iterable[Mapping[str, float]]) -> str:
    """Compare empirical (FPR, ADR) frontiers without nominal-FPR substitution."""
    left = [(float(r["actual_fpr"]), float(r["adr"])) for r in lola]
    right = [(float(r["actual_fpr"]), float(r["adr"])) for r in mirabel]
    if not left or not right:
        return "PARETO_MIXED"

    def dominates(a, b):
        return all(any(fa <= fb + 1e-15 and ta >= tb - 1e-15 for fa, ta in a)
                   for fb, tb in b)

    lola_dominates = dominates(left, right)
    mirabel_dominates = dominates(right, left)
    if lola_dominates and not mirabel_dominates:
        return "LOLA_PARETO_DOMINANT"
    if mirabel_dominates and not lola_dominates:
        return "MIRABEL_PARETO_DOMINANT"
    return "PARETO_MIXED"


def paper_gate(metrics: Mapping[str, float | bool]) -> tuple[bool, list[str]]:
    requirements = {
        "nb_adr": float(metrics["nb_adr"]) >= .70,
        "query_fpr": float(metrics["query_fpr"]) <= .02,
        "session_fpr": float(metrics["session_fpr"]) <= .02,
        "rag_q1": float(metrics["rag_q1"]) >= .80,
        "mba_q1": float(metrics["mba_q1"]) >= .80,
        "dcmi_q2": float(metrics["dcmi_q2"]) >= .80,
        "s2_q1": float(metrics["s2_q1"]) >= .50,
        "menta_q5": float(metrics["menta_q5"]) >= .60,
        "ia_q15": float(metrics["ia_q15"]) >= .60,
        "mean_effective_auc": float(metrics["mean_effective_auc"]) <= .58,
        "utility": float(metrics["utility"]) >= .95,
        "intervention": float(metrics["intervention"]) <= .02,
        "real_fpr_competitive": bool(metrics["real_fpr_competitive"]),
    }
    failures = [name for name, passed in requirements.items() if not passed]
    return not failures, failures

