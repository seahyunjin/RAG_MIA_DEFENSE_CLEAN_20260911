"""Public statistical primitives for the Exp47 submission repair.

The module is deliberately free of private paths, experiment artifacts, model
weights, and API clients so it can be distributed and tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


FAMILIES = ("RAG-MIA", "S²-MIA", "MBA", "DCMI", "MEntA", "IA")
TARGET_FPRS = (0.01, 0.03, 0.05)


def wilson_interval(positives: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= positives <= total:
        raise ValueError("require 0 <= positives <= total and total > 0")
    z = 1.959963984540054
    p = positives / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def family_macro_member_detection_rate(
    member_decisions: Mapping[str, Sequence[bool]],
) -> float:
    """Equal-family mean P(blocked | family, member=1)."""
    if set(member_decisions) != set(FAMILIES):
        raise ValueError("all six registered families are required")
    rates = []
    for family in FAMILIES:
        values = np.asarray(member_decisions[family], dtype=bool)
        if not len(values):
            raise ValueError(f"empty member cell: {family}")
        rates.append(float(values.mean()))
    return float(np.mean(rates))


def effective_auc(raw_auc: float) -> float:
    value = float(raw_auc)
    if not 0 <= value <= 1:
        raise ValueError("AUROC must be in [0,1]")
    return max(value, 1.0 - value)


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(labels_array) != len(scores_array) or len(np.unique(labels_array)) != 2:
        raise ValueError("AUROC requires aligned values and both labels")
    if not np.isfinite(scores_array).all():
        raise ValueError("AUROC scores must be finite")
    return float(roc_auc_score(labels_array, scores_array))


def block_signal(
    first_block_query: int | None, native_budget: int, signal: str,
) -> float:
    budget = int(native_budget)
    if budget < 1:
        raise ValueError("native budget must be positive")
    first = None if first_block_query is None else int(first_block_query)
    if first is not None and not 1 <= first <= budget:
        raise ValueError("first block query lies outside the native budget")
    blocked_count = 0 if first is None else budget - first + 1
    if signal == "block_bit":
        return float(first is not None)
    if signal == "first_block_position":
        # Earlier block means greater membership evidence; never maps to zero.
        return float(0 if first is None else budget + 1 - first)
    if signal == "queries_answered_before_block":
        # Negated so a higher value consistently means more membership evidence.
        return float(-(budget if first is None else first - 1))
    if signal == "fraction_queries_blocked":
        return float(blocked_count / budget)
    raise ValueError(f"unknown block signal: {signal}")


@dataclass(frozen=True)
class BlockRateStatistics:
    member_rate: float
    nonmember_rate: float
    balanced_block_rate: float
    absolute_gap: float
    gap_ci_low: float
    gap_ci_high: float
    risk_ratio: float
    risk_ratio_ci_low: float
    risk_ratio_ci_high: float
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float


def block_rate_statistics(member_blocked: Sequence[bool], nonmember_blocked: Sequence[bool]) -> BlockRateStatistics:
    member = np.asarray(member_blocked, dtype=bool)
    nonmember = np.asarray(nonmember_blocked, dtype=bool)
    if not len(member) or not len(nonmember):
        raise ValueError("both membership classes are required")
    a, n1 = int(member.sum()), len(member)
    c, n0 = int(nonmember.sum()), len(nonmember)
    member_rate, nonmember_rate = a / n1, c / n0
    member_ci, nonmember_ci = wilson_interval(a, n1), wilson_interval(c, n0)
    gap_low = member_ci[0] - nonmember_ci[1]
    gap_high = member_ci[1] - nonmember_ci[0]
    # Haldane-Anscombe correction keeps RR/OR finite for zero cells.
    aa, bb, cc, dd = a + .5, n1 - a + .5, c + .5, n0 - c + .5
    corrected_member, corrected_nonmember = aa / (n1 + 1), cc / (n0 + 1)
    rr = corrected_member / corrected_nonmember
    rr_se = math.sqrt(max(0.0, 1 / aa - 1 / (n1 + 1) + 1 / cc - 1 / (n0 + 1)))
    odds = (aa * dd) / (bb * cc)
    odds_se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return BlockRateStatistics(
        member_rate, nonmember_rate, (member_rate + nonmember_rate) / 2,
        member_rate - nonmember_rate, gap_low, gap_high, rr,
        math.exp(math.log(rr) - 1.959963984540054 * rr_se),
        math.exp(math.log(rr) + 1.959963984540054 * rr_se), odds,
        math.exp(math.log(odds) - 1.959963984540054 * odds_se),
        math.exp(math.log(odds) + 1.959963984540054 * odds_se),
    )


def sidechannel_label(macro_effective_auc: float, retriever: str) -> str:
    value = float(macro_effective_auc)
    suffix = retriever.replace("-", "_")
    if value <= .55:
        return f"BLOCK_SIDECHANNEL_SAFE_{suffix}"
    if value <= .60:
        return f"BLOCK_SIDECHANNEL_RESIDUAL_{suffix}"
    return f"BLOCK_SIDECHANNEL_FAILED_{suffix}"


def generator_cell_gate(effective_full: float, advantage_full: float,
                        advantage_early: float) -> dict[str, bool]:
    absolute = float(effective_full) <= .55
    relative = float(advantage_full) < float(advantage_early)
    return {"absolute_gate": absolute, "relative_gate": relative, "cell_pass": absolute or relative}


def generator_overall_label(cell_passes: Sequence[bool]) -> str:
    values = [bool(value) for value in cell_passes]
    if not values:
        raise ValueError("at least one required generator cell is required")
    if all(values):
        return "ACTUAL_GENERATOR_DEFENSE_CONFIRMED"
    if any(values):
        return "ACTUAL_GENERATOR_DEFENSE_PARTIAL"
    return "ACTUAL_GENERATOR_DEFENSE_FAILED"


def low_fpr_valid(observed_q30_fpr: float, ci_upper: float, member_rate: float) -> bool:
    return bool(
        .0025 <= float(observed_q30_fpr) <= .02 and float(ci_upper) <= .03 and
        not (float(observed_q30_fpr) == 0 and float(member_rate) == 0)
    )


def paired_cluster_family_bootstrap(
    records: Sequence[Mapping[str, object]], *, iterations: int = 10000,
    seed: int = 20260814,
) -> dict[str, float | int]:
    """Paired cluster bootstrap of an equal-family mean detector difference.

    Each record contains family, cluster_id, detector_a, and detector_b. The
    same sampled cluster multiplicity is applied to both detector outcomes.
    """
    if iterations < 2000:
        raise ValueError("at least 2,000 iterations are required")
    by_family: dict[str, dict[str, list[tuple[float, float]]]] = {family: {} for family in FAMILIES}
    for row in records:
        family = str(row["family"])
        if family not in by_family:
            raise ValueError(f"unregistered family: {family}")
        cluster = str(row["cluster_id"])
        by_family[family].setdefault(cluster, []).append((float(row["detector_a"]), float(row["detector_b"])))
    if any(not clusters for clusters in by_family.values()):
        raise ValueError("all registered families require clusters")
    point_family = {}
    for family, clusters in by_family.items():
        values = np.asarray([item for rows in clusters.values() for item in rows], dtype=float)
        point_family[family] = float(np.mean(values[:, 0] - values[:, 1]))
    point = float(np.mean(list(point_family.values())))
    rng = np.random.default_rng(seed)
    samples = np.zeros(iterations, dtype=float)
    # A sampled cluster contributes every session it contains.  Pre-aggregating
    # its paired difference sum and row count is exactly equivalent to
    # repeatedly flattening its rows, but avoids billions of Python objects on
    # the 14,639-session audit cohort.
    for family in FAMILIES:
        clusters = by_family[family]
        keys = list(clusters)
        cluster_sum = np.asarray([
            sum(left - right for left, right in clusters[key]) for key in keys
        ], dtype=float)
        cluster_count = np.asarray([len(clusters[key]) for key in keys], dtype=float)
        family_samples = np.empty(iterations, dtype=float)
        chunk_size = 256
        for start in range(0, iterations, chunk_size):
            stop = min(iterations, start + chunk_size)
            chosen = rng.integers(0, len(keys), size=(stop - start, len(keys)))
            family_samples[start:stop] = (
                cluster_sum[chosen].sum(axis=1) / cluster_count[chosen].sum(axis=1)
            )
        samples += family_samples
    samples /= len(FAMILIES)
    wins = sum(value > 0 for value in point_family.values())
    ties = sum(abs(value) <= 1e-15 for value in point_family.values())
    return {
        "point_difference": point,
        "ci_low": float(np.quantile(samples, .025)),
        "ci_high": float(np.quantile(samples, .975)),
        "iterations": int(iterations), "seed": int(seed),
        "family_wins": int(wins), "family_ties": int(ties),
        "family_losses": int(len(FAMILIES) - wins - ties),
    }
