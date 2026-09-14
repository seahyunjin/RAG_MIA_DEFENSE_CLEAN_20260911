"""Statistical primitives for Exp73's frozen LoLA Running-MinP policy.

The functions in this module are deliberately model- and attack-agnostic.
They accept only a scalar LoLA score, benign calibration scores, and a linked
session identifier.  Attack family and native budget are reporting metadata,
never detector inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def strict_alarm(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Return the frozen strict-``>`` decision used by LoLA-v1."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional scores required")
    return values > float(threshold)


def empirical_upper_tail_p(
    scores: Sequence[float], reference_scores: Sequence[float]
) -> np.ndarray:
    """Conservative empirical upper-tail p-values, including the +1 correction."""
    values = np.asarray(scores, dtype=float)
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    if values.ndim != 1 or reference.ndim != 1 or not len(reference):
        raise ValueError("scores/reference must be non-empty one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("scores/reference must be finite")
    greater_equal = len(reference) - np.searchsorted(reference, values, side="left")
    return (greater_equal + 1.0) / (len(reference) + 1.0)


def threshold_higher(scores: Sequence[float], target_fpr: float) -> float:
    """Benign-only empirical threshold whose strict alarm rate is at most alpha."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty scores required")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between zero and one")
    return float(np.quantile(values, 1.0 - target_fpr, method="higher"))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    low = center - radius
    high = center + radius
    if abs(low) < 1e-15:
        low = 0.0
    if abs(high - 1.0) < 1e-15:
        high = 1.0
    return max(0.0, low), min(1.0, high)


def harmonic_adr(member_tpr: float, nonmember_tpr: float) -> float:
    total = float(member_tpr) + float(nonmember_tpr)
    return 0.0 if total == 0 else 2.0 * float(member_tpr) * float(nonmember_tpr) / total


def attack_metrics(labels: Sequence[int], alarms: Sequence[bool]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    a = np.asarray(alarms, dtype=bool)
    if y.shape != a.shape or set(np.unique(y)) - {0, 1}:
        raise ValueError("aligned member/nonmember binary labels required")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("both member and nonmember examples required")
    member = float(a[y == 1].mean())
    nonmember = float(a[y == 0].mean())
    return {
        "member_tpr": member,
        "nonmember_tpr": nonmember,
        "adr": harmonic_adr(member, nonmember),
        "member_nonmember_gap": abs(member - nonmember),
    }


def add_empirical_p_values(
    frame: pd.DataFrame,
    *,
    reference_role: str = "cdf_reference",
    role_column: str = "exp58_normal_role",
    domain_column: str = "domain",
    score_column: str = "intent_score",
) -> pd.DataFrame:
    """Attach domain-calibrated p-values without reading attack labels."""
    result = frame.copy()
    result["intent_p"] = np.nan
    for domain, cell in result.groupby(domain_column, sort=False):
        reference = cell.loc[cell[role_column] == reference_role, score_column].to_numpy(float)
        if not len(reference):
            raise ValueError(f"no benign p-value reference for {domain}")
        index = cell.index
        result.loc[index, "intent_p"] = empirical_upper_tail_p(
            result.loc[index, score_column].to_numpy(float), reference
        )
    if result.intent_p.isna().any():
        raise RuntimeError("p-value assignment incomplete")
    return result


def add_running_minp(
    frame: pd.DataFrame,
    *,
    session_column: str = "session_id",
    turn_column: str = "turn",
    p_column: str = "intent_p",
) -> pd.DataFrame:
    """Attach chronological raw Running-MinP and its monotone risk transform."""
    result = frame.sort_values([session_column, turn_column], kind="stable").copy()
    result["running_minp"] = result.groupby(session_column, sort=False)[p_column].cummin()
    result["running_minp_risk"] = -np.log10(result.running_minp.clip(lower=np.finfo(float).tiny))
    return result


def calibrate_horizon_thresholds(
    frame: pd.DataFrame,
    *,
    horizons: Iterable[int],
    target_fprs: Iterable[float],
    role_column: str = "exp58_normal_role",
    lock_role: str = "threshold_lock",
    confirmation_role: str = "normal_confirmation",
) -> pd.DataFrame:
    """Calibrate each horizon on benign lock sessions and audit confirmation FPR."""
    rows: list[dict[str, float | int]] = []
    for horizon in sorted(set(map(int, horizons))):
        lock = frame[(frame[role_column] == lock_role) & (frame.turn == horizon)]
        confirmation = frame[(frame[role_column] == confirmation_role) & (frame.turn == horizon)]
        if lock.empty or confirmation.empty:
            raise ValueError(f"benign linked sessions do not support Q{horizon}")
        for target in target_fprs:
            threshold = threshold_higher(lock.running_minp_risk, float(target))
            lock_alarm = strict_alarm(lock.running_minp_risk, threshold)
            confirmation_alarm = strict_alarm(confirmation.running_minp_risk, threshold)
            low, upper = wilson_interval(int(confirmation_alarm.sum()), len(confirmation_alarm))
            rows.append({
                "horizon": horizon,
                "target_fpr": float(target),
                "threshold": threshold,
                "lock_sessions": int(len(lock)),
                "lock_fpr": float(lock_alarm.mean()),
                "confirmation_sessions": int(len(confirmation)),
                "confirmation_fpr": float(confirmation_alarm.mean()),
                "confirmation_wilson_low": low,
                "confirmation_wilson_upper": upper,
                "attack_examples_used": 0,
                "strict_comparator": ">",
            })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class OnlinePolicy:
    """Frozen stateless thresholds plus benign-only per-turn session thresholds."""

    stateless_thresholds: Mapping[str, float]
    session_thresholds: Mapping[int, float]


def apply_online_policy(
    frame: pd.DataFrame,
    policy: OnlinePolicy,
    *,
    immediate_families: frozenset[str] = frozenset({"RAG-MIA", "S²-MIA", "MBA"}),
) -> pd.DataFrame:
    """Simulate chronological alarms.

    Attack family is used only to label truly one-query evaluation sessions.
    Every first query uses the stateless LoLA decision.  From query two onward,
    any linkable session uses only its turn-indexed Running-MinP threshold and
    sticky state.  Thus no future query budget or family is exposed at runtime.
    """
    del immediate_families  # retained only for an explicit API-level threat-model reminder
    result = frame.sort_values(["session_id", "turn"], kind="stable").copy()
    result["stateless_alarm"] = [
        float(score) > float(policy.stateless_thresholds[str(domain)])
        for score, domain in zip(result.intent_score, result.domain)
    ]
    result["online_alarm"] = False
    result["alarm_turn"] = np.nan
    for _, cell in result.groupby("session_id", sort=False):
        sticky = False
        first_alarm = math.nan
        for index, row in cell.sort_values("turn", kind="stable").iterrows():
            turn = int(row.turn)
            if turn == 1:
                current = bool(row.stateless_alarm)
            else:
                threshold = policy.session_thresholds.get(turn)
                if threshold is None:
                    raise KeyError(f"no frozen Running-MinP threshold for Q{turn}")
                current = float(row.running_minp_risk) > float(threshold)
            if current and not sticky:
                first_alarm = turn
            sticky = sticky or current
            result.loc[index, "online_alarm"] = sticky
            result.loc[index, "alarm_turn"] = first_alarm
    return result


def endpoint_rows(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One last-observed row per session at or before a reporting horizon."""
    eligible = frame[frame.turn <= int(horizon)]
    return eligible.sort_values(["session_id", "turn"], kind="stable").groupby(
        "session_id", as_index=False, sort=False
    ).tail(1)


__all__ = [
    "OnlinePolicy", "add_empirical_p_values", "add_running_minp",
    "apply_online_policy", "attack_metrics", "calibrate_horizon_thresholds",
    "empirical_upper_tail_p", "endpoint_rows", "harmonic_adr", "strict_alarm",
    "threshold_higher", "wilson_interval",
]
