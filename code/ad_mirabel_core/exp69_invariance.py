"""Pure helpers for Exp69 LoLA-v2 invariant-representation training.

Family, domain, and membership metadata are accepted only for constructing
development groups and evaluation summaries.  None of these values are runtime
features of the detector.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from src.exp61_intent_mirabel import wilson_interval


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def invariant_pairs(rows: Sequence[Mapping[str, Any]], mode: str) -> list[tuple[int, int]]:
    """Create deterministic cross-domain/family positive pairs.

    Rows require ``index``, ``family``, ``domain``, ``member`` and ``length``.
    Pairing uses metadata only and never a learned or final-evaluation score.
    """
    if mode not in {"same_family_cross_domain", "cross_family_attack"}:
        raise ValueError("unknown invariant pair mode")
    data = [dict(row) for row in rows]
    output: set[tuple[int, int]] = set()
    for left in data:
        eligible = []
        for right in data:
            if int(left["index"]) == int(right["index"]) or int(left["member"]) != int(right["member"]):
                continue
            if mode == "same_family_cross_domain":
                if left["family"] != right["family"] or left["domain"] == right["domain"]:
                    continue
            else:
                if left["family"] == right["family"]:
                    continue
            distance = abs(float(left["length"]) - float(right["length"]))
            tie = stable_hash(f"{mode}|{left['index']}|{right['index']}")
            eligible.append((distance, tie, int(right["index"])))
        if eligible:
            right_index = min(eligible)[2]
            pair = tuple(sorted((int(left["index"]), right_index)))
            output.add(pair)
    return sorted(output)


def strict_group_threshold(groups: Mapping[str, Sequence[float]], target_fpr: float) -> dict[str, Any]:
    """Single strict ``>`` threshold meeting the target in every calibration group."""
    if not 0 < float(target_fpr) < 1 or not groups:
        raise ValueError("non-empty groups and target_fpr in (0,1) required")
    thresholds = {}
    for name, raw in sorted(groups.items()):
        values = np.asarray(raw, dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"invalid group: {name}")
        candidates = np.unique(values)
        feasible = [(int(np.count_nonzero(values > threshold)), float(threshold)) for threshold in candidates
                    if np.mean(values > threshold) <= target_fpr + 1e-15]
        alarms, threshold = max(feasible, key=lambda item: (item[0], -item[1]))
        thresholds[name] = {"threshold": threshold, "alarms": alarms, "n": len(values),
                            "observed_fpr": alarms / len(values)}
    global_threshold = max(row["threshold"] for row in thresholds.values())
    return {"threshold": global_threshold, "strict_comparator": ">", "target_fpr": target_fpr,
            "per_group": thresholds, "attack_examples_used": 0}


def binary_metrics(values: Sequence[bool]) -> dict[str, float | int]:
    alarms = np.asarray(values, dtype=bool)
    if not len(alarms):
        raise ValueError("non-empty decisions required")
    low, high = wilson_interval(int(alarms.sum()), len(alarms))
    return {"alarms": int(alarms.sum()), "n": len(alarms), "fpr": float(alarms.mean()),
            "wilson_low": float(low), "wilson_upper": float(high)}


def harmonic(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else float(2 * left * right / (left + right))


def family_metrics(labels: Sequence[int], alarms: Sequence[bool]) -> dict[str, float]:
    y, a = np.asarray(labels, int), np.asarray(alarms, bool)
    if y.shape != a.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned member and nonmember observations required")
    member, nonmember = float(a[y == 1].mean()), float(a[y == 0].mean())
    return {"member_tpr": member, "nonmember_tpr": nonmember,
            "adr": harmonic(member, nonmember), "member_nonmember_gap": abs(member - nonmember)}


def fit_linear_head(features: np.ndarray, labels: Sequence[int], groups: Sequence[str], *,
                    group_dro: bool, regularization: float = 1e-3, rounds: int = 8,
                    eta: float = .5) -> dict[str, Any]:
    """Fit the sole linear runtime head with balanced ERM or GroupDRO."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    group = np.asarray(groups, str)
    if x.ndim != 2 or len(x) != len(y) or len(group) != len(y) or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("aligned binary features required")
    names = sorted(np.unique(group).tolist())
    mass = np.ones(len(names), dtype=np.float64) / len(names)
    parameter = np.zeros(x.shape[1] + 1, dtype=np.float64)
    completed_rounds = rounds if group_dro else 1
    for _ in range(completed_rounds):
        weights = np.ones(len(y), dtype=np.float64)
        if group_dro:
            for index, name in enumerate(names):
                mask = group == name
                weights[mask] = mass[index] / max(mask.sum(), 1)
            weights *= len(weights) / weights.sum()
        else:
            # B0 is explicitly class-balanced ERM, not prevalence-weighted ERM.
            for label in (0.0, 1.0):
                mask = y == label
                weights[mask] = .5 / max(mask.sum(), 1)
            weights *= len(weights) / weights.sum()

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            coefficient, intercept = value[:-1], value[-1]
            score = x @ coefficient + intercept
            loss = np.logaddexp(0.0, score) - y * score
            probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
            residual = weights * (probability - y)
            result = float(np.mean(weights * loss) + .5 * regularization * coefficient @ coefficient)
            gradient = np.r_[x.T @ residual / len(y) + regularization * coefficient, residual.mean()]
            return result, gradient

        fitted = minimize(objective, parameter, method="L-BFGS-B", jac=True,
                          options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-7})
        if not fitted.success and not np.isfinite(fitted.x).all():
            raise RuntimeError(f"linear head failed: {fitted.message}")
        parameter = fitted.x
        if group_dro:
            score = x @ parameter[:-1] + parameter[-1]
            losses = np.logaddexp(0.0, score) - y * score
            group_loss = np.asarray([losses[group == name].mean() for name in names])
            mass *= np.exp(eta * group_loss); mass /= mass.sum()
    return {"coefficient": parameter[:-1].astype(np.float32), "intercept": float(parameter[-1]),
            "group_weights": {name: float(mass[index]) for index, name in enumerate(names)},
            "objective": "GroupDRO" if group_dro else "balanced ERM", "hidden_layers": 0}


def candidate_feasible(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    for key in ("known_mean_fpr", "known_max_fpr", "qrecc_original_fpr", "qrecc_rewrite_fpr",
                "quac_fpr", "topiocqa_fpr"):
        if float(row.get(key, math.inf)) > .02:
            failures.append(key)
    if abs(float(row.get("qrecc_rewrite_fpr", math.inf)) - float(row.get("qrecc_original_fpr", -math.inf))) > .01:
        failures.append("qrecc_delta")
    if float(row.get("max_member_nonmember_gap", math.inf)) > .10:
        failures.append("member_nonmember_gap")
    return not failures, failures


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audited = []
    for source in rows:
        row = dict(source); feasible, failures = candidate_feasible(row)
        row["feasible"] = feasible; row["failures"] = ";".join(failures)
        row["robust_min"] = min(float(row["worst_known_domain_adr"]), float(row["lofo_worst"]),
                                float(row["finance_fm_adr"])) if feasible else -math.inf
        audited.append(row)
    eligible = [row for row in audited if row["feasible"]]
    if not eligible:
        raise RuntimeError("NO_FEASIBLE_LOLA_V2_CANDIDATE")
    selected = sorted(eligible, key=lambda row: (-float(row["robust_min"]),
        float(row["finance_fpr"]), float(row["qrecc_rewrite_fpr"]), -float(row["lofo_median"]),
        -float(row["known_fm_adr"]), int(row["parameter_count"]), str(row["candidate"])))[0]
    return dict(selected), audited


__all__ = ["binary_metrics", "candidate_feasible", "family_metrics", "fit_linear_head",
           "invariant_pairs", "select_candidate", "stable_hash", "strict_group_threshold"]
