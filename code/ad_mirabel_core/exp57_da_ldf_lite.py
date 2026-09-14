"""Privacy-safe linear primitives for DA-LDF-Lite (Exp57)."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from .exp55_domain_adaptive_detector import FORBIDDEN_INFERENCE_FIELDS, validate_inference_payload


EPSILON = 1e-12


def stable_seed(value: object) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16) % (2**32)


def empirical_upper_tail(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float).reshape(-1))
    if ref.size == 0 or not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError("finite values and nonempty reference required")
    count_ge = ref.size - np.searchsorted(ref, x, side="left")
    return (1.0 + count_ge) / (ref.size + 1.0)


def target_normal_transform(matrix: Sequence[Sequence[float]], references: Sequence[Sequence[float]]) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(references) or not np.isfinite(x).all():
        raise ValueError("aligned finite feature matrix/reference list required")
    p = np.column_stack([empirical_upper_tail(x[:, j], references[j]) for j in range(x.shape[1])])
    return -np.log10(np.clip(p, EPSILON, 1.0))


def linear_score(matrix: Sequence[Sequence[float]], coefficient: Sequence[float]) -> np.ndarray:
    x, w = np.asarray(matrix, float), np.asarray(coefficient, float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("finite aligned matrix/coefficient required")
    return x @ w


def _logistic_loss_and_gradient(w: np.ndarray, x0: np.ndarray, x1: np.ndarray, regularization: float,
                                offset0: np.ndarray | None = None, offset1: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    s0 = x0 @ w + (0.0 if offset0 is None else offset0)
    s1 = x1 @ w + (0.0 if offset1 is None else offset1)
    loss0 = np.logaddexp(0.0, s0).mean()
    loss1 = np.logaddexp(0.0, -s1).mean()
    sigmoid0 = 1.0 / (1.0 + np.exp(-np.clip(s0, -40, 40)))
    sigmoid_neg1 = 1.0 / (1.0 + np.exp(np.clip(s1, -40, 40)))
    gradient = x0.T @ sigmoid0 / len(x0) - x1.T @ sigmoid_neg1 / len(x1) + regularization * w
    loss = loss0 + loss1 + .5 * regularization * float(w @ w)
    return float(loss), np.asarray(gradient, float)


def fit_linear_logistic(normal: Sequence[Sequence[float]], attack: Sequence[Sequence[float]],
                        regularization: float, *, initial: Sequence[float] | None = None,
                        frozen_offset: Sequence[float] | None = None) -> np.ndarray:
    x0, x1 = np.asarray(normal, float), np.asarray(attack, float)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1] or not len(x0) or not len(x1):
        raise ValueError("nonempty aligned normal/attack matrices required")
    if not np.isfinite(x0).all() or not np.isfinite(x1).all() or regularization <= 0:
        raise ValueError("finite matrices and positive regularization required")
    start = np.zeros(x0.shape[1], float) if initial is None else np.asarray(initial, float).copy()
    if frozen_offset is None:
        offset0 = offset1 = None
    else:
        base = np.asarray(frozen_offset, float)
        if base.shape != start.shape: raise ValueError("frozen offset shape mismatch")
        offset0, offset1 = x0 @ base, x1 @ base
    result = minimize(lambda w: _logistic_loss_and_gradient(w, x0, x1, regularization, offset0, offset1),
                      start, jac=True, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-11})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"linear logistic optimization failed: {result.message}")
    return np.asarray(result.x, float)


def balanced_pair_indices(families: Sequence[str], hard_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(families, str)
    if len(labels) == 0 or hard_count < 1: raise ValueError("attacks and hard normals required")
    attack_indices, hard_indices = [], []
    for family in sorted(set(labels)):
        available = np.flatnonzero(labels == family)
        count = min(len(available), hard_count)
        rng = np.random.default_rng(stable_seed(f"{seed}|{family}"))
        attack_indices.extend(rng.permutation(available)[:count].tolist())
        hard_indices.extend(rng.permutation(hard_count)[:count].tolist())
    return np.asarray(attack_indices, int), np.asarray(hard_indices, int)


def fit_pairwise_residual(global_coefficient: Sequence[float], attack: Sequence[Sequence[float]],
                          hard_normal: Sequence[Sequence[float]], families: Sequence[str],
                          regularization: float, seed: int) -> np.ndarray:
    w0, xa, xh = np.asarray(global_coefficient, float), np.asarray(attack, float), np.asarray(hard_normal, float)
    if xa.ndim != 2 or xh.ndim != 2 or xa.shape[1] != xh.shape[1] or xa.shape[1] != len(w0):
        raise ValueError("aligned global/attack/hard-normal features required")
    if not np.isfinite(xa).all() or not np.isfinite(xh).all() or regularization <= 0:
        raise ValueError("finite matrices and positive regularization required")
    ai, hi = balanced_pair_indices(families, len(xh), seed)
    difference = xa[ai] - xh[hi]
    base = difference @ w0

    def objective(delta: np.ndarray) -> tuple[float, np.ndarray]:
        margin = base + difference @ delta
        loss = np.logaddexp(0.0, -margin).mean() + .5 * regularization * float(delta @ delta)
        probability = 1.0 / (1.0 + np.exp(np.clip(margin, -40, 40)))
        gradient = -(difference.T @ probability) / len(difference) + regularization * delta
        return float(loss), np.asarray(gradient, float)

    result = minimize(objective, np.zeros_like(w0), jac=True, method="L-BFGS-B",
                      options={"maxiter": 400, "ftol": 1e-11})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"pairwise residual optimization failed: {result.message}")
    return np.asarray(result.x, float)


def member_nonmember_family_metrics(families: Sequence[str], member: Sequence[int], decisions: Sequence[bool]) -> dict:
    f, m, y = np.asarray(families, str), np.asarray(member, int), np.asarray(decisions, bool)
    if not (len(f) == len(m) == len(y)) or not len(f): raise ValueError("aligned nonempty inputs required")
    rows = {}
    for family in sorted(set(f)):
        mask = f == family
        rates = []
        for label in (0, 1):
            cell = y[mask & (m == label)]
            rates.append(float(cell.mean()) if len(cell) else math.nan)
        adr = float(np.nanmean(rates)); gap = float(abs(rates[1] - rates[0])) if np.isfinite(rates).all() else math.nan
        rows[family] = {"nonmember_tpr": rates[0], "member_tpr": rates[1], "adr": adr, "gap": gap}
    return {"fm_adr": float(np.mean([v["adr"] for v in rows.values()])),
            "member_tpr": float(np.nanmean([v["member_tpr"] for v in rows.values()])),
            "nonmember_tpr": float(np.nanmean([v["nonmember_tpr"] for v in rows.values()])),
            "member_nonmember_gap": float(np.nanmean([v["gap"] for v in rows.values()])), "families": rows}


def minimum_budget(rows: Sequence[Mapping[str, float]], order: Sequence[str] = ("5", "10", "25", "50", "100", "FULL")) -> str | None:
    by_budget = {str(row["budget"]): row for row in rows}
    if "FULL" not in by_budget: return None
    full = by_budget["FULL"]
    for budget in order[:-1]:
        row = by_budget.get(budget)
        if row and float(row["fm_adr"]) >= .9 * float(full["fm_adr"]) and float(row["lofo_median"]) >= .8 * float(full["lofo_median"]) and bool(row["fpr_pass"]):
            return budget
    return None


def coefficient_stability(coefficients: Sequence[Sequence[float]]) -> dict:
    matrix = np.asarray(coefficients, float)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all(): raise ValueError("at least two finite coefficient vectors required")
    majority = np.sign(np.median(matrix, axis=0))
    sign = float(np.mean(np.sign(matrix) == majority))
    correlations = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            correlations.append(float(np.corrcoef(rankdata(matrix[i]), rankdata(matrix[j]))[0, 1]))
    return {"sign_stability": sign, "rank_correlation_mean": float(np.nanmean(correlations)),
            "coefficient_norm_mean": float(np.linalg.norm(matrix, axis=1).mean()),
            "coefficient_norm_std": float(np.linalg.norm(matrix, axis=1).std())}


__all__ = [
    "FORBIDDEN_INFERENCE_FIELDS", "balanced_pair_indices", "coefficient_stability",
    "empirical_upper_tail", "fit_linear_logistic", "fit_pairwise_residual", "linear_score",
    "member_nonmember_family_metrics", "minimum_budget", "stable_seed", "target_normal_transform",
    "validate_inference_payload",
]
