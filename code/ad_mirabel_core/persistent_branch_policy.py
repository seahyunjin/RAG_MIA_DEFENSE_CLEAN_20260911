"""Persistent branch-calibrated Early-to-CCT deployment policy.

The policy keeps Mirabel, Q1--Q4 heuristic-free Early, and Q5+ rolling CCT as
three separately thresholded branches.  Client-side conversation boundaries do
not clear server state when a stable linkage key is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .heuristic_free_early import HeuristicFreeCalibration, branch_risk_curves


BRANCHES = ("Mirabel", "Early", "Rolling CCT")


@dataclass(frozen=True)
class BranchThresholds:
    mirabel: float
    early: float
    rolling_cct: float
    target_fpr: float
    observed_validation_fpr: float
    alpha_ratios: tuple[float, float, float]
    alpha_scale: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "Mirabel": self.mirabel,
            "Early": self.early,
            "Rolling CCT": self.rolling_cct,
        }


def threshold_at_tail_probability(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("threshold reference must be a nonempty vector")
    if alpha <= 0:
        return float(np.max(values))
    ordered = np.sort(values)
    probability = 1.0 - min(float(alpha), 0.5)
    # NumPy's ``method="higher"`` is the order statistic at
    # ceil(probability * (n - 1)).  Writing it explicitly avoids thousands of
    # repeated quantile allocations during the branch-budget grid search.
    index = int(np.ceil(probability * (len(ordered) - 1)))
    return float(ordered[index])


def branch_matrix(
    sessions: Sequence[Sequence[Mapping[str, object]]],
    calibration: HeuristicFreeCalibration,
    *,
    reset_every: int | None = None,
) -> dict[str, np.ndarray]:
    if not sessions:
        raise ValueError("sessions cannot be empty")
    rows = [
        branch_risk_curves(session, calibration, reset_every=reset_every)
        for session in sessions
    ]
    return {
        branch: np.asarray([row[branch] for row in rows], dtype=float)
        for branch in BRANCHES
    }


def weighted_branch_fusion_scores(
    matrices: Mapping[str, np.ndarray],
    benign_reference: Mapping[str, np.ndarray],
    *,
    alpha_ratios: Sequence[float],
    reference_mode: str = "terminal",
    sticky: bool = True,
) -> np.ndarray:
    """Fuse branch evidence as a weighted minimum empirical p-value.

    Each branch is first converted to an empirical upper-tail p-value using
    benign reference sessions.  The fused p-value is ``min(p_i / w_i, 1)``,
    which is the weighted Bonferroni union test.  Consequently the allocation
    is interpretable as an FPR budget rather than an arbitrary score weight.

    ``terminal`` uses each benign session's final cumulative branch score as
    the reference for every turn and is appropriate for an anytime session
    decision.  ``prefix`` uses a separate reference at every query budget and
    is intended for apples-to-apples Q-specific ROC analysis only.
    """
    ratios = np.asarray(alpha_ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isfinite(ratios).all():
        raise ValueError("three finite positive alpha ratios are required")
    ratios = ratios / ratios.sum()
    shape = np.asarray(matrices["Mirabel"], dtype=float).shape
    if len(shape) != 2:
        raise ValueError("branch matrices must be two-dimensional")
    combined_p = np.ones(shape, dtype=float)
    for branch_index, branch in enumerate(BRANCHES):
        values = np.asarray(matrices[branch], dtype=float)
        reference = np.asarray(benign_reference[branch], dtype=float)
        if values.shape != shape or reference.ndim != 2:
            raise ValueError("branch matrices and references must be aligned matrices")
        if reference.shape[1] < shape[1]:
            raise ValueError("reference horizon must cover the evaluated horizon")
        for turn in range(shape[1]):
            reference_turn = reference[:, -1] if reference_mode == "terminal" else reference[:, turn]
            ordered = np.sort(reference_turn)
            ge = len(ordered) - np.searchsorted(
                ordered, values[:, turn], side="left"
            )
            branch_p = (ge.astype(float) + 1.0) / (len(ordered) + 1.0)
            combined_p[:, turn] = np.minimum(
                combined_p[:, turn], branch_p / ratios[branch_index]
            )
    if reference_mode not in {"terminal", "prefix"}:
        raise ValueError("reference_mode must be 'terminal' or 'prefix'")
    risk = -np.log10(np.clip(combined_p, 1e-15, 1.0))
    return np.maximum.accumulate(risk, axis=1) if sticky else risk


def combined_flags(
    matrices: Mapping[str, np.ndarray], thresholds: BranchThresholds
) -> np.ndarray:
    reference_shape = np.asarray(matrices["Mirabel"]).shape
    output = np.zeros(reference_shape, dtype=bool)
    for branch, threshold in thresholds.as_mapping().items():
        values = np.asarray(matrices[branch], dtype=float)
        if values.shape != reference_shape:
            raise ValueError("branch matrices must be aligned")
        output |= values > threshold
    return np.maximum.accumulate(output, axis=1)


def fit_branch_thresholds(
    benign_matrices: Mapping[str, np.ndarray],
    *,
    alpha_ratios: Sequence[float],
    target_fpr: float,
    scale_grid: Sequence[float] | None = None,
) -> BranchThresholds:
    """Fit branch thresholds using benign validation sessions only.

    Ratios are frozen model structure.  A scalar multiplier is selected using
    benign validation only so the union of all branch decisions is as close as
    possible to, without exceeding, the cumulative target FPR.
    """
    ratios = np.asarray(alpha_ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isfinite(ratios).all():
        raise ValueError("three finite positive alpha ratios are required")
    ratios = ratios / ratios.sum()
    if not 0 < target_fpr < 0.5:
        raise ValueError("target_fpr must be between zero and 0.5")
    if scale_grid is None:
        scale_grid = np.linspace(0.05, 4.0, 396)
    final = {
        branch: np.asarray(benign_matrices[branch], dtype=float)[:, -1]
        for branch in BRANCHES
    }
    ordered = {branch: np.sort(values) for branch, values in final.items()}
    best: BranchThresholds | None = None
    best_key: tuple[float, float] | None = None
    for scale in scale_grid:
        alpha = np.minimum(target_fpr * ratios * float(scale), 0.49)
        values = {}
        for index, branch in enumerate(BRANCHES):
            probability = 1.0 - min(float(alpha[index]), 0.5)
            order_index = int(
                np.ceil(probability * (len(ordered[branch]) - 1))
            )
            values[branch] = float(ordered[branch][order_index])
        union = np.zeros(len(final["Mirabel"]), dtype=bool)
        for branch in BRANCHES:
            union |= final[branch] > values[branch]
        observed = float(union.mean())
        if observed > target_fpr + 1e-12:
            continue
        key = (observed, float(scale))
        if best is None or key > best_key:
            best_key = key
            best = BranchThresholds(
                mirabel=values["Mirabel"],
                early=values["Early"],
                rolling_cct=values["Rolling CCT"],
                target_fpr=float(target_fpr),
                observed_validation_fpr=observed,
                alpha_ratios=tuple(map(float, ratios)),
                alpha_scale=float(scale),
            )
    if best is None:
        raise RuntimeError("no branch thresholds satisfy the validation FPR")
    return best


class PersistentEarlyCCTDetector:
    """Stateful detector keyed by a server-controlled linkage identifier."""

    def __init__(
        self,
        calibration: HeuristicFreeCalibration,
        thresholds: BranchThresholds,
        *,
        max_history: int = 30,
    ) -> None:
        if max_history < 5:
            raise ValueError("max_history must be at least five")
        self.calibration = calibration
        self.thresholds = thresholds
        self.max_history = int(max_history)
        self._history: dict[str, list[Mapping[str, object]]] = {}
        self._blocked: set[str] = set()

    def observe(self, linkage_key: str, row: Mapping[str, object]) -> dict[str, object]:
        key = str(linkage_key)
        history = self._history.setdefault(key, [])
        history.append(row)
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]
        branches = branch_risk_curves(history, self.calibration)
        index = len(history) - 1
        hits = {
            branch: bool(branches[branch][index] > threshold)
            for branch, threshold in self.thresholds.as_mapping().items()
        }
        if any(hits.values()):
            self._blocked.add(key)
        active = [branch for branch, hit in hits.items() if hit]
        return {
            "blocked": key in self._blocked,
            "branch": "+".join(active) if active else None,
            "query_count": len(history),
            "mirabel_risk": float(branches["Mirabel"][index]),
            "early_risk": float(branches["Early"][index]),
            "rolling_cct_risk": float(branches["Rolling CCT"][index]),
        }

    def client_reset(self, linkage_key: str) -> None:
        """A UI/client reset intentionally does not clear server history."""
        if str(linkage_key) not in self._history:
            self._history[str(linkage_key)] = []

    def unlinkable_reset(self, old_key: str, new_key: str) -> None:
        """Start a new identity; state cannot be transferred without linkage."""
        del old_key  # explicit documentation: no cross-identity information exists
        self._history.setdefault(str(new_key), [])

    def clear_for_retention_policy(self, linkage_key: str) -> None:
        key = str(linkage_key)
        self._history.pop(key, None)
        self._blocked.discard(key)


class PersistentFusedEarlyCCTDetector:
    """Stateful weighted-p fusion used by the robust any-query policy."""

    def __init__(
        self,
        calibration: HeuristicFreeCalibration,
        benign_reference: Mapping[str, np.ndarray],
        ratios_by_target: Mapping[float, Sequence[float]],
        *,
        max_history: int,
    ) -> None:
        if max_history < 5:
            raise ValueError("max_history must be at least five")
        if any(np.asarray(benign_reference[branch]).shape[1] < max_history for branch in BRANCHES):
            raise ValueError("benign reference must cover max_history")
        self.calibration = calibration
        self.benign_reference = {
            branch: np.asarray(benign_reference[branch], dtype=float)
            for branch in BRANCHES
        }
        self.ratios_by_target = {
            float(target): tuple(map(float, ratios))
            for target, ratios in ratios_by_target.items()
        }
        self.max_history = int(max_history)
        self._thresholds: dict[float, float] = {}
        for target, ratios in self.ratios_by_target.items():
            reference_score = weighted_branch_fusion_scores(
                self.benign_reference,
                self.benign_reference,
                alpha_ratios=ratios,
                reference_mode="terminal",
                sticky=True,
            )
            self._thresholds[target] = threshold_at_tail_probability(
                reference_score[:, self.max_history - 1], target
            )
        self._history: dict[str, list[Mapping[str, object]]] = {}
        self._blocked: set[tuple[str, float]] = set()

    def observe(
        self, linkage_key: str, row: Mapping[str, object], *, target_fpr: float
    ) -> dict[str, object]:
        target = float(target_fpr)
        if target not in self.ratios_by_target:
            raise KeyError(f"unconfigured target FPR: {target}")
        key = str(linkage_key)
        history = self._history.setdefault(key, [])
        history.append(row)
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]
        matrices = branch_matrix([history], self.calibration)
        score = weighted_branch_fusion_scores(
            matrices,
            self.benign_reference,
            alpha_ratios=self.ratios_by_target[target],
            reference_mode="terminal",
            sticky=True,
        )
        current = float(score[0, -1])
        state_key = (key, target)
        if current > self._thresholds[target]:
            self._blocked.add(state_key)
        return {
            "blocked": state_key in self._blocked,
            "query_count_in_retention_window": len(history),
            "fusion_risk": current,
            "deployment_threshold": self._thresholds[target],
            "target_fpr": target,
        }

    def client_reset(self, linkage_key: str) -> None:
        """A presentation/UI boundary does not erase server-linked evidence."""
        self._history.setdefault(str(linkage_key), [])

    def unlinkable_reset(self, new_key: str) -> None:
        """A genuinely unlinkable identity necessarily starts without history."""
        self._history.setdefault(str(new_key), [])

    def clear_for_retention_policy(self, linkage_key: str) -> None:
        key = str(linkage_key)
        self._history.pop(key, None)
        self._blocked = {item for item in self._blocked if item[0] != key}
