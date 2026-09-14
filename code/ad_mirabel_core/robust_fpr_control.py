"""Distribution-free cumulative-FPR control for the deployment detector.

This module deliberately separates *ranking* from *deployment calibration*.
The retrieval detector produces an anomaly score, but a score alone cannot
guarantee a low false-positive rate after a domain change.  Deployment uses an
independent benign threshold-lock split and chooses the least conservative
threshold whose one-sided exact binomial upper bound is below the requested
cumulative FPR.

For a Q30 policy, every threshold is fitted to the sticky maximum through Q30.
Consequently, a single threshold controls ``ever blocked by Q30`` rather than
thirty unrelated per-query error rates.  Group-specific thresholds are used
for service/retriever domains; an unseen group is explicitly uncalibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta

from .persistent_branch_policy import BRANCHES


@dataclass(frozen=True)
class PACThreshold:
    """A threshold and its validation-only false-positive certificate."""

    threshold: float
    target_fpr: float
    validation_fpr: float
    fpr_upper_bound: float
    exceedances: int
    validation_sessions: int
    confidence: float
    familywise_comparisons: int


def clopper_pearson_upper(successes: int, total: int, confidence: float) -> float:
    """Return a one-sided exact binomial upper confidence bound."""

    k = int(successes)
    n = int(total)
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def fit_pac_threshold(
    validation_terminal_scores: Sequence[float],
    target_fpr: float,
    *,
    confidence: float = 0.95,
    familywise_comparisons: int = 1,
) -> PACThreshold:
    """Fit the most powerful validation threshold with an exact FPR bound.

    ``familywise_comparisons`` applies a Bonferroni confidence correction.  It
    is intended to cover all predeclared domain x FPR certificates, not query
    prefixes: the terminal sticky score already covers every prefix jointly.
    """

    values = np.asarray(validation_terminal_scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("validation scores must be a finite nonempty vector")
    if not 0.0 < float(target_fpr) < 0.5:
        raise ValueError("target_fpr must lie between zero and 0.5")
    comparisons = int(familywise_comparisons)
    if comparisons < 1:
        raise ValueError("familywise_comparisons must be positive")
    adjusted_confidence = 1.0 - (1.0 - float(confidence)) / comparisons

    # A score is blocked iff score > threshold.  Testing unique observed
    # values is therefore sufficient; +inf provides a safe no-block fallback.
    candidates = np.concatenate([np.unique(values), [np.inf]])
    best: PACThreshold | None = None
    for threshold in candidates:
        exceedances = int(np.sum(values > threshold))
        upper = clopper_pearson_upper(
            exceedances, len(values), adjusted_confidence
        )
        if upper <= float(target_fpr) + 1e-15:
            best = PACThreshold(
                threshold=float(threshold),
                target_fpr=float(target_fpr),
                validation_fpr=exceedances / len(values),
                fpr_upper_bound=upper,
                exceedances=exceedances,
                validation_sessions=len(values),
                confidence=float(confidence),
                familywise_comparisons=comparisons,
            )
            break
    if best is None:  # pragma: no cover - +inf with enough data always works
        raise RuntimeError("no PAC-valid threshold exists")
    return best


def fit_groupwise_pac_thresholds(
    validation_scores_by_group: Mapping[str, Sequence[float]],
    targets: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, dict[float, PACThreshold]]:
    """Fit simultaneous domain x target certificates using benign data only."""

    if not validation_scores_by_group:
        raise ValueError("at least one calibration group is required")
    target_values = tuple(float(value) for value in targets)
    comparisons = len(validation_scores_by_group) * len(target_values)
    return {
        str(group): {
            target: fit_pac_threshold(
                scores,
                target,
                confidence=confidence,
                familywise_comparisons=comparisons,
            )
            for target in target_values
        }
        for group, scores in validation_scores_by_group.items()
    }


def empirical_upper_p_matrix(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert every score column to an empirical upper-tail p-value."""

    array = np.asarray(values, dtype=float)
    null = np.asarray(reference, dtype=float)
    if array.ndim != 2 or null.ndim != 2 or array.shape[1] != null.shape[1]:
        raise ValueError("values and reference must be aligned 2D matrices")
    output = np.ones_like(array, dtype=float)
    for turn in range(array.shape[1]):
        ordered = np.sort(null[:, turn])
        ge = len(ordered) - np.searchsorted(ordered, array[:, turn], side="left")
        output[:, turn] = (ge + 1.0) / (len(ordered) + 1.0)
    return output


def fuse_branch_pvalues(
    matrices: Mapping[str, np.ndarray],
    benign_reference: Mapping[str, np.ndarray],
    *,
    method: str = "cct",
    sticky: bool = True,
) -> np.ndarray:
    """Fuse three branch p-values without hand-set feature/branch weights.

    The returned quantity is an anomaly score (larger is more suspicious).
    It is always empirically thresholded downstream, so approximate analytic
    p-value validity of CCT/HMP is not used as the deployment guarantee.
    """

    p_cube = np.stack(
        [
            empirical_upper_p_matrix(
                np.asarray(matrices[branch], dtype=float),
                np.asarray(benign_reference[branch], dtype=float),
            )
            for branch in BRANCHES
        ],
        axis=2,
    )
    clipped = np.clip(p_cube, 1e-12, 1.0 - 1e-12)
    normalized = method.lower().replace("_", "-")
    if normalized == "cct":
        statistic = np.mean(np.tan((0.5 - clipped) * math.pi), axis=2)
        combined_p = 0.5 - np.arctan(statistic) / math.pi
    elif normalized in {"equal-bonferroni", "bonferroni"}:
        combined_p = np.minimum(1.0, len(BRANCHES) * np.min(clipped, axis=2))
    elif normalized in {"harmonic-mean", "hmp"}:
        combined_p = len(BRANCHES) / np.sum(1.0 / clipped, axis=2)
    elif normalized == "fisher":
        # A monotone Fisher statistic is enough because the final scale is
        # calibrated empirically; no chi-square independence assumption is made.
        statistic = -2.0 * np.sum(np.log(clipped), axis=2)
        score = statistic
        return np.maximum.accumulate(score, axis=1) if sticky else score
    else:
        raise KeyError(method)
    score = -np.log10(np.clip(combined_p, 1e-15, 1.0))
    return np.maximum.accumulate(score, axis=1) if sticky else score


class CalibratedDomainRegistry:
    """Fail closed on claims, not users, when a deployment domain is unseen.

    An unseen domain is reported as ``calibration_required``.  The registry
    does not silently reuse a threshold from another corpus, which is the
    behavior that produced the large Experiment-28 FPR shifts.
    """

    def __init__(self, certificates: Mapping[str, Mapping[float, PACThreshold]]):
        self._certificates = {
            str(group): {float(target): cert for target, cert in values.items()}
            for group, values in certificates.items()
        }

    def decide(self, group: str, score: float, target_fpr: float) -> dict[str, object]:
        name = str(group)
        target = float(target_fpr)
        if name not in self._certificates or target not in self._certificates[name]:
            return {
                "status": "calibration_required",
                "blocked": False,
                "fpr_certificate": None,
            }
        certificate = self._certificates[name][target]
        return {
            "status": "calibrated",
            "blocked": bool(float(score) > certificate.threshold),
            "fpr_certificate": certificate,
        }
