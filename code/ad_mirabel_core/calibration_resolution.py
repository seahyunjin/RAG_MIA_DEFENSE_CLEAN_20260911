"""Resolution checks for finite-sample empirical low-FPR calibration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


class InsufficientCalibrationSessions(RuntimeError):
    """Raised when an empirical operating point cannot be represented."""


@dataclass(frozen=True)
class CalibrationResolution:
    num_independent_sessions: int
    target_fpr: float
    minimum_empirical_p: float
    minimum_required_sessions: int
    expected_tail_count: float
    resolvable: bool
    status: str


def assess_empirical_resolution(
    *, num_independent_sessions: int, target_fpr: float
) -> CalibrationResolution:
    if num_independent_sessions <= 0:
        raise ValueError("num_independent_sessions must be positive")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between zero and one")
    p_min = 1.0 / (num_independent_sessions + 1.0)
    minimum_required = ceil(1.0 / target_fpr) - 1
    expected_tail_count = num_independent_sessions * target_fpr
    resolvable = target_fpr >= p_min
    if not resolvable:
        status = "UNRESOLVABLE"
    elif expected_tail_count < 5:
        status = "HIGHLY_UNSTABLE"
    elif expected_tail_count < 10:
        status = "DEVELOPMENT_ONLY"
    else:
        status = "LOW_FPR_READY"
    return CalibrationResolution(
        num_independent_sessions=num_independent_sessions,
        target_fpr=target_fpr,
        minimum_empirical_p=p_min,
        minimum_required_sessions=minimum_required,
        expected_tail_count=expected_tail_count,
        resolvable=resolvable,
        status=status,
    )


def require_empirical_resolution(
    *, num_independent_sessions: int, target_fpr: float
) -> CalibrationResolution:
    result = assess_empirical_resolution(
        num_independent_sessions=num_independent_sessions,
        target_fpr=target_fpr,
    )
    if not result.resolvable:
        raise InsufficientCalibrationSessions(
            f"FPR={target_fpr:g} is unresolvable with "
            f"n={num_independent_sessions}; minimum n={result.minimum_required_sessions}"
        )
    return result
