"""Exact query-subset attacks with an explicit detector/attacker boundary.

The detector receives only canonical retrieval statistics.  The attack utility
evaluator may use the attacker's target document identifier because that is an
input to a document-target membership attack, but never receives the ground
truth membership label during subset selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import itertools
import math


@dataclass(frozen=True)
class SubsetEvaluation:
    session_id: str
    subset_size: int
    query_indices: tuple[int, ...]
    detector_score: float
    detector_blocked: bool
    attack_native_score: float
    attack_utility_proxy: float
    attack_family: str
    selection_policy: str


def enumerate_query_subsets(
    rows: Sequence[Mapping[str, object]],
    *,
    max_subset_size: int = 5,
) -> Iterable[tuple[int, ...]]:
    """Enumerate every non-empty subset up to ``max_subset_size`` exactly."""

    if max_subset_size < 1:
        raise ValueError("max_subset_size must be positive")
    upper = min(len(rows), int(max_subset_size))
    for size in range(1, upper + 1):
        yield from itertools.combinations(range(len(rows)), size)


class TargetRetrievalUtility:
    """Label-free native MIA score based on target-document retrieval exposure.

    A document-target attacker knows the target document and sees retrieval
    identifiers/scores.  For a subset, the native score is the strongest
    target-document similarity returned by any selected query.  The proxy adds
    a reciprocal-rank factor, preferring queries that expose the target near the
    top.  Non-exposure has score zero.  Membership labels are used only later to
    compute cohort AUC and never by this evaluator.
    """

    name = "target_retrieval_exposure"

    def __init__(self, target_document_id: str) -> None:
        self.target_document_id = str(target_document_id)

    def query_values(self, row: Mapping[str, object]) -> tuple[float, float]:
        identifiers = [str(value) for value in row.get("doc_ids", ())]
        scores = [float(value) for value in row.get("scores", ())]
        if len(identifiers) != len(scores):
            raise ValueError("doc_ids and scores must have equal lengths")
        if self.target_document_id not in identifiers:
            return 0.0, 0.0
        index = identifiers.index(self.target_document_id)
        native = scores[index]
        proxy = native / math.sqrt(index + 1.0)
        return float(native), float(proxy)

    def evaluate(
        self,
        rows: Sequence[Mapping[str, object]],
        subset: Sequence[int],
    ) -> tuple[float, float]:
        values = [self.query_values(rows[int(index)]) for index in subset]
        return (
            max((value[0] for value in values), default=0.0),
            max((value[1] for value in values), default=0.0),
        )


def evaluate_subset(
    rows: Sequence[Mapping[str, object]],
    subset: Sequence[int],
    detector: object,
    attack_evaluator: object,
    *,
    session_id: str = "unknown",
    attack_family: str = "unknown",
    selection_policy: str = "unselected_exact_candidate",
) -> SubsetEvaluation:
    """Evaluate one subset without granting the selector a membership label."""

    indices = tuple(int(value) for value in subset)
    selected = [rows[index] for index in indices]
    detector_result = detector.score_set(selected)
    native, proxy = attack_evaluator.evaluate(rows, indices)
    return SubsetEvaluation(
        session_id=str(session_id),
        subset_size=len(indices),
        query_indices=indices,
        detector_score=float(detector_result["score"]),
        detector_blocked=bool(detector_result["blocked"]),
        attack_native_score=float(native),
        attack_utility_proxy=float(proxy),
        attack_family=str(attack_family),
        selection_policy=str(selection_policy),
    )


def select_detector_only_weakest(
    candidates: Sequence[SubsetEvaluation],
) -> SubsetEvaluation:
    if not candidates:
        raise ValueError("candidates cannot be empty")
    return min(candidates, key=lambda row: (row.detector_score, row.query_indices))


def select_utility_constrained(
    candidates: Sequence[SubsetEvaluation],
    *,
    full_utility_proxy: float,
    retention: float,
) -> SubsetEvaluation:
    """Minimize detector score subject to a label-free utility constraint."""

    if not 0.0 <= float(retention) <= 1.0:
        raise ValueError("retention must be in [0, 1]")
    required = float(retention) * float(full_utility_proxy)
    feasible = [row for row in candidates if row.attack_utility_proxy + 1e-15 >= required]
    if not feasible:
        # Finite precision can make rho=1 miss by a few ulps.  Falling back to
        # maximum proxy remains label-free and is recorded by the caller.
        best = max(row.attack_utility_proxy for row in candidates)
        feasible = [row for row in candidates if row.attack_utility_proxy == best]
    return min(feasible, key=lambda row: (row.detector_score, row.query_indices))


def select_oracle_diagnostic(
    candidates: Sequence[SubsetEvaluation], *, member_label: int
) -> SubsetEvaluation:
    """Label-using diagnostic upper bound; never a realizable attack result."""

    if not candidates:
        raise ValueError("candidates cannot be empty")
    direction = -1.0 if int(member_label) == 1 else 1.0
    return min(
        candidates,
        key=lambda row: (
            direction * row.attack_native_score,
            row.detector_score,
            row.query_indices,
        ),
    )
