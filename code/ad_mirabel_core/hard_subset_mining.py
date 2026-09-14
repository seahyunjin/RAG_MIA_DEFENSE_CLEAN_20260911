"""Leakage-safe exact hard-subset mining for Robust-LDF.

Mining operates on one Q15 session at a time and enumerates every one of the
4,943 subsets with size at most five.  Utility-constrained selectors receive
retrieval-native utility values but never the membership label.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import itertools

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class HardSubset:
    session_id: str
    subset_size: int
    query_indices: tuple[int, ...]
    detector_score: float
    full_utility_proxy: float
    subset_utility_proxy: float
    utility_retention_target: float | None
    actual_proxy_retention: float | None
    source_kind: str
    fold: int
    miner_training_session_ids: tuple[str, ...]


def exact_subsets(query_count: int = 15, max_subset_size: int = 5) -> Iterable[tuple[int, ...]]:
    if query_count < 1 or max_subset_size < 1:
        raise ValueError("query_count and max_subset_size must be positive")
    for size in range(1, min(query_count, max_subset_size) + 1):
        yield from itertools.combinations(range(query_count), size)


def choose_utility_valid_hard_positive(
    *,
    session_id: str,
    query_count: int,
    subset_size: int,
    score_subset: Callable[[tuple[int, ...]], float],
    query_utility_proxy: Sequence[float],
    retention: float = 0.90,
    fold: int = -1,
    miner_training_session_ids: Sequence[str] = (),
) -> HardSubset:
    """Select the weakest detector subset while preserving label-free utility."""

    utility = np.asarray(query_utility_proxy, dtype=float)
    if utility.shape != (query_count,) or not np.isfinite(utility).all() or np.any(utility < 0):
        raise ValueError("query utility must be a finite nonnegative Q-vector")
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0,1]")
    full = float(utility.max())
    required = float(retention) * full
    feasible = []
    for subset in itertools.combinations(range(query_count), int(subset_size)):
        subset_utility = float(utility[list(subset)].max())
        if subset_utility + 1e-15 >= required:
            feasible.append((float(score_subset(subset)), subset, subset_utility))
    if not feasible:
        raise RuntimeError("no utility-valid subset exists; utility implementation is inconsistent")
    score, indices, subset_utility = min(feasible, key=lambda row: (row[0], row[1]))
    proxy_retention = 1.0 if full == 0.0 else subset_utility / full
    return HardSubset(
        session_id=str(session_id), subset_size=int(subset_size), query_indices=tuple(indices),
        detector_score=score, full_utility_proxy=full, subset_utility_proxy=subset_utility,
        utility_retention_target=float(retention), actual_proxy_retention=float(proxy_retention),
        source_kind="hard_positive", fold=int(fold),
        miner_training_session_ids=tuple(map(str, miner_training_session_ids)),
    )


def choose_hard_negative(
    *,
    session_id: str,
    query_count: int,
    subset_size: int,
    score_subset: Callable[[tuple[int, ...]], float],
    fold: int = -1,
    miner_training_session_ids: Sequence[str] = (),
) -> HardSubset:
    candidates = [
        (float(score_subset(subset)), subset)
        for subset in itertools.combinations(range(query_count), int(subset_size))
    ]
    score, indices = max(candidates, key=lambda row: (row[0], tuple(-v for v in row[1])))
    return HardSubset(
        session_id=str(session_id), subset_size=int(subset_size), query_indices=tuple(indices),
        detector_score=score, full_utility_proxy=0.0, subset_utility_proxy=0.0,
        utility_retention_target=None, actual_proxy_retention=None,
        source_kind="hard_negative", fold=int(fold),
        miner_training_session_ids=tuple(map(str, miner_training_session_ids)),
    )


def grouped_three_folds(sessions: Sequence[Mapping[str, object]]) -> list[tuple[list[int], list[int]]]:
    """Return deterministic group-disjoint folds using target/source identities."""

    if len(sessions) < 3:
        raise ValueError("at least three sessions are required")
    groups = np.asarray([
        str(row.get("target_document_id") or row.get("source_conversation_id") or row["session_id"])
        for row in sessions
    ])
    splitter = GroupKFold(n_splits=3)
    indices = np.arange(len(sessions))
    return [(train.tolist(), held.tolist()) for train, held in splitter.split(indices, groups=groups)]


def assert_cross_fitted(subsets: Sequence[HardSubset]) -> None:
    for item in subsets:
        if item.session_id in set(item.miner_training_session_ids):
            raise ValueError(f"cross-fit leakage: {item.session_id} was mined by its training model")


def unique_hard_query_indices(subsets: Sequence[HardSubset]) -> dict[str, tuple[int, ...]]:
    """Deduplicate queries within a session; frequency remains diagnostic only."""

    by_session: dict[str, set[int]] = {}
    for item in subsets:
        by_session.setdefault(item.session_id, set()).update(item.query_indices)
    return {key: tuple(sorted(values)) for key, values in by_session.items()}


def mining_statistics(subsets: Sequence[HardSubset]) -> list[dict[str, object]]:
    output = []
    for session_id in sorted({row.session_id for row in subsets}):
        selected = [row for row in subsets if row.session_id == session_id]
        frequency = Counter(index for row in selected for index in row.query_indices)
        sets = [set(row.query_indices) for row in selected]
        jaccard = []
        for left, right in itertools.combinations(sets, 2):
            jaccard.append(len(left & right) / len(left | right))
        output.append({
            "session_id": session_id,
            "source_kind": selected[0].source_kind,
            "subsets": len(selected),
            "unique_hard_query_count": len(frequency),
            "maximum_query_selection_frequency": max(frequency.values(), default=0),
            "mean_subset_jaccard": float(np.mean(jaccard)) if jaccard else 1.0,
            "query_frequency": ";".join(f"{key}:{value}" for key, value in sorted(frequency.items())),
        })
    return output
