"""Label-free order and subset attacks against public detector evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools

import numpy as np


DEFAULT_RANDOM_SEED = 20260810


def _stable_seed(session_id: str, replicate: int, seed: int) -> int:
    payload = f"{seed}:{session_id}:{replicate}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def basic_order_indices(
    rows: Sequence[Mapping[str, object]],
    ordering: str,
    *,
    session_id: str,
) -> list[int]:
    indices = list(range(len(rows)))
    if ordering == "original":
        return indices
    if ordering == "highest-top1-first":
        return sorted(indices, key=lambda i: (-float(rows[i]["top1"]), i))
    if ordering == "lowest-top1-first":
        return sorted(indices, key=lambda i: (float(rows[i]["top1"]), i))
    if ordering == "lowest-canonical-margin-first":
        return sorted(indices, key=lambda i: (float(rows[i]["mirabel_margin"]), i))
    if ordering.startswith("random-"):
        replicate = int(ordering.rsplit("-", 1)[1])
        return list(
            map(
                int,
                np.random.default_rng(
                    _stable_seed(session_id, replicate, DEFAULT_RANDOM_SEED)
                ).permutation(len(rows)),
            )
        )
    raise KeyError(ordering)


def random_100_indices(
    rows: Sequence[Mapping[str, object]], *, session_id: str, seed: int = DEFAULT_RANDOM_SEED
) -> list[list[int]]:
    return [
        list(
            map(
                int,
                np.random.default_rng(_stable_seed(session_id, replicate, seed)).permutation(
                    len(rows)
                ),
            )
        )
        for replicate in range(100)
    ]


def evidence_order_indices(
    evidence: Sequence[float], *, weakest_first: bool = True
) -> list[int]:
    values = np.asarray(evidence, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("query evidence must be a finite vector")
    return sorted(
        range(len(values)),
        key=lambda i: (values[i] if weakest_first else -values[i], i),
    )


def greedy_white_box_min_prefix(
    rows: Sequence[Mapping[str, object]],
    score_prefix: Callable[[Sequence[Mapping[str, object]]], float],
) -> list[int]:
    """Choose each next query to minimize the public current-prefix score."""

    prefix: list[Mapping[str, object]] = []
    chosen: list[int] = []
    remaining = list(range(len(rows)))
    while remaining:
        candidate = min(
            remaining,
            key=lambda index: (float(score_prefix(prefix + [rows[index]])), index),
        )
        chosen.append(candidate)
        prefix.append(rows[candidate])
        remaining.remove(candidate)
    return chosen


def weakest_k_subset_indices(evidence: Sequence[float], k: int) -> list[int]:
    """Exact weakest additive-evidence subset (largest query p / lowest evidence)."""

    if not 1 <= int(k) <= len(evidence):
        raise ValueError("k must be between one and the available query count")
    return evidence_order_indices(evidence, weakest_first=True)[: int(k)]


@dataclass(frozen=True)
class ExactPermutationResult:
    best_permutation_score: float
    median_permutation_score: float
    worst_permutation_score: float
    score_range: float
    worst_permutation: tuple[int, ...]
    best_first_detection_turn: int | None
    worst_first_detection_turn: int | None
    permutations_evaluated: int


def exact_permutation_summary(
    rows: Sequence[Mapping[str, object]],
    score_and_first_detection: Callable[
        [Sequence[Mapping[str, object]]], tuple[float, int | None]
    ],
) -> ExactPermutationResult:
    if not 1 <= len(rows) <= 5:
        raise ValueError("exact enumeration is restricted to one through five queries")
    records = []
    for permutation in itertools.permutations(range(len(rows))):
        score, first = score_and_first_detection([rows[index] for index in permutation])
        records.append((float(score), first, tuple(permutation)))
    scores = np.asarray([row[0] for row in records], dtype=float)
    worst_index = int(np.argmin(scores))
    first_values = [row[1] for row in records]
    finite_first = [value for value in first_values if value is not None]
    return ExactPermutationResult(
        best_permutation_score=float(scores.max()),
        median_permutation_score=float(np.median(scores)),
        worst_permutation_score=float(scores.min()),
        score_range=float(np.ptp(scores)),
        worst_permutation=records[worst_index][2],
        best_first_detection_turn=min(finite_first) if finite_first else None,
        worst_first_detection_turn=(
            None if any(value is None for value in first_values) else max(finite_first)
        ),
        permutations_evaluated=len(records),
    )


def beam_search_min_prefix(
    rows: Sequence[Mapping[str, object]],
    score_prefix: Callable[[Sequence[Mapping[str, object]]], float],
    *,
    width: int = 32,
) -> list[int]:
    """Bounded white-box search for horizons where exact enumeration is infeasible."""

    if int(width) < 1:
        raise ValueError("beam width must be positive")
    beam: list[tuple[tuple[int, ...], tuple[int, ...], float]] = [
        ((), tuple(range(len(rows))), 0.0)
    ]
    for _ in range(len(rows)):
        candidates = []
        for prefix, remaining, _ in beam:
            for index in remaining:
                proposal = prefix + (index,)
                score = float(score_prefix([rows[i] for i in proposal]))
                candidates.append(
                    (proposal, tuple(i for i in remaining if i != index), score)
                )
        candidates.sort(key=lambda item: (item[2], item[0]))
        beam = candidates[: int(width)]
    return list(beam[0][0])

