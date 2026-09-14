"""Experiment-39 helpers for comparing OOF hard session subsets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import itertools

import numpy as np


def subset_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def round_stability(
    previous: Sequence[Mapping[str, object]], current: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    def keyed(rows):
        return {
            (str(row["session_id"]), str(row["source_kind"]), int(row["subset_size"])): row
            for row in rows
        }
    old, new = keyed(previous), keyed(current)
    output = []
    for key in sorted(set(old) & set(new)):
        left, right = old[key], new[key]
        output.append({
            "session_id": key[0], "source_kind": key[1], "subset_size": key[2],
            "subset_jaccard": subset_jaccard(left["query_indices"], right["query_indices"]),
            "previous_risk": float(left["detector_score"]),
            "current_risk": float(right["detector_score"]),
            "actual_utility_retention": right.get("actual_proxy_retention"),
        })
    return output


def query_frequency(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    frequency = Counter(int(i) for row in rows for i in row["query_indices"])
    return {
        "unique_selected_query_positions": len(frequency),
        "maximum_query_position_frequency": max(frequency.values(), default=0),
        "frequency_json": {str(k): int(v) for k, v in sorted(frequency.items())},
    }


def exact_subset_count(query_count: int = 15, max_size: int = 5) -> int:
    return int(sum(len(list(itertools.combinations(range(query_count), k))) for k in range(1, max_size + 1)))
