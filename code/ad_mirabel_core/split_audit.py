"""Group-level leakage checks shared by Experiment 36 and unit tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def audit_group_leakage(
    rows: Sequence[Mapping[str, object]],
    *,
    split_field: str,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for field in group_fields:
        locations: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get(field) not in (None, ""):
                locations[str(row[field])].add(str(row[split_field]))
        overlaps = {value: splits for value, splits in locations.items() if len(splits) > 1}
        findings.append(
            {
                "group_field": field,
                "unique_groups": len(locations),
                "overlap_count": len(overlaps),
                "overlap_examples": sorted(overlaps)[:10],
                "passed": len(overlaps) == 0,
            }
        )
    return findings

