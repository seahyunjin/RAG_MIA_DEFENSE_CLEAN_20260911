"""Fail-closed retrieval-lineage checks for score-table joins.

Every score artifact participating in a query-id join must declare the hash of
the retrieval database that produced it. A mismatch is an exception, not a
warning.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

class RetrievalLineageMismatch(ValueError):
    """Raised when score artifacts come from different retrieval databases."""

def require_same_retrieval_db(*artifacts: Mapping[str, Any]) -> str:
    """Return the common database hash or raise on missing/mismatched lineage."""
    if len(artifacts) < 2:
        raise ValueError("at least two artifacts are required")
    declared=[]
    for index, artifact in enumerate(artifacts):
        name=str(artifact.get("name",f"artifact_{index}"))
        value=artifact.get("retrieval_db_hash")
        if not isinstance(value,str) or not value.strip():
            raise RetrievalLineageMismatch(f"missing retrieval_db_hash for {name}")
        declared.append((name,value.strip().lower()))
    hashes={value for _,value in declared}
    if len(hashes)!=1:
        detail=", ".join(f"{name}={value}" for name,value in declared)
        raise RetrievalLineageMismatch(f"retrieval lineage mismatch: {detail}")
    return declared[0][1]

def join_rows_by_query_id(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]], *, left_lineage: Mapping[str, Any], right_lineage: Mapping[str, Any]):
    """Inner-join rows only after retrieval-database identity is verified."""
    require_same_retrieval_db(left_lineage,right_lineage)
    right={str(row["query_id"]):row for row in right_rows}
    return [(row,right[str(row["query_id"])]) for row in left_rows if str(row["query_id"]) in right]
