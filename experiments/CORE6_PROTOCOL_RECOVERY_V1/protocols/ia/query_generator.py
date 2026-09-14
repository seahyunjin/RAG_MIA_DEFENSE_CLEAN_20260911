"""Fail-closed IA query adapter.

The paper's immutable 30-query/GPT-4o artifact is unavailable.  This module
validates an author-provided bundle if one is supplied later; it never creates
a replacement and never relabels the 15-query repository variant as paper IA.
"""

from __future__ import annotations

from protocols.common import ProtocolInputError, ProtocolUnavailableError, require_membership_label


PAPER_QUERY_COUNT = 30
PAPER_QUERY_MODEL = "gpt-4o"
SHADOW_GT_MODEL = "gpt-4o-mini"


def generate_queries(*_args, **_kwargs):
    raise ProtocolUnavailableError(
        "IA paper-exact 30-query generation is unavailable without the immutable author bundle; "
        "the repository 15-query data is a separate IA_REPOSITORY_VARIANT"
    )


def validate_immutable_bundle(records: list[dict], target_id: str, membership: str) -> list[dict]:
    require_membership_label(membership)
    if len(records) != PAPER_QUERY_COUNT:
        raise ProtocolInputError("paper IA bundle must contain exactly 30 query/GT records")
    expected = set(range(1, PAPER_QUERY_COUNT + 1))
    indices = set()
    for record in records:
        if record.get("target_id") != target_id or record.get("membership") != membership:
            raise ProtocolInputError("IA bundle target/membership mismatch")
        if not record.get("query") or record.get("ground_truth") not in {"Yes", "No"}:
            raise ProtocolInputError("IA bundle requires non-empty query and immutable Yes/No GT")
        indices.add(record.get("query_index"))
        if not record.get("query_sha256") or not record.get("gt_sha256"):
            raise ProtocolInputError("IA immutable query and GT hashes are required")
    if indices != expected:
        raise ProtocolInputError("IA query indices must be exactly 1..30")
    return records

