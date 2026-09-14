"""Paper-faithful MEntA document-level aggregation."""

from __future__ import annotations

from protocols.common import ProtocolInputError


QUERY_COUNT = 5
SCORE_POLARITY = "higher_is_more_member"


def score_session(query_evidence: list[dict]) -> float:
    """Aggregate frozen NLI outputs.

    Every item must contain boolean ``entailed`` and ``idk``.  NLI and claim
    extraction are executed by the preserved MEntA implementation pinned in
    protocol_config.json; this function only performs the paper aggregation.
    """

    if len(query_evidence) != QUERY_COUNT:
        raise ProtocolInputError("MEntA scorer requires exactly five query records")
    contributions: list[int] = []
    for item in query_evidence:
        if type(item.get("entailed")) is not bool or type(item.get("idk")) is not bool:
            raise ProtocolInputError("entailed and idk must be explicit booleans")
        if item["entailed"] and item["idk"]:
            raise ProtocolInputError("a query cannot be both an entailment hit and an abstention")
        contributions.append(-1 if item["idk"] else int(item["entailed"]))
    return sum(contributions) / QUERY_COUNT

