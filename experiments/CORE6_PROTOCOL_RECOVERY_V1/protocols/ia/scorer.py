"""Paper IA formula; usable only with a validated immutable bundle."""

from __future__ import annotations

import re

from protocols.common import ProtocolInputError


PAPER_QUERY_COUNT = 30
UNK_PENALTY = 5.0
SCORE_POLARITY = "higher_is_more_member"


def normalize_response(response: str) -> str:
    if not isinstance(response, str):
        raise ProtocolInputError("IA response must be text")
    lowered = response.lower()
    if any(term in lowered for term in ("i don't know", "i do not know", "not sure", "unable to")):
        return "UNK"
    tokens = {token.lower() for token in re.findall(r"\b(?:yes|no)\b", response, re.I)}
    if tokens == {"yes"}:
        return "Yes"
    if tokens == {"no"}:
        return "No"
    return "UNK"


def score_session(responses: list[str], ground_truth: list[str]) -> float:
    if len(responses) != PAPER_QUERY_COUNT or len(ground_truth) != PAPER_QUERY_COUNT:
        raise ProtocolInputError("paper IA scorer requires exactly 30 responses and 30 GT labels")
    if any(gt not in {"Yes", "No"} for gt in ground_truth):
        raise ProtocolInputError("IA ground truth must be immutable Yes/No labels")
    score = 0.0
    for response, gt in zip(responses, ground_truth):
        normalized = normalize_response(response)
        score += float(normalized == gt)
        score -= UNK_PENALTY * float(normalized == "UNK")
    return score / PAPER_QUERY_COUNT

