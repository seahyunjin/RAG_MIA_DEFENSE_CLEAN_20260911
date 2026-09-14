"""RAG-MIA black-box discrete scorer.

The paper maps outputs containing neither Yes nor No to nonmember.  The paper
does not specify outputs containing both; those fail closed rather than being
assigned an invented label.
"""

from __future__ import annotations

import re

from protocols.common import ProtocolInputError


SCORE_POLARITY = "one_is_member"


def black_box_membership_score(response: str) -> int:
    if not isinstance(response, str):
        raise ProtocolInputError("RAG-MIA response must be text")
    tokens = {token.lower() for token in re.findall(r"\b(?:yes|no)\b", response, re.I)}
    if tokens == {"yes"}:
        return 1
    if tokens in ({"no"}, set()):
        return 0
    raise ProtocolInputError("paper does not define responses containing both Yes and No")


def aggregate_single(response: str) -> int:
    return black_box_membership_score(response)

