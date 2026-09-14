"""DCMI black-box differential-calibration scorer."""

from __future__ import annotations

import re

from protocols.common import ProtocolInputError


SCORE_POLARITY = "higher_is_more_member"


def parse_binary(response: str) -> int:
    if not isinstance(response, str):
        raise ProtocolInputError("DCMI response must be text")
    tokens = {token.lower() for token in re.findall(r"\b(?:yes|no)\b", response, re.I)}
    if tokens == {"yes"}:
        return 1
    if tokens == {"no"}:
        return 0
    raise ProtocolInputError("DCMI requires one unambiguous Yes/No response")


def calibrated_score(original_response: str, perturbed_response: str) -> int:
    return parse_binary(original_response) - parse_binary(perturbed_response)

