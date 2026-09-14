"""MBA mask-reconstruction accuracy scorer."""

from __future__ import annotations

import re

from protocols.common import ProtocolInputError


SCORE_POLARITY = "higher_is_more_member"


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", value.lower()))


def parse_predictions(text: str, expected_count: int) -> dict[str, str]:
    if not isinstance(text, str):
        raise ProtocolInputError("MBA response must be text")
    predictions: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*\[?Mask_(\d+)\]?\s*:\s*(\S(?:.*\S)?)\s*", line, re.I)
        if not match:
            if line.strip():
                raise ProtocolInputError(f"malformed MBA prediction line: {line!r}")
            continue
        key = f"Mask_{int(match.group(1))}"
        if key in predictions:
            raise ProtocolInputError(f"duplicate MBA prediction: {key}")
        predictions[key] = match.group(2)
    expected_keys = {f"Mask_{i}" for i in range(1, expected_count + 1)}
    if set(predictions) != expected_keys:
        raise ProtocolInputError("MBA output must contain every mask exactly once")
    return predictions


def reconstruction_accuracy(response: str, ground_truth: dict[str, list[str]]) -> float:
    if not ground_truth:
        raise ProtocolInputError("MBA ground truth cannot be empty")
    expected_keys = {f"Mask_{i}" for i in range(1, len(ground_truth) + 1)}
    if set(ground_truth) != expected_keys:
        raise ProtocolInputError("MBA mask indices must be contiguous from Mask_1")
    predictions = parse_predictions(response, len(ground_truth))
    hits = 0
    for key, accepted in ground_truth.items():
        if not isinstance(accepted, list) or not accepted:
            raise ProtocolInputError(f"{key} must have accepted answers")
        if _normalize(predictions[key]) in {_normalize(item) for item in accepted}:
            hits += 1
    return hits / len(ground_truth)


def membership_decision(accuracy: float, gamma: float, mask_count: int) -> int:
    if mask_count <= 0 or not 0 < gamma <= 1 or not 0 <= accuracy <= 1:
        raise ProtocolInputError("invalid MBA accuracy/gamma/mask_count")
    correct_count = accuracy * mask_count
    return int(correct_count > gamma * mask_count)

