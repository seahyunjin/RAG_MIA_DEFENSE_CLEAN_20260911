#!/usr/bin/env python3
"""Paper-aligned MBA mask reconstruction scorer.

Only a complete, line-anchored ``[Mask_i]: answer`` entry is parsed.  Missing
or malformed entries are not repaired; they count as incorrect in the fixed
M-mask denominator.  This matches the paper's reconstruction-accuracy
definition and avoids conditioning the ROC cohort on model format compliance.
"""

from __future__ import annotations

import re


LINE = re.compile(r"^\s*\[Mask_(\d+)\]\s*:\s*(\S(?:.*\S)?)\s*$", re.I)
TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def normalize(value: str) -> str:
    return " ".join(TOKEN.findall(value.lower()))


def parse_predictions(response: str, mask_count: int) -> dict[str, str]:
    if not isinstance(response, str):
        raise TypeError("MBA response must be text")
    if mask_count <= 0:
        raise ValueError("mask_count must be positive")
    predictions: dict[str, str] = {}
    for line in response.splitlines():
        match = LINE.fullmatch(line)
        if not match:
            continue
        index = int(match.group(1))
        if 1 <= index <= mask_count:
            # Preserve the official evaluator's last-occurrence behavior.
            predictions[f"Mask_{index}"] = match.group(2)
    return predictions


def reconstruction_accuracy(response: str, ground_truth: dict[str, list[str]]) -> float:
    if not ground_truth:
        raise ValueError("MBA ground truth cannot be empty")
    expected = [f"Mask_{index}" for index in range(1, len(ground_truth) + 1)]
    if set(ground_truth) != set(expected):
        raise ValueError("MBA ground-truth indices must be contiguous from Mask_1")
    predictions = parse_predictions(response, len(expected))
    correct = 0
    for key in expected:
        accepted = ground_truth[key]
        if not isinstance(accepted, list) or not accepted:
            raise ValueError(f"{key} must contain at least one accepted answer")
        prediction = predictions.get(key)
        if prediction is not None and normalize(prediction) in {normalize(value) for value in accepted}:
            correct += 1
    return correct / len(expected)


def format_compliance(response: str, mask_count: int) -> tuple[bool, str]:
    """Diagnostic only; never used to exclude a score from the ROC cohort."""
    parsed = parse_predictions(response, mask_count)
    expected = {f"Mask_{index}" for index in range(1, mask_count + 1)}
    if set(parsed) != expected:
        return False, f"parsed {len(parsed)}/{mask_count} required indexed masks"
    nonempty = [line for line in response.splitlines() if line.strip()]
    if len(nonempty) != mask_count or any(LINE.fullmatch(line) is None for line in nonempty):
        return False, "response contains non-mask prose or extra lines"
    return True, "complete exact indexed-mask output"
