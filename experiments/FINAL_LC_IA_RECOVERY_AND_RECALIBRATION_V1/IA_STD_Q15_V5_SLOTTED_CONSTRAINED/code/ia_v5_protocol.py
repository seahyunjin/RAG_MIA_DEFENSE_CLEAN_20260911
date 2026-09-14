"""Pure validation helpers for IA-Std-Q15-v5-SlottedConstrained."""
from __future__ import annotations

import re
from dataclasses import dataclass

STARTERS = ("Is", "Are", "Was", "Were", "Does", "Do", "Did", "Can", "Could",
            "Has", "Have", "Had", "Will", "Would", "Should")
LABELS = ("Yes", "No", "Unknown")
FORBIDDEN_REFERENCES = ("the study", "the passage")


def normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class QuestionResult:
    valid: bool
    value: str
    reason: str


def validate_question(raw: str, previous: list[str]) -> QuestionResult:
    value = raw.strip()
    if not value:
        return QuestionResult(False, value, "EMPTY")
    if "\n" in value or "\r" in value:
        return QuestionResult(False, value, "MULTILINE")
    if any(marker in value for marker in ('"', "{", "}", "[", "]")):
        return QuestionResult(False, value, "WRAPPER_OR_QUOTE")
    if not value.endswith("?") or value.count("?") != 1:
        return QuestionResult(False, value, "QUESTION_TERMINATOR")
    if not re.match(r"^(?:" + "|".join(STARTERS) + r")\b", value):
        return QuestionResult(False, value, "NOT_BINARY_QUESTION")
    normalized = normalize_question(value)
    if any(phrase in normalized for phrase in FORBIDDEN_REFERENCES):
        return QuestionResult(False, value, "FORBIDDEN_REFERENCE")
    if normalized in {normalize_question(item) for item in previous}:
        return QuestionResult(False, value, "EXACT_DUPLICATE")
    return QuestionResult(True, value, "OK")


def validate_label(raw: str) -> tuple[bool, str, str]:
    value = raw.strip()
    if value not in LABELS:
        return False, value, "INVALID_JUDGMENT"
    return True, value, "OK"
