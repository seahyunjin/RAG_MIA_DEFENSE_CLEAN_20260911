#!/usr/bin/env python3
"""Deterministic, syntax-only parser for frozen IA-Std-Q15 raw outputs."""
from __future__ import annotations

import re
from dataclasses import dataclass


START = re.compile(r"^\s*(?:question\s*)?(\d{1,2})\s*[\).:\-]\s*(.*)\s*$", re.I)
QUESTION_THEN_ANSWER = re.compile(
    r"^\s*(?:question\s*:\s*)?(?P<question>.+\?)\s*"
    r"(?:answer\s*(?:\d{1,2})?\s*[:\-]\s*)?(?P<label>yes|no)\s*[.!]?\s*$", re.I)
ANSWER_THEN_QUESTION = re.compile(
    r"^\s*(?:answer\s*(?:\d{1,2})?\s*[:\-]\s*)?(?P<label>yes|no)\s*"
    r"(?:question\s*:\s*)?(?P<question>.+\?)\s*$", re.I)


@dataclass(frozen=True)
class ParseOutcome:
    valid: bool
    items: tuple[tuple[str, str], ...]
    reason: str
    record_count: int
    inferred_fields: int = 0


def normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


def split_numbered_records(raw: str) -> tuple[list[tuple[int, list[str]]], bool]:
    records: list[tuple[int, list[str]]] = []
    prefix_or_footer = False
    current: tuple[int, list[str]] | None = None
    for source_line in raw.splitlines():
        line = source_line.strip().strip("` ")
        if not line:
            continue
        match = START.fullmatch(line)
        if match:
            if current is not None:
                records.append(current)
            current = (int(match.group(1)), [match.group(2).strip()])
        elif current is None:
            prefix_or_footer = True
        else:
            current[1].append(line)
    if current is not None:
        records.append(current)
    return records, prefix_or_footer


def parse_attempt(raw: str) -> ParseOutcome:
    records, outside = split_numbered_records(raw)
    if outside:
        return ParseOutcome(False, (), "UNNUMBERED_TEXT_OUTSIDE_RECORDS", len(records))
    if len(records) != 15:
        return ParseOutcome(False, (), "EXPECTED_EXACTLY_15_NUMBERED_RECORDS", len(records))
    if [number for number, _ in records] != list(range(1, 16)):
        return ParseOutcome(False, (), "NONSEQUENTIAL_OR_DUPLICATE_NUMBERING", len(records))
    items: list[tuple[str, str]] = []
    for number, lines in records:
        text = " ".join(part for part in lines if part).strip()
        # Some frozen outputs repeat the explicit record number ("1. 1. ...").
        # Removing an exactly repeated current index is formatting-only.
        text = re.sub(rf"^\s*{number}\s*[\).:\-]\s*", "", text, count=1)
        match = QUESTION_THEN_ANSWER.fullmatch(text) or ANSWER_THEN_QUESTION.fullmatch(text)
        if not match:
            if "?" not in text:
                reason = "QUESTION_CONTENT_MISSING"
            elif not re.search(r"\b(?:yes|no)\b", text, re.I):
                reason = "BINARY_JUDGMENT_MISSING"
            elif len(re.findall(r"\b(?:yes|no)\b", text, re.I)) != 1:
                reason = "MULTIPLE_BINARY_JUDGMENTS"
            else:
                reason = "UNSUPPORTED_EXTRA_CONTENT_OR_FIELD_LAYOUT"
            return ParseOutcome(False, (), reason, len(records))
        question = " ".join(match.group("question").split())
        label = match.group("label").title()
        if not question.endswith("?") or not question[:-1].strip():
            return ParseOutcome(False, (), "QUESTION_CONTENT_MISSING", len(records))
        items.append((question, label))
    if len({normalize_question(question) for question, _ in items}) != 15:
        return ParseOutcome(False, (), "DUPLICATE_QUESTION", 15)
    return ParseOutcome(True, tuple(items), "VALID_EXPLICIT_Q15", 15, 0)


def parse_session(raw_attempts: list[str]) -> tuple[ParseOutcome, int | None]:
    """Select the first of the already-frozen attempts that parses exactly; never synthesize fields."""
    outcomes = [parse_attempt(raw) for raw in raw_attempts]
    for index, outcome in enumerate(outcomes, 1):
        if outcome.valid:
            return outcome, index
    # Prefer a concrete missing-content reason; otherwise preserve the final syntactic reason.
    content_reasons = {"EXPECTED_EXACTLY_15_NUMBERED_RECORDS", "QUESTION_CONTENT_MISSING", "BINARY_JUDGMENT_MISSING"}
    selected = next((outcome for outcome in outcomes if outcome.reason in content_reasons), outcomes[-1] if outcomes else ParseOutcome(False, (), "NO_RAW_ATTEMPT", 0))
    return selected, None


def audit_category(outcome: ParseOutcome) -> str:
    if outcome.valid:
        return "FORMAT_ONLY"
    if outcome.reason in {"EXPECTED_EXACTLY_15_NUMBERED_RECORDS", "QUESTION_CONTENT_MISSING", "BINARY_JUDGMENT_MISSING"}:
        return "CONTENT_MISSING"
    return "OTHER"
