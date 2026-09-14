#!/usr/bin/env python3
"""Exact JSON-only validators for IA-Std-Q15-v4-TwoStage."""
from __future__ import annotations

import json
from dataclasses import dataclass

LABELS = {"Yes", "No", "Unknown"}


@dataclass(frozen=True)
class QuestionsResult:
    valid: bool
    items: tuple[tuple[int, str], ...]
    reason: str
    inferred_fields: int = 0


@dataclass(frozen=True)
class JudgmentsResult:
    valid: bool
    items: tuple[tuple[int, str], ...]
    reason: str
    inferred_fields: int = 0


def parse_questions(raw: str) -> QuestionsResult:
    try:
        value = json.loads(raw)
    except Exception:
        return QuestionsResult(False, (), "INVALID_JSON")
    if not isinstance(value, dict) or set(value) != {"queries"} or not isinstance(value["queries"], list):
        return QuestionsResult(False, (), "ROOT_SCHEMA_MISMATCH")
    rows = value["queries"]
    if len(rows) != 15:
        return QuestionsResult(False, (), "ENTRY_COUNT_NOT_15")
    parsed = []
    for expected, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"id", "question"}:
            return QuestionsResult(False, (), "ENTRY_SCHEMA_MISMATCH")
        if type(row["id"]) is not int or row["id"] != expected:
            return QuestionsResult(False, (), "ID_SEQUENCE_MISMATCH")
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            return QuestionsResult(False, (), "EMPTY_QUESTION")
        parsed.append((expected, question.strip()))
    normalized = [" ".join(question.casefold().split()) for _, question in parsed]
    if len(set(normalized)) != 15:
        return QuestionsResult(False, (), "DUPLICATE_QUESTION")
    return QuestionsResult(True, tuple(parsed), "VALID_EXACT_SCHEMA", 0)


def parse_judgments(raw: str) -> JudgmentsResult:
    try:
        value = json.loads(raw)
    except Exception:
        return JudgmentsResult(False, (), "INVALID_JSON")
    if not isinstance(value, dict) or set(value) != {"judgments"} or not isinstance(value["judgments"], list):
        return JudgmentsResult(False, (), "ROOT_SCHEMA_MISMATCH")
    rows = value["judgments"]
    if len(rows) != 15:
        return JudgmentsResult(False, (), "ENTRY_COUNT_NOT_15")
    parsed = []
    for expected, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"id", "judgment"}:
            return JudgmentsResult(False, (), "ENTRY_SCHEMA_MISMATCH")
        if type(row["id"]) is not int or row["id"] != expected:
            return JudgmentsResult(False, (), "ID_SEQUENCE_MISMATCH")
        if type(row["judgment"]) is not str or row["judgment"] not in LABELS:
            return JudgmentsResult(False, (), "INVALID_JUDGMENT")
        parsed.append((expected, row["judgment"]))
    return JudgmentsResult(True, tuple(parsed), "VALID_EXACT_SCHEMA", 0)
