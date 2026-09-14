#!/usr/bin/env python3
"""Exact JSON-only IA-v3 schema validator; no fuzzy or semantic repair."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    valid: bool
    items: tuple[tuple[int, str, str], ...]
    reason: str
    inferred_fields: int = 0


def parse_exact(raw: str) -> Result:
    try:
        value = json.loads(raw)
    except Exception:
        return Result(False, (), "INVALID_JSON")
    if not isinstance(value, dict) or set(value) != {"queries"} or not isinstance(value["queries"], list):
        return Result(False, (), "ROOT_SCHEMA_MISMATCH")
    rows = value["queries"]
    if len(rows) != 15:
        return Result(False, (), "ENTRY_COUNT_NOT_15")
    parsed = []
    for expected, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"id", "question", "judgment"}:
            return Result(False, (), "ENTRY_SCHEMA_MISMATCH")
        if type(row["id"]) is not int or row["id"] != expected:
            return Result(False, (), "ID_SEQUENCE_MISMATCH")
        question = row["question"]
        judgment = row["judgment"]
        if not isinstance(question, str) or not question.strip():
            return Result(False, (), "EMPTY_QUESTION")
        if judgment not in {"Yes", "No", "Unknown"}:
            return Result(False, (), "INVALID_JUDGMENT")
        parsed.append((expected, question.strip(), judgment))
    normalized = [" ".join(question.casefold().split()) for _, question, _ in parsed]
    if len(set(normalized)) != 15:
        return Result(False, (), "DUPLICATE_QUESTION")
    return Result(True, tuple(parsed), "VALID_EXACT_SCHEMA", 0)
