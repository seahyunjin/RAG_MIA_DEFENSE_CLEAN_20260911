#!/usr/bin/env python3
"""Frozen deterministic output segmentation and assembly for BC-CGD.

This module is new for BC-CGD.  It does not claim compatibility with, or
reproduction of, any deleted legacy segmenter.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


FALLBACK = "I don't know."
_BULLET_OR_LIST = re.compile(r"^(?:[-*+•]\s+|\d+[.)]\s+)")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+")


def segment_units(answer: str) -> list[str]:
    """Apply the pre-specified BC-CGD unit segmentation contract."""
    text = str(answer).strip()
    if not text:
        return []
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _BULLET_OR_LIST.match(line):
            units.append(line)
        else:
            units.extend(piece.strip() for piece in _SENTENCE_BOUNDARY.split(line)
                         if piece.strip())
    return units or [text]


def assemble_kept(units: Iterable[str], keep: Iterable[bool]) -> str:
    """Return kept original units in order, or the frozen fallback."""
    source = list(units)
    mask = list(keep)
    if len(source) != len(mask):
        raise ValueError("unit/keep length mismatch")
    selected = [unit for unit, retained in zip(source, mask) if retained]
    return " ".join(selected) if selected else FALLBACK
