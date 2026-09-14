"""Shared validation only; no common attack score is defined here."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence


class ProtocolInputError(ValueError):
    """Raised when an input cannot be scored without inventing protocol behavior."""


class ProtocolUnavailableError(RuntimeError):
    """Raised when an immutable input required by the paper is unavailable."""


def require_membership_label(label: str) -> str:
    if label not in {"member", "nonmember"}:
        raise ProtocolInputError("membership must be exactly 'member' or 'nonmember'")
    return label


def require_nonempty_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolInputError(f"{field} must be non-empty text")
    return value.strip()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text.lower())


def perplexity_from_logprobs(logprobs: Sequence[float]) -> float:
    if not logprobs or any(not math.isfinite(x) for x in logprobs):
        raise ProtocolInputError("finite token log probabilities are required")
    return math.exp(-sum(logprobs) / len(logprobs))


def require_exact_count(values: Iterable[object], expected: int, field: str) -> list[object]:
    result = list(values)
    if len(result) != expected:
        raise ProtocolInputError(f"{field} must contain exactly {expected} items")
    return result

