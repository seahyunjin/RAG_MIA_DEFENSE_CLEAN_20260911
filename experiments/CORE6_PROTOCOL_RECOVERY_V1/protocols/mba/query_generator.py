"""Paper-faithful core of MBA difficult-word masking.

Proxy-LM ranks and accepted spelling variants are explicit inputs.  They must
be produced by the model revisions frozen in protocol_config.json; this module
does not replace them with NER, random, or ``important`` masking.
"""

from __future__ import annotations

import math
import re

from protocols.common import ProtocolInputError, require_membership_label, require_nonempty_text


ALLOWED_MASK_COUNTS = {5, 10, 15, 20}


def _group_bounds(length: int, groups: int, index: int) -> tuple[int, int]:
    return math.floor(index * length / groups), math.floor((index + 1) * length / groups)


def select_difficult_words(word_records: list[dict], mask_count: int) -> list[int]:
    """Select one highest-rank eligible word from each equal-length subtext."""

    if mask_count not in ALLOWED_MASK_COUNTS:
        raise ProtocolInputError("MBA mask_count must be one of 5, 10, 15, 20")
    if len(word_records) < mask_count:
        raise ProtocolInputError("target is too short for the requested number of masks")
    selected: list[int] = []
    for group_index in range(mask_count):
        start, end = _group_bounds(len(word_records), mask_count, group_index)
        candidates: list[tuple[float, int]] = []
        for position in range(start, end):
            record = word_records[position]
            if record.get("eligible") is not True:
                continue
            try:
                rank = float(record["difficulty_rank"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolInputError("every eligible MBA word needs a numeric difficulty_rank") from exc
            if not math.isfinite(rank):
                raise ProtocolInputError("MBA difficulty_rank must be finite")
            candidates.append((rank, position))
        if not candidates:
            raise ProtocolInputError(f"MBA subtext {group_index + 1} has no eligible mask target")
        selected.append(max(candidates, key=lambda item: (item[0], -item[1]))[1])
    if any(b - a == 1 for a, b in zip(selected, selected[1:])):
        raise ProtocolInputError("adjacent masks are forbidden by the MBA protocol")
    return selected


def make_query(target_id: str, membership: str, word_records: list[dict], mask_count: int) -> dict:
    target_id = require_nonempty_text(target_id, "target_id")
    require_membership_label(membership)
    selected = select_difficult_words(word_records, mask_count)
    selected_set = set(selected)
    masked_words: list[str] = []
    answers: dict[str, list[str]] = {}
    mask_number = 1
    for index, record in enumerate(word_records):
        word = require_nonempty_text(record.get("word", ""), f"word_records[{index}].word")
        if index in selected_set:
            key = f"Mask_{mask_number}"
            variants = record.get("accepted_answers") or [word]
            if not isinstance(variants, list) or not variants or any(not isinstance(v, str) or not v.strip() for v in variants):
                raise ProtocolInputError("accepted_answers must be a non-empty list of strings")
            masked_words.append(f"[{key}]")
            answers[key] = [v.strip() for v in variants]
            mask_number += 1
        else:
            masked_words.append(word)
    masked_text = " ".join(masked_words)
    query = (
        "Fill every indexed mask in the following text using the retrieved context. "
        "Return one line per mask in the format '[Mask_i]: answer'.\n\n" + masked_text
    )
    return {
        "session_id": f"mba::{target_id}",
        "query_id": f"mba::{target_id}::q1",
        "target_id": target_id,
        "membership": membership,
        "query": query,
        "masked_text": masked_text,
        "mask_answers": answers,
        "mask_count": mask_count,
    }


def tokenize_without_proxy_ranks(*_args, **_kwargs):
    raise ProtocolInputError(
        "MBA requires proxy-LM word ranks and fragmentation/spelling metadata; "
        "random or important-word fallback is forbidden"
    )

