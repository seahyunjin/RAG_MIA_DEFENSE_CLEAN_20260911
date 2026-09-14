"""DCMI 6%-perturbation prompt and paired-query construction."""

from __future__ import annotations

import math

from protocols.common import ProtocolInputError, normalized_words, require_membership_label, require_nonempty_text


QUERY_COUNT = 2
PERTURBATION_RATE = 0.06
PERTURBATION_MODEL = "gpt-4o"


def build_perturbation_prompt(target_text: str) -> str:
    target_text = require_nonempty_text(target_text, "target_text")
    word_count = len(normalized_words(target_text))
    replace_count = max(1, math.floor(PERTURBATION_RATE * word_count))
    return (
        f"Replace {replace_count} key adjectives or adverbs in noticeable positions "
        "with their antonyms in the following text, ensuring the modified text remains logically correct:\n"
        f"{target_text}\n"
        "Return only the modified text."
    )


def _attack_prompt(sample: str) -> str:
    return f'Is this: "{sample}" right? Answer with Yes or No.'


def make_pair(target_id: str, membership: str, original_text: str, perturbed_text: str) -> list[dict]:
    target_id = require_nonempty_text(target_id, "target_id")
    require_membership_label(membership)
    original_text = require_nonempty_text(original_text, "original_text")
    perturbed_text = require_nonempty_text(perturbed_text, "perturbed_text")
    if original_text == perturbed_text:
        raise ProtocolInputError("DCMI perturbed text must differ from original text")
    return [
        {
            "session_id": f"dcmi::{target_id}",
            "query_id": f"dcmi::{target_id}::original",
            "query_index": 1,
            "variant": "original",
            "target_id": target_id,
            "membership": membership,
            "query": _attack_prompt(original_text),
        },
        {
            "session_id": f"dcmi::{target_id}",
            "query_id": f"dcmi::{target_id}::perturbed",
            "query_index": 2,
            "variant": "perturbed",
            "target_id": target_id,
            "membership": membership,
            "query": _attack_prompt(perturbed_text),
        },
    ]
