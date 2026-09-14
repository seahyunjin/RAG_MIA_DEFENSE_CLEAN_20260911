"""Pure constraints and validators for IA-Std-Q15-v6."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

import torch
from transformers import LogitsProcessor

STARTERS = ("Is", "Are", "Was", "Were", "Does", "Do", "Did", "Can", "Could",
            "Has", "Have", "Had", "Will", "Would", "Should")
LABELS = ("Yes", "No", "Unknown")
FORBIDDEN_REFERENCES = ("the study", "the passage")


def normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


@dataclass(frozen=True)
class QuestionResult:
    valid: bool
    value: str
    reason: str


def validate_question(raw: str, token_ids: list[int], previous_texts: list[str],
                      previous_token_ids: list[list[int]]) -> QuestionResult:
    value = raw.strip()
    if not value:
        return QuestionResult(False, value, "EMPTY")
    if "\n" in value or "\r" in value:
        return QuestionResult(False, value, "MULTILINE")
    if value[:1] in ("{", "[") or value[-1:] in ("}", "]"):
        return QuestionResult(False, value, "JSON_WRAPPER")
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return QuestionResult(False, value, "QUOTE_WRAPPER")
    if not value.endswith("?") or value.count("?") != 1:
        return QuestionResult(False, value, "QUESTION_TERMINATOR")
    if not re.match(r"^(?:" + "|".join(STARTERS) + r")\b", value):
        return QuestionResult(False, value, "NOT_BINARY_QUESTION")
    normalized = normalize_question(value)
    if any(phrase in normalized.casefold() for phrase in FORBIDDEN_REFERENCES):
        return QuestionResult(False, value, "FORBIDDEN_REFERENCE")
    if list(token_ids) in [list(seq) for seq in previous_token_ids]:
        return QuestionResult(False, value, "TOKEN_EXACT_DUPLICATE")
    if normalized in {normalize_question(item) for item in previous_texts}:
        return QuestionResult(False, value, "NORMALIZED_EXACT_DUPLICATE")
    return QuestionResult(True, value, "OK")


def validate_label(raw: str) -> tuple[bool, str, str]:
    value = raw.strip()
    if value not in LABELS:
        return False, value, "INVALID_JUDGMENT"
    return True, value, "OK"


class DuplicateSequenceLogitsProcessor(LogitsProcessor):
    """Block only the token that would exactly complete a prior row-specific sequence."""

    def __init__(self, input_length: int, forbidden_by_row: list[list[list[int]]]):
        self.input_length = int(input_length)
        self.forbidden_by_row = [[tuple(int(x) for x in seq) for seq in row] for row in forbidden_by_row]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for row_index, sequences in enumerate(self.forbidden_by_row):
            generated = tuple(int(x) for x in input_ids[row_index, self.input_length:].tolist())
            blocked = {seq[-1] for seq in sequences if len(seq) == len(generated) + 1 and seq[:-1] == generated}
            if blocked:
                scores[row_index, list(blocked)] = float("-inf")
        return scores


class FirstTokenMask(LogitsProcessor):
    def __init__(self, input_length: int, allowed: list[int]):
        self.input_length = int(input_length)
        self.allowed = tuple(sorted(set(int(x) for x in allowed)))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids.shape[1] == self.input_length:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, list(self.allowed)] = scores[:, list(self.allowed)]
            return mask
        return scores


class AllowedSequenceTrieLogitsProcessor(LogitsProcessor):
    """Permit exactly one frozen sequence followed by EOS."""

    def __init__(self, input_length: int, sequences: list[list[int]], eos_token_id: int):
        self.input_length = int(input_length)
        self.sequences = tuple(tuple(int(x) for x in seq) for seq in sequences)
        self.eos_token_id = int(eos_token_id)
        if not self.sequences or any(not seq for seq in self.sequences):
            raise ValueError("allowed sequences must be non-empty")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = torch.full_like(scores, float("-inf"))
        for row_index in range(input_ids.shape[0]):
            generated = tuple(int(x) for x in input_ids[row_index, self.input_length:].tolist())
            allowed = set()
            for sequence in self.sequences:
                if generated == sequence:
                    allowed.add(self.eos_token_id)
                elif len(generated) < len(sequence) and sequence[:len(generated)] == generated:
                    allowed.add(sequence[len(generated)])
            if not allowed:
                allowed.add(self.eos_token_id)
            mask[row_index, list(allowed)] = scores[row_index, list(allowed)]
        return mask


def serialize_session(target_id: str, questions: list[str]) -> str:
    payload = {"target_id": target_id,
               "queries": [{"id": index, "question": question} for index, question in enumerate(questions, 1)]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
