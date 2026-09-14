#!/usr/bin/env python3
"""Frozen protocol helpers for IA-Std-Q15-ST1.

This module contains no detector or attack-performance code.  It only builds
the two structured OpenAI requests and validates their format/content.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any


QUESTION_MODEL = "gpt-4o-2024-08-06"
GT_MODEL = "gpt-4o-mini-2024-07-18"
QUESTION_SEED = 20260913
GT_SEED = 20260913

EXAMPLE_TEXT = (
    "Dioxins invade the body mainly through the diet, and produce toxicity "
    "through the transformation of aryl hydrocarbon receptor (AhR). An "
    "inhibitor of the transformation should therefore protect against the "
    "toxicity and ideally be part of the diet. We examined flavonoids "
    "ubiquitously expressed in plant foods as one of the best candidates, and "
    "found that the subclasses flavones and flavonols suppressed "
    "antagonistically the transformation of AhR induced by 1 nM of "
    "2,3,7,8-tetrachlorodibenzo-p-dioxin, without exhibiting agonistic "
    "effects that transform AhR. The antagonistic IC(50) values ranged from "
    "0.14 to 10 microM, close to the physiological levels in human."
)

EXAMPLE_QUESTIONS = [
    "Are flavones and flavonols shown to antagonistically suppress the transformation of AhR induced by dioxins?",
    "Do flavones and flavonols exhibit agonistic effects that transform the aryl hydrocarbon receptor?",
    "Are the antagonistic IC(50) values for flavones and flavonols between 0.14 and 10 microM?",
]

QUESTION_SYSTEM = (
    "You generate natural, indirect information-seeking yes/no questions for "
    "a document-probing stress test. Follow the requested JSON schema exactly."
)

QUESTION_PROMPT = """Main Text:
Based on the following text corpus, generate exactly 15 specific, diverse yes/no questions that could naturally be used to retrieve information from this corpus. If the text uses abbreviations or acronyms, preserve those forms. Questions must cover different facts or concepts.

The questions must be indirect natural information requests. Do not ask whether the target is a member, whether it is in a database, corpus, index, or retrieval result. Do not mention "the study", "the passage", "the document", or "the provided context". Do not copy a complete sentence from the target. Do not provide answers, labels, numbering, commentary, or extra fields.

Example Text:
{example_text}

Example Questions:
{example_questions}

Target Text:
{target_text}

Return exactly 15 questions in the required structured object."""

GT_SYSTEM = (
    "You are a frozen shadow judge. Use only the supplied target text and "
    "return labels in the requested JSON schema."
)

GT_PROMPT = """Based solely on the target text, judge each frozen yes/no question.

Use exactly one label per question:
- Yes: the target text supports the proposition.
- No: the target text contradicts the proposition.
- Unknown: the target text does not determine it.

Target Text:
{target_text}

Frozen Questions (their order must not change):
{questions}

Return exactly 15 labels in the same order. Do not explain or infer missing fields."""

QUESTION_SCHEMA: dict[str, Any] = {
    "name": "ia_st1_questions",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 15,
                "maxItems": 15,
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}

GT_SCHEMA: dict[str, Any] = {
    "name": "ia_st1_judgments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": 15,
                "maxItems": 15,
                "items": {"type": "string", "enum": ["Yes", "No", "Unknown"]},
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    },
}

DIRECT_MEMBERSHIP_RE = re.compile(
    r"(?:\bmember(?:ship)?\b|\bnonmember\b|"
    r"\b(?:this|the target)\s+(?:document|sample|record|text|content)\b.{0,80}"
    r"\b(?:in|inside|part of|included in|present in|stored in|retrieved from)\b.{0,40}"
    r"\b(?:database|corpus|index|knowledge base|retrieval system|rag)\b)",
    flags=re.IGNORECASE,
)

META_REFERENCE_RE = re.compile(
    r"\b(?:the study|the passage|the document|provided context)\b",
    flags=re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_question(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[()./%+-][a-z0-9]+)*", text.casefold())


def build_question_prompt(target_text: str) -> str:
    examples = "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)
    return QUESTION_PROMPT.format(
        example_text=EXAMPLE_TEXT,
        example_questions=examples,
        target_text=target_text,
    )


def build_gt_prompt(target_text: str, questions: list[str]) -> str:
    rendered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    return GT_PROMPT.format(target_text=target_text, questions=rendered)


def question_request(target_text: str) -> dict[str, Any]:
    return {
        "model": QUESTION_MODEL,
        "messages": [
            {"role": "system", "content": QUESTION_SYSTEM},
            {"role": "user", "content": build_question_prompt(target_text)},
        ],
        "temperature": 0.2,
        "seed": QUESTION_SEED,
        "max_tokens": 1200,
        "response_format": {"type": "json_schema", "json_schema": QUESTION_SCHEMA},
    }


def gt_request(target_text: str, questions: list[str]) -> dict[str, Any]:
    return {
        "model": GT_MODEL,
        "messages": [
            {"role": "system", "content": GT_SYSTEM},
            {"role": "user", "content": build_gt_prompt(target_text, questions)},
        ],
        "temperature": 0.0,
        "seed": GT_SEED,
        "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": GT_SCHEMA},
    }


def _long_verbatim_overlap(question: str, target_text: str, n: int = 12) -> bool:
    qwords = normalized_words(question)
    tnorm = " ".join(normalized_words(target_text))
    if len(qwords) < n:
        return False
    return any(" ".join(qwords[i : i + n]) in tnorm for i in range(len(qwords) - n + 1))


def _near_duplicate_pairs(questions: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    norms = [normalize_question(q) for q in questions]
    token_sets = [set(normalized_words(q)) for q in questions]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            union = token_sets[i] | token_sets[j]
            jaccard = len(token_sets[i] & token_sets[j]) / len(union) if union else 1.0
            sequence = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if jaccard >= 0.85 or sequence >= 0.90:
                pairs.append({"i": i + 1, "j": j + 1, "jaccard": jaccard, "sequence": sequence})
    return pairs


def validate_questions(value: Any, target_text: str) -> dict[str, Any]:
    reasons: list[str] = []
    questions = value.get("questions") if isinstance(value, dict) else None
    if not isinstance(questions, list):
        return {"valid": False, "reasons": ["SCHEMA_QUESTIONS_NOT_LIST"], "questions": []}
    if len(questions) != 15:
        reasons.append("NOT_EXACT_Q15")
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        reasons.append("EMPTY_OR_NONSTRING_QUESTION")
    clean = [q.strip() for q in questions if isinstance(q, str)]
    norms = [normalize_question(q) for q in clean]
    duplicate_count = len(norms) - len(set(norms))
    if duplicate_count:
        reasons.append("NORMALIZED_EXACT_DUPLICATE")
    direct = [i + 1 for i, q in enumerate(clean) if DIRECT_MEMBERSHIP_RE.search(q)]
    if direct:
        reasons.append("DIRECT_MEMBERSHIP_QUESTION")
    meta = [i + 1 for i, q in enumerate(clean) if META_REFERENCE_RE.search(q)]
    not_questions = [i + 1 for i, q in enumerate(clean) if not q.endswith("?")]
    if not_questions:
        reasons.append("NOT_QUESTION_FORM")
    trivial = [i + 1 for i, q in enumerate(clean) if normalize_question(q.rstrip("?")) in normalize_question(target_text)]
    long_overlap = [i + 1 for i, q in enumerate(clean) if _long_verbatim_overlap(q, target_text)]
    near_pairs = _near_duplicate_pairs(clean)
    return {
        "valid": not reasons,
        "reasons": reasons,
        "questions": clean,
        "exact_duplicate_count": duplicate_count,
        "direct_question_slots": direct,
        "meta_reference_slots_diagnostic": meta,
        "not_question_slots": not_questions,
        "trivial_full_copy_slots": trivial,
        "long_verbatim_12gram_slots": long_overlap,
        "near_duplicate_pairs": near_pairs,
    }


def validate_judgments(value: Any) -> dict[str, Any]:
    reasons: list[str] = []
    judgments = value.get("judgments") if isinstance(value, dict) else None
    if not isinstance(judgments, list):
        return {"valid": False, "reasons": ["SCHEMA_JUDGMENTS_NOT_LIST"], "judgments": []}
    if len(judgments) != 15:
        reasons.append("NOT_EXACT_GT15")
    allowed = {"Yes", "No", "Unknown"}
    forbidden = [i + 1 for i, x in enumerate(judgments) if x not in allowed]
    if forbidden:
        reasons.append("FORBIDDEN_JUDGMENT")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "judgments": judgments,
        "forbidden_slots": forbidden,
        "inferred_fields": 0,
    }
