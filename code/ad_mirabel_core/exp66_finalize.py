"""Pure audit helpers for Exp66-Finalize.

The helpers in this module never fit or mutate the frozen LoLA detector.  They
only construct/validate the human-label packet and summarize IA missingness.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


ALLOWED_HUMAN_LABELS = {
    "PRESERVED",
    "PARTIALLY_PRESERVED",
    "NOT_PRESERVED",
    "AMBIGUOUS",
}


def text_sha256(value: object) -> str:
    """Hash the exact UTF-8 representation used in the frozen CSV packet."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def build_human_label_template(packet: pd.DataFrame) -> pd.DataFrame:
    """Return an unlabelled, hash-bound template for the 1,043-pair packet."""
    required = {
        "annotation_id",
        "source_index",
        "domain",
        "attack_family",
        "member_nonmember",
        "original_query",
        "paraphrased_query",
        "source_document_reference",
        "target_document_reference",
    }
    missing = required.difference(packet.columns)
    if missing:
        raise ValueError(f"annotation packet missing columns: {sorted(missing)}")
    if packet.annotation_id.isna().any() or packet.annotation_id.astype(str).duplicated().any():
        raise ValueError("annotation IDs must be complete and unique")
    output = packet[[
        "annotation_id", "source_index", "domain", "attack_family", "member_nonmember",
        "source_document_reference", "target_document_reference",
    ]].copy()
    output["original_query_sha256"] = packet.original_query.map(text_sha256)
    output["paraphrased_query_sha256"] = packet.paraphrased_query.map(text_sha256)
    output["pair_sha256"] = [
        text_sha256(f"{annotation_id}\0{original}\0{paraphrase}")
        for annotation_id, original, paraphrase in zip(
            packet.annotation_id.astype(str), packet.original_query.astype(str),
            packet.paraphrased_query.astype(str)
        )
    ]
    output["label"] = ""
    output["notes"] = ""
    output["annotator_id"] = ""
    return output


def validate_completed_human_labels(labels: pd.DataFrame, template: pd.DataFrame) -> dict[str, object]:
    """Validate exact IDs/hashes and the closed label vocabulary."""
    required = {"annotation_id", "original_query_sha256", "paraphrased_query_sha256", "pair_sha256", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"human label file missing columns: {sorted(missing)}")
    if labels.annotation_id.astype(str).duplicated().any():
        raise ValueError("human label file contains duplicate annotation IDs")
    expected = template.set_index("annotation_id")
    actual = labels.set_index("annotation_id")
    if set(actual.index.astype(str)) != set(expected.index.astype(str)):
        raise ValueError("human label IDs do not exactly match the frozen packet")
    actual = actual.loc[expected.index]
    for column in ("original_query_sha256", "paraphrased_query_sha256", "pair_sha256"):
        if not actual[column].astype(str).equals(expected[column].astype(str)):
            raise ValueError(f"human label {column} values do not match the frozen packet")
    normalized = actual.label.fillna("").astype(str).str.strip().str.upper()
    completed = normalized.ne("")
    invalid = sorted(set(normalized[completed]).difference(ALLOWED_HUMAN_LABELS))
    if invalid:
        raise ValueError(f"invalid human labels: {invalid}")
    return {
        "expected": int(len(expected)),
        "completed": int(completed.sum()),
        "complete": bool(completed.all()),
        "label_distribution": normalized[completed].value_counts().sort_index().to_dict(),
    }


def missingness_probability(rows: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    """Calculate P(undefined | group) without dropping informative missingness."""
    keys = list(by)
    required = set(keys) | {"score_missing"}
    if not required.issubset(rows.columns):
        raise ValueError(f"missing columns: {sorted(required.difference(rows.columns))}")
    grouped = rows.groupby(keys, dropna=False).score_missing.agg(["sum", "count"]).reset_index()
    grouped = grouped.rename(columns={"sum": "undefined", "count": "sessions"})
    grouped["defined"] = grouped.sessions - grouped.undefined
    grouped["p_undefined"] = grouped.undefined / grouped.sessions
    return grouped


__all__ = [
    "ALLOWED_HUMAN_LABELS",
    "build_human_label_template",
    "missingness_probability",
    "text_sha256",
    "validate_completed_human_labels",
]
