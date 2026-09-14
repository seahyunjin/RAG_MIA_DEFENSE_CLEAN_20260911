"""Schema and leakage audit helpers for private generated-response cohorts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib


REQUIRED_FIELDS = (
    "response_generation_id", "session_id", "query_id", "parent_query_id",
    "query_hash", "retriever_id", "retriever_schema_version",
    "retrieved_document_ids", "retrieved_document_hashes", "retrieval_scores",
    "corpus_snapshot_hash", "generator_model_id", "generator_checkpoint_hash",
    "prompt_template_hash", "generation_config_hash", "generation_seed",
    "response_text", "response_text_hash", "token_count",
    "token_logprobs_available", "created_at", "split_role",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def validate_response_record(record: Mapping[str, object]) -> list[str]:
    errors = [f"missing:{name}" for name in REQUIRED_FIELDS if name not in record]
    if "response_text" in record and record.get("response_text_hash") != sha256_text(str(record["response_text"])):
        errors.append("response_text_hash_mismatch")
    ids = list(record.get("retrieved_document_ids", ()))
    hashes = list(record.get("retrieved_document_hashes", ()))
    scores = list(record.get("retrieval_scores", ()))
    if not (len(ids) == len(hashes) == len(scores)):
        errors.append("retrieved_context_alignment_mismatch")
    return errors


def audit_response_records(
    records: Sequence[Mapping[str, object]],
    *,
    split_group_fields: Sequence[str] = (
        "source_id", "target_document_id", "canonical_document_id",
        "source_conversation_id", "generated_query_set_id", "attack_session_id",
        "parent_query_id", "canonical_query_hash",
    ),
) -> dict[str, object]:
    if not records:
        return {"records": 0, "schema_errors": 0, "generation_success_rate": 0.0, "split_leakage": 0}
    schema_errors = sum(bool(validate_response_record(row)) for row in records)
    empty = sum(not str(row.get("response_text", "")).strip() for row in records)
    failures = sum(bool(row.get("generation_failed", False)) for row in records)
    truncated = sum(bool(row.get("truncated", False)) for row in records)
    prompt_hashes = {str(row.get("prompt_template_hash")) for row in records}
    model_hashes = {str(row.get("generator_checkpoint_hash")) for row in records}
    config_hashes = {str(row.get("generation_config_hash")) for row in records}
    duplicate_response = sum(value - 1 for value in Counter(str(r.get("response_text_hash")) for r in records).values() if value > 1)
    duplicate_query = sum(value - 1 for value in Counter((str(r.get("retriever_id")), str(r.get("query_hash"))) for r in records).values() if value > 1)
    leakage = 0
    for field in split_group_fields:
        roles: dict[str, set[str]] = defaultdict(set)
        for row in records:
            value = row.get(field)
            if value not in (None, ""):
                roles[str(value)].add(str(row.get("split_role")))
        leakage += sum(len(value) > 1 for value in roles.values())
    by_retriever = Counter(str(row.get("retriever_id")) for row in records)
    by_seed = Counter(str(row.get("generation_seed")) for row in records)
    return {
        "records": len(records), "schema_errors": schema_errors,
        "missing_response_rate": empty / len(records),
        "generation_failure_rate": failures / len(records),
        "generation_success_rate": (len(records) - failures - empty) / len(records),
        "truncation_rate": truncated / len(records),
        "prompt_hash_mismatch": max(0, len(prompt_hashes) - 1),
        "model_hash_mismatch": max(0, len(model_hashes) - 1),
        "config_hash_mismatch": max(0, len(config_hashes) - 1),
        "split_leakage": leakage,
        "duplicate_response_hash_count": duplicate_response,
        "duplicate_query_hash_count": duplicate_query,
        "retriever_counts": dict(by_retriever), "seed_counts": dict(by_seed),
    }
