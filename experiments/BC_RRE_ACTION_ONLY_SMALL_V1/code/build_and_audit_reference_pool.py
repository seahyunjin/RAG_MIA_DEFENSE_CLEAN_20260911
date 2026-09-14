#!/usr/bin/env python3
"""Materialize the precommitted SciFact reference pool and fail closed on overlap."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unicodedata


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"
RAW = CAMPAIGN / "inputs/reference_raw/scifact/corpus.jsonl"
POOL = CAMPAIGN / "inputs/REFERENCE_POOL.jsonl"
AUDIT = CAMPAIGN / "audits/REFERENCE_POOL_AUDIT.json"
RULE = CAMPAIGN / "configs/REFERENCE_RULE_PRECOMMIT.json"
EXPECTED_RULE_SHA = "a9b971d5cd5d9a919e7a17e62ed98b9475678a2366d19ce2ae452c447a61dd31"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def normalized_hash(value: str) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if sha256(RULE) != EXPECTED_RULE_SHA:
        raise RuntimeError("reference-rule precommit checksum mismatch")
    rows = []
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            local_id = str(raw.get("_id") or raw.get("id"))
            title = str(raw.get("title") or "").strip()
            text = str(raw.get("text") or "").strip()
            source = "\n".join(value for value in (title, text) if value)
            rows.append({
                "document_id": f"BeIR_scifact::{local_id}",
                "local_document_id": local_id,
                "domain": "scifact",
                "title": title,
                "text": text,
                "source_text": source,
                "normalized_text_hash": normalized_hash(source),
            })
    if not rows or len({row["document_id"] for row in rows}) != len(rows):
        raise RuntimeError("empty or duplicate-ID reference corpus")
    with POOL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    protected = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    targets = list(csv.DictReader((PARENT / "inputs/SHARED_TARGETS.csv").open(encoding="utf-8")))
    protected_ids = {row["document_id"] for row in protected}
    protected_hashes = {row["normalized_text_hash"] for row in protected}
    member = [row for row in targets if row["membership"] == "member"]
    nonmember = [row for row in targets if row["membership"] == "nonmember"]
    reference_ids = {row["document_id"] for row in rows}
    reference_hashes = {row["normalized_text_hash"] for row in rows}
    checks = {
        "reference_intersection_protected_id": sorted(reference_ids & protected_ids),
        "reference_intersection_member_id": sorted(reference_ids & {row["document_id"] for row in member}),
        "reference_intersection_nonmember_id": sorted(reference_ids & {row["document_id"] for row in nonmember}),
        "reference_intersection_protected_normalized_text": sorted(reference_hashes & protected_hashes),
        "reference_intersection_member_normalized_text": sorted(reference_hashes & {row["normalized_text_hash"] for row in member}),
        "reference_intersection_nonmember_normalized_text": sorted(reference_hashes & {row["normalized_text_hash"] for row in nonmember}),
    }
    passed = all(len(value) == 0 for value in checks.values())
    result = {
        "verdict": "REFERENCE_POOL_AUDIT_PASS" if passed else "REFERENCE_SPLIT_CONTAMINATED",
        "construction_rule": json.loads(RULE.read_text(encoding="utf-8"))["reference_rule"],
        "target_independent": True,
        "post_hoc_exclusions": 0,
        "archive_sha256": sha256(CAMPAIGN / "inputs/scifact.zip"),
        "raw_corpus_sha256": sha256(RAW),
        "reference_pool_sha256": sha256(POOL),
        "reference_size": len(rows),
        "reference_domain": "scifact",
        "protected_size": len(protected),
        "member_targets": len(member),
        "nonmember_targets": len(nonmember),
        "overlap": {key: {"count": len(value), "values": value} for key, value in checks.items()},
    }
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 17)


if __name__ == "__main__":
    main()
