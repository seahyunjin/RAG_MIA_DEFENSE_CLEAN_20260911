#!/usr/bin/env python3
"""Freeze the action-only RRE experiment before any RRE answer generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    ref_audit = json.loads((CAMPAIGN / "audits/REFERENCE_POOL_AUDIT.json").read_text())
    retrieval_audit = json.loads((CAMPAIGN / "audits/REFERENCE_RETRIEVAL_AUDIT.json").read_text())
    mba_audit = json.loads((CAMPAIGN / "audits/MBA_PARSER_AUDIT.json").read_text())
    if ref_audit["verdict"] != "REFERENCE_POOL_AUDIT_PASS" or retrieval_audit["verdict"] != "REFERENCE_RETRIEVAL_PASS":
        raise RuntimeError("reference preflight failed")
    if mba_audit["verdict"] != "MBA_PARSER_IMPLEMENTATION_BUG_FIXED":
        raise RuntimeError("MBA audit unresolved")
    parent_cache = read_jsonl(PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")
    relevant = [row for row in parent_cache if row["cohort"] == "ATTACK" or row["split"] == "HOLDOUT"]
    risk = [row for row in relevant if bool(row["bc_q97_alarm"])]
    safe = [row for row in relevant if not bool(row["bc_q97_alarm"])]
    retrieval = read_jsonl(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl")
    if {row["query_id"] for row in risk} != {row["query_id"] for row in retrieval}:
        raise RuntimeError("front-end alarm IDs differ")
    parent_locators = {row["query_id"]: row["selected_source_id"] for row in risk}
    rre_locators = {row["query_id"]: row["selected_source_id"] for row in retrieval}
    if parent_locators != rre_locators:
        raise RuntimeError("front-end locator mapping differs")
    threshold = json.loads((PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE_MANIFEST.json").read_text())["thresholds"]["q97"]
    code_files = sorted((CAMPAIGN / "code").glob("*.py"))
    payload = {
        "campaign": "BC_RRE_ACTION_ONLY_SMALL_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_purpose": "At identical BC-q97 detection/privacy coverage, test whether one-for-one disjoint reference retrieval equalization reduces false-positive answer damage relative to simple source hide.",
        "development_only": True,
        "frozen_front_end": {
            "detector": "BC-MIRABEL q97",
            "strict_operator": ">",
            "numeric_threshold": threshold,
            "risk_query_count": len(risk),
            "safe_query_count": len(safe),
            "alarm_ids_sha256": retrieval_audit["alarm_ids_sha256"],
            "locator_mapping_sha256": retrieval_audit["locator_mapping_sha256"],
            "selected_source": "frozen MIRABEL top-1 locator",
        },
        "single_change": {
            "baseline": "Remove selected source and generate with the remaining three sources.",
            "candidate": "Remove the same selected source, insert frozen-BGE Top-1 SciFact reference document into the same slot, and generate with four sources.",
            "reference_top_k": 20,
            "replacement_selection": "first eligible document; every precommitted SciFact row is eligible after global disjointness PASS",
        },
        "frozen_generation": {"generator": "Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "source_token_budget": 2048, "max_prompt_tokens": 3072, "max_new_tokens": 160, "do_sample": False, "num_beams": 1, "seed": 20260911},
        "hard_gates": {
            "benign": {"simple_fp_preservation_reference": 0.590, "rre_improvement_absolute_or_level": "+0.15 OR >=0.75", "new_refusal": "<5/19", "answer_change_count": "<= Simple Hide"},
            "MEntA_native_auc": "RRE <= Simple Hide + 0.05",
            "RAG_MIA_native_auc": "RRE <= 0.55",
            "MBA_native_auc_if_valid": "RRE <= Simple Hide + 0.05",
        },
        "reference": ref_audit,
        "replacement_availability": retrieval_audit,
        "mba_audit": {"verdict": mba_audit["verdict"], "final_valid_n_each_condition": mba_audit["final_valid_n_each_condition"], "scorer_sha256": mba_audit["paper_aligned_scorer_sha256"]},
        "parent_artifact_hashes": {str(path.relative_to(PARENT)): sha256(path) for path in [
            PARENT / "configs/CLEAN_CORE3_DEV_V1_PRECOMMIT.json",
            PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl",
            PARENT / "runtime/CLEAN_CORE3_GENERATED_ANSWERS.jsonl",
            PARENT / "tables/MENTA_SESSION_SCORES.csv",
            PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl",
        ]},
        "code_sha256": {path.name: sha256(path) for path in code_files},
        "prohibited": ["threshold recalibration", "alarm-mask change", "locator change", "reference retry", "learned reranker", "QLL", "LOO", "privacy prompt", "fixed refusal", "NLI filtering", "extractive fallback", "token clipping"],
    }
    output = CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output)
    (CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    audit = {"verdict": "PRECOMMIT_PASS", "precommit_sha256": digest,
             "alarm_ids_identical": True, "safe_ids_identical": True,
             "locator_mapping_identical": True, "risk_queries": len(risk), "safe_queries": len(safe)}
    (CAMPAIGN / "audits/FRONT_END_EQUALITY_PREGENERATION.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
