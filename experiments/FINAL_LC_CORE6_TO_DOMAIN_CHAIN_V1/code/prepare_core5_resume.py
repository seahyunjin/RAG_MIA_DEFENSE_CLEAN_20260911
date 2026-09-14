#!/usr/bin/env python3
"""Freeze the mandated Core5 branch after IA parser-v2 recovery fails."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

from common import EXP, FINAL8, LC, ROOT, atomic_json, checkpoint, freeze_json, now, sha_file, sha_text, verify_hashed_json, write_jsonl


CORE5 = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")


def artifact(path: Path, **extra: object) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing Core5 input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def main() -> None:
    parser_result = json.loads((EXP / "IA_PARSER_V2_RESULT.json").read_text(encoding="utf-8"))
    if parser_result["verdict"] != "IA_STD_Q15_FORMAT_RECOVERY_FAILED":
        raise RuntimeError("Core5 branch is allowed only after the frozen IA parser gate fails")
    if any((EXP / name).exists() for name in ("PHASE_A_RESULT.json", "PHASE_B_RESULT.json", "PHASE_C_RESULT.json")):
        raise RuntimeError("performance result already exists; refusing to rewrite the Core5 precommit")
    verify_hashed_json(EXP / "configs" / "IA_STD_Q15_PARSER_V2_SPEC.json")
    verify_hashed_json(EXP / "configs" / "IA_STD_Q15_PARSER_V2_OUTPUT_MANIFEST.json")
    verify_hashed_json(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json")
    verify_hashed_json(EXP / "configs" / "DCMI_STD_Q2_PROTOCOL.json")

    targets = {row["document_id"]: row for row in csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))}
    con = sqlite3.connect(EXP / "runtime" / "standardized_query_generation.sqlite3")
    rows = list(con.execute(
        "SELECT target_id,membership,raw_output,parsed_json,valid,error,prompt_sha256,output_sha256 "
        "FROM generation WHERE attack='DCMI-Std-Q2' ORDER BY target_id"))
    if len(rows) != 2000 or Counter(row[1] for row in rows) != {"member": 1000, "nonmember": 1000}:
        raise RuntimeError("DCMI frozen session cohort drift")
    if any(not row[4] for row in rows):
        raise RuntimeError("DCMI contains invalid frozen sessions")

    queries, provenance = [], []
    for target_id, membership, raw_output, parsed_json, valid, error, prompt_sha, output_sha in rows:
        target = targets[target_id]
        perturbed = json.loads(parsed_json)["perturbed_text"]
        for index, (variant, sample) in enumerate((("original", target["source_text"]), ("perturbed", perturbed)), 1):
            query = f'Is this: "{sample}" right? Answer with Yes or No.'
            queries.append({
                "attack": "DCMI-Std-Q2", "session_id": f"dcmi_std::{target_id}",
                "query_id": f"dcmi_std::{target_id}::{variant}", "query_index": index,
                "variant": variant, "target_id": target_id, "membership": membership,
                "domain": target["domain"], "query": query, "query_sha256": sha_text(query),
                "perturbed_text_sha256": sha_text(perturbed), "generation_output_sha256": output_sha,
                "protocol": "DCMI-Std-Q2", "claim_boundary": "STANDARDIZED_NOT_ORIGINAL_DCMI",
            })
        provenance.append({
            "attack": "DCMI-Std-Q2", "target_id": target_id, "membership": membership,
            "raw_output_sha256": sha_text(raw_output), "parsed_json_sha256": sha_text(parsed_json),
            "prompt_sha256": prompt_sha, "output_sha256": output_sha, "valid": bool(valid), "error": error,
        })
    if len(queries) != 4000 or len({row["query_id"] for row in queries}) != 4000:
        raise RuntimeError("DCMI Q2 export drift")
    query_path = EXP / "inputs" / "DCMI_STD_Q2_QUERIES.jsonl"
    provenance_path = EXP / "runtime" / "DCMI_STD_Q2_FROZEN_PROVENANCE.jsonl"
    write_jsonl(query_path, queries)
    write_jsonl(provenance_path, provenance)

    code_names = (
        "common.py", "run_phase_a_detection.py", "prepare_core5_resume.py", "run_core5_detection.py",
        "prepare_core5_phase_b.py", "run_phase_b_generation.py", "run_core5_generation.py",
        "run_phase_b_scoring.py", "run_core5_scoring.py", "prepare_phase_c.py", "run_phase_c_gold.py",
    )
    code_paths = [EXP / "code" / name for name in code_names]
    if not all(path.is_file() for path in code_paths):
        raise RuntimeError("Core5 code set incomplete")
    precommit = {
        "campaign": EXP.name, "branch": "CORE5_AFTER_IA_PARSER_FAILURE", "created_utc": now(),
        "parser_failure": artifact(EXP / "IA_PARSER_V2_RESULT.json"),
        "parser_spec": artifact(EXP / "configs" / "IA_STD_Q15_PARSER_V2_SPEC.json"),
        "frozen_model": artifact(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "core5": list(CORE5), "ia_policy": "excluded; no regeneration; no valid-only mixing",
        "dcmi_queries": artifact(query_path, sessions=2000, queries=4000, member_sessions=1000, nonmember_sessions=1000),
        "dcmi_provenance": artifact(provenance_path),
        "old_detection": artifact(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl", n=17000),
        "old_answers": artifact(FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl", n=68000),
        "old_query_manifest": artifact(FINAL8 / "inputs" / "FINAL_SUPPORTED_QUERY_MANIFEST.jsonl", n=17000),
        "old_menta_evidence": artifact(FINAL8 / "tables" / "MENTA_QUERY_EVIDENCE.csv"),
        "targets": artifact(LC / "inputs" / "LARGE_SHARED_TARGETS.csv", n=2000),
        "negative": "same frozen benign holdout 1,000", "budgets": [0.01, 0.025, 0.03, 0.05],
        "primary_budget": 0.025, "strict_operator": ">",
        "continuation_policy": "Core5 detection, matched-budget E2E, then Gold QA even though IA parser recovery failed",
        "performance_results_opened": False,
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    digest = freeze_json(EXP / "configs" / "CORE5_RESUME_PRECOMMIT.json", precommit)
    atomic_json(EXP / "CORE5_BRANCH_NOTICE.json", {
        "reason": "IA deterministic parser-v2 failed its precommitted validity gates",
        "ia_regenerated": False, "ia_performance_used": False, "core5": list(CORE5),
        "precommit_sha256": digest,
    })
    checkpoint("CORE5_RESUME_PRECOMMITTED", precommit_sha256=digest, dcmi_sessions=2000,
               dcmi_queries=4000, ia_excluded=True, ia_regenerated=False, next_stage="CORE5_DETECTION")


if __name__ == "__main__":
    main()
