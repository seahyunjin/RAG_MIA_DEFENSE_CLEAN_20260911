#!/usr/bin/env python3
"""Precommit the single permitted structured-generation IA-Q15 recovery."""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from common import EXP, LC, QWEN, REQUEST, ROOT, SEED, SOURCE, TORCH_PYTHON, checkpoint, freeze_json, now, sha_file, verify_hashed_json


def artifact(path: Path, **extra: object) -> dict:
    if not path.is_file(): raise RuntimeError(f"missing IA input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


QUERY_PROMPT = """Based on the following text corpus, generate exactly 15 specific, diverse yes/no questions as indirect natural queries, together with each question's explicit judgment. Preserve abbreviations exactly. Questions must contain concrete keywords from the text, cover different facts, avoid mentioning 'the study' or 'the passage', and must not be generic.

Corpus:
{target_text}

Return one JSON object and no prose. It must contain only the key \"queries\". The value must be an array of exactly 15 objects. Each object must contain exactly: \"id\" (integer 1 through 15 in order), \"question\" (non-empty string), and \"judgment\" (exactly \"Yes\", \"No\", or \"Unknown\")."""

QUERY_RETRY = """Your previous response failed only the exact JSON schema validator. Repeat the same task and semantics for the same corpus. Return one JSON object and no prose, with exactly 15 ordered entries and only id, question, judgment fields. Do not explain.

Corpus:
{target_text}"""

GT_PROMPT = """Based solely on the provided context, assign an explicit judgment to each of the 15 supplied questions. Preserve every supplied question exactly; do not rewrite it.

Context:
{target_text}

Questions:
{questions_json}

Return one JSON object and no prose. It must contain only the key \"queries\". The value must be an array of exactly 15 objects. Each object must contain exactly: \"id\" (integer 1 through 15 in order), \"question\" (the exact supplied question), and \"judgment\" (exactly \"Yes\", \"No\", or \"Unknown\")."""

GT_RETRY = """Your previous response failed only the exact JSON schema or exact-question validator. Repeat the same judgment task. Preserve all supplied questions byte-for-byte and return only the required JSON object with exactly 15 entries.

Context:
{target_text}

Questions:
{questions_json}"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["queries"], "properties": {
    "queries": {"type": "array", "minItems": 15, "maxItems": 15, "items": {"type": "object",
                "additionalProperties": False, "required": ["id", "question", "judgment"], "properties": {
                    "id": {"type": "integer", "minimum": 1, "maximum": 15},
                    "question": {"type": "string", "minLength": 1},
                    "judgment": {"enum": ["Yes", "No", "Unknown"]}}}}}}


def main() -> None:
    if (EXP / "IA_V3_VALIDITY_RESULT.json").exists(): raise RuntimeError("IA-v3 result already exists")
    old = json.loads((SOURCE / "IA_PARSER_V2_RESULT.json").read_text(encoding="utf-8"))
    if old["verdict"] != "IA_STD_Q15_FORMAT_RECOVERY_FAILED": raise RuntimeError("old IA failure drift")
    frozen = verify_hashed_json(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json")
    targets = list(csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    if len(targets) != 2000 or Counter(row["membership"] for row in targets) != {"member": 1000, "nonmember": 1000}:
        raise RuntimeError("IA-v3 target balance drift")
    subprocess.run([str(TORCH_PYTHON), "-m", "unittest", "tests/test_ia_schema.py", "-v"], check=True, cwd=EXP)
    code = [EXP / "code" / name for name in ("common.py", "ia_schema.py", "prepare_ia_v3.py", "run_ia_v3.py")]
    tests = EXP / "tests" / "test_ia_schema.py"
    if not all(path.is_file() for path in code): raise RuntimeError("IA-v3 code incomplete")
    pre = {"campaign": EXP.name, "phase": "A_IA_STD_Q15_V3", "created_utc": now(),
           "request": artifact(REQUEST), "final_lc_frozen_manifest": artifact(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
           "final_lc_frozen_sha256": sha_file(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
           "source_of_truth": frozen["source_of_truth"], "old_v2_failure": artifact(SOURCE / "IA_PARSER_V2_RESULT.json"),
           "old_v2_verdict_preserved": "IA_STD_Q15_FORMAT_RECOVERY_FAILED",
           "name": "IA-Std-Q15-v3", "claim_boundary": "standardized structured Q15; not paper-exact IA Q30",
           "targets": artifact(LC / "inputs" / "LARGE_SHARED_TARGETS.csv", sessions=2000, member=1000, nonmember=1000),
           "query_semantics": "same target-level indirect natural Yes/No probing intent and Q15 budget as the frozen standardized protocol",
           "query_prompt": QUERY_PROMPT, "query_schema": SCHEMA, "query_retry_prompt": QUERY_RETRY,
           "gt_prompt": GT_PROMPT, "gt_schema": SCHEMA, "gt_retry_prompt": GT_RETRY,
           "generation_model": {"name": "Qwen2.5-3B-Instruct", "snapshot": QWEN.name, "config_sha256": sha_file(QWEN / "config.json")},
           "decoding": {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None,
                        "seed": SEED, "input_truncation": 4096, "query_max_new_tokens": 1500,
                        "gt_max_new_tokens": 1000, "answer_max_new_tokens": 12},
           "retry": "one retry only after exact schema validation failure; same corpus/task; no result/content-based retry",
           "parser": "json.loads whole output + exact keys/types/id sequence; no fuzzy parsing, inference, extraction, or imputation",
           "validity_gate": {"coverage": ">=0.95", "member_nonmember_gap": "<=0.02 absolute",
                             "exact_q15_among_valid": True, "inferred_missing_fields": 0, "fuzzy_parsing": 0},
           "detection": {"negative": "same frozen 1,000 benign holdout", "methods": ["MIRABEL", "Final LC"],
                         "budgets": [.01, .025, .03, .05], "primary": .025, "operator": ">",
                         "bootstrap": {"unit": "target/session", "iterations": 2000}},
           "e2e": {"conditions": ["NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5"],
                   "score": "(sum correct - 5*sum response-UNK)/15; target/session ROC-AUC",
                   "gt_unknown_policy": "a GT session containing Unknown is invalid for E2E; no inferred label and no selective mixing",
                   "native": "fixed-polarity ROC-AUC", "secondary": "E-AUC=max(AUC,1-AUC)"},
           "frozen_components": ["BGE-M3", "MIRABEL raw margin", "LC formula", "k=200", "semantic-neighbor procedure",
                                 "locator", "Simple Hide", "Qwen", "context builder", "prompt", "decoding", "protection action"],
           "training_steps": 0, "gradient_steps": 0,
           "unit_tests": {"path": str(tests), "sha256": sha_file(tests), "status": "PASS", "count": 8},
           "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code},
           "performance_results_opened": False}
    digest = freeze_json(EXP / "configs" / "IA_STD_Q15_V3_PRECOMMIT.json", pre)
    checkpoint("IA_V3_PRECOMMITTED", precommit_sha256=digest, sessions=2000, member=1000, nonmember=1000,
               model="Qwen2.5-3B-Instruct", next_stage="IA_V3_STRUCTURED_QUERY_GENERATION")


if __name__ == "__main__": main()
