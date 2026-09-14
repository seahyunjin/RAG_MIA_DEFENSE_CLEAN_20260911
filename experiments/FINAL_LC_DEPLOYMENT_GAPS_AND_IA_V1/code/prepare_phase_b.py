#!/usr/bin/env python3
"""Freeze the graceful closed-book fallback screen after Phase A2."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (CORE6, EXP, FINAL8, LC, QWEN, RECAL, ROOT, checkpoint,
                    freeze_json, now, sha_file)


def item(path: Path, **extra) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing Phase-B input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def main() -> None:
    phase_a2 = item(EXP / "PHASE_A2_RESULT.json")
    fp_path = RECAL / "tables" / "GOLD_REFRESH_FP_SUBSET_AUDIT.csv"
    with fp_path.open(encoding="utf-8") as handle:
        fp = list(csv.DictReader(handle))
    if len(fp) != 27 or len({row["query_id"] for row in fp}) != 27:
        raise RuntimeError(f"Gold FP cohort drift: {len(fp)}")
    code_names = ("common.py", "prepare_phase_b.py", "run_phase_b_generation.py", "run_phase_b_scoring.py")
    code = [EXP / "code" / name for name in code_names]
    pre = {
        "campaign": EXP.name, "phase": "B_FP_SAFE_CLOSED_BOOK_FALLBACK", "created_utc": now(),
        "status": "PRECOMMITTED_BEFORE_CLOSED_BOOK_GENERATION", "phase_a2": phase_a2,
        "frozen_final_lc": item(CORE6 / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "gold_recalibration_result": item(RECAL / "GOLD_RECALIBRATION_RESULT.json"),
        "gold_fp": item(fp_path, n=27, query_ids=sorted(row["query_id"] for row in fp)),
        "gold_queries": item(CORE6 / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl"),
        "gold_refreshed_answers": item(RECAL / "runtime" / "GOLD_REFRESHED_ANSWERS.jsonl"),
        "a2_retrieval": item(EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl"),
        "a2_answers": item(EXP / "runtime" / "PHASE_A2_GENERATED_ANSWERS.jsonl"),
        "a2_generation_manifest": item(EXP / "runtime" / "PHASE_A2_GENERATION_MANIFEST.json"),
        "a2_native_privacy": item(EXP / "tables" / "PHASE_A2_NATIVE_PRIVACY.csv"),
        "a2_menta_evidence": item(EXP / "runtime" / "PHASE_A2_MENTA_EVIDENCE.jsonl"),
        "targets": item(LC / "inputs" / "LARGE_SHARED_TARGETS.csv"),
        "old_menta_evidence": item(FINAL8 / "tables" / "MENTA_QUERY_EVIDENCE.csv"),
        "candidate": "GRACEFUL_CLOSED_BOOK_FALLBACK_V1",
        "trigger": "the frozen refusal parser OR exact empty answer on A_hide; only when Final LC alarmed",
        "refusal_terms": ["i don't know", "i do not know", "cannot determine", "insufficient information", "not enough information"],
        "closed_book_prompt": {
            "system": "Answer the user's question. Follow any required output format exactly. If you do not know the answer, answer exactly: I don't know.",
            "user": "User query:\n<exact frozen query>", "protected_documents": 0, "remaining_documents": 0,
            "rank5_or_external_reference": 0, "gold_at_runtime": False, "nli_selection": False,
        },
        "generator": {"model": "Qwen/Qwen2.5-3B-Instruct", "snapshot": str(QWEN),
                      "config": item(QWEN / "config.json"), "do_sample": False, "num_beams": 1,
                      "max_prompt_tokens": 3072,
                      "max_new_tokens": {"GOLD": 160, "MEntA": 160, "MBA": 160, "RAG-MIA": 12,
                                         "S²-MIA": 160, "DCMI-Std-Q2": 12}},
        "privacy_subset": "the exact Phase-A2 V0 primary 200-member/200-nonmember sessions per attack; all S2 reference retained only for scorer fitting",
        "utility_gate": {"fp_f1_min": .238, "or_recover_fraction_of_0_039_loss": .50,
                         "refusal": "strictly less than Simple Hide"},
        "privacy_gate": "each frozen native attack metric <= V0 Simple Hide +0.02",
        "attacks": ["MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2"],
        "promotion": "small PASS is not automatic full-scale deployment promotion; report first",
        "prohibitions": ["retrieved document", "removed source restoration", "remaining source", "rank5",
                         "external reference", "attack classifier", "gold runtime", "NLI selection", "keyword tuning"],
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code},
    }
    digest = freeze_json(EXP / "configs" / "FP_CLOSED_BOOK_FALLBACK_PRECOMMIT.json", pre)
    checkpoint("PHASE_B_PRECOMMITTED", precommit_sha256=digest, gold_fp=27,
               privacy_sessions=2000, next="PHASE_B_CLOSED_BOOK_GENERATION")
    print(json.dumps({"precommit_sha256": digest, "gold_fp": 27}, indent=2))


if __name__ == "__main__":
    main()
