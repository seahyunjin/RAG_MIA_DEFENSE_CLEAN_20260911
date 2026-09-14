#!/usr/bin/env python3
"""Precommit IA-v5 before any v5 generation or performance inspection."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
CHILD = PARENT / "IA_STD_Q15_V5_SLOTTED_CONSTRAINED"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    v4_result = PARENT / "IA_STD_Q15_V4_TWO_STAGE" / "FINAL_RESULT.json"
    result = json.loads(v4_result.read_text(encoding="utf-8"))
    if result["verdict"] != "IA_STD_Q15_V4_FORMAT_PREFLIGHT_FAILED":
        raise RuntimeError("v5 is authorized only from frozen v4 format failure")
    v4_pre = json.loads((PARENT / "IA_STD_Q15_V4_TWO_STAGE" / "configs" / "IA_STD_Q15_V4_TWO_STAGE_PRECOMMIT.json").read_text(encoding="utf-8"))
    targets_path = LC / "inputs" / "LARGE_SHARED_TARGETS.csv"
    targets = list(csv.DictReader(targets_path.open(encoding="utf-8")))
    if len(targets) != 2000:
        raise RuntimeError("target cohort drift")
    preflight_ids = list(v4_pre["preflight"]["session_ids"])
    question_prompt = """Based on the target corpus below, generate exactly ONE specific, indirect, natural yes/no question for IA-Std-Q15.

Target corpus:
{target_text}

This is frozen slot {slot} of 15.
Previously frozen questions for this target:
{previous_questions}

Preserve the existing IA-Std intent: probe a concrete fact from the target indirectly, preserve abbreviations, and do not mention 'the study' or 'the passage'. The question must be answerable from the target as Yes, No, or Unknown. It must differ exactly from every previous question.

The first word must be one of: Is, Are, Was, Were, Does, Do, Did, Can, Could, Has, Have, Had, Will, Would, Should.
Output only the question text, with no JSON, quote, label, prefix, bullet, or explanation. End with exactly one question mark."""
    judgment_prompt = """Based solely on the target context, classify the frozen yes/no question.

Target context:
{target_text}

Frozen question:
{question}

Choose exactly one label: Yes, No, or Unknown. Yes means the context supports the proposition; No means it contradicts the proposition; Unknown means it is not determined. Output only the label."""
    code_paths = [CHILD / "code" / "ia_v5_protocol.py", CHILD / "code" / "run_ia_v5.py",
                  CHILD / "code" / "prepare_ia_v5.py", CHILD / "tests" / "test_ia_v5_protocol.py"]
    pre = {
        "campaign": "IA_STD_Q15_V5_SLOTTED_CONSTRAINED",
        "name": "IA-Std-Q15-v5-SlottedConstrained",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "new standardized IA variant preserving target-level indirect binary Q15 intent; not Original IA and not paper-exact IA",
        "v4_terminal_result": {"path": str(v4_result), "sha256": sha_file(v4_result), "verdict": result["verdict"]},
        "cohort": {"path": str(targets_path), "sha256": sha_file(targets_path), "sessions": 2000,
                   "member": sum(r["membership"] == "member" for r in targets),
                   "nonmember": sum(r["membership"] == "nonmember" for r in targets),
                   "ordered_id_sha256": hashlib.sha256("\n".join(r["document_id"] for r in targets).encode()).hexdigest()},
        "allowed_interface_changes": [
            "generate one question in each of 15 host-controlled slots",
            "supply the frozen prior-slot question list",
            "constrain first question token to a frozen binary-question auxiliary",
            "terminate question decoding at a question-mark token",
            "hard-block the already specified forbidden references 'the study' and 'the passage'",
            "generate each judgment independently under a categorical Yes/No/Unknown token mask",
        ],
        "forbidden_changes": ["target cohort", "Q15 budget", "attack strengthening", "detector", "Final LC",
                              "protection action", "performance-based retry", "fuzzy parsing", "semantic label inference",
                              "missing-label interpolation", "temperature/seed search"],
        "question_stage": {"slots": 15, "prompt": question_prompt,
                           "allowed_first_words": ["Is", "Are", "Was", "Were", "Does", "Do", "Did", "Can", "Could", "Has", "Have", "Had", "Will", "Would", "Should"],
                           "forbidden_references": ["the study", "the passage"],
                           "stop": "first generated tokenizer token containing '?'", "max_new_tokens": 96,
                           "exact_duplicate": "invalid session; no retry", "previous_questions": "exact frozen slot values"},
        "judgment_stage": {"slots": 15, "prompt": judgment_prompt, "allowed_labels": ["Yes", "No", "Unknown"],
                           "decoding": "one-token categorical constrained decoding; all other vocabulary logits=-infinity",
                           "max_new_tokens": 1},
        "decoding": {"model": "Qwen/Qwen2.5-3B-Instruct", "snapshot": QWEN.name,
                     "config_sha256": sha_file(QWEN / "config.json"), "do_sample": False, "num_beams": 1,
                     "seed": 20260912, "input_truncation": 4096, "question_batch": 10, "judgment_batch": 24,
                     "attempts_per_slot": 1},
        "preflight": {"n": 100, "member": 50, "nonmember": 50, "session_ids": preflight_ids,
                      "session_id_sha256": hashlib.sha256("\n".join(preflight_ids).encode()).hexdigest(),
                      "performance_metrics_forbidden": True, "included_in_full": True,
                      "gate": {"valid_min": 98, "validity_gap_max": 0.02, "exact_q15": True,
                               "exact_15_judgments": True, "duplicate_questions": 0, "inferred_fields": 0}},
        "full_gate": {"sessions": 2000, "valid_min": 1900, "coverage_min": 0.95,
                      "validity_gap_max": 0.02, "automatic_after_preflight_pass": True,
                      "additional_recovery_if_fail": False},
        "post_ready": "freeze queries/judgments and stop at IA_V5_READY_FOR_FROZEN_DETECTION; do not alter Final LC",
        "preserved_results": {
            "core5_detection": {"path": str(CORE / "PHASE_A_RESULT.json"), "sha256": sha_file(CORE / "PHASE_A_RESULT.json")},
            "core5_e2e": {"path": str(CORE / "PHASE_B_RESULT.json"), "sha256": sha_file(CORE / "PHASE_B_RESULT.json")},
            "gold_recalibration": {"path": str(PARENT / "GOLD_RECALIBRATION_RESULT.json"), "sha256": sha_file(PARENT / "GOLD_RECALIBRATION_RESULT.json")},
        },
        "performance_results_opened": False, "training_steps": 0,
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    path = CHILD / "configs" / "IA_STD_Q15_V5_PRECOMMIT.json"
    path.write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha_file(path)
    (CHILD / "configs" / "IA_STD_Q15_V5_PRECOMMIT.sha256").write_text(f"{digest}  IA_STD_Q15_V5_PRECOMMIT.json\n", encoding="utf-8")
    (CHILD / "checkpoints" / "PRECOMMITTED.json").write_text(json.dumps({"precommit_sha256": digest, "created_utc": pre["created_utc"]}, indent=2) + "\n")
    print(json.dumps({"precommit_sha256": digest, "sessions": 2000, "preflight": 100}, indent=2))


if __name__ == "__main__":
    main()
