#!/usr/bin/env python3
"""Freeze the Core5 matched-budget E2E execution after Core5 detection."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, FINAL8, LC, QWEN, ROOT, checkpoint, freeze_json, now, sha_file, verify_hashed_json


def item(path: Path, **extra: object) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing Core5 E2E input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def main() -> None:
    detection = json.loads((EXP / "PHASE_A_RESULT.json").read_text(encoding="utf-8"))
    if detection["phase"] != "A_CORE5_DETECTION":
        raise RuntimeError("Core5 detection result missing")
    frozen = verify_hashed_json(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json")
    branch = verify_hashed_json(EXP / "configs" / "CORE5_RESUME_PRECOMMIT.json")
    thresholds = {}
    for row in detection["benign_alarm_rows"]:
        thresholds[f"{row['method']}@{row['nominal_budget']}"] = {
            "threshold": row["threshold"], "strict_operator": ">", "benign_alarms": row["benign_alarms"],
            "actual_benign_fpr": row["actual_benign_fpr"],
        }
    code_names = ("common.py", "run_phase_b_generation.py", "run_core5_generation.py",
                  "run_phase_b_scoring.py", "run_core5_scoring.py")
    code_paths = [EXP / "code" / name for name in code_names]
    pre = {
        "campaign": EXP.name, "phase": "B_CORE5_MATCHED_BUDGET_E2E", "created_utc": now(),
        "status": "CORE5_E2E_PRECOMMITTED", "detection_result": item(EXP / "PHASE_A_RESULT.json"),
        "core5_resume_precommit": item(EXP / "configs" / "CORE5_RESUME_PRECOMMIT.json"),
        "final_lc_manifest": item(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "final_lc_source_manifest": frozen["source_of_truth"],
        "new_detection": item(EXP / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl", n=4000),
        "old_detection": branch["old_detection"], "old_answers": branch["old_answers"],
        "old_query_manifest": branch["old_query_manifest"], "old_menta_evidence": branch["old_menta_evidence"],
        "dcmi_queries": branch["dcmi_queries"], "targets": branch["targets"],
        "conditions": ["NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5", "ORIGINAL_MIRABEL_FIXED"],
        "thresholds": thresholds, "primary_budget": .025,
        "branching": {"A0": "frozen normal Top-4 context", "A_HIDE": "remove selected retriever/MIRABEL top-1 and deterministic waterfill",
                      "old_core4": "reuse exact prior A0/A_HIDE by text/hash when available",
                      "new_dcmi": "generate A0 and only union-required A_HIDE"},
        "generator": frozen["generator"], "generator_config": item(QWEN / "config.json"),
        "core5": ["MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2"],
        "ia_policy": "excluded after parser-v2 failure; no regeneration; no partial valid-only result",
        "native_scoring": {"MEntA": "paper-faithful frozen NLI Q5", "MBA": "mask reconstruction; malformed/incomplete=0",
                           "RAG-MIA": "paper-faithful Yes/No", "S²-MIA": "reference-fitted BLEU/perplexity balanced accuracy",
                           "DCMI-Std-Q2": "original-vs-perturbed Yes differential session ROC-AUC"},
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260912},
        "continuation": "Gold QA runs after Core5 scoring regardless of Core5 aggregate gate; privacy failure is retained and reported",
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    digest = freeze_json(EXP / "configs" / "CORE5_MATCHED_BUDGET_E2E_PRECOMMIT.json", pre)
    checkpoint("CORE5_PHASE_B_PRECOMMIT_COMPLETE", precommit_sha256=digest,
               detection_verdict=detection["verdict"], ia_excluded=True, next_stage="CORE5_E2E_GENERATION")


if __name__ == "__main__":
    main()
