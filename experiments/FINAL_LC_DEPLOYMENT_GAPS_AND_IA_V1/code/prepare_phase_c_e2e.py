#!/usr/bin/env python3
"""Freeze matched-budget IA API1 E2E after the detection gate passes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import CORE3, CORE6, EXP, LC, QWEN, ROOT, freeze_json, now, sha_file


IA = EXP / "IA_STD_Q15_API1"
POST = IA / "post_ready"


def item(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing IA E2E input: {path}")
    return {"path": str(path), "sha256": sha_file(path)}


def main() -> None:
    detection_path = POST / "IA_API1_DETECTION_RESULT.json"
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    if not detection.get("checks", {}).get("no_catastrophic_stealth_miss", False):
        raise RuntimeError("IA E2E prohibited: catastrophic stealth-detection failure")
    threshold_rows = list(csv.DictReader((POST / "tables" / "IA_ST1_BENIGN_THRESHOLDS.csv").open(encoding="utf-8")))
    thresholds = {}
    for row in threshold_rows:
        if abs(float(row["nominal_fpr"]) - 0.025) < 1e-12:
            thresholds[row["method"]] = {"threshold": float(row["threshold"]), "operator": ">", "actual_benign_fpr": float(row["actual_fpr"])}
    if set(thresholds) != {"MIRABEL", "Final LC"}:
        raise RuntimeError("matched 2.5% thresholds missing")
    full = json.loads((IA / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    generation_wrapper = EXP / "code" / "run_phase_c_e2e_generation.py"
    scoring_wrapper = EXP / "code" / "run_phase_c_e2e_scoring.py"
    legacy_generation = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1" / "post_ready" / "run_e2e_generation.py"
    legacy_scoring = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1" / "post_ready" / "run_e2e_scoring.py"
    pre = {
        "campaign": EXP.name, "phase": "IA_API1_MATCHED_BUDGET_E2E", "created_utc": now(),
        "attack": "IA-Std-Q15-API1; standardized stress test, not Original/paper-exact IA",
        "detection_result": item(detection_path),
        "detection_cache": item(POST / "cache" / "IA_ST1_RETRIEVAL_AND_DETECTION.jsonl"),
        "queries": item(Path(full["artifacts"]["queries"]["path"])), "gt": item(Path(full["artifacts"]["gt"]["path"])),
        "corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "targets": item(LC / "inputs" / "LARGE_SHARED_TARGETS.csv"),
        "final_lc_manifest": item(CORE6 / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "generator_config": item(QWEN / "config.json"),
        "conditions": ["NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5"],
        "thresholds": thresholds,
        "branches": {"A0": "unaltered Top-4 context", "A_HIDE": "remove frozen MIRABEL top-1 source and deterministic waterfill"},
        "generator": {"model": "Qwen/Qwen2.5-3B-Instruct", "snapshot": str(QWEN), "source_token_budget": 2048,
                      "max_prompt_tokens": 3072, "max_new_tokens": 12, "do_sample": False, "num_beams": 1, "generation_count": 1},
        "scorer": {"labels": ["Yes", "No", "UNK"], "unknown_penalty_lambda": 5,
                   "session_score": "sum(1[response==GT] if response!=UNK else -5)/15",
                   "native_metric": "fixed-polarity session ROC-AUC", "secondary": "E-AUC=max(AUC,1-AUC)"},
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260913},
        "hard_gate": {"privacy_reduction": "Final LC native AUC <= No Defense native AUC -0.03",
                      "noninferior": "Final LC native AUC <= MIRABEL native AUC +0.03"},
        "forbidden": ["condition-specific branch regeneration", "threshold tuning", "scorer changes", "invalid imputation"],
        "code": {str(path): sha_file(path) for path in (generation_wrapper, scoring_wrapper, legacy_generation, legacy_scoring)},
    }
    path = POST / "configs" / "IA_API1_E2E_PRECOMMIT.json"
    digest = freeze_json(path, pre)
    print(json.dumps({"precommit": str(path), "sha256": digest, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
