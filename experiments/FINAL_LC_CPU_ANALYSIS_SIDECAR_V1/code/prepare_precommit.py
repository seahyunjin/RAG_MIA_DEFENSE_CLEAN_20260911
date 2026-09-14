#!/usr/bin/env python3
"""Freeze the CPU-sidecar inputs and deterministic controls before results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
RECAL = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
CLEAN3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    paths = [
        FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl",
        CORE / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl",
        LC / "cache" / "LARGE_DETECTION_SCORES.jsonl",
        CORE / "tables" / "CORE5_E2E_ALL_BUDGETS.csv",
        CORE / "tables" / "CORE5_MATCHED_BUDGET_E2E_PRIVACY.csv",
        CORE / "PHASE_A_RESULT.json", CORE / "PHASE_B_RESULT.json", CORE / "PHASE_C_RESULT.json",
        CORE / "runtime" / "GOLD_QA_DETAIL.jsonl",
        CORE / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl",
        CORE / "inputs" / "TOPIOCQA_GOLD_CORPUS.jsonl",
        RECAL / "GOLD_RECALIBRATION_RESULT.json",
        RECAL / "cache" / "GOLD_REFRESHED_SCORES.jsonl",
        RECAL / "runtime" / "GOLD_REFRESHED_ANSWERS.jsonl",
        CLEAN3 / "manifests" / "CLEAN_CORE3_DB_MANIFEST.json",
        CLEAN3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl",
        CLEAN3 / "inputs" / "COMMON_ELIGIBLE_POOL.csv.gz",
        CLEAN3 / "inputs" / "BENIGN_CALIBRATION.jsonl",
        CLEAN3 / "inputs" / "BENIGN_HOLDOUT.jsonl",
        FINAL8 / "reports" / "PHASE_A_PROTOCOL_AUDIT_KO.md",
        FINAL8 / "runtime" / "FINAL_GENERATION_MANIFEST.json",
        FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl",
        CORE / "runtime" / "PHASE_B_GENERATION_MANIFEST.json",
        CORE / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl",
    ]
    manifest = json.loads((CORE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json").read_text())
    matched = json.loads((CORE / "configs" / "CORE5_MATCHED_BUDGET_E2E_PRECOMMIT.json").read_text())
    pre = {
        "campaign": EXP.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {"device": "CPU_ONLY", "worker_limit": 4, "model_loading": False,
                      "retrieval": False, "answer_generation": False, "training": False,
                      "ia_gpu_session_policy": "DO_NOT_SIGNAL_STOP_OR_MODIFY"},
        "bootstrap": {"iterations": 2000, "seed": 20260912, "unit": "target/session",
                      "lineage": "exact frozen Core5 Phase-A seed; attack offset follows frozen attack order"},
        "s2_detection_scope": "S2_EVALUATION only; exclude frozen 402-query scorer-reference split; expected member N=799",
        "random_neighbor": {
            "k": 200, "reference_n": 1000,
            "rule": "PCG64(first64(SHA256('FINAL_LC_CPU_ANALYSIS_SIDECAR_V1|RANDOM_LC_V1|' + query_id))); choose 200 of 1000 without replacement",
            "seed_search": False, "attack_result_used": False,
        },
        "fpr_budgets": [0.01, 0.025, 0.03, 0.05], "strict_operator": ">",
        "matched_thresholds": {
            "MIRABEL@0.025": matched["thresholds"]["MIRABEL@0.025"]["threshold"],
            "Final LC@0.025": matched["thresholds"]["Final LC@0.025"]["threshold"],
        },
        "fixed_thresholds": {"Original MIRABEL": 0.0, "Global BC": 3.428580914764567,
                             "Final LC": manifest["detector"]["outer_threshold"]},
        "fp_harm_rule_order": ["REFUSAL_DRIVEN", "UNIQUE_EVIDENCE_REMOVED", "REDUNDANT_EVIDENCE_AVAILABLE", "UNRESOLVED"],
        "refusal_rule": ["i don't know", "i do not know", "cannot determine", "insufficient information", "not enough information"],
        "fp_evidence_rule": "NFKC + lowercase + alphanumeric whitespace normalized exact gold-answer substring; no semantic inference",
        "db_churn": {"versions": {"V0": 0.0, "V10": 0.10, "V25": 0.25, "V50": 0.50},
                     "protected": "member targets + benign calibration/holdout primary qrels IDs",
                     "replacement": "per-domain deterministic SHA256 order from frozen common eligible pool"},
        "optional_k_sensitivity": {"values": [50, 100, 200, 400], "selection_forbidden": True,
                                   "execution_policy": "omit if frozen embeddings incomplete or concurrent IA protection warrants"},
        "code_sha256": sha_file(EXP / "code" / "run_cpu_sidecar.py"),
        "frozen_inputs": [{"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size} for path in paths],
    }
    path = EXP / "configs" / "SIDECAR_PRECOMMIT.json"
    path.write_text(json.dumps(pre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha_file(path)
    (EXP / "configs" / "SIDECAR_PRECOMMIT.sha256").write_text(f"{digest}  SIDECAR_PRECOMMIT.json\n", encoding="utf-8")
    print(json.dumps({"precommit": str(path), "sha256": digest, "inputs": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
