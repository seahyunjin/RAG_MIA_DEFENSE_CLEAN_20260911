#!/usr/bin/env python3
"""Precommit the CPU-only IA streaming and DB-churn audit."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_LC_IA_AUDIT_AND_DB_CHURN_CPU_V1"
V6 = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1" / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED"
OLD = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
RECAL = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"


def sha_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    inputs = [
        V6 / "configs" / "IA_STD_Q15_V6_PRECOMMIT.json",
        V6 / "../IA_STD_Q15_V5_SLOTTED_CONSTRAINED" / "FINAL_RESULT.json",
        OLD / "tables" / "CORE5_DETECTION_STATISTICS.csv",
        OLD / "tables" / "BENIGN_DOMAIN_FPR_WILSON.csv",
        RECAL / "GOLD_RECALIBRATION_RESULT.json",
        RECAL / "cache" / "GOLD_REFRESHED_SCORES.jsonl",
        RECAL / "cache" / "GOLD_RECALIBRATION_QUERY_EMBEDDINGS.float16.npy",
        RECAL / "inputs" / "TOPIOCQA_BENIGN_RECALIBRATION_500.jsonl",
        LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy",
    ] + [OLD / "manifests" / f"DB_CHURN_{label}.json" for label in ("V0", "V10", "V25", "V50")]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    code = [EXP / "code" / "prepare_sidecar.py", EXP / "code" / "run_sidecar.py"]
    precommit = {
        "campaign": "FINAL_LC_IA_AUDIT_AND_DB_CHURN_CPU_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resource_policy": {"gpu": False, "model_loading": False, "answer_generation": False,
                            "cpu_affinity": [0, 1, 2, 3], "workers_max": 4,
                            "blas_threads": 1, "stream_interval_seconds": 120},
        "ia_policy": {"database": str(V6 / "runtime" / "ia_v6.sqlite3"),
                      "access": "SQLite URI mode=ro; never write, signal, stop, or modify IA-v6",
                      "performance_metrics_forbidden": ["MIRABEL TPR", "Final LC TPR", "AUC", "E-AUC", "privacy"],
                      "normalization": "Unicode NFKC, strip, collapse whitespace",
                      "tokenization_for_diagnostic": "Unicode alphanumeric word tokens, casefolded",
                      "near_duplicate_diagnostic": "pair is suspicious when (token Jaccard>=0.85 AND character SequenceMatcher>=0.90) OR common-prefix ratio>=0.90",
                      "near_duplicate_is_validity_gate": False},
        "churn_policy": {"versions": ["V0", "V10", "V25", "V50"],
                         "mode_a": "V0 frozen Final LC calibration/reference; no refresh",
                         "mode_b": "benign-only score/reference/outer-threshold refresh; attack samples=0",
                         "primary_budget": 0.025, "promising_fpr_max": 0.05,
                         "promising_tpr_degradation_max_pp": 5.0,
                         "missing_document_embedding": "DB_CHURN_GPU_EMBEDDING_REQUIRED; do not encode or build affected index"},
        "calibration_sensitivity": {"sizes": [100, 200, 300, 500],
                                    "subset": "ascending SHA256(query_id) prefix", "seed_search": False,
                                    "selection_use": False, "locked_test_n": 1000},
        "input_sha256": {str(path): sha_file(path) for path in inputs},
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code},
        "frozen": {"Final LC": True, "threshold_retuning": False, "new_detector": False,
                   "retrieval_training": False, "trainable_updates": 0},
    }
    output = EXP / "configs" / "SIDECAR_PRECOMMIT.json"
    output.write_text(json.dumps(precommit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha_file(output)
    (EXP / "configs" / "SIDECAR_PRECOMMIT.sha256").write_text(
        f"{digest}  SIDECAR_PRECOMMIT.json\n", encoding="utf-8")
    print(json.dumps({"precommit_sha256": digest, "inputs": len(inputs), "gpu": False}, indent=2))


if __name__ == "__main__":
    main()
