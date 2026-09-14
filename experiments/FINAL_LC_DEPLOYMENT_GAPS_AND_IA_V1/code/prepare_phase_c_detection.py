#!/usr/bin/env python3
"""Freeze Final-LC versus MIRABEL detection on a READY IA API1 artifact."""
from __future__ import annotations

import json
from pathlib import Path

from common import CORE3, CORE6, EXP, FINAL8, LC, ROOT, freeze_json, now, sha_file


IA = EXP / "IA_STD_Q15_API1"
POST = IA / "post_ready"


def item(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing IA detection input: {path}")
    return {"path": str(path), "sha256": sha_file(path)}


def main() -> None:
    result_path = IA / "FINAL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("verdict") != "IA_STD_Q15_API1_READY":
        raise RuntimeError("IA detection prohibited: API1 artifact is not READY")
    for key in ("sessions", "queries", "gt"):
        if sha_file(Path(result["artifacts"][key]["path"])) != result["artifacts"][key]["sha256"]:
            raise RuntimeError(f"IA API1 artifact drift: {key}")
    wrapper = EXP / "code" / "run_phase_c_detection.py"
    legacy = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1" / "post_ready" / "run_detection.py"
    pre = {
        "campaign": EXP.name, "phase": "IA_API1_SAME_FPR_DETECTION", "created_utc": now(),
        "attack": "IA-Std-Q15-API1; standardized stealth stress test, not Original/paper-exact IA",
        "full_result": item(result_path), "queries": item(Path(result["artifacts"]["queries"]["path"])),
        "gt": item(Path(result["artifacts"]["gt"]["path"])), "sessions": item(Path(result["artifacts"]["sessions"]["path"])),
        "final_lc_manifest": item(CORE6 / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "frozen_reference_scores": item(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl"),
        "frozen_reference_embeddings": item(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"),
        "frozen_corpus_embeddings": item(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"),
        "frozen_corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "frozen_benign_detection": item(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"),
        "positive_definition": "valid member-target IA-Std-Q15-API1 query",
        "negative_definition": "same frozen 1,000-query benign deployment holdout",
        "budgets": [0.01, 0.025, 0.03, 0.05], "primary_budget": 0.025,
        "threshold_rule": "strict >, at most floor(alpha*N) alarms on the identical benign cohort",
        "detectors": {"MIRABEL": "canonical full-corpus Gumbel margin", "Final LC": "frozen semantic-local calibration, k=200"},
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260913},
        "hard_gate": {"noninferiority": "Final LC TPR >= MIRABEL TPR -0.03", "catastrophic_floor": "Final LC TPR >=0.05"},
        "forbidden": ["threshold tuning", "formula changes", "attack training", "invalid imputation"],
        "code": item(wrapper), "legacy_computation_module": item(legacy),
    }
    path = POST / "configs" / "IA_API1_DETECTION_PRECOMMIT.json"
    digest = freeze_json(path, pre)
    print(json.dumps({"precommit": str(path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
