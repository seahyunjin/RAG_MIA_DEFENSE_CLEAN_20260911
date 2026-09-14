#!/usr/bin/env python3
"""Freeze IA-ST1 matched-budget E2E after the detection gate passes."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
POST = EXP / "post_ready"
FINAL_LC = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def item(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required input missing: {path}")
    return {"path": str(path), "sha256": sha(path)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    detection_path = POST / "IA_ST1_DETECTION_RESULT.json"
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    if detection.get("verdict") != "IA_STEALTH_DETECTION_PASS":
        raise RuntimeError("E2E prohibited: IA detection gate did not pass")
    threshold_rows = list(csv.DictReader((POST / "tables" / "IA_ST1_BENIGN_THRESHOLDS.csv").open(encoding="utf-8")))
    thresholds = {}
    for row in threshold_rows:
        if abs(float(row["nominal_fpr"]) - 0.025) < 1e-12:
            thresholds[row["method"]] = {"threshold": float(row["threshold"]), "operator": ">", "actual_benign_fpr": float(row["actual_fpr"])}
    if set(thresholds) != {"MIRABEL", "Final LC"}:
        raise RuntimeError("primary matched-budget thresholds missing")

    code_paths = [POST / "run_e2e_generation.py", POST / "run_e2e_scoring.py"]
    pre = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "phase": "IA_MATCHED_BUDGET_E2E",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attack": "IA-Std-Q15-ST1; standardized stealth stress test, not Original/paper-exact IA",
        "detection_result": item(detection_path),
        "detection_cache": item(POST / "cache" / "IA_ST1_RETRIEVAL_AND_DETECTION.jsonl"),
        "queries": item(EXP / "inputs" / "IA_ST1_FULL_QUERIES.jsonl"),
        "gt": item(EXP / "inputs" / "IA_ST1_FULL_GT.jsonl"),
        "corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "targets": item(LC / "inputs" / "LARGE_SHARED_TARGETS.csv"),
        "final_lc_manifest": item(FINAL_LC / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "generator_config": item(QWEN / "config.json"),
        "conditions": ["NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5", "ORIGINAL_MIRABEL_FIXED_REFERENCE"],
        "thresholds": thresholds,
        "branches": {"A0": "unaltered frozen Top-4 context", "A_HIDE": "remove MIRABEL/retriever top-1 source and deterministic waterfill to the same 2048 source-token budget"},
        "generator": {"model": "Qwen/Qwen2.5-3B-Instruct", "snapshot": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "max_prompt_tokens": 3072, "source_token_budget": 2048, "max_new_tokens": 12, "do_sample": False, "num_beams": 1, "generation_count": 1},
        "scorer": {"response_labels": ["Yes", "No", "UNK"], "gt_labels": ["Yes", "No", "Unknown"], "session_score": "sum(1[response==GT] if response!=UNK else -5)/15", "unknown_penalty_lambda": 5, "native_metric": "fixed-polarity session ROC-AUC", "secondary": "E-AUC=max(AUC,1-AUC)"},
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260913},
        "hard_gate": {"clear_reduction_vs_no_defense": "Final LC native AUC <= No Defense native AUC - 0.03", "noninferior_to_same_budget_mirabel": "Final LC native AUC <= MIRABEL native AUC + 0.03"},
        "preferred_not_gate": {"attack_performance": "<=0.60", "strong": "<=0.55"},
        "forbidden": ["branch regeneration by defense condition", "threshold tuning", "scorer changes", "invalid-session imputation"],
        "code": {str(path): sha(path) for path in code_paths},
    }
    out = POST / "configs" / "IA_ST1_E2E_PRECOMMIT.json"
    write_json(out, pre)
    digest = sha(out)
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"precommit": str(out), "sha256": digest, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
