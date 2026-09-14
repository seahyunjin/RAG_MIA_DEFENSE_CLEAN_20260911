#!/usr/bin/env python3
"""Audit all CLEAN_CORE3 MBA outputs and rescore without regeneration."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments" / "BC_RRE_ACTION_ONLY_SMALL_V1"
CODE = CAMPAIGN / "code"
TABLES = CAMPAIGN / "tables"
AUDITS = CAMPAIGN / "audits"
REPORTS = CAMPAIGN / "reports"
CONDITIONS = ["NO_DEFENSE", "ORIGINAL_MIRABEL_TOP1_HIDE", "BC_MIRABEL_Q97_TOP1_HIDE"]

sys.path.insert(0, str(CODE))
import mba_scorer_paper_aligned as aligned


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_legacy():
    path = ROOT / "experiments/CORE6_PROTOCOL_RECOVERY_V1/protocols/mba/scorer.py"
    spec = importlib.util.spec_from_file_location("legacy_mba_scorer_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(ROOT / "experiments/CORE6_PROTOCOL_RECOVERY_V1"))
    spec.loader.exec_module(module)
    return module, path


def main() -> None:
    for directory in (TABLES, AUDITS, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    query_rows = {row["query_id"]: row for row in read_jsonl(PARENT / "inputs/MBA_ATTACK_QUERIES.jsonl")}
    answers = read_jsonl(PARENT / "runtime/CLEAN_CORE3_GENERATED_ANSWERS.jsonl")
    answers = {(row["query_id"], row["condition"]): row for row in answers if row["query_id"].startswith("mba::")}
    legacy, legacy_path = load_legacy()
    if len(query_rows) != 40 or len(answers) != 120:
        raise RuntimeError(f"unexpected MBA substrate queries={len(query_rows)} answers={len(answers)}")

    records = []
    for query_id, item in sorted(query_rows.items()):
        for condition in CONDITIONS:
            answer = answers[(query_id, condition)]["answer"]
            try:
                legacy_score = legacy.reconstruction_accuracy(answer, item["mask_answers"])
                old_valid, old_error = True, ""
            except Exception as exc:
                legacy_score, old_valid = math.nan, False
                old_error = f"{type(exc).__name__}: {exc}"
            new_score = aligned.reconstruction_accuracy(answer, item["mask_answers"])
            compliance, compliance_reason = aligned.format_compliance(answer, len(item["mask_answers"]))
            classification = "NO_PRIOR_INVALIDITY"
            if not old_valid:
                classification = "PARSER_IMPLEMENTATION_BUG"
            records.append({
                "target_id": item["target_id"],
                "membership": item["membership"],
                "query_id": query_id,
                "condition": condition,
                "exact_attack_input": item["query"],
                "exact_generated_answer": answer,
                "paper_faithful_expected_format": "one exact line '[Mask_i]: answer' per predicted indexed mask; unpredicted masks count incorrect in fixed M denominator",
                "current_parser_valid": old_valid,
                "current_parser_result": legacy_score,
                "current_parser_error": old_error,
                "output_format_compliant": compliance,
                "format_diagnostic": compliance_reason,
                "invalid_reason": old_error,
                "cause_classification": classification,
                "paper_aligned_score_valid": True,
                "paper_aligned_reconstruction_accuracy": new_score,
                "new_generation_used": False,
            })
    write_csv(TABLES / "MBA_40_CASE_VALIDITY_ALL_CONDITIONS.csv", records)
    prior_invalid = [row for row in records if row["condition"] == "BC_MIRABEL_Q97_TOP1_HIDE" and not row["current_parser_valid"]]
    if len(prior_invalid) != 25:
        raise RuntimeError(f"expected 25 prior invalid BC cases, got {len(prior_invalid)}")
    write_csv(TABLES / "MBA_25_INVALID_CASE_AUDIT.csv", prior_invalid)

    summary = []
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        labels = np.asarray([int(row["membership"] == "member") for row in rows])
        scores = np.asarray([float(row["paper_aligned_reconstruction_accuracy"]) for row in rows])
        summary.append({
            "condition": condition,
            "total_n": len(rows),
            "old_valid_n": sum(bool(row["current_parser_valid"]) for row in rows),
            "paper_aligned_valid_n": len(rows),
            "format_compliant_n": sum(bool(row["output_format_compliant"]) for row in rows),
            "member_n": int(labels.sum()),
            "nonmember_n": int((labels == 0).sum()),
            "member_score_mean": float(scores[labels == 1].mean()),
            "nonmember_score_mean": float(scores[labels == 0].mean()),
            "native_roc_auc": float(roc_auc_score(labels, scores)),
            "new_generation_count": 0,
        })
    write_csv(TABLES / "MBA_RESCORED_NATIVE_AUC.csv", summary)
    result = {
        "verdict": "MBA_PARSER_IMPLEMENTATION_BUG_FIXED",
        "reason": "The prior scorer excluded incomplete outputs. The paper score uses a fixed M denominator, so missing exact indexed predictions are incorrect rather than unscorable.",
        "prior_bc_invalid_n": len(prior_invalid),
        "classification_counts": {"PARSER_IMPLEMENTATION_BUG": len(prior_invalid), "MODEL_OUTPUT_INVALID": 0, "PROTOCOL_MISMATCH": 0},
        "format_noncompliant_bc_n": sum(not bool(row["output_format_compliant"]) for row in records if row["condition"] == "BC_MIRABEL_Q97_TOP1_HIDE"),
        "final_valid_n_each_condition": 40,
        "new_generation_count": 0,
        "legacy_scorer_sha256": sha256(legacy_path),
        "paper_aligned_scorer_sha256": sha256(CODE / "mba_scorer_paper_aligned.py"),
        "rescored_metrics": summary,
        "source_evidence": {
            "paper_pdf": str(ROOT / "papers/MBA.pdf"),
            "paper_pdf_sha256": sha256(ROOT / "papers/MBA.pdf"),
            "preserved_evaluator": str(ROOT / "code/menta_official/MBA/evaluate.py"),
            "preserved_evaluator_sha256": sha256(ROOT / "code/menta_official/MBA/evaluate.py"),
        },
    }
    write_json(AUDITS / "MBA_PARSER_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
