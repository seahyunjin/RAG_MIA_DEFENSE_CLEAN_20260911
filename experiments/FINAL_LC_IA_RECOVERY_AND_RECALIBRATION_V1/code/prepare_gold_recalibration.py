#!/usr/bin/env python3
"""Precommit a benign-only TopiOCQA recalibration without reading gold utility."""
from __future__ import annotations

import json
from pathlib import Path

from common import BGE, EXP, K_LOCAL, QWEN, REQUEST, ROOT, SOURCE, TOPI, checkpoint, freeze_json, now, read_jsonl, sha_file, sha_text, verify_hashed_json, write_jsonl


def artifact(path: Path, **extra: object) -> dict:
    if not path.is_file(): raise RuntimeError(f"missing input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def topi_candidates() -> list[dict]:
    rows = []
    for line in (TOPI / "topiocqa_valid.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        answer = row.get("Answer", "")
        if not answer or answer == "UNANSWERABLE": continue
        history = row.get("Context") or []
        history_text = "\n".join(("User: " if index % 2 == 0 else "Assistant: ") + str(value)
                                 for index, value in enumerate(history))
        query = f"Conversation history:\n{history_text}\n\nCurrent question: {row['Question']}" if history else row["Question"]
        row_id = f"topi-valid::{row['Conversation_no']}::{row['Turn_no']}"
        passage = row["Gold_passage"]
        key = sha_text(json.dumps([passage["id"], passage["title"], passage["text"]], ensure_ascii=False, separators=(",", ":")))
        rows.append({"query_id": row_id, "query": query, "raw_question": row["Question"],
                     "gold_document_id_for_provenance_only": f"TopiOCQA::{key}", "selection_key": sha_text(row_id)})
    return sorted(rows, key=lambda row: (row["selection_key"], row["query_id"]))


def main() -> None:
    if (EXP / "GOLD_RECALIBRATION_RESULT.json").exists(): raise RuntimeError("Gold result already exists")
    frozen = verify_hashed_json(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json")
    strict = json.loads((SOURCE / "PHASE_C_RESULT.json").read_text(encoding="utf-8"))
    if strict["verdict"] != "GOLD_QA_UTILITY_FAILED": raise RuntimeError("strict-transfer baseline drift")
    test = read_jsonl(SOURCE / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl")
    if len(test) != 1000: raise RuntimeError("locked Gold test drift")
    test_ids = {row["query_id"] for row in test}
    candidates = topi_candidates()
    calibration = [row for row in candidates if row["query_id"] not in test_ids][:500]
    if len(calibration) != 500 or test_ids & {row["query_id"] for row in calibration}:
        raise RuntimeError("deterministic calibration/test split failed")
    calibration_path = EXP / "inputs" / "TOPIOCQA_BENIGN_RECALIBRATION_500.jsonl"
    write_jsonl(calibration_path, calibration)
    audit = {"dataset": "McGill-NLP/TopiOCQA", "snapshot": TOPI.parent.name,
             "calibration_n": 500, "locked_test_n": 1000, "query_id_overlap": 0,
             "locked_test": "unchanged 1,000-query test used for the preserved strict-transfer baseline",
             "calibration_selection": "lowest SHA256(query_id) among valid candidates after excluding all locked-test IDs",
             "calibration_fields_used": ["benign query text", "query embedding", "retrieval scores", "MIRABEL margin", "LC scores"],
             "calibration_gold_answers_used": 0, "attack_samples_used": 0, "utility_metrics_used": 0,
             "known_test_status": "not an untouched blind; the strict-transfer test was already evaluated"}
    audit_hash = freeze_json(EXP / "audits" / "GOLD_RECALIBRATION_SPLIT_AUDIT.json", audit)
    code = [EXP / "code" / name for name in ("common.py", "prepare_gold_recalibration.py", "run_gold_recalibration.py")]
    if not all(path.is_file() for path in code): raise RuntimeError("Gold recalibration code incomplete")
    pre = {"campaign": EXP.name, "phase": "B_GOLD_BENIGN_ONLY_RECALIBRATION", "created_utc": now(),
           "request": artifact(REQUEST), "final_lc_frozen_manifest": artifact(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
           "final_lc_frozen_sha256": sha_file(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
           "source_of_truth": frozen["source_of_truth"], "strict_transfer_result": artifact(SOURCE / "PHASE_C_RESULT.json"),
           "strict_transfer_frozen": {"intervention": .692, "no_defense_f1": .20894493709122594,
                                      "final_lc_f1": .1779442924752994, "f1_delta": -.031000644615926554,
                                      "new_refusal": .058, "verdict": "STRICT_TRANSFER_FAILED"},
           "calibration": artifact(calibration_path, n=500), "locked_test": artifact(SOURCE / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl", n=1000),
           "corpus": artifact(SOURCE / "inputs" / "TOPIOCQA_GOLD_CORPUS.jsonl"),
           "old_test_retrieval": artifact(SOURCE / "cache" / "GOLD_RETRIEVAL_AND_DETECTION.jsonl", n=1000),
           "old_test_answers": artifact(SOURCE / "runtime" / "GOLD_QA_DETAIL.jsonl", n=4000),
           "old_factuality": artifact(SOURCE / "tables" / "GOLD_FACTUALITY_SENTENCES.csv"),
           "split_audit_sha256": audit_hash, "retriever": artifact(BGE / "config.json"),
           "generator": artifact(QWEN / "config.json"), "k_local": K_LOCAL,
           "refresh": ["benign reference bank", "MIRABEL-margin distribution", "local LC neighborhood cache", "outer thresholds"],
           "frozen": ["LC formula", "k", "BGE-M3", "locator", "Simple Hide", "Qwen", "context builder", "prompt", "decoding"],
           "threshold_rule": "most permissive calibration-observed threshold satisfying strict > and floor(alpha*N) alarms",
           "budgets": [.01, .025, .03, .05], "primary_budget": .025,
           "calibration_local_rule": "leave-self-out 200 nearest calibration-query embeddings; add-one empirical upper tail",
           "test_local_rule": "200 nearest calibration-query embeddings; add-one empirical upper tail",
           "gates": {"intervention": "<=0.05", "gold_f1_drop": "<=0.02 absolute", "new_refusal": "<=0.01 preferred and verdict requirement"},
           "training_steps": 0, "gradient_steps": 0, "attack_samples": 0,
           "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code},
           "performance_results_opened": False}
    digest = freeze_json(EXP / "configs" / "GOLD_BENIGN_RECALIBRATION_PRECOMMIT.json", pre)
    checkpoint("GOLD_RECALIBRATION_PRECOMMITTED", precommit_sha256=digest, calibration_n=500,
               locked_test_n=1000, attack_samples=0, next_stage="GOLD_BENIGN_SCORE_REFRESH")


if __name__ == "__main__": main()
