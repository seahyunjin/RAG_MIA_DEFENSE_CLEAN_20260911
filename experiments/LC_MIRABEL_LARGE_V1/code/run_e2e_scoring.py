#!/usr/bin/env python3
"""Paper-faithful native scoring and benign-harm audit for large LC-MIRABEL E2E."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from common import EXP, PARENT, RECOVERY, atomic_json, atomic_text, checkpoint, now, read_jsonl, sha_file, write_csv


CONDITIONS = ["NO_DEFENSE", "ORIGINAL_MIRABEL_SIMPLE_HIDE", "GLOBAL_BC_SIMPLE_HIDE", "LC_MIRABEL_SIMPLE_HIDE"]
DISPLAY = {"NO_DEFENSE": "No Defense", "ORIGINAL_MIRABEL_SIMPLE_HIDE": "Original MIRABEL",
           "GLOBAL_BC_SIMPLE_HIDE": "Global BC-MIRABEL", "LC_MIRABEL_SIMPLE_HIDE": "LC-MIRABEL"}
PRECOMMIT = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"


def load_parent_evaluator():
    path = PARENT / "code/run_final_evaluation.py"
    spec = importlib.util.spec_from_file_location("lc_large_native_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load preserved evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CAMPAIGN = EXP
    module.INPUTS = EXP / "inputs"
    module.CACHE = EXP / "cache"
    module.RUNTIME = EXP / "runtime"
    module.TABLES = EXP / "tables"
    module.AUDITS = EXP / "audits"
    module.REPORTS = EXP / "reports"
    module.CONFIGS = EXP / "configs"
    module.CHECKPOINTS = EXP / "checkpoints"
    module.CONDITIONS = CONDITIONS
    module.DISPLAY = DISPLAY
    module.BOOTSTRAP_ITERATIONS = 2000
    return module


def token_f1(left: str, right: str) -> float:
    a, b = left.lower().split(), right.lower().split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap/len(b), overlap/len(a)
    return 2*precision*recall/(precision+recall)


def is_refusal(value: str) -> bool:
    text = " ".join(value.lower().split())
    return any(item in text for item in ("i don't know", "i do not know", "cannot determine", "can't determine",
                                          "insufficient information", "not enough information"))


def verify() -> None:
    expected = PRECOMMIT.with_suffix(".sha256").read_text().split()[0]
    pre = json.loads(PRECOMMIT.read_text())
    if sha_file(PRECOMMIT) != expected or pre["lineage"]["code_sha256"].get(Path(__file__).name) != sha_file(Path(__file__)):
        raise RuntimeError("precommit/code checksum mismatch")
    generation = EXP / "runtime/LARGE_GENERATED_ANSWERS.jsonl"
    manifest = json.loads((EXP / "runtime/LARGE_GENERATION_MANIFEST.json").read_text())
    if sha_file(generation) != manifest["output_sha256"]:
        raise RuntimeError("generated answers drift")


def main() -> None:
    verify()
    evaluator = load_parent_evaluator()
    retrieval = [row for row in read_jsonl(EXP / "cache/LARGE_DETECTION_SCORES.jsonl") if row["split"] != "REFERENCE"]
    answer_rows = read_jsonl(EXP / "runtime/LARGE_GENERATED_ANSWERS.jsonl")
    answer_map = {(row["query_id"], row["condition"]): row for row in answer_rows}
    expected = {(row["query_id"], condition) for row in retrieval for condition in CONDITIONS}
    if set(answer_map) != expected:
        raise RuntimeError(f"answer key mismatch actual={len(answer_map)} expected={len(expected)}")
    attack_inputs = {}
    for filename in ("LARGE_MENTA_ATTACK_QUERIES.jsonl", "LARGE_MBA_ATTACK_QUERIES.jsonl", "LARGE_RAG_MIA_ATTACK_QUERIES.jsonl"):
        for row in read_jsonl(EXP / "inputs" / filename):
            attack_inputs[row["query_id"]] = row
    targets = {row["document_id"]: row for row in csv.DictReader((EXP / "inputs/LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))}
    bundle = {"attacks": attack_inputs, "targets": targets}
    checkpoint("LARGE_NATIVE_SCORING_STARTED", attack_queries=len(attack_inputs), conditions=4)
    discrete = evaluator.score_discrete(answer_map, bundle)
    _, menta_sessions = evaluator.score_menta(answer_map, bundle)
    privacy = evaluator.summarize_privacy(discrete, menta_sessions)

    benign = [row for row in retrieval if row["split"] == "HOLDOUT"]
    harm_rows = []
    summary_rows = []
    for condition in CONDITIONS[1:]:
        for row in benign:
            base = answer_map[(row["query_id"], "NO_DEFENSE")]["answer"]
            answer = answer_map[(row["query_id"], condition)]["answer"]
            alarm = bool(answer_map[(row["query_id"], condition)]["detector_alarm"])
            harm_rows.append({"query_id": row["query_id"], "domain": row["domain"], "condition": condition,
                              "intervened": alarm, "query": row["query"], "no_defense_answer": base,
                              "condition_answer": answer, "answer_changed": answer != base,
                              "answer_preservation_token_f1": token_f1(base, answer),
                              "new_refusal": is_refusal(answer) and not is_refusal(base),
                              "empty": not bool(answer.strip()), "no_defense_words": len(base.split()),
                              "condition_words": len(answer.split())})
        subset = [item for item in harm_rows if item["condition"] == condition]
        fp = [item for item in subset if item["intervened"]]
        summary_rows.append({"condition": condition, "benign_n": len(subset), "intervention_count": len(fp),
                             "intervention_rate": len(fp)/len(subset),
                             "overall_answer_preservation_f1": statistics_fmean(item["answer_preservation_token_f1"] for item in subset),
                             "overall_answer_change_rate": statistics_fmean(item["answer_changed"] for item in subset),
                             "overall_new_refusal_rate": statistics_fmean(item["new_refusal"] for item in subset),
                             "fp_subset_preservation_f1": statistics_fmean((item["answer_preservation_token_f1"] for item in fp), math.nan),
                             "fp_subset_answer_change_rate": statistics_fmean((item["answer_changed"] for item in fp), math.nan),
                             "fp_subset_new_refusal_rate": statistics_fmean((item["new_refusal"] for item in fp), math.nan),
                             "gold_correctness_available": False})
    write_csv(EXP / "tables/BENIGN_ANSWER_HARM.csv", harm_rows)
    write_csv(EXP / "tables/BENIGN_ANSWER_HARM_SUMMARY.csv", summary_rows)
    atomic_json(EXP / "audits/GOLD_QA_AUDIT.json", {
        "verdict": "GOLD_QA_NOT_AVAILABLE", "reason": "Locked BEIR benign queries provide relevance qrels but no reference answer strings.",
        "answer_preservation_is_not_correctness": True})
    atomic_json(EXP / "audits/FACTUALITY_AUDIT_SCOPE.json", {
        "verdict": "FORMAL_HALLUCINATION_CLAIM_NOT_SUPPORTED",
        "reported": ["native attack score", "answer preservation", "refusal", "empty answer"],
        "not_claimed": ["hallucination-free", "gold QA correctness", "formal groundedness"]})

    pmap = {(row["attack"], row["condition"]): row for row in privacy}
    checks = {
        "menta_privacy_improves_over_global": pmap[("MEntA", "LC_MIRABEL_SIMPLE_HIDE")]["native_roc_auc"] < pmap[("MEntA", "GLOBAL_BC_SIMPLE_HIDE")]["native_roc_auc"],
        "mba_privacy_not_worse_than_global": pmap[("MBA", "LC_MIRABEL_SIMPLE_HIDE")]["native_roc_auc"] <= pmap[("MBA", "GLOBAL_BC_SIMPLE_HIDE")]["native_roc_auc"],
        "rag_mia_privacy_not_worse_than_global": pmap[("RAG-MIA", "LC_MIRABEL_SIMPLE_HIDE")]["native_roc_auc"] <= pmap[("RAG-MIA", "GLOBAL_BC_SIMPLE_HIDE")]["native_roc_auc"],
    }
    passed = all(checks.values())
    result = {"campaign": "LC_MIRABEL_LARGE_V1", "phase": "LARGE_E2E",
              "verdict": "LC_MIRABEL_E2E_PASS" if passed else "LC_MIRABEL_PRIVACY_FAILED",
              "completed_utc": now(), "privacy": privacy, "checks": checks,
              "benign_harm": summary_rows, "gold_qa": "GOLD_QA_NOT_AVAILABLE",
              "factuality": "FORMAL_HALLUCINATION_CLAIM_NOT_SUPPORTED", "churn_allowed": passed}
    atomic_json(EXP / "E2E_RESULT.json", result)
    lines = ["# LC-MIRABEL Large E2E Report", "", f"- 판정: `{result['verdict']}`", "",
             "| Attack | No Defense AUC | Original AUC | Global BC AUC | LC AUC |", "|---|---:|---:|---:|---:|"]
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        lines.append(f"| {attack} | {pmap[(attack, CONDITIONS[0])]['native_roc_auc']:.4f} | {pmap[(attack, CONDITIONS[1])]['native_roc_auc']:.4f} | {pmap[(attack, CONDITIONS[2])]['native_roc_auc']:.4f} | {pmap[(attack, CONDITIONS[3])]['native_roc_auc']:.4f} |")
    lines += ["", "Gold answer가 없으므로 answer preservation을 QA correctness로 해석하지 않는다."]
    atomic_text(EXP / "reports/LC_MIRABEL_E2E_REPORT_KO.md", "\n".join(lines)+"\n")
    checkpoint(result["verdict"], churn_allowed=passed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def statistics_fmean(values, default=0.0):
    values = list(values)
    return float(np.mean(values)) if values else default


if __name__ == "__main__":
    main()
