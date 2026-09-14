#!/usr/bin/env python3
"""Write both result-blind precommits before Phase A/B execution."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, PARENT, QWEN, REQUEST, V1, V2, atomic_json, atomic_text, now, sha_file


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def main() -> None:
    for name in ("configs", "tables", "reports", "audits", "cache", "private", "logs", "checkpoints"):
        (EXP / name).mkdir(parents=True, exist_ok=True)
    old = json.loads((V2 / "PHASE1_RESULT.json").read_text())
    if old["verdict"] != "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED":
        raise RuntimeError("frozen V2 verdict drift")
    scripts = {p.name: sha_file(p) for p in sorted((EXP / "code").glob("*.py"))}
    shared = {
        "campaign": "ORTHOGONAL_EXPOSURE_SEARCH_V1",
        "created_utc": now(),
        "development_notice": "THIS_COHORT_IS_NOW_DEVELOPMENT_DATA",
        "frozen_v2_verdict": old["verdict"],
        "frozen_v2_result": fingerprint(V2 / "PHASE1_RESULT.json"),
        "fresh_retrieval": fingerprint(V2 / "cache/FRESH_REPLICATION_RETRIEVAL_SCORES.jsonl"),
        "protected_db": fingerprint(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "legacy_retrieval": fingerprint(PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"),
        "legacy_score_rows": fingerprint(V1 / "tables/SCORE_ROWS.csv"),
        "request": fingerprint(REQUEST), "code_sha256": scripts,
        "primary": {"positive": "member-target attack query", "negative": "fresh benign query",
                    "nonmember_attack": "diagnostic only", "unit": "query", "fpr": 0.03,
                    "secondary_fpr": [0.01, 0.05], "comparison": "empirical ROC linear interpolation"},
        "attacks": ["MEntA", "MBA", "RAG-MIA"],
    }
    qwen_index = QWEN / "model.safetensors.index.json"
    phase_a = {**shared, "phase": "A_EXISTING_SIGNAL_COMPLEMENTARITY_AUDIT",
        "signals": {
            "MIRABEL": "frozen canonical full-corpus margin M",
            "DualTail": "frozen V1 max(-ln p_M,-ln p_C)",
            "QLL": "max softmax over four mean query-token log-likelihoods; ANALYSIS_ONLY_NON_BLACKBOX",
            "Answer-LOO": "SIGNAL_UNAVAILABLE unless exact fresh answer/ablation lineage exists",
        },
        "qll": {"model_snapshot": str(QWEN), "snapshot_commit": QWEN.name,
                "model_index": fingerprint(qwen_index),
                "canonical_implementation": fingerprint(PARENT.parents[1] / "code/qll_release/final_model/qll_source_hide_final.py"),
                "formula": "mean query-token log P(query|Context: source; User query: query), then max softmax",
                "eligibility": "ANALYSIS_ONLY_NON_BLACKBOX"},
        "selection_prohibitions": ["new feature", "threshold search beyond fixed FPR", "fusion", "attack-label tuning"]}
    pa = EXP / "configs/COMPLEMENTARITY_AUDIT_PRECOMMIT.json"
    atomic_json(pa, phase_a); atomic_text(pa.with_suffix(".sha256"), sha_file(pa)+"\n")
    phase_b = {**shared, "phase": "B_TRAINING_FREE_SPARSE_EXPOSURE",
        "normalization": ["Unicode NFKC", "lowercase", "Unicode alphanumeric tokens", "punctuation separator",
                          "numbers retained", "no stopwords/stemming/lemmatization/entity/fuzzy matching", "unique query tokens"],
        "idf": "log((3000+1)/(df(t)+1))+1",
        "lexical_score": "sum_idf(U(q) intersection U(d1))/sum_idf(U(q)); d1=BGE top1; zero denominator => 0",
        "tail_calibration": "legacy benign 500 deterministically frozen by V1 as 250 TAIL_REFERENCE/250 THRESHOLD_CALIBRATION",
        "combined_score": "max(-ln p_M,-ln p_L)", "comparison_operator": "strict >",
        "gate": {"MEntA_delta_tpr3": ">=0.05", "MEntA_cluster_bootstrap_ci_low": ">0",
                 "MBA_delta": ">=-0.05", "RAG-MIA_delta": ">=-0.05", "core3_macro_delta": ">0"},
        "if_fail": ["SPARSE_EXPOSURE_SCREEN_FAILED", "TRAINING_FREE_HANDCRAFTED_DETECTOR_SEARCH_CLOSED"],
        "if_pass": "write separate frozen confirmation precommit; select fresh 100/100 targets with zero overlap"}
    pb = EXP / "configs/SPARSE_EXPOSURE_V1_PRECOMMIT.json"
    atomic_json(pb, phase_b); atomic_text(pb.with_suffix(".sha256"), sha_file(pb)+"\n")
    atomic_json(EXP / "checkpoints/PRECOMMITS_WRITTEN.json", {
        "created_utc": now(), "phase_a_sha256": sha_file(pa), "phase_b_sha256": sha_file(pb),
        "frozen_v2_verdict": old["verdict"]})
    print(json.dumps({"phase_a": sha_file(pa), "phase_b": sha_file(pb)}, indent=2))


if __name__ == "__main__": main()

