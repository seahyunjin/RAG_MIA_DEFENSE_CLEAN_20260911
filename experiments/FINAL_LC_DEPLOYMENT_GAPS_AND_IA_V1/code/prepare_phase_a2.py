#!/usr/bin/env python3
"""Freeze Phase-A2 cohort, scorer, generator, and churn inputs before E2E."""
from __future__ import annotations

import json

from common import (ATTACKS, CHURN, CONDITIONS, CORE3, CORE6, EXP, FINAL8, LC,
                    QWEN, ROOT, SEED, SIDECAR, VERSIONS, checkpoint,
                    freeze_json, now, sha_file)


def item(path, **extra):
    if not path.is_file():
        raise RuntimeError(f"missing Phase-A2 input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def main() -> None:
    for folder in ("configs", "inputs", "cache", "runtime", "tables", "reports", "logs", "checkpoints", "tests"):
        (EXP / folder).mkdir(parents=True, exist_ok=True)
    code_names = ("common.py", "prepare_phase_a2.py", "build_phase_a2_substrate.py",
                  "run_phase_a2_generation.py", "run_phase_a2_scoring.py")
    code = [EXP / "code" / name for name in code_names]
    pre = {
        "campaign": EXP.name,
        "phase": "A2_CHURN_E2E_CAUSAL_SCREEN",
        "created_utc": now(),
        "status": "PRECOMMITTED_BEFORE_A2_SUBSTRATE_OR_GENERATION",
        "final_lc_modified": False,
        "frozen_final_lc": item(CORE6 / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "phase_a1": item(EXP / "PHASE_A1_RESULT.json"),
        "phase_a1_detail": item(EXP / "tables" / "PHASE_A1_MEMBER_QUERY_DETAIL.csv"),
        "churn_result": item(CHURN / "DB_CHURN_V2_CORRECTED_RESULT.json"),
        "query_manifest": item(CHURN / "inputs" / "CHURN_QUERY_MANIFEST.jsonl", n=22000),
        "s2_provenance": item(FINAL8 / "inputs" / "FINAL_SUPPORTED_QUERY_MANIFEST.jsonl"),
        "targets": item(LC / "inputs" / "LARGE_SHARED_TARGETS.csv"),
        "base_corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "base_corpus_embeddings": item(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"),
        "added_corpus": item(CHURN / "inputs" / "MISSING_EMBEDDING_UNION.jsonl"),
        "added_corpus_embeddings": item(CHURN / "cache" / "CHURN_ADDED_DOCUMENT_EMBEDDINGS.float16.npy"),
        "large_query_embeddings": item(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"),
        "dcmi_query_embeddings": item(CORE6 / "cache" / "STANDARDIZED_QUERY_EMBEDDINGS.float16.npy"),
        "s2_query_embeddings": item(CHURN / "cache" / "S2_QUERY_EMBEDDINGS.float16.npy"),
        "versions": {version: {
            "manifest": item(SIDECAR / "manifests" / f"DB_CHURN_{version}.json"),
            "scores": item(CHURN / "cache" / f"{version}_QUERY_SCORES.npz"),
        } for version in VERSIONS},
        "old_answers": item(FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl"),
        "old_dcmi_answers": item(CORE6 / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl"),
        "old_menta_evidence": item(FINAL8 / "tables" / "MENTA_QUERY_EVIDENCE.csv"),
        "generator": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "snapshot": str(QWEN),
            "config": item(QWEN / "config.json"),
            "do_sample": False,
            "num_beams": 1,
            "source_token_budget": 2048,
            "max_prompt_tokens": 3072,
            "max_new_tokens": {"MEntA": 160, "MBA": 160, "RAG-MIA": 12, "S²-MIA": 160, "DCMI-Std-Q2": 12},
        },
        "attacks": list(ATTACKS),
        "versions_order": list(VERSIONS),
        "conditions": list(CONDITIONS),
        "primary_selection": {
            "unit": "session/target",
            "member_per_attack": 200,
            "nonmember_per_attack": 200,
            "key": f"SHA256('{SEED}|A2|<attack>|<membership>|<session_id>'), ascending; session_id tie-break",
            "s2_scope": "S2_EVALUATION only",
        },
        "s2_reference": {
            "role": "attacker-side frozen native-scorer fitting only; excluded from primary 200/200 metrics",
            "scope": "all 402 pre-existing S2_REFERENCE sessions per DB/condition",
        },
        "native_scoring": {
            "MEntA": "paper-faithful frozen claim extraction + document NLI, Q5 session ROC-AUC",
            "MBA": "mask reconstruction accuracy; malformed/incomplete=0",
            "RAG-MIA": "Yes/No membership score",
            "S²-MIA": "reference-fitted BLEU/perplexity balanced accuracy",
            "DCMI-Std-Q2": "original-minus-perturbed Yes score session ROC-AUC",
        },
        "classification": {
            "case_a": "Retrieval@4 materially decreases; No-Defense moves toward chance; TPR|EXPOSED drop <=5pp",
            "case_b": "Retrieval@4 remains; No-Defense remains strong; TPR|EXPOSED materially decreases",
            "case_c": "both exposure loss and conditional detector loss",
            "material_exposure_drop": ">=0.10 V0-to-V50 macro Retrieval@4",
            "no_defense_toward_chance": ">=0.02 V0-to-V50 reduction in macro mean abs(native metric-0.5)",
            "conditional_detector_failure": ">0.05 V0-to-V50 macro TPR|EXPOSED drop",
            "canary": "record DB_UPDATE_EXPOSURE_CANARY_REQUIRED only for Case B/mixed; do not implement",
        },
        "prohibitions": ["detector change", "new LC feature", "new protection", "threshold search", "attack-specific calibration"],
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code},
    }
    digest = freeze_json(EXP / "configs" / "PHASE_A2_CHURN_E2E_PRECOMMIT.json", pre)
    checkpoint("PHASE_A2_PRECOMMITTED", precommit_sha256=digest, primary_sessions=5 * 400,
               s2_reference_sessions=402, next="BUILD_PHASE_A2_SUBSTRATE")
    print(json.dumps({"precommit_sha256": digest, "primary_sessions": 2000}, indent=2))


if __name__ == "__main__":
    main()
