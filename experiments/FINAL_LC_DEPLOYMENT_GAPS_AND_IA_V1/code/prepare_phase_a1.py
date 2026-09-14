#!/usr/bin/env python3
"""Precommit the frozen DB-churn retrieval/exposure decomposition."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1"
OLD = ROOT / "experiments" / "FINAL_LC_CHURN_AND_FP_SAFE_V1"
SIDECAR = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
CORE6 = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def item(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}")
    return {"path": str(path), "sha256": sha(path)}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    code = EXP / "code" / "run_phase_a1.py"
    versions = {}
    for version in ("V0", "V10", "V25", "V50"):
        versions[version] = {
            "manifest": item(SIDECAR / "manifests" / f"DB_CHURN_{version}.json"),
            "scores": item(OLD / "cache" / f"{version}_QUERY_SCORES.npz"),
        }
    precommit = {
        "campaign": "FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1",
        "phase": "A1_DB_CHURN_CAUSAL_SCORE_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_final_lc": item(CORE6 / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "prior_churn_result": item(OLD / "DB_CHURN_V2_CORRECTED_RESULT.json"),
        "query_manifest": item(OLD / "inputs" / "CHURN_QUERY_MANIFEST.jsonl"),
        "base_corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "base_corpus_embeddings": item(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"),
        "added_corpus": item(OLD / "inputs" / "MISSING_EMBEDDING_UNION.jsonl"),
        "added_corpus_embeddings": item(OLD / "cache" / "CHURN_ADDED_DOCUMENT_EMBEDDINGS.float16.npy"),
        "large_query_embeddings": item(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"),
        "dcmi_query_embeddings": item(CORE6 / "cache" / "STANDARDIZED_QUERY_EMBEDDINGS.float16.npy"),
        "s2_query_embeddings": item(OLD / "cache" / "S2_QUERY_EMBEDDINGS.float16.npy"),
        "s2_provenance": item(FINAL8 / "inputs" / "FINAL_SUPPORTED_QUERY_MANIFEST.jsonl"),
        "versions": versions,
        "attacks": ["MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2"],
        "membership_scope": "member queries only for retrieval/exposure and TPR; S2_EVALUATION only",
        "exposed": "target document ID is in the exact generation Top-4 retrieval list",
        "effective_protection_opportunity": "Final LC Refresh alarm AND target in Top-4 AND locator Top-1 equals target",
        "runtime_feature": False,
        "training_steps": 0,
        "prohibitions": ["new detector", "new LC feature", "new action", "threshold search", "attack-specific calibration"],
        "code": item(code),
    }
    out = EXP / "configs" / "PHASE_A1_CHURN_CAUSAL_PRECOMMIT.json"
    atomic_json(out, precommit)
    digest = sha(out)
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"precommit": str(out), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

