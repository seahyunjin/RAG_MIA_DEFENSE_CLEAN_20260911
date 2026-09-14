#!/usr/bin/env python3
"""Freeze the IA-ST1 detection evaluation only after the full artifact is READY."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
POST = EXP / "post_ready"
FINAL_LC = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"


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
    result_path = EXP / "IA_ST1_FULL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("verdict") != "IA_STD_Q15_ST1_READY":
        raise RuntimeError("detection prohibited: full IA-ST1 artifact is not READY")
    for key in ("queries", "gt", "sessions"):
        path = Path(result["artifacts"][key]["path"])
        if sha(path) != result["artifacts"][key]["sha256"]:
            raise RuntimeError(f"frozen IA artifact drift: {key}")

    code = POST / "run_detection.py"
    manifest = FINAL_LC / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"
    expected_manifest = (manifest.with_suffix(".sha256")).read_text(encoding="utf-8").split()[0]
    if sha(manifest) != expected_manifest:
        raise RuntimeError("Final LC manifest hash mismatch")

    precommit = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "phase": "IA_SAME_FPR_DETECTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attack": "IA-Std-Q15-ST1; standardized stealth stress test, not Original or paper-exact IA",
        "full_result": item(result_path),
        "queries": item(Path(result["artifacts"]["queries"]["path"])),
        "gt": item(Path(result["artifacts"]["gt"]["path"])),
        "sessions": item(Path(result["artifacts"]["sessions"]["path"])),
        "final_lc_manifest": item(manifest),
        "frozen_reference_scores": item(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl"),
        "frozen_reference_embeddings": item(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"),
        "frozen_corpus_embeddings": item(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"),
        "frozen_corpus": item(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "frozen_benign_detection": item(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"),
        "positive_definition": "valid member-target IA-ST1 query",
        "negative_definition": "same frozen 1000-query benign deployment holdout used by Final LC",
        "budgets": [0.01, 0.025, 0.03, 0.05],
        "primary_budget": 0.025,
        "threshold_rule": "most permissive observed benign threshold with strict > alarms <= floor(alpha*N)",
        "detectors": {
            "MIRABEL": "canonical full-corpus Gumbel margin",
            "Final LC": "R_LC=-log((1+count of 200 semantic-local benign neighbors with M_b>=M_q)/201)",
        },
        "session_metrics": ["alarms/15", "protected/15", "any alarm", "all-15 alarm", "first alarm index"],
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260913},
        "hard_gate": {
            "noninferiority": "Final LC TPR@2.5% >= MIRABEL TPR@2.5% - 0.03",
            "catastrophic_miss_floor": "Final LC TPR@2.5% >= 0.05",
        },
        "preferred_not_gate": "Final LC TPR@2.5% > MIRABEL TPR@2.5%",
        "forbidden": ["threshold tuning", "formula changes", "attack-specific training", "invalid-session imputation"],
        "code": item(code),
    }
    out = POST / "configs" / "IA_ST1_DETECTION_PRECOMMIT.json"
    write_json(out, precommit)
    digest = sha(out)
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"precommit": str(out), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
