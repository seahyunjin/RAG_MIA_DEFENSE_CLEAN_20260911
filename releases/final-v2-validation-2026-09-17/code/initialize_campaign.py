#!/usr/bin/env python3
"""Freeze Final V2 and its result lineage before remaining validation.

This script is deliberately CPU-only.  It never imports torch, never runs
retrieval/generation, and never derives a new detector or threshold.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1"
FREEZE = ROOT / "experiments" / "FINAL_LC_V2_FINAL_FREEZE_V1"
FINQA = ROOT / "experiments" / "FINAL_V2_FINQA_ABLATION_ADAPTIVE_V1"
FINQA_ADD = ROOT / "experiments" / "FINAL_V2_FINQA_VALIDATION_ADDENDUM_20260916"
FINQA_MENTA = ROOT / "experiments" / "FINAL_V2_FINQA_MENTA_RESUME_V1"
ROBUST = ROOT / "experiments" / "FINAL_V2_ROBUST_THRESHOLD_CONTAMINATION_V1"
CORE_GEOM = ROOT / "experiments" / "LOCAL_RETRIEVAL_GEOMETRY_FPR_MATCHED_V1"
FUSION = ROOT / "experiments" / "S1_G4_FUSION_DIAG_CPU_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE6 = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
IA = ROOT / "experiments" / "FINAL_LC_IA_RESOLUTION_V1" / "fpr3_sensitivity"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def id_manifest(path: Path, candidates=("query_id", "session_id", "document_id")) -> dict:
    rows = read_jsonl(path)
    ids = []
    for row in rows:
        value = next((row.get(key) for key in candidates if row.get(key) is not None), None)
        if value is not None:
            ids.append(str(value))
    return {"path": str(path), "file_sha256": sha(path), "rows": len(rows),
            "ordered_id_sha256": text_sha(ids), "unique_ids": len(set(ids))}


def verify_precommit_inputs(precommit: dict) -> list[dict]:
    output = []
    for name, record in precommit["inputs"].items():
        path = Path(record["path"])
        exists = path.is_file()
        actual = sha(path) if exists else None
        output.append({"name": name, "path": str(path), "expected_sha256": record["sha256"],
                       "actual_sha256": actual, "exists": exists, "match": exists and actual == record["sha256"]})
    return output


def verify_finqa() -> dict:
    rows = list(csv.DictReader((FINQA_ADD / "tables" / "TABLE_FINQA_CORE5_E2E.csv").open()))
    condition_alias = {"MIRABEL_2_5": "MIRABEL", "V2_2_5": "FINAL_V2"}
    lookup = {("S2-MIA" if row["attack"] == "S²-MIA" else row["attack"], condition_alias.get(row["condition"], row["condition"])): float(row["e_auc"])
              for row in rows if row.get("environment") == "FINQA"}
    expected = {
        ("DCMI-Std-Q2", "NO_DEFENSE"): 0.557442,
        ("DCMI-Std-Q2", "MIRABEL"): 0.524001,
        ("DCMI-Std-Q2", "FINAL_V2"): 0.529501,
        ("MBA", "NO_DEFENSE"): 0.632368,
        ("MBA", "MIRABEL"): 0.5417125,
        ("MBA", "FINAL_V2"): 0.541803,
        ("MEntA", "NO_DEFENSE"): 0.7636139643,
        ("MEntA", "MIRABEL"): 0.6926861797,
        ("MEntA", "FINAL_V2"): 0.6477708940,
        ("RAG-MIA", "NO_DEFENSE"): 0.5745,
        ("RAG-MIA", "MIRABEL"): 0.5305,
        ("RAG-MIA", "FINAL_V2"): 0.5385,
        ("S2-MIA", "NO_DEFENSE"): 0.54,
        ("S2-MIA", "MIRABEL"): 0.52375,
        ("S2-MIA", "FINAL_V2"): 0.5025,
    }
    detail = []
    for key, value in expected.items():
        actual = lookup.get(key)
        detail.append({"attack": key[0], "condition": key[1], "expected": value, "actual": actual,
                       "abs_error": None if actual is None else abs(actual - value),
                       "match": actual is not None and abs(actual - value) <= 1e-9})
    return {"verdict": "PASS" if all(x["match"] for x in detail) else "FAIL", "detail": detail,
            "source": str(FINQA_ADD / "tables" / "TABLE_FINQA_CORE5_E2E.csv")}


def main() -> None:
    for name in ("configs", "audits", "tables", "figures", "reports", "runtime", "logs", "paper", "package"):
        (EXP / name).mkdir(parents=True, exist_ok=True)
    frozen = read_json(FREEZE / "configs" / "FINAL_V2_FROZEN.json")
    precommit = read_json(FREEZE / "configs" / "PRECOMMIT.json")
    checks = verify_precommit_inputs(precommit)
    if not all(row["match"] for row in checks):
        write_csv(EXP / "audits" / "UPSTREAM_SHA256_AUDIT.csv", checks)
        raise RuntimeError("upstream SHA-256 mismatch")
    finqa_audit = verify_finqa()
    if finqa_audit["verdict"] != "PASS":
        atomic_json(EXP / "audits" / "FINQA_NUMERIC_AUDIT.json", finqa_audit)
        raise RuntimeError("FinQA frozen numeric mismatch")

    method = {
        "campaign": "FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1",
        "frozen_utc": utc(),
        "method_aliases": ["Final V2", "G4", "LRC-4"],
        "score": "G4(q)=s1-mean(s1,s2,s3,s4)",
        "top_k": 4,
        "operator": ">",
        "core_numeric_threshold": frozen["threshold"],
        "operating_protocol": "benign-only strict empirical 97.5th percentile; target FPR 2.5%; no attack samples",
        "domain_transfer_protocol": "same formula/action; benign-only target-domain recalibration allowed and reported separately from numeric zero-shot",
        "action": "on alarm hide current-turn rank-1 source, deterministic context backfill, generate once",
        "forbidden": ["formula changes", "new detector search", "s1 fusion", "k changes", "attack-specific routing",
                      "family-specific threshold", "post-result threshold retuning", "failed-candidate resurrection"],
        "model_identifiers": {
            "retriever_primary": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
            "generator_primary": "Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "generator_transfer": "unsloth/Llama-3.2-3B-Instruct (frozen artifact)",
        },
        "anchor_hashes": {
            "v2_precommit": sha(FREEZE / "configs" / "PRECOMMIT.json"),
            "v2_code": precommit["code_sha256"],
            "bge_weights": precommit["inputs"]["bge_weights"]["sha256"],
            "qwen_tokenizer": precommit["inputs"]["qwen_tokenizer"]["sha256"],
            "core_scorer": precommit["inputs"]["f8_scorer"]["sha256"],
            "dcmi_scorer": precommit["inputs"]["dcmi_scorer"]["sha256"],
            "ia_parser": precommit["inputs"]["ia_parser"]["sha256"],
        },
    }
    atomic_json(EXP / "configs" / "FINAL_V2_METHOD_FREEZE.json", method)

    lineage_inputs = {
        "core_detection": Path(precommit["inputs"]["core_detection"]["path"]),
        "ia_detection": Path(precommit["inputs"]["ia_detection"]["path"]),
        "benign_calibration": Path(precommit["inputs"]["gold_calibration"]["path"]),
        "benign_locked_test": Path(precommit["inputs"]["gold_test"]["path"]),
        "finqa_calibration": FINQA / "inputs" / "FINQA_BENIGN_CALIBRATION_1000.jsonl",
        "finqa_locked_test": FINQA / "inputs" / "FINQA_BENIGN_LOCKED_TEST_1000.jsonl",
        "finqa_attacks": FINQA / "inputs" / "FINQA_ATTACK_QUERIES.jsonl",
        "core_answers": Path(precommit["inputs"]["core_answers"]["path"]),
        "finqa_answers": FINQA / "runtime" / "FINQA_QWEN_BRANCH_ANSWERS.jsonl",
        "finqa_menta_evidence": FINQA_MENTA / "runtime" / "MENTA_EVIDENCE_COMPLETE.jsonl",
    }
    lineage = {name: id_manifest(path) for name, path in lineage_inputs.items()}
    result_lineage = {
        "created_utc": utc(), "method_freeze_sha256": sha(EXP / "configs" / "FINAL_V2_METHOD_FREEZE.json"),
        "artifacts": lineage, "upstream_hash_audit": "PASS", "finqa_numeric_audit": finqa_audit,
        "frozen_results": {
            "core": {"member_tpr": 0.9460, "nonmember_intervention": 0.0292, "benign_fpr": 0.0250},
            "finqa": {"member_tpr": 0.5450, "nonmember_intervention": 0.1120, "benign_fpr": 0.0280,
                      "mean_e_auc": {"NO_DEFENSE": 0.6136, "MIRABEL": 0.5625, "FINAL_V2": 0.5520},
                      "worst_e_auc": {"NO_DEFENSE": 0.7636, "MIRABEL": 0.6927, "FINAL_V2": 0.6478}},
            "member_g4": {"Core": 0.1731, "FinQA": 0.0907},
        },
    }
    atomic_json(EXP / "configs" / "FINAL_V2_RESULT_LINEAGE.json", result_lineage)
    write_csv(EXP / "audits" / "UPSTREAM_SHA256_AUDIT.csv", checks)
    atomic_json(EXP / "audits" / "FINQA_NUMERIC_AUDIT.json", finqa_audit)

    anchors = [EXP / "configs" / "FINAL_V2_METHOD_FREEZE.json",
               EXP / "configs" / "FINAL_V2_RESULT_LINEAGE.json",
               EXP / "audits" / "UPSTREAM_SHA256_AUDIT.csv",
               EXP / "audits" / "FINQA_NUMERIC_AUDIT.json"]
    lines = [f"{sha(path)}  {path.relative_to(EXP)}" for path in anchors]
    atomic_text(EXP / "configs" / "FINAL_V2_SHA256SUMS.txt", "\n".join(lines) + "\n")
    status = {"campaign": method["campaign"], "stage": "CURRENT_RESULTS_FROZEN", "updated_utc": utc(),
              "method_modified": False, "upstream_hash_audit": "PASS", "finqa_numeric_audit": "PASS",
              "next": "CONSTRUCTED_HARD_BENIGN_V1_BANK_FREEZE"}
    atomic_json(EXP / "STATUS.json", status)
    atomic_text(EXP / "STATUS.md", "# Status\n\n- Stage: `CURRENT_RESULTS_FROZEN`\n- Upstream SHA-256: `PASS`\n- FinQA numeric lineage: `PASS`\n- Method changed: `false`\n- Next: `CONSTRUCTED_HARD_BENIGN_V1_BANK_FREEZE`\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
