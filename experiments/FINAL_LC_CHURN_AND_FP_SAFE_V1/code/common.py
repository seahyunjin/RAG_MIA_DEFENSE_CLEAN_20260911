#!/usr/bin/env python3
"""Shared, non-learning utilities for FINAL_LC_CHURN_AND_FP_SAFE_V1."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments/FINAL_LC_CHURN_AND_FP_SAFE_V1"
LC = ROOT / "experiments/LC_MIRABEL_LARGE_V1"
FINAL8 = ROOT / "experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE6 = ROOT / "experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
RECAL = ROOT / "experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
V6 = RECAL / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED"
OLD_SIDECAR = ROOT / "experiments/FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
CLEAN = ROOT / "experiments/CLEAN_CORE3_DEV_V1"

BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
DOMAINS = ("nfcorpus", "scidocs", "trec-covid")
VERSIONS = ("V0", "V10", "V25", "V50")
ATTACKS = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")
K_LOCAL = 200
OUTER_BUDGET = 0.025
STRICT_THRESHOLD = 3.80543877128208
SEED = 20260913


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows, fields: list[str] | None = None) -> None:
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **fields) -> None:
    payload = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), **fields}
    atomic_json(EXP / "STATUS.json", payload)
    atomic_text(EXP / "STATUS.md", "# FINAL_LC_CHURN_AND_FP_SAFE_V1\n\n" +
                f"- Stage: `{stage}`\n- Updated UTC: `{payload['updated_utc']}`\n" +
                "".join(f"- {key}: `{value}`\n" for key, value in fields.items()))


def verify_hashed_json(path: Path) -> dict:
    sidecar = path.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha_file(path)
    if actual != expected:
        raise RuntimeError(f"hash drift: {path}: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def matched_threshold(values, budget: float) -> tuple[float, int]:
    ordered = sorted(map(float, values), reverse=True)
    allowance = math.floor(budget * len(ordered) + 1e-12)
    if not ordered:
        raise ValueError("empty calibration values")
    if allowance >= len(ordered):
        return -math.inf, len(ordered)
    threshold = ordered[allowance]
    alarms = sum(value > threshold for value in ordered)
    if alarms > allowance:
        raise RuntimeError("strict threshold exceeds alarm budget")
    return threshold, alarms


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin
