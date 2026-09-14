#!/usr/bin/env python3
"""Shared fail-closed helpers for the deployment-gap campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
CHURN = ROOT / "experiments" / "FINAL_LC_CHURN_AND_FP_SAFE_V1"
SIDECAR = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE6 = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
RECAL = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
TORCH_PYTHON = Path("/home/traffic_3/workspace/miniconda3/envs/torch/bin/python")
VERSIONS = ("V0", "V10", "V25", "V50")
ATTACKS = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")
CONDITIONS = ("NO_DEFENSE", "FINAL_LC_REFRESH")
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_json(path: Path, value: object) -> str:
    atomic_json(path, value)
    digest = sha_file(path)
    atomic_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return digest


def verify_hashed_json(path: Path) -> dict:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha_file(path)
    if expected != actual:
        raise RuntimeError(f"hash drift: {path}: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(stage: str, **extra: object) -> None:
    payload = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), **extra}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_json(EXP / "checkpoints" / f"{stage}.json", payload)
    lines = [f"# {EXP.name}", "", f"- Current stage: `{stage}`", f"- Updated UTC: `{payload['updated_utc']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in extra.items())
    atomic_text(EXP / "STATUS.md", "\n".join(lines) + "\n")
