#!/usr/bin/env python3
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
SOURCE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
REQUEST = Path("/home/cau_lab/.codex/attachments/2e9f1731-d8ad-4e92-af46-5e28b3af042f/pasted-text.txt")
TORCH_PYTHON = Path("/home/traffic_3/workspace/miniconda3/envs/torch/bin/python")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
TOPI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/datasets--McGill-NLP--TopiOCQA/snapshots/66cd1dbf5577c653ecb99b385200f08e15e12f30/data")
SEED = 20260912
K_LOCAL = 200
BUDGETS = (0.01, 0.025, 0.03, 0.05)


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
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, ""); return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def freeze_json(path: Path, value: object) -> str:
    atomic_json(path, value)
    digest = sha_file(path)
    atomic_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return digest


def verify_hashed_json(path: Path) -> dict:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file(): raise RuntimeError(f"missing sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha_file(path)
    if actual != expected: raise RuntimeError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(stage: str, **extra: object) -> None:
    payload = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), **extra}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_json(EXP / "checkpoints" / f"{stage}.json", payload)
    lines = [f"# {EXP.name}", "", f"- Current stage: `{stage}`", f"- Updated UTC: `{payload['updated_utc']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in extra.items())
    atomic_text(EXP / "STATUS.md", "\n".join(lines) + "\n")
