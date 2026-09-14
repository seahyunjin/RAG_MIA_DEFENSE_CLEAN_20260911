#!/usr/bin/env python3
"""Shared utilities for FINAL_8ATTACK_E2E_FRAMEWORK_V1."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
REQUEST = Path("/home/cau_lab/.codex/attachments/f1537eb5-14f3-45de-923b-c0787690af26/pasted-text.txt")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
CONDITIONS = (
    "NO_DEFENSE",
    "ORIGINAL_MIRABEL_SIMPLE_HIDE",
    "GLOBAL_BC_SIMPLE_HIDE",
    "FINAL_LC_OUTER_SIMPLE_HIDE",
)
DISPLAY = {
    "NO_DEFENSE": "No Defense",
    "ORIGINAL_MIRABEL_SIMPLE_HIDE": "Original MIRABEL",
    "GLOBAL_BC_SIMPLE_HIDE": "Global BC-MIRABEL",
    "FINAL_LC_OUTER_SIMPLE_HIDE": "Final LC+outer MIRABEL",
}


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_json(EXP / "checkpoints" / f"{stage}.json", payload)
    lines = [f"# {EXP.name}", "", f"- 현재 단계: `{stage}`", f"- 갱신(UTC): `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(EXP / "STATUS.md", "\n".join(lines) + "\n")
    log = EXP / "logs" / "pipeline.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

