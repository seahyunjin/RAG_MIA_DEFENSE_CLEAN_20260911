#!/usr/bin/env python3
"""Shared frozen utilities for the Output-Conditioned Selective LOO Phase 1 audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/traffic_3/workspace/workspace/SH")
DATA_ROOT = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir")
RAW_ROOT = DATA_ROOT / "_raw"
REQUEST = Path("/home/cau_lab/.codex/attachments/316a9159-6f9c-464e-aa52-8b2a064ef47a/pasted-text.txt")
ATTACK_SCORES = WORK / "unified_benign_calibrated_multiview_gate_20260910/private/ATTACK_PER_QUERY.csv.gz"
ATTACK_QUERIES = WORK / "b2_counterfactual_min_20260910/phase_02_core6/QUERY_RESPONSE_REVIEW.csv.gz"
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")

DOMAINS = ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid")
FOLDERS = {"BeIR_nfcorpus": "nfcorpus", "BeIR_scidocs": "scidocs", "BeIR_trec-covid": "trec-covid"}
FAMILIES = ("MEntA", "S²-MIA", "MBA")
NATIVE_BUDGET = {"MEntA": 5, "S²-MIA": 1, "MBA": 1}
TARGET_FPRS = (0.01, 0.03, 0.05)
SEED = 20260911
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required "
                 "output format exactly. If the context is insufficient, answer exactly: I don't know.")
TOP_K = 4
CORPUS_SIZE = 1000
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW_TOKENS = 160


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n")


def atomic_csv(frame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {
        "experiment": "OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_PHASE1",
        "stage": stage,
        "updated_utc": now(),
        "pid": os.getpid(),
        "phase": 1,
        "paid_api_calls": 0,
        **details,
    }
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    lines = [
        "# Output-Conditioned Selective LOO — Phase 1",
        "",
        f"- Stage: **{stage}**",
        f"- Updated UTC: `{payload['updated_utc']}`",
        f"- PID: `{payload['pid']}`",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def strict_threshold(values, target_fpr: float) -> tuple[float, int, float]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite one-dimensional calibration scores required")
    allowed = int(math.floor(target_fpr * len(scores) + 1e-12))
    ordered = np.sort(scores)[::-1]
    threshold = float(ordered[allowed]) if allowed < len(scores) else float(np.nextafter(ordered[-1], -np.inf))
    false_positives = int(np.sum(scores > threshold))
    if false_positives > allowed:
        raise RuntimeError("strict-FPR contract violated")
    return threshold, false_positives, false_positives / len(scores)


def waterfill(lengths: list[int], total: int = SOURCE_TOKEN_BUDGET) -> list[int]:
    values = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(values), dtype=int)
    remaining = int(total)
    active = [index for index, length in enumerate(values) if length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(values[index] - caps[index]), remaining)
            if add > 0:
                caps[index] += add
                remaining -= add
                changed = True
            if caps[index] >= values[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def normalize_family(value: object) -> str:
    aliases = {"S2-MIA": "S²-MIA", "S²-MIA": "S²-MIA", "IA-MIA": "IA"}
    return aliases.get(str(value), str(value))
