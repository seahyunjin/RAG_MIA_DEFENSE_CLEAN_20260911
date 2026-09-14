#!/usr/bin/env python3
"""Frozen shared utilities for LC_MIRABEL_LARGE_V1."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
V2 = ROOT / "experiments" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
ORTH = ROOT / "experiments" / "ORTHOGONAL_EXPOSURE_SEARCH_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
REQUEST = Path("/home/cau_lab/.codex/attachments/963c18c8-5708-457d-a8ce-f3ff0a44aaab/pasted-text.txt")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
GPT2 = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2")
SPELL = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--oliverguhr--spelling-correction-english-base/snapshots/0e3958355a09d2816ed2701fdc2f4471d46c320e")
DOMAINS = ("nfcorpus", "scidocs", "trec-covid")
ATTACKS = ("MEntA", "MBA", "RAG-MIA")
ALPHAS = (0.01, 0.03, 0.05)
K_LOCAL = 200
SEED = 20260912


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).lower().split())


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "LC_MIRABEL_LARGE_V1", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), **details}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_json(EXP / "checkpoints" / f"{stage}.json", payload)
    lines = ["# LC_MIRABEL_LARGE_V1", "", f"- 현재 단계: `{stage}`",
             f"- 갱신(UTC): `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(EXP / "STATUS.md", "\n".join(lines) + "\n")


def empirical_upper_tail(sorted_reference: list[float], value: float) -> float:
    count = len(sorted_reference) - bisect.bisect_left(sorted_reference, float(value))
    return (1.0 + count) / (len(sorted_reference) + 1.0)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for index in order[i:j]:
            result[index] = rank
        i = j
    return result


def roc_points(positive: list[float], negative: list[float]) -> list[tuple[float, float]]:
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for value in positive:
        groups[float(value)][0] += 1
    for value in negative:
        groups[float(value)][1] += 1
    tp = fp = 0
    raw = [(0.0, 0.0)]
    for score in sorted(groups, reverse=True):
        tp += groups[score][0]
        fp += groups[score][1]
        raw.append((fp / len(negative), tp / len(positive)))
    best: dict[float, float] = {}
    for fpr, tpr in raw:
        best[fpr] = max(best.get(fpr, 0.0), tpr)
    return sorted(best.items())


def tpr_at_fpr(positive: list[float], negative: list[float], target: float) -> float:
    points = roc_points(positive, negative)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            return max(y0, y1) if x1 == x0 else y0 + (target - x0) * (y1 - y0) / (x1 - x0)
    return points[-1][1]


def matched_threshold(negative: list[float], alpha: float) -> tuple[float, int]:
    budget = math.floor(alpha * len(negative) + 1e-12)
    unique = sorted(set(map(float, negative)), reverse=True)
    if not unique:
        raise ValueError("empty negative score set")
    best_threshold, best_count = unique[0], 0
    for high, low in zip(unique, unique[1:]):
        threshold = (high + low) / 2.0
        count = sum(value > threshold for value in negative)
        if best_count <= count <= budget:
            best_threshold, best_count = threshold, count
    return best_threshold, best_count


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high-position) + ordered[high] * (position-low)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)

