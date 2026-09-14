#!/usr/bin/env python3
"""Shared, frozen utilities for ORTHOGONAL_EXPOSURE_SEARCH_V1."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "ORTHOGONAL_EXPOSURE_SEARCH_V1"
V2 = ROOT / "experiments" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
V1 = ROOT / "experiments" / "BC_DUALTAIL_DETECTOR_V1_STRICT"
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
REQUEST = Path("/home/cau_lab/.codex/attachments/2c2cddc6-e05b-4ac3-8fc8-be88baf2fdd7/pasted-text.txt")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
ATTACKS = ("MEntA", "MBA", "RAG-MIA")
ALPHAS = (0.01, 0.03, 0.05)
PRIMARY_ALPHA = 0.03
SEED = 20260912


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "ORTHOGONAL_EXPOSURE_SEARCH_V1", "stage": stage,
               "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_json(EXP / "checkpoints" / f"{stage}.json", payload)
    atomic_text(EXP / "STATUS.md", "# ORTHOGONAL_EXPOSURE_SEARCH_V1\n\n" +
                f"- 현재 단계: `{stage}`\n- 갱신: `{payload['updated_utc']}`\n" +
                "".join(f"- {k}: `{v}`\n" for k, v in details.items()))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("empty percentile input")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi-pos) + ordered[hi] * (pos-lo)


def empirical_cutoff(values: list[float], alpha: float) -> float:
    descending = sorted(map(float, values), reverse=True)
    k = math.ceil(alpha * len(descending))
    if k < 1:
        raise ValueError("calibration sample too small")
    return descending[k-1]


def empirical_upper_tail(reference_sorted: list[float], value: float) -> float:
    count_ge = len(reference_sorted) - bisect.bisect_left(reference_sorted, float(value))
    return (1.0 + count_ge) / (len(reference_sorted) + 1.0)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i]); ranks = [0.0] * len(values); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j): ranks[order[k]] = rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y): return None
    a, b = average_ranks(x), average_ranks(y)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    da, db = [v-ma for v in a], [v-mb for v in b]
    den = math.sqrt(sum(v*v for v in da) * sum(v*v for v in db))
    return None if den == 0 else sum(u*v for u, v in zip(da, db)) / den


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0: return None, None
    p = successes/n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, center-radius), min(1.0, center+radius)


def roc_points(positive: list[float], negative: list[float]) -> list[tuple[float, float]]:
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for x in positive: groups[float(x)][0] += 1
    for x in negative: groups[float(x)][1] += 1
    tp = fp = 0; raw = [(0.0, 0.0)]
    for score in sorted(groups, reverse=True):
        tp += groups[score][0]; fp += groups[score][1]
        raw.append((fp/len(negative), tp/len(positive)))
    best: dict[float, float] = {}
    for fpr, tpr in raw: best[fpr] = max(best.get(fpr, 0.0), tpr)
    return sorted(best.items())


def tpr_at_fpr(positive: list[float], negative: list[float], target: float) -> float:
    points = roc_points(positive, negative)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            return max(y0, y1) if x1 == x0 else y0 + (target-x0)/(x1-x0)*(y1-y0)
    return points[0][1] if target <= points[0][0] else points[-1][1]


def matched_binary_threshold(negative: list[float], alpha: float) -> tuple[float, int]:
    budget = math.floor(alpha * len(negative) + 1e-12)
    unique = sorted(set(map(float, negative)), reverse=True)
    if budget <= 0: return max(unique), 0
    best_t, best_n = max(unique), 0
    for high, low in zip(unique, unique[1:]):
        threshold = (high + low)/2
        count = sum(x > threshold for x in negative)
        if best_n <= count <= budget: best_t, best_n = threshold, count
    return best_t, best_n


def normalize_tokens(text: str) -> set[str]:
    """NFKC/lowercase Unicode letter-or-number tokens; punctuation separates."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    out: list[str] = []; current: list[str] = []
    for char in normalized:
        if unicodedata.category(char)[0] in {"L", "N"}:
            current.append(char)
        elif current:
            out.append("".join(current)); current = []
    if current: out.append("".join(current))
    return set(out)


def bootstrap_delta(rows: list[dict], negative_m: list[float], negative_x: list[float],
                    x_key: str, iterations: int, seed: int) -> tuple[float, float, float]:
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_unit[row["session_id"]].append(row)
    units = sorted(by_unit); rng = random.Random(seed); deltas = []
    for _ in range(iterations):
        sampled = [r for _ in units for r in by_unit[rng.choice(units)]]
        deltas.append(tpr_at_fpr([r[x_key] for r in sampled], negative_x, PRIMARY_ALPHA) -
                      tpr_at_fpr([r["M"] for r in sampled], negative_m, PRIMARY_ALPHA))
    return statistics.fmean(deltas), percentile(deltas, .025), percentile(deltas, .975)

