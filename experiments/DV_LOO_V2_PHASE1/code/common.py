#!/usr/bin/env python3
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
PARENT = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
REQUEST = Path("/home/cau_lab/.codex/attachments/9687e7d1-4d9c-4dab-9119-83d497a2d8d3/pasted-text.txt")
DOMAINS = ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid")
FAMILIES = ("MEntA", "S²-MIA", "MBA")
TARGET_FPRS = (0.01, 0.03, 0.05)
RHO = 0.05
SEED = 20260911


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


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
    payload = {"campaign": "DV_LOO_V2_PHASE1", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), "phase": 1, "protected_generation_count": 0,
               "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    lines = ["# DV-LOO V2 Phase 1", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{payload['pid']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def strict_threshold(values, target_fpr: float) -> tuple[float, int, float]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite 1-D calibration scores required")
    allowed = int(math.floor(target_fpr * len(scores) + 1e-12))
    ordered = np.sort(scores)[::-1]
    threshold = float(ordered[allowed]) if allowed < len(scores) else float(np.nextafter(ordered[-1], -np.inf))
    fp = int(np.sum(scores > threshold))
    if fp > allowed:
        raise RuntimeError("strict-FPR contract violated")
    return threshold, fp, fp / len(scores)


def empirical_cdf(calibration, values):
    reference = np.sort(np.asarray(calibration, dtype=np.float64))
    query = np.asarray(values, dtype=np.float64)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("finite empirical-CDF inputs required")
    return np.searchsorted(reference, query, side="right") / float(len(reference))


def mirabel_margin(similarities, rho: float = RHO):
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Mirabel requires finite full-corpus similarities")
    top = int(np.argmax(values))
    s_max = float(values[top])
    remainder = np.delete(values, top)
    mu_q = float(remainder.mean())
    sigma_q = float(remainder.std(ddof=0))
    n = len(values)
    root = math.sqrt(2.0 * math.log(n))
    mu_n = mu_q + sigma_q * root
    c = -math.log(-math.log(1.0 - rho))
    tau_q = mu_n + c * sigma_q / root
    return top, s_max, mu_q, sigma_q, float(tau_q), float(s_max - tau_q)
