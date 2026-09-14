#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/traffic_3/workspace/workspace/SH")
PARENT = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
DV = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DV_LOO_V2_PHASE1"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DEFENSE_NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7")
MENTA_NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots/04dcf11f844b07bc57015169fca2b7d6df8299d5")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required "
                 "output format exactly. If the context is insufficient, answer exactly: I don't know.")
MAX_PROMPT_TOKENS = 3072
MAX_NEW_TOKENS = 160
SEED = 20260911
CONDITIONS = ("NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL", "BC_CGD")
FAMILIES = ("MEntA", "S²-MIA", "MBA")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


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
    payload = {"campaign": "BC_CGD_SMALL", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    lines = ["# BC-MIRABEL + Counterfactual Grounded Disclosure", "",
             f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def strict_threshold(values, target_fpr: float) -> tuple[float, int, float]:
    scores = np.asarray(values, dtype=np.float64)
    allowed = int(math.floor(target_fpr * len(scores) + 1e-12))
    ordered = np.sort(scores)[::-1]
    threshold = float(ordered[allowed])
    false_positives = int(np.sum(scores > threshold))
    if false_positives > allowed:
        raise RuntimeError("strict-FPR contract violated")
    return threshold, false_positives, false_positives / len(scores)


def token_f1(left: object, right: object) -> float:
    a = TOKEN_RE.findall(str(left).casefold())
    b = TOKEN_RE.findall(str(right).casefold())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    return 2.0 * common / (len(a) + len(b))


def refusal(value: object) -> bool:
    text = str(value).strip().casefold()
    phrases = ("i don't know", "i do not know", "cannot determine", "insufficient context",
               "not enough information", "unable to answer", "cannot answer")
    return any(phrase in text for phrase in phrases)


def effective_auc(labels, scores) -> tuple[float, float]:
    raw = float(roc_auc_score(np.asarray(labels, int), np.asarray(scores, float)))
    return raw, max(raw, 1.0 - raw)


def bootstrap_auc(labels, scores, iterations: int = 5000, seed: int = SEED) -> tuple[float, float, float, float]:
    y = np.asarray(labels, int)
    values = np.asarray(scores, float)
    positive = values[y == 1]
    negative = values[y == 0]
    rng = np.random.default_rng(seed)
    raw_values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        p = positive[rng.integers(0, len(positive), len(positive))]
        n = negative[rng.integers(0, len(negative), len(negative))]
        pair = p[:, None] - n[None, :]
        raw_values[index] = (pair > 0).mean() + 0.5 * (pair == 0).mean()
    effective_values = np.maximum(raw_values, 1.0 - raw_values)
    return (float(np.quantile(raw_values, .025)), float(np.quantile(raw_values, .975)),
            float(np.quantile(effective_values, .025)), float(np.quantile(effective_values, .975)))


def tpr_at_nonmember_fpr(member, nonmember, target: float) -> tuple[float, float, float]:
    threshold, _, actual = strict_threshold(nonmember, target)
    return float(np.mean(np.asarray(member, float) > threshold)), actual, threshold


def render_prompt(tokenizer, query: str, source_ids: list[str], visible: list[str], removed_rank: int) -> tuple[str, list[int]]:
    blocks = [f"[Document {rank}]\n{text}" for rank, text in enumerate(visible, 1)
              if rank != int(removed_rank)]
    joined = "\n\n".join(blocks)
    user = f"Retrieved context:\n{joined}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids[-MAX_PROMPT_TOKENS:]
    return rendered, ids
