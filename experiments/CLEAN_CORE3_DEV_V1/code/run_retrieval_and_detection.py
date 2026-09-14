#!/usr/bin/env python3
"""Fresh Top-4 retrieval plus full-corpus MIRABEL/BC calibration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
INPUTS = CAMPAIGN / "inputs"
AUDITS = CAMPAIGN / "audits"
TABLES = CAMPAIGN / "tables"
CACHE = CAMPAIGN / "cache"
CONFIGS = CAMPAIGN / "configs"
RETRIEVER = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def empirical_cutoff(values: list[float], alpha: float) -> float:
    descending = sorted((float(x) for x in values), reverse=True)
    k = math.ceil(alpha * len(descending))
    if k < 1:
        raise ValueError("calibration sample too small")
    return descending[k - 1]


def main() -> None:
    precommit = CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.json"
    checksum = (CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256").read_text(encoding="utf-8").split()[0]
    if sha_file(precommit) != checksum:
        raise RuntimeError("precommit checksum mismatch")
    docs = read_jsonl(INPUTS / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    query_rows = []
    for split, path in (("CALIBRATION", INPUTS / "BENIGN_CALIBRATION.jsonl"), ("HOLDOUT", INPUTS / "BENIGN_HOLDOUT.jsonl")):
        for row in read_jsonl(path):
            query_rows.append({**row, "cohort": "BENIGN", "split": split, "attack": None, "membership": None, "target_id": None, "session_id": row["query_id"], "query_index": 1})
    for attack, path in (("MEntA", INPUTS / "MENTA_ATTACK_QUERIES.jsonl"), ("MBA", INPUTS / "MBA_ATTACK_QUERIES.jsonl"), ("RAG-MIA", INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl")):
        for row in read_jsonl(path):
            query_rows.append({**row, "cohort": "ATTACK", "split": "DEVELOPMENT", "attack": attack})

    if len({r["query_id"] for r in query_rows}) != len(query_rows):
        raise RuntimeError("query IDs are not unique")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    model = SentenceTransformer(str(RETRIEVER), device=device)
    model.max_seq_length = 512
    doc_texts = [r["source_text"] for r in docs]
    query_texts = [r["query"] for r in query_rows]
    doc_embeddings = model.encode(doc_texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    query_embeddings = model.encode(query_texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "CORPUS_EMBEDDINGS.float16.npy", doc_embeddings.astype(np.float16))
    np.save(CACHE / "QUERY_EMBEDDINGS.float16.npy", query_embeddings.astype(np.float16))

    doc_ids = [r["document_id"] for r in docs]
    rows = []
    batch_size = 128
    for start in range(0, len(query_rows), batch_size):
        score_batch = query_embeddings[start:start + batch_size] @ doc_embeddings.T
        for offset, scores in enumerate(score_batch):
            query = query_rows[start + offset]
            order = np.argsort(-scores, kind="stable")
            top = order[:4]
            top_ids = [doc_ids[int(i)] for i in top]
            top_scores = [float(scores[int(i)]) for i in top]
            top1 = top_scores[0]
            stats = canonical_mirabel_from_moments(
                top1=top1,
                sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(doc_ids),
                confidence=0.95,
            )
            target_id = query.get("target_id")
            target_rank = top_ids.index(target_id) + 1 if target_id in top_ids else 0
            rows.append({
                "query_id": query["query_id"],
                "session_id": query.get("session_id", query["query_id"]),
                "query_index": int(query.get("query_index", 1)),
                "cohort": query["cohort"],
                "split": query["split"],
                "attack": query.get("attack"),
                "domain": query["domain"],
                "membership": query.get("membership"),
                "target_id": target_id,
                "query": query["query"],
                "top_document_ids": top_ids,
                "top_scores": top_scores,
                "target_retrieved_at1": bool(target_rank == 1),
                "target_retrieved_at4": bool(target_rank > 0),
                "target_rank": target_rank,
                "selected_source_id": top_ids[0],
                "s_max": stats.top1,
                "mirabel_threshold": stats.threshold,
                "mirabel_margin": stats.margin,
                "mirabel_background_mean": stats.background_mean,
                "mirabel_background_std": stats.background_std,
                "mirabel_corpus_n": stats.corpus_size,
                "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
                "original_alarm": bool(stats.margin > 0.0),
            })
        write_json(CAMPAIGN / "HEARTBEAT.json", {"stage": "FRESH_RETRIEVAL_MIRABEL", "completed_queries": min(start + batch_size, len(query_rows)), "total_queries": len(query_rows), "updated_utc": datetime.now(timezone.utc).isoformat()})

    calibration = [r["mirabel_margin"] for r in rows if r["cohort"] == "BENIGN" and r["split"] == "CALIBRATION"]
    if len(calibration) != 500:
        raise RuntimeError(f"expected 500 calibration margins, got {len(calibration)}")
    thresholds = {"q99": empirical_cutoff(calibration, 0.01), "q97": empirical_cutoff(calibration, 0.03), "q95": empirical_cutoff(calibration, 0.05)}
    for row in rows:
        for name, threshold in thresholds.items():
            row[f"bc_{name}_alarm"] = bool(row["mirabel_margin"] > threshold)

    cache_path = CACHE / "CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(CACHE / "CLEAN_CORE3_RETRIEVAL_CACHE_MANIFEST.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "benign": sum(r["cohort"] == "BENIGN" for r in rows),
        "attack": sum(r["cohort"] == "ATTACK" for r in rows),
        "retriever": str(RETRIEVER),
        "top_k": 4,
        "corpus_sha256": sha_file(INPUTS / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
        "cache_sha256": sha_file(cache_path),
        "thresholds": thresholds,
        "runtime_seconds": time.monotonic() - started,
    })

    holdout = [r for r in rows if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    TABLES.mkdir(parents=True, exist_ok=True)
    fpr_rows = []
    points = [("ORIGINAL", 0.05, 0.0, "original_alarm"), ("BC_Q99", 0.01, thresholds["q99"], "bc_q99_alarm"), ("BC_Q97", 0.03, thresholds["q97"], "bc_q97_alarm"), ("BC_Q95", 0.05, thresholds["q95"], "bc_q95_alarm")]
    for name, nominal, threshold, field in points:
        for domain in ("ALL", "nfcorpus", "scidocs", "trec-covid"):
            subset = holdout if domain == "ALL" else [r for r in holdout if r["domain"] == domain]
            fp = sum(bool(r[field]) for r in subset)
            fpr_rows.append({"operating_point": name, "nominal_calibration_fpr": nominal, "threshold": threshold, "domain": domain, "holdout_n": len(subset), "false_positives": fp, "measured_holdout_fpr": fp / len(subset) if subset else None})
    with (TABLES / "BENIGN_FPR.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fpr_rows[0]))
        writer.writeheader(); writer.writerows(fpr_rows)
    print(json.dumps({"rows": len(rows), "thresholds": thresholds, "holdout_fpr": [r for r in fpr_rows if r["domain"] == "ALL"], "runtime_seconds": time.monotonic()-started}, indent=2))


if __name__ == "__main__":
    main()
