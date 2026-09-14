#!/usr/bin/env python3
"""Frozen BGE-M3 retrieval over the precommitted disjoint reference pool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"
RETRIEVER = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    audit = json.loads((CAMPAIGN / "audits/REFERENCE_POOL_AUDIT.json").read_text(encoding="utf-8"))
    if audit["verdict"] != "REFERENCE_POOL_AUDIT_PASS":
        raise RuntimeError("reference pool did not pass disjointness audit")
    parent_rows = read_jsonl(PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")
    parent_embedding = np.load(PARENT / "cache/QUERY_EMBEDDINGS.float16.npy").astype(np.float32)
    if len(parent_rows) != len(parent_embedding) or len(parent_rows) != 1280:
        raise RuntimeError("parent query embedding alignment failure")
    risk_indices = [index for index, row in enumerate(parent_rows)
                    if bool(row["bc_q97_alarm"]) and (row["cohort"] == "ATTACK" or row["split"] == "HOLDOUT")]
    if not risk_indices:
        raise RuntimeError("no q97 alarm queries")
    reference = read_jsonl(CAMPAIGN / "inputs/REFERENCE_POOL.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    model = SentenceTransformer(str(RETRIEVER), device=device)
    model.max_seq_length = 512
    reference_embedding = model.encode(
        [row["source_text"] for row in reference], batch_size=32,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.save(CAMPAIGN / "cache/REFERENCE_EMBEDDINGS.float16.npy", reference_embedding.astype(np.float16))
    protected_embedding = np.load(PARENT / "cache/CORPUS_EMBEDDINGS.float16.npy").astype(np.float32)
    protected_rows = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    protected_index = {row["document_id"]: index for index, row in enumerate(protected_rows)}
    targets = {row["document_id"]: row for row in __import__("csv").DictReader(
        (PARENT / "inputs/SHARED_TARGETS.csv").open(encoding="utf-8"))}
    records = []
    for index in risk_indices:
        parent = parent_rows[index]
        qvec = parent_embedding[index]
        scores = reference_embedding @ qvec
        order = np.argsort(-scores, kind="stable")[:20]
        top20 = [reference[int(i)]["document_id"] for i in order]
        top20_scores = [float(scores[int(i)]) for i in order]
        selected_index = int(order[0])
        replacement = reference[selected_index]
        removed_id = parent["selected_source_id"]
        removed_similarity = float(reference_embedding[selected_index] @ protected_embedding[protected_index[removed_id]])
        target_id = parent.get("target_id")
        target_similarity = None
        if target_id in protected_index:
            target_similarity = float(reference_embedding[selected_index] @ protected_embedding[protected_index[target_id]])
        elif target_id in targets:
            # Nonmembers are absent from the protected embedding.  No new
            # embedding is computed for this diagnostic; report unavailable.
            target_similarity = None
        records.append({
            "query_id": parent["query_id"], "session_id": parent["session_id"],
            "query_index": parent["query_index"], "cohort": parent["cohort"],
            "split": parent["split"], "attack": parent.get("attack"),
            "membership": parent.get("membership"), "target_id": target_id,
            "query": parent["query"], "bc_q97_alarm": True,
            "selected_source_id": removed_id,
            "original_top_document_ids": parent["top_document_ids"],
            "replacement_source_id": replacement["document_id"],
            "replacement_reference_rank": 1,
            "reference_top20_ids": top20,
            "reference_top20_scores": top20_scores,
            "replacement_query_similarity": top20_scores[0],
            "replacement_removed_source_similarity": removed_similarity,
            "replacement_target_similarity": target_similarity,
            "replacement_normalized_text_hash": replacement["normalized_text_hash"],
            "replacement_exact_duplicate": False,
            "replacement_near_duplicate_cosine_ge_0_95": bool(removed_similarity >= 0.95 or (target_similarity is not None and target_similarity >= 0.95)),
        })
    output = CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    availability = len(records) == len(risk_indices) and all(row["replacement_source_id"] for row in records)
    alarm_ids = "\n".join(sorted(row["query_id"] for row in records))
    locator = "\n".join(f"{row['query_id']}\t{row['selected_source_id']}" for row in sorted(records, key=lambda x: x["query_id"]))
    result = {
        "verdict": "REFERENCE_RETRIEVAL_PASS" if availability else "REFERENCE_REPLACEMENT_UNAVAILABLE",
        "risk_queries": len(risk_indices), "replacement_available": len(records),
        "availability_rate": len(records) / len(risk_indices),
        "by_cohort": {name: sum(row["cohort"] == name for row in records) for name in ("ATTACK", "BENIGN")},
        "by_attack": {name: sum(row.get("attack") == name for row in records) for name in ("MEntA", "MBA", "RAG-MIA")},
        "alarm_ids_sha256": sha_text(alarm_ids),
        "locator_mapping_sha256": sha_text(locator),
        "parent_query_embeddings_sha256": sha256(PARENT / "cache/QUERY_EMBEDDINGS.float16.npy"),
        "reference_embeddings_sha256": sha256(CAMPAIGN / "cache/REFERENCE_EMBEDDINGS.float16.npy"),
        "reference_retrieval_sha256": sha256(output),
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(CAMPAIGN / "audits/REFERENCE_RETRIEVAL_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
