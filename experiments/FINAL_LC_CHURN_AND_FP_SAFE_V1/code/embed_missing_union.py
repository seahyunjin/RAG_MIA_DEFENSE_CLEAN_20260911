#!/usr/bin/env python3
"""One frozen BGE-M3 GPU pass for churn-added docs and uncached S² queries."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import BGE, EXP, atomic_json, checkpoint, now, read_jsonl, sha_file, sha_text, verify_hashed_json


def verify(pre: dict) -> None:
    for path, digest in pre["input_sha256"].items():
        if sha_file(Path(path)) != digest:
            raise RuntimeError(f"frozen input drift: {path}")
    for path, digest in pre["derived_input_sha256"].items():
        if sha_file(Path(path)) != digest:
            raise RuntimeError(f"derived input drift: {path}")
    for relative, digest in pre["code_sha256"].items():
        if sha_file(Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911") / relative) != digest:
            raise RuntimeError(f"code drift: {relative}")


def main() -> None:
    pre = verify_hashed_json(EXP / "configs/DB_CHURN_V2_PRECOMMIT.json")
    verify(pre)
    docs = read_jsonl(EXP / "inputs/MISSING_EMBEDDING_UNION.jsonl")
    queries = [row for row in read_jsonl(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl")
               if row["embedding_source"] == "NEW_BGE_S2"]
    doc_cache = EXP / "cache/CHURN_ADDED_DOCUMENT_EMBEDDINGS.float16.npy"
    query_cache = EXP / "cache/S2_QUERY_EMBEDDINGS.float16.npy"
    manifest_path = EXP / "cache/BGE_EMBEDDING_UNION_MANIFEST.json"
    if doc_cache.exists() and query_cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["document_cache_sha256"] == sha_file(doc_cache)
                and manifest["query_cache_sha256"] == sha_file(query_cache)
                and manifest["documents"] == len(docs) and manifest["queries"] == len(queries)):
            checkpoint("BGE_EMBEDDING_UNION_CACHE_REUSED", documents=len(docs), queries=len(queries))
            print(json.dumps(manifest, indent=2))
            return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; frozen BGE-M3 GPU embedding required")
    checkpoint("BGE_EMBEDDING_UNION_STARTED", documents=len(docs), queries=len(queries), gpu=True)
    started = time.monotonic()
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    document_embeddings = np.asarray(model.encode(
        [row["source_text"] for row in docs], batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    query_embeddings = np.asarray(model.encode(
        [row["query"] for row in queries], batch_size=48, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    del model
    torch.cuda.empty_cache()
    np.save(doc_cache, document_embeddings.astype(np.float16))
    np.save(query_cache, query_embeddings.astype(np.float16))
    manifest = {
        "campaign": EXP.name, "created_utc": now(),
        "model": "BAAI/bge-m3", "snapshot": BGE.name,
        "model_config_sha256": sha_file(BGE / "config.json"),
        "batch_config": {"documents": 32, "queries": 48, "max_seq_length": 512,
                         "normalize_embeddings": True, "stored_dtype": "float16"},
        "documents": len(docs), "queries": len(queries),
        "document_ids_sha256": sha_text("\n".join(row["document_id"] for row in docs)),
        "document_text_hashes_sha256": sha_text("\n".join(row["normalized_text_hash"] for row in docs)),
        "query_ids_sha256": sha_text("\n".join(row["query_id"] for row in queries)),
        "document_cache": str(doc_cache), "document_cache_sha256": sha_file(doc_cache),
        "query_cache": str(query_cache), "query_cache_sha256": sha_file(query_cache),
        "runtime_seconds": time.monotonic() - started,
        "training_steps": 0, "gradient_updates": 0,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(EXP / "checkpoints/BGE_EMBEDDING_UNION_COMPLETE.json", manifest)
    checkpoint("BGE_EMBEDDING_UNION_COMPLETE", documents=len(docs), queries=len(queries),
               runtime_seconds=round(manifest["runtime_seconds"], 2), gpu=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
