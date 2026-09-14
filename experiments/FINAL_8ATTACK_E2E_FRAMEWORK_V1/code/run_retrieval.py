#!/usr/bin/env python3
"""Build one shared retrieval/detection artifact, reusing only verified identical rows."""
from __future__ import annotations

import bisect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import BGE, CORE3, EXP, LC, ROOT, atomic_json, checkpoint, read_jsonl, sha_file, write_jsonl

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION


def p_upper(sorted_values: list[float], value: float) -> float:
    return (1 + len(sorted_values) - bisect.bisect_left(sorted_values, value)) / (len(sorted_values) + 1)


def verify() -> dict:
    path = EXP / "configs" / "FINAL_8ATTACK_E2E_PRECOMMIT.json"
    expected = (path.with_suffix(".sha256")).read_text().split()[0]
    if sha_file(path) != expected:
        raise RuntimeError("precommit hash mismatch")
    pre = json.loads(path.read_text())
    if pre["code_sha256"][str(Path(__file__).relative_to(ROOT))] != sha_file(Path(__file__)):
        raise RuntimeError("retrieval code changed after freeze")
    for item in (pre["queries"], pre["targets"], pre["final_defense"], pre["membership_manifest"]):
        if sha_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen artifact drift: {item['path']}")
    return pre


def main() -> None:
    pre = verify()
    checkpoint("PHASE_D_SHARED_RETRIEVAL_STARTED")
    final_queries = read_jsonl(Path(pre["queries"]["path"]))
    prior_rows = read_jsonl(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl")
    prior_map = {row["query_id"]: row for row in prior_rows if row["split"] != "REFERENCE"}
    reference_rows = [row for row in prior_rows if row["split"] == "REFERENCE"]
    if len(prior_map) != 15000 or len(reference_rows) != 1000:
        raise RuntimeError("LC detection cache count drift")
    output = []
    s2_queries = []
    for row in final_queries:
        if row["query_id"] in prior_map:
            old = prior_map[row["query_id"]]
            if old["query"] != row["query"]:
                raise RuntimeError(f"unsafe retrieval reuse: {row['query_id']}")
            output.append({**row, **{key: old[key] for key in (
                "top_document_ids", "top_scores", "selected_source_id", "target_rank", "target_retrieved_at1",
                "target_retrieved_at4", "M", "mirabel_threshold", "mirabel_background_mean",
                "mirabel_background_std", "mirabel_formula_version", "p_local", "R_LC", "p_global", "R_GLOBAL",
                "local_neighbor_ids_sha256", "local_neighbor_similarity_max", "local_neighbor_similarity_min")},
                "retrieval_reuse": "VERIFIED_LC_CACHE"})
        elif row["attack"] == "S²-MIA":
            s2_queries.append(row)
        else:
            raise RuntimeError(f"query has no retrieval path: {row['query_id']}")

    documents = read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    doc_embeddings = np.asarray(np.load(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"), dtype=np.float32)
    all_query_embeddings = np.asarray(np.load(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if doc_embeddings.shape[0] != len(documents) or all_query_embeddings.shape[0] != len(prior_rows):
        raise RuntimeError("embedding cache shape drift")
    ref_embeddings = all_query_embeddings[:1000]
    ref_margins = [float(row["M"]) for row in reference_rows]
    global_sorted = sorted(ref_margins)
    started = time.monotonic()
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    s2_embeddings = np.asarray(model.encode([row["query"] for row in s2_queries], batch_size=32,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    del model
    torch.cuda.empty_cache()
    doc_ids = [row["document_id"] for row in documents]
    for start in range(0, len(s2_queries), 128):
        query_batch = s2_embeddings[start:start+128]
        scores_batch = query_batch @ doc_embeddings.T
        local_similarity = query_batch @ ref_embeddings.T
        for offset, scores in enumerate(scores_batch):
            source = s2_queries[start + offset]
            top_idx = np.argpartition(-scores, 4)[:4]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
            top_ids = [doc_ids[int(index)] for index in top_idx]
            top_scores = [float(scores[int(index)]) for index in top_idx]
            stats = canonical_mirabel_from_moments(top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)), corpus_size=len(doc_ids), confidence=.95)
            similarities = local_similarity[offset]
            neighbors = np.argpartition(-similarities, 200)[:200]
            neighbors = neighbors[np.argsort(-similarities[neighbors], kind="stable")]
            local_sorted = sorted(ref_margins[int(index)] for index in neighbors)
            p_local = p_upper(local_sorted, float(stats.margin))
            p_global = p_upper(global_sorted, float(stats.margin))
            target = source["target_id"]
            target_rank = top_ids.index(target) + 1 if target in top_ids else 0
            output.append({**source, "top_document_ids": top_ids, "top_scores": top_scores,
                "selected_source_id": top_ids[0], "target_rank": target_rank,
                "target_retrieved_at1": target_rank == 1, "target_retrieved_at4": target_rank > 0,
                "M": float(stats.margin), "mirabel_threshold": float(stats.threshold),
                "mirabel_background_mean": float(stats.background_mean), "mirabel_background_std": float(stats.background_std),
                "mirabel_formula_version": MIRABEL_FORMULA_VERSION, "p_local": p_local, "R_LC": -math.log(p_local),
                "p_global": p_global, "R_GLOBAL": -math.log(p_global),
                "local_neighbor_ids_sha256": __import__("hashlib").sha256("\n".join(reference_rows[int(i)]["query_id"] for i in neighbors).encode()).hexdigest(),
                "local_neighbor_similarity_max": float(similarities[neighbors[0]]),
                "local_neighbor_similarity_min": float(similarities[neighbors[-1]]), "retrieval_reuse": "NEW_S2_ENCODING"})
        checkpoint("PHASE_D_S2_RETRIEVAL_PROGRESS", completed=min(start+128, len(s2_queries)), total=len(s2_queries))
    output.sort(key=lambda row: row["query_id"])
    if len(output) != len(final_queries) or {row["query_id"] for row in output} != {row["query_id"] for row in final_queries}:
        raise RuntimeError("final retrieval key mismatch")
    path = EXP / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"
    write_jsonl(path, output)
    manifest = {"rows": len(output), "verified_reuses": sum(row["retrieval_reuse"] == "VERIFIED_LC_CACHE" for row in output),
                "new_s2_rows": len(s2_queries), "sha256": sha_file(path), "runtime_seconds": time.monotonic()-started,
                "member_target_top4": {attack: sum(r["target_retrieved_at4"] for r in output if r["attack"] == attack and r.get("membership") == "member") /
                    max(1, sum(1 for r in output if r["attack"] == attack and r.get("membership") == "member"))
                    for attack in ("MEntA", "MBA", "RAG-MIA", "S²-MIA")}}
    atomic_json(EXP / "cache" / "FINAL_RETRIEVAL_MANIFEST.json", manifest)
    checkpoint("PHASE_D_SHARED_RETRIEVAL_COMPLETE", **manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

