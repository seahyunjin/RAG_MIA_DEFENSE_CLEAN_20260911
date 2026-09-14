#!/usr/bin/env python3
"""Build the frozen 200/200-per-attack, four-DB Phase-A2 substrate."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (ATTACKS, CHURN, CORE3, CORE6, EXP, FINAL8, LC, ROOT, SEED,
                    VERSIONS, checkpoint, read_jsonl, sha_file, sha_text,
                    verify_hashed_json, write_jsonl, atomic_json)


def verify(pre: dict) -> None:
    for key in ("frozen_final_lc", "phase_a1", "phase_a1_detail", "churn_result", "query_manifest",
                "s2_provenance", "targets", "base_corpus", "base_corpus_embeddings", "added_corpus",
                "added_corpus_embeddings", "large_query_embeddings", "dcmi_query_embeddings",
                "s2_query_embeddings", "old_answers", "old_dcmi_answers", "old_menta_evidence"):
        value = pre[key]
        if sha_file(Path(value["path"])) != value["sha256"]:
            raise RuntimeError(f"Phase-A2 input drift: {key}")
    for version in VERSIONS:
        for key in ("manifest", "scores"):
            value = pre["versions"][version][key]
            if sha_file(Path(value["path"])) != value["sha256"]:
                raise RuntimeError(f"Phase-A2 input drift: {version}/{key}")
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"Phase-A2 code drift: {relative}")


def selection_key(attack: str, membership: str, session_id: str) -> tuple[str, str]:
    return sha_text(f"{SEED}|A2|{attack}|{membership}|{session_id}"), session_id


def assemble_query_embeddings(all_rows: list[dict], selected: list[dict], pre: dict) -> np.ndarray:
    large = np.load(Path(pre["large_query_embeddings"]["path"]), mmap_mode="r")
    dcmi = np.load(Path(pre["dcmi_query_embeddings"]["path"]), mmap_mode="r")
    s2 = np.load(Path(pre["s2_query_embeddings"]["path"]), mmap_mode="r")
    s2_rows = [row for row in all_rows if row["embedding_source"] == "NEW_BGE_S2"]
    s2_index = {row["query_id"]: index for index, row in enumerate(s2_rows)}
    output = np.empty((len(selected), large.shape[1]), dtype=np.float32)
    for out_index, row in enumerate(selected):
        if row["embedding_source"] == "LC_LARGE_FROZEN":
            output[out_index] = large[int(row["embedding_index"])]
        elif row["embedding_source"] == "DCMI_FROZEN":
            output[out_index] = dcmi[int(row["embedding_index"])]
        elif row["embedding_source"] == "NEW_BGE_S2":
            output[out_index] = s2[s2_index[row["query_id"]]]
        else:
            raise RuntimeError(f"unknown query embedding source: {row['embedding_source']}")
    if np.max(np.abs(np.linalg.norm(output, axis=1) - 1.0)) > 0.01:
        raise RuntimeError("selected query embeddings are not normalized")
    return output


def documents(pre: dict, version: str) -> tuple[list[str], np.ndarray]:
    base_rows = read_jsonl(Path(pre["base_corpus"]["path"]))
    added_rows = read_jsonl(Path(pre["added_corpus"]["path"]))
    base_vectors = np.asarray(np.load(Path(pre["base_corpus_embeddings"]["path"]), mmap_mode="r"), dtype=np.float32)
    added_vectors = np.asarray(np.load(Path(pre["added_corpus_embeddings"]["path"]), mmap_mode="r"), dtype=np.float32)
    vector = {row["document_id"]: base_vectors[index] for index, row in enumerate(base_rows)}
    vector.update({row["document_id"]: added_vectors[index] for index, row in enumerate(added_rows)})
    manifest = json.loads(Path(pre["versions"][version]["manifest"]["path"]).read_text(encoding="utf-8"))
    ids = manifest["ordered_document_ids"]
    matrix = np.asarray([vector[item] for item in ids], dtype=np.float32)
    if matrix.shape != (3000, 1024):
        raise RuntimeError(f"document matrix drift: {version}: {matrix.shape}")
    return ids, matrix


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "PHASE_A2_CHURN_E2E_PRECOMMIT.json")
    verify(pre)
    rows = read_jsonl(Path(pre["query_manifest"]["path"]))
    row_index = {row["query_id"]: index for index, row in enumerate(rows)}
    s2_split = {row["query_id"]: row.get("evaluation_split") for row in read_jsonl(Path(pre["s2_provenance"]["path"]))
                if row.get("attack") == "S²-MIA"}
    sessions = defaultdict(set)
    for row in rows:
        if row["split"] != "ATTACK":
            continue
        if row["attack"] == "S²-MIA" and s2_split.get(row["query_id"]) != "S2_EVALUATION":
            continue
        sessions[(row["attack"], row["membership"])].add(row["session_id"])
    chosen = set()
    selection = []
    for attack in ATTACKS:
        for membership in ("member", "nonmember"):
            ordered = sorted(sessions[(attack, membership)], key=lambda value: selection_key(attack, membership, value))
            if len(ordered) < 200:
                raise RuntimeError(f"insufficient sessions: {attack}/{membership}: {len(ordered)}")
            for rank, session_id in enumerate(ordered[:200], 1):
                chosen.add(session_id)
                selection.append({"attack": attack, "membership": membership, "session_id": session_id,
                                  "selection_rank": rank, "selection_key_sha256": selection_key(attack, membership, session_id)[0]})
    primary = []
    selection_by_session = {row["session_id"]: row for row in selection}
    for row in rows:
        if row["session_id"] not in chosen:
            continue
        primary.append({**row, "manifest_row_index": row_index[row["query_id"]], "primary_evaluation": True,
                        "s2_evaluation_split": s2_split.get(row["query_id"]),
                        "selection_rank": selection_by_session[row["session_id"]]["selection_rank"]})
    reference = []
    for row in rows:
        if row.get("attack") == "S²-MIA" and s2_split.get(row["query_id"]) == "S2_REFERENCE":
            reference.append({**row, "manifest_row_index": row_index[row["query_id"]], "primary_evaluation": False,
                              "s2_evaluation_split": "S2_REFERENCE", "selection_rank": None})
    selected = primary + reference
    expected_queries = {"MEntA": 2000, "MBA": 400, "RAG-MIA": 400, "S²-MIA": 400, "DCMI-Std-Q2": 800}
    actual_queries = Counter(row["attack"] for row in primary)
    if dict(actual_queries) != expected_queries:
        raise RuntimeError(f"primary query-count drift: {actual_queries}")
    if len(reference) != 402:
        raise RuntimeError(f"S2 reference drift: {len(reference)}")
    write_jsonl(EXP / "inputs" / "PHASE_A2_SELECTED_SESSIONS.jsonl", sorted(selection, key=lambda r: (r["attack"], r["membership"], r["selection_rank"])))
    write_jsonl(EXP / "inputs" / "PHASE_A2_SELECTED_QUERIES.jsonl", selected)
    query_vectors = assemble_query_embeddings(rows, selected, pre)
    prior = json.loads(Path(pre["churn_result"]["path"]).read_text(encoding="utf-8"))
    retrieval_rows = []
    for version in VERSIONS:
        ids, matrix = documents(pre, version)
        score_cache = np.load(Path(pre["versions"][version]["scores"]["path"]), allow_pickle=True)
        risk = score_cache["R_LC_REFRESH"]
        margins = score_cache["M"]
        tau = float(prior["versions"][version]["refresh_threshold"])
        for start in range(0, len(selected), 128):
            similarities = query_vectors[start:start + 128] @ matrix.T
            top = np.argpartition(-similarities, 4, axis=1)[:, :4]
            order = np.argsort(-np.take_along_axis(similarities, top, axis=1), axis=1, kind="stable")
            top = np.take_along_axis(top, order, axis=1)
            for offset, indices in enumerate(top):
                row = selected[start + offset]
                top_ids = [ids[int(value)] for value in indices]
                target_rank = top_ids.index(row["target_id"]) + 1 if row["target_id"] in top_ids else 0
                original_index = int(row["manifest_row_index"])
                retrieval_rows.append({**row, "db": version, "top_document_ids": top_ids,
                                       "selected_source_id": top_ids[0], "target_rank": target_rank,
                                       "target_retrieved_at4": target_rank > 0, "M": float(margins[original_index]),
                                       "R_LC_REFRESH": float(risk[original_index]), "threshold": tau,
                                       "final_lc_alarm": bool(float(risk[original_index]) > tau)})
            checkpoint("PHASE_A2_SUBSTRATE_PROGRESS", db=version, completed=min(start + 128, len(selected)), total=len(selected))
        del matrix
    write_jsonl(EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl", retrieval_rows)
    audit = {
        "verdict": "PHASE_A2_SUBSTRATE_READY",
        "primary_sessions": len(selection),
        "primary_queries": len(primary),
        "s2_reference_queries": len(reference),
        "retrieval_rows": len(retrieval_rows),
        "session_counts": {f"{attack}/{membership}": 200 for attack in ATTACKS for membership in ("member", "nonmember")},
        "primary_query_counts": dict(actual_queries),
        "selection_id_sha256": sha_text("\n".join(row["session_id"] for row in sorted(selection, key=lambda r: (r["attack"], r["membership"], r["selection_rank"])))),
        "selected_queries_sha256": sha_file(EXP / "inputs" / "PHASE_A2_SELECTED_QUERIES.jsonl"),
        "retrieval_sha256": sha_file(EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl"),
        "member_nonmember_symmetric": True,
        "s2_reference_excluded_from_primary": True,
    }
    atomic_json(EXP / "PHASE_A2_SUBSTRATE_AUDIT.json", audit)
    checkpoint("PHASE_A2_SUBSTRATE_READY", primary_sessions=2000, primary_queries=len(primary),
               s2_reference=402, retrieval_rows=len(retrieval_rows), next="PHASE_A2_GENERATION")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
