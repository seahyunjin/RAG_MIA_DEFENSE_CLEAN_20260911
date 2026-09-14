#!/usr/bin/env python3
"""Decompose Final-LC churn TPR into exposure and conditional detector behavior."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1"
PRECOMMIT = EXP / "configs" / "PHASE_A1_CHURN_CAUSAL_PRECOMMIT.json"
VERSIONS = ("V0", "V10", "V25", "V50")
ATTACKS = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, ""); return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checkpoint(stage: str, **extra: object) -> None:
    payload = {"campaign": "FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1", "phase": "A1_DB_CHURN_CAUSAL_SCORE_AUDIT", "stage": stage, "updated_utc": now(), **extra}
    atomic_json(EXP / "HEARTBEAT.json", payload)
    atomic_text(EXP / "STATUS.md", "\n".join(["# FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1", "", f"- phase: `A1`", f"- stage: `{stage}`", f"- updated_utc: `{payload['updated_utc']}`"] + [f"- {key}: `{value}`" for key, value in extra.items()]) + "\n")


def verify() -> dict:
    expected = PRECOMMIT.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha(PRECOMMIT) != expected: raise RuntimeError("precommit hash mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    keys = ["frozen_final_lc", "prior_churn_result", "query_manifest", "base_corpus", "base_corpus_embeddings", "added_corpus", "added_corpus_embeddings", "large_query_embeddings", "dcmi_query_embeddings", "s2_query_embeddings", "s2_provenance", "code"]
    for key in keys:
        if sha(Path(pre[key]["path"])) != pre[key]["sha256"]: raise RuntimeError(f"input drift: {key}")
    for version in VERSIONS:
        for key in ("manifest", "scores"):
            value = pre["versions"][version][key]
            if sha(Path(value["path"])) != value["sha256"]: raise RuntimeError(f"input drift: {version}/{key}")
    return pre


def assemble_queries(rows: list[dict], pre: dict) -> np.ndarray:
    large = np.load(Path(pre["large_query_embeddings"]["path"]), mmap_mode="r")
    dcmi = np.load(Path(pre["dcmi_query_embeddings"]["path"]), mmap_mode="r")
    s2 = np.load(Path(pre["s2_query_embeddings"]["path"]), mmap_mode="r")
    s2_rows = [row for row in rows if row["embedding_source"] == "NEW_BGE_S2"]
    s2_index = {row["query_id"]: index for index, row in enumerate(s2_rows)}
    output = np.empty((len(rows), large.shape[1]), dtype=np.float32)
    for index, row in enumerate(rows):
        if row["embedding_source"] == "LC_LARGE_FROZEN": output[index] = large[int(row["embedding_index"])]
        elif row["embedding_source"] == "DCMI_FROZEN": output[index] = dcmi[int(row["embedding_index"])]
        elif row["embedding_source"] == "NEW_BGE_S2": output[index] = s2[s2_index[row["query_id"]]]
        else: raise RuntimeError(f"unknown embedding source: {row['embedding_source']}")
    if np.max(np.abs(np.linalg.norm(output, axis=1) - 1.0)) > 0.01: raise RuntimeError("query embeddings not normalized")
    return output


def assemble_documents(pre: dict, version: str) -> tuple[list[str], np.ndarray]:
    base_rows = read_jsonl(Path(pre["base_corpus"]["path"]))
    added_rows = read_jsonl(Path(pre["added_corpus"]["path"]))
    base_embeddings = np.asarray(np.load(Path(pre["base_corpus_embeddings"]["path"]), mmap_mode="r"), dtype=np.float32)
    added_embeddings = np.asarray(np.load(Path(pre["added_corpus_embeddings"]["path"]), mmap_mode="r"), dtype=np.float32)
    vectors = {row["document_id"]: base_embeddings[index] for index, row in enumerate(base_rows)}
    vectors.update({row["document_id"]: added_embeddings[index] for index, row in enumerate(added_rows)})
    manifest = json.loads(Path(pre["versions"][version]["manifest"]["path"]).read_text(encoding="utf-8"))
    ids = manifest["ordered_document_ids"]
    matrix = np.asarray([vectors[document_id] for document_id in ids], dtype=np.float32)
    if matrix.shape != (3000, 1024): raise RuntimeError(f"document matrix drift: {version}/{matrix.shape}")
    return ids, matrix


def safe_mean(values: list[bool | float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    pre = verify()
    started = time.monotonic()
    rows = read_jsonl(Path(pre["query_manifest"]["path"]))
    s2_meta = {row["query_id"]: row.get("evaluation_split") for row in read_jsonl(Path(pre["s2_provenance"]["path"])) if row.get("attack") == "S²-MIA"}
    query_embeddings = assemble_queries(rows, pre)
    detail_rows, summary_rows = [], []
    for version in VERSIONS:
        ids, document_embeddings = assemble_documents(pre, version)
        scores_npz = np.load(Path(pre["versions"][version]["scores"]["path"]), allow_pickle=True)
        margins = scores_npz["M"]
        risks = scores_npz["R_LC_REFRESH"]
        prior = json.loads(Path(pre["prior_churn_result"]["path"]).read_text(encoding="utf-8"))
        threshold = float(prior["versions"][version]["refresh_threshold"])
        version_details = []
        for start in range(0, len(rows), 128):
            matrix = query_embeddings[start:start + 128] @ document_embeddings.T
            top = np.argpartition(-matrix, 4, axis=1)[:, :4]
            order = np.argsort(-np.take_along_axis(matrix, top, axis=1), axis=1, kind="stable")
            top = np.take_along_axis(top, order, axis=1)
            for offset, indices in enumerate(top):
                index = start + offset
                row = rows[index]
                if row["split"] != "ATTACK" or row["membership"] != "member": continue
                if row["attack"] == "S²-MIA" and s2_meta.get(row["query_id"]) != "S2_EVALUATION": continue
                top_ids = [ids[int(value)] for value in indices]
                target_rank = top_ids.index(row["target_id"]) + 1 if row["target_id"] in top_ids else 0
                exposed = target_rank > 0
                alarm = bool(float(risks[index]) > threshold)
                item = {"db": version, "attack": row["attack"], "query_id": row["query_id"], "session_id": row["session_id"], "query_index": row["query_index"], "target_id": row["target_id"], "top1_id": top_ids[0], "top2_id": top_ids[1], "top3_id": top_ids[2], "top4_id": top_ids[3], "target_rank": target_rank, "exposed": exposed, "target_not_retrieved": not exposed, "locator_hit": top_ids[0] == row["target_id"], "mirabel_margin": float(margins[index]), "final_lc_risk_refresh": float(risks[index]), "final_lc_threshold_refresh": threshold, "final_lc_alarm": alarm, "effective_protection_opportunity": bool(alarm and exposed and top_ids[0] == row["target_id"])}
                detail_rows.append(item); version_details.append(item)
            checkpoint("TOP4_PROGRESS", db=version, completed=min(start + 128, len(rows)), total=len(rows))
        for attack in ATTACKS:
            subset = [row for row in version_details if row["attack"] == attack]
            exposed = [row for row in subset if row["exposed"]]
            not_exposed = [row for row in subset if not row["exposed"]]
            ranks = Counter(row["target_rank"] for row in subset)
            summary_rows.append({"db": version, "attack": attack, "member_queries": len(subset), "member_sessions": len({row['session_id'] for row in subset}), "retrieval_at_1": safe_mean([row["target_rank"] == 1 for row in subset]), "retrieval_at_4": safe_mean([row["exposed"] for row in subset]), "target_rank_1": ranks[1], "target_rank_2": ranks[2], "target_rank_3": ranks[3], "target_rank_4": ranks[4], "target_not_retrieved": ranks[0], "locator_hit_at_1_overall": safe_mean([row["locator_hit"] for row in subset]), "locator_hit_given_exposed": safe_mean([row["locator_hit"] for row in exposed]), "mirabel_margin_mean": safe_mean([row["mirabel_margin"] for row in subset]), "mirabel_margin_median": float(np.median([row["mirabel_margin"] for row in subset])) if subset else None, "mirabel_margin_std": float(np.std([row["mirabel_margin"] for row in subset])) if subset else None, "final_lc_tpr_overall": safe_mean([row["final_lc_alarm"] for row in subset]), "final_lc_tpr_given_exposed": safe_mean([row["final_lc_alarm"] for row in exposed]), "final_lc_tpr_given_not_exposed": safe_mean([row["final_lc_alarm"] for row in not_exposed]), "effective_protection_opportunity": safe_mean([row["effective_protection_opportunity"] for row in subset])})
        checkpoint("VERSION_COMPLETE", db=version, member_queries=len(version_details))
        del document_embeddings
    write_csv(EXP / "tables" / "PHASE_A1_CHURN_RETRIEVAL_EXPOSURE.csv", summary_rows)
    write_csv(EXP / "tables" / "PHASE_A1_MEMBER_QUERY_DETAIL.csv", detail_rows)
    result = {"campaign": "FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1", "phase": "A1_DB_CHURN_CAUSAL_SCORE_AUDIT", "verdict": "PHASE_A1_COMPLETE", "completed_utc": now(), "runtime_seconds": time.monotonic() - started, "member_query_rows": len(detail_rows), "summary_rows": len(summary_rows), "definitions": {"EXPOSED": "target in actual Top-4", "effective_protection_opportunity": "alarm AND target retrieved AND locator equals target"}, "outputs": {"summary": str(EXP / "tables" / "PHASE_A1_CHURN_RETRIEVAL_EXPOSURE.csv"), "detail": str(EXP / "tables" / "PHASE_A1_MEMBER_QUERY_DETAIL.csv")}, "performance_interpretation": "deferred until Phase A2 No-Defense/Final-LC E2E", "final_lc_modified": False}
    atomic_json(EXP / "PHASE_A1_RESULT.json", result)
    checkpoint("PHASE_A1_COMPLETE", runtime_seconds=round(result["runtime_seconds"], 2), rows=len(detail_rows), next="PHASE_A2_CHURN_E2E")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

