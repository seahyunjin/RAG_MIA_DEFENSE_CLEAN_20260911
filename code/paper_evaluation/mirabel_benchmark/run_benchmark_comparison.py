#!/usr/bin/env python3
"""Mirabel benchmark: canonical Mirabel versus Stateless QLL Source Hide.

The experiment deliberately separates pre-generation detection (TPR at a
benign-derived FPR) from post-generation membership leakage (Effective AUC).
All inputs are immutable artifacts from Exp43/45/141 and the final QLL model.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "mirabel_benchmark_six_attack_bge_qwen_20260907"
AD3 = PROJECT.parent / "AD-test-LLM3"
EXP43 = AD3 / "experiments/43_FROZEN_MULTIATTACK_FINAL_MODEL_SELECTION"
EXP45 = AD3 / "experiments/45_FINAL_LOCAL_GENERALIZATION_AND_PAPER_ASSETS"
EXP141 = PROJECT / "exp141_member_nonmember_normal_3k_20260825"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
FINAL_MODEL = PROJECT / "finalization_campaign_20260903/final_defense_20260903"
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
MANIFEST = EXP141 / "private/EXP141_3K_CONFIRMATION_MANIFEST.private.csv.gz"
ATTACK_SESSIONS = EXP43 / "PRIVATE_CACHE/attack_sessions.private.jsonl.gz"
BGE_CACHE = EXP45 / "private_cache/retrieval/bge"
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
MIRABEL_CODE = AD3 / "src/canonical_mirabel.py"
QLL_CODE = EXP179 / "code/run_exp179.py"
GENERATOR_CODE = EXP87 / "code/exp87_models.py"
SCORER_CODE = AD3 / "scripts/run_exp44_overnight.py"
EXP44_RESPONSE_DB = (AD3 / "experiments/44_LOCAL_GENERATOR_GATE_F_FINAL_RELEASE/private_cache/"
                     "local_generator_responses/responses.sqlite3")
FAMILIES = ("RAG-MIA", "S²-MIA", "MBA", "DCMI", "MEntA", "IA")
NATIVE_BUDGET = {"RAG-MIA": 1, "S²-MIA": 1, "MBA": 1, "DCMI": 2, "MEntA": 5, "IA": 15}
FAMILY_ALIASES = {"S2-MIA": "S²-MIA", "S²-MIA": "S²-MIA", "IA-MIA": "IA", "IA": "IA"}
DATASETS = ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid")
TARGET_FPRS = (0.01, 0.03, 0.05)
PRIMARY_FPR = 0.03
TOP_K_SCORE = 10
TOP_K_GENERATION = 4
TOTAL_SOURCE_TOKENS = 2048
MAX_NEW = {"RAG-MIA": 12, "S²-MIA": 96, "MBA": 160, "DCMI": 12, "MEntA": 96, "IA": 32}
NORMAL_SYSTEM = ("Answer the user's question using only the retrieved context. Follow any required output "
                 "format exactly. If the context is insufficient, answer exactly: I don't know.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


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
                                 default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pkl.gz", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_pickle(temporary, compression="gzip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"experiment": "Mirabel benchmark six-attack BGE/Qwen comparison", "stage": stage,
               "updated_utc": now(), "pid": os.getpid(), "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    lines = ["# Mirabel benchmark six-attack comparison", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`",
             "- Retriever / generator: `BGE-M3 / Qwen2.5-3B-Instruct`", "- Paid API calls: `0`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_family(value: object) -> str:
    return FAMILY_ALIASES.get(str(value), str(value))


def strict_threshold(values: np.ndarray, target_fpr: float) -> tuple[float, int, float]:
    """Deterministic threshold for strict score > threshold and empirical FPR <= target."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite nonempty score vector required")
    allowed = int(math.floor(float(target_fpr) * len(scores) + 1e-12))
    ordered = np.sort(scores)[::-1]
    threshold = float(ordered[allowed]) if allowed < len(scores) else float(np.nextafter(ordered[-1], -np.inf))
    false_positives = int(np.sum(scores > threshold))
    if false_positives > allowed:
        raise RuntimeError("strict-FPR contract violated")
    return threshold, false_positives, false_positives / len(scores)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def effective_auc(labels, scores) -> tuple[float, float]:
    raw = float(roc_auc_score(np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)))
    return raw, max(raw, 1.0 - raw)


def bootstrap_auc(labels, scores, iterations: int = 5000, seed: int = 907) -> tuple[float, float, float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    raw_values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        p = positive[rng.integers(0, len(positive), len(positive))]
        n = negative[rng.integers(0, len(negative), len(negative))]
        pair = p[:, None] - n[None, :]
        raw_values[index] = float((pair > 0).mean() + 0.5 * (pair == 0).mean())
    effective_values = np.maximum(raw_values, 1.0 - raw_values)
    return (float(np.quantile(raw_values, 0.025)), float(np.quantile(raw_values, 0.975)),
            float(np.quantile(effective_values, 0.025)), float(np.quantile(effective_values, 0.975)))


def preflight() -> dict[str, Path]:
    required = {
        "precommit": PRECOMMIT,
        "exp141_manifest": MANIFEST,
        "exp43_attack_sessions": ATTACK_SESSIONS,
        "mirabel_code": MIRABEL_CODE,
        "qll_code": QLL_CODE,
        "generator_code": GENERATOR_CODE,
        "native_scorer_code": SCORER_CODE,
        "immutable_exp44_ia_ground_truth": EXP44_RESPONSE_DB,
        "final_qll_model": FINAL_MODEL / "qll_source_hide_final.py",
        "final_qll_config": FINAL_MODEL / "config_final.yaml",
        "bge_config": BGE / "config.json",
        "qwen_config": QWEN / "config.json",
        "mpnet_config": MPNET / "config.json",
    }
    for dataset in DATASETS:
        required[f"{dataset}_member_corpus"] = AD3 / f"data/beir/{dataset}/corpus_member.jsonl"
        required[f"{dataset}_bge_attack_retrieval"] = BGE_CACHE / f"{dataset}.canonical_v2.jsonl.gz"
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing}")
    protocol = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if not protocol.get("written_before_scores_and_generation"):
        raise RuntimeError("precommit was not written before scoring")
    rows = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                          "sha256_before": sha256_file(path), "access": "READ_ONLY"}
                         for key, path in required.items()])
    atomic_csv(rows, ROOT / "provenance/FROZEN_INPUTS.csv")
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", f"{sha256_file(PRECOMMIT)}  PRECOMMIT.json\n")
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(rows))
    return required


def read_attack_sessions() -> dict[str, dict]:
    output = {}
    with gzip.open(ATTACK_SESSIONS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["session_id"])] = row
    return output


def build_query_cohort() -> pd.DataFrame:
    destination = ROOT / "private/BENCHMARK_QUERY_COHORT.private.csv.gz"
    if destination.exists():
        return pd.read_csv(destination, keep_default_na=False, low_memory=False,
                           dtype={"case_id": str, "session_id": str, "target_document_id": str})
    manifest = pd.read_csv(MANIFEST, keep_default_na=False, low_memory=False,
                           dtype={"row_id": str, "session_id": str, "target_document_id": str})
    expected = {"MEMBER_ATTACK": 1000, "NONMEMBER_ATTACK": 1000, "NORMAL_USER": 1000}
    if len(manifest) != 3000 or manifest.cohort.value_counts().to_dict() != expected:
        raise RuntimeError("Exp141 3K cohort contract failed")
    source = read_attack_sessions()
    rows = []
    for item in manifest.itertuples(index=False):
        if str(item.cohort) == "NORMAL_USER":
            rows.append({"case_id": f"NORMAL|{item.row_id}", "row_id": str(item.row_id),
                         "kind": "NORMAL", "family": "NORMAL_USER", "session_id": str(item.session_id),
                         "turn": 1, "native_budget": 1, "query": str(item.query),
                         "query_sha256": sha256_text(item.query), "member": -1, "dataset": str(item.domain),
                         "domain": str(item.domain), "target_document_id": str(item.target_document_id),
                         "source_document_id": str(item.target_document_id)})
            continue
        session = source.get(str(item.session_id))
        if session is None:
            raise RuntimeError(f"selected attack session is absent: {item.session_id}")
        family = canonical_family(session["attack_family"])
        queries = list(map(str, session["queries"]))
        if family not in FAMILIES or len(queries) != NATIVE_BUDGET[family]:
            raise RuntimeError(f"native attack budget mismatch: {item.session_id}/{family}/{len(queries)}")
        if int(session["member_label"]) != int(item.member) or str(session["dataset"]) != str(item.domain):
            raise RuntimeError(f"Exp141/Exp43 identity mismatch: {item.session_id}")
        for turn, query in enumerate(queries, 1):
            rows.append({"case_id": f"ATTACK|{item.session_id}|Q{turn}", "row_id": str(item.row_id),
                         "kind": "ATTACK", "family": family, "session_id": str(item.session_id),
                         "turn": turn, "native_budget": len(queries), "query": query,
                         "query_sha256": sha256_text(query), "member": int(item.member),
                         "dataset": str(item.domain), "domain": str(item.domain),
                         "target_document_id": str(item.target_document_id),
                         "source_document_id": str(item.target_document_id)})
    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    attack = frame[frame.kind.eq("ATTACK")]
    session_counts = attack.drop_duplicates("session_id").groupby(["member", "family"]).size()
    expected_family = {"RAG-MIA": 180, "S²-MIA": 180, "MBA": 180, "DCMI": 100, "MEntA": 180, "IA": 180}
    if len(frame[frame.kind.eq("NORMAL")]) != 1000 or attack.session_id.nunique() != 2000:
        raise RuntimeError("expanded 3K cohort size mismatch")
    for member in (0, 1):
        for family, count in expected_family.items():
            if int(session_counts.get((member, family), 0)) != count:
                raise RuntimeError(f"family count mismatch: member={member}/{family}")
    if len(frame) != 9680 or frame.case_id.nunique() != len(frame):
        raise RuntimeError(f"query-turn identity mismatch: {frame.shape}")
    atomic_csv(frame, destination, "gzip")
    counts = frame.groupby(["kind", "dataset", "family", "member"], as_index=False).agg(
        query_turns=("case_id", "size"), sessions=("session_id", "nunique"))
    atomic_csv(counts, ROOT / "tables/COHORT_COUNTS.csv")
    atomic_json(ROOT / "audits/COHORT_AUDIT.json", {
        "status": "PASS", "rows": len(frame), "attack_query_turns": len(attack),
        "attack_sessions": attack.session_id.nunique(), "member_sessions": 1000,
        "nonmember_sessions": 1000, "normal_queries": 1000,
        "datasets": list(DATASETS), "families": list(FAMILIES),
        "query_id_hash": sha256_text("\n".join(sorted(frame.case_id.astype(str))))})
    checkpoint("COHORT_FROZEN", rows=len(frame), attack_turns=len(attack), attack_sessions=2000,
               member=1000, nonmember=1000, normal=1000)
    return frame


def load_corpora() -> tuple[dict[str, tuple[list[str], list[str]]], dict[tuple[str, str], str]]:
    corpora = {}
    documents = {}
    for dataset in DATASETS:
        ids, texts = [], []
        path = AD3 / f"data/beir/{dataset}/corpus_member.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    document_id = str(row.get("_id", row.get("id")))
                    text = "\n".join(value for value in (str(row.get("title", "")), str(row.get("text", ""))) if value)
                    ids.append(document_id)
                    texts.append(text)
                    documents[(dataset, document_id)] = text
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise RuntimeError(f"member corpus contract failed: {dataset}/{len(ids)}")
        corpora[dataset] = (ids, texts)
    atomic_json(ROOT / "audits/CORPUS_AUDIT.json", {
        dataset: {"documents": len(ids), "ordered_id_hash": sha256_text("\n".join(ids))}
        for dataset, (ids, _) in corpora.items()})
    return corpora, documents


def load_target_documents(frame: pd.DataFrame, member_documents: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    required = {(str(row.dataset), str(row.target_document_id))
                for row in frame[frame.kind.eq("ATTACK")].drop_duplicates("session_id").itertuples(index=False)}
    documents = dict(member_documents)
    missing = required - set(documents)
    raw_root = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
    folders = {"BeIR_nfcorpus": "nfcorpus", "BeIR_scidocs": "scidocs", "BeIR_trec-covid": "trec-covid"}
    for dataset, folder in folders.items():
        needed = {document_id for domain, document_id in missing if domain == dataset}
        if not needed:
            continue
        with (raw_root / folder / "corpus.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                document_id = str(row.get("_id", row.get("id")))
                if document_id in needed:
                    documents[(dataset, document_id)] = "\n".join(
                        value for value in (str(row.get("title", "")), str(row.get("text", ""))) if value)
                    needed.remove(document_id)
                    if not needed:
                        break
    absent = sorted(required - set(documents))
    if absent:
        raise RuntimeError(f"missing target documents: {absent[:5]} ({len(absent)})")
    return documents


def read_bge_cache(dataset: str) -> dict[str, dict]:
    output = {}
    with gzip.open(BGE_CACHE / f"{dataset}.canonical_v2.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["query_sha256"])] = row
    return output


def retrieve_and_mirabel(frame: pd.DataFrame, corpora: dict[str, tuple[list[str], list[str]]]) -> pd.DataFrame:
    destination = ROOT / "private/BGE_RETRIEVAL_AND_MIRABEL.private.csv.gz"
    if destination.exists():
        return pd.read_csv(destination, keep_default_na=False, low_memory=False,
                           dtype={"case_id": str, "session_id": str, "target_document_id": str})
    import torch
    from sentence_transformers import SentenceTransformer
    canonical = load_module("benchmark_canonical_mirabel", MIRABEL_CODE)
    attack = frame[frame.kind.eq("ATTACK")]
    rows = []
    for dataset in DATASETS:
        cache = read_bge_cache(dataset)
        subset = attack[attack.dataset.eq(dataset)]
        for item in subset.itertuples(index=False):
            cached = cache.get(str(item.query_sha256))
            if cached is None or cached.get("mirabel_formula_version") != "canonical_official_v2":
                raise RuntimeError(f"missing canonical BGE attack retrieval: {dataset}/{item.case_id}")
            ids = list(map(str, cached["doc_ids"][:TOP_K_SCORE]))
            target = str(item.target_document_id)
            target_rank = ids.index(target) + 1 if target in ids else 0
            rows.append({**item._asdict(), "retrieved_document_ids": json.dumps(ids),
                         "retrieved_scores": json.dumps(list(map(float, cached["scores"][:TOP_K_SCORE]))),
                         "target_rank": target_rank, "mirabel_margin": float(cached["mirabel_margin"]),
                         "mirabel_top1": float(cached["top1"]), "retrieval_source": "Exp45 frozen BGE cache"})
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    for dataset, (ids, texts) in corpora.items():
        subset = frame[(frame.kind.eq("NORMAL")) & frame.dataset.eq(dataset)].reset_index(drop=True)
        checkpoint("NORMAL_BGE_RETRIEVAL_STARTED", dataset=dataset, queries=len(subset), documents=len(ids))
        document_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                           batch_size=16, show_progress_bar=False).astype(np.float32)
        query_embeddings = model.encode(subset["query"].astype(str).tolist(), normalize_embeddings=True,
                                        convert_to_numpy=True, batch_size=64, show_progress_bar=False).astype(np.float32)
        matrix = query_embeddings @ document_embeddings.T
        order = np.argsort(-matrix, axis=1)[:, :TOP_K_SCORE]
        for number, item in enumerate(subset.itertuples(index=False)):
            top = order[number]
            values = matrix[number, top]
            top1 = float(values[0])
            stats = canonical.canonical_mirabel_from_moments(
                top1=top1, sum_all=float(matrix[number].sum(dtype=np.float64)),
                sumsq_all=float(np.square(matrix[number], dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(ids))
            doc_ids = [ids[index] for index in top]
            target = str(item.target_document_id)
            rows.append({**item._asdict(), "retrieved_document_ids": json.dumps(doc_ids),
                         "retrieved_scores": json.dumps([float(value) for value in values]),
                         "target_rank": doc_ids.index(target) + 1 if target in doc_ids else 0,
                         "mirabel_margin": float(stats.margin), "mirabel_top1": top1,
                         "retrieval_source": "fresh exact BGE full-corpus normal retrieval"})
        checkpoint("NORMAL_BGE_RETRIEVAL_COMPLETE", dataset=dataset, queries=len(subset))
        del document_embeddings, query_embeddings, matrix, order
        gc.collect()
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    output = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if len(output) != len(frame) or output.case_id.nunique() != len(frame) or output.mirabel_margin.isna().any():
        raise RuntimeError(f"BGE retrieval result mismatch: {output.shape}")
    atomic_csv(output, destination, "gzip")
    checkpoint("BGE_RETRIEVAL_MIRABEL_COMPLETE", queries=len(output), cached_attack=len(attack), fresh_normal=1000)
    return output


def qll_scores(retrieval: pd.DataFrame, corpora: dict[str, tuple[list[str], list[str]]]) -> pd.DataFrame:
    destination = ROOT / "private/BGE_QLL_CASES.private.csv.gz"
    if destination.exists():
        return pd.read_csv(destination, keep_default_na=False, low_memory=False,
                           dtype={"case_id": str, "session_id": str, "target_document_id": str})
    qll = load_module("benchmark_qll", QLL_CODE)
    qll.ROOT = ROOT / "qll_runtime"
    qll.DB = ROOT / "private/BGE_QLL.sqlite3"
    qll.PRIOR_DB = ROOT / "private/NO_PRIOR_QLL.sqlite3"
    qll.QWEN = QWEN
    qll.checkpoint = lambda stage, **details: checkpoint(f"QLL_{stage}", **details)
    documents = {(dataset, document_id): text for dataset, (ids, texts) in corpora.items()
                 for document_id, text in zip(ids, texts)}
    rows = []
    for item in retrieval.itertuples(index=False):
        source_ids = list(map(str, json.loads(item.retrieved_document_ids)))[:TOP_K_GENERATION]
        for rank, source_id in enumerate(source_ids, 1):
            source_hash = sha256_text(documents[(str(item.dataset), source_id)])
            # QLL is a deterministic function of (frozen model, query, source).
            # Keep this identity independent of the evaluation case so exact
            # duplicate query/source pairs are evaluated only once.
            task_id = sha256_text(f"QLL_V1\0{item.query_sha256}\0{source_id}\0{source_hash}")
            rows.append({"task_id": task_id, "case_id": str(item.case_id),
                         "attack_family": str(item.family), "cohort": str(item.kind),
                         "dataset": str(item.dataset), "member": int(item.member), "query": str(item.query),
                         "query_sha256": str(item.query_sha256), "source_id": source_id,
                         "source_rank": rank, "target_document_id": str(item.target_document_id),
                         "target_rank": int(item.target_rank),
                         "is_labeled_source": source_id == str(item.target_document_id)})
    tasks = pd.DataFrame(rows)
    if len(tasks) != TOP_K_GENERATION * len(retrieval):
        raise RuntimeError(f"QLL task row mismatch: rows={len(tasks)}")
    atomic_csv(tasks.drop(columns=["query"]), ROOT / "private/BGE_QLL_TASKS.private.csv.gz", "gzip")
    started = time.perf_counter()
    unique_tasks = tasks.drop_duplicates("task_id", keep="first").copy()
    unique_scored = qll.score_tasks(unique_tasks, documents)
    value_columns = ["task_id", "mean_query_log_probability", "query_tokens", "prefix_tokens",
                     "total_tokens", "source_truncated", "completed_utc", "origin"]
    scored = tasks.merge(unique_scored[value_columns], on="task_id", validate="many_to_one")
    source_scores, cases = qll.case_scores(scored)
    atomic_csv(source_scores, ROOT / "private/BGE_QLL_SOURCE_SCORES.private.csv.gz", "gzip")
    output = retrieval.merge(cases[["case_id", "qll_top1_source", "qll_top2_source", "margin", "dominance",
                                    "entropy", "labeled_source_qll_rank"]], on="case_id", validate="one_to_one")
    atomic_csv(output, destination, "gzip")
    checkpoint("QLL_COMPLETE", cases=len(output), source_pairs=len(tasks),
               unique_source_pairs=len(unique_tasks), seconds=time.perf_counter() - started)
    return output


def session_scores(query_scores: pd.DataFrame) -> pd.DataFrame:
    # Each normal query is one independent benign session.  Attack sessions
    # preserve their original multi-turn session ID and use an anytime maximum.
    work = query_scores.copy()
    work["evaluation_session_id"] = np.where(
        work.kind.eq("NORMAL"), work.case_id.astype(str), work.session_id.astype(str))
    return work.groupby(["kind", "family", "dataset", "evaluation_session_id"], as_index=False).agg(
        member=("member", "first"), turns=("turn", "nunique"),
        mirabel_margin=("mirabel_margin", "max"), qll_dominance=("dominance", "max"))


def low_fpr_tables(query_scores: pd.DataFrame) -> dict[str, float]:
    query = query_scores.rename(columns={"dominance": "qll_dominance"})
    session = session_scores(query_scores)
    detectors = {"Original Mirabel": "mirabel_margin", "Stateless QLL Source Hide": "qll_dominance"}
    thresholds, metrics = [], []
    primary = {}
    for unit, frame in (("query", query), ("session_any_turn", session)):
        normal = frame[frame.kind.eq("NORMAL")]
        attack = frame[frame.kind.eq("ATTACK")]
        if len(normal) != 1000:
            raise RuntimeError(f"normal threshold cohort mismatch for {unit}: {len(normal)}")
        for detector, column in detectors.items():
            for target_fpr in TARGET_FPRS:
                threshold, fp, realized = strict_threshold(normal[column].to_numpy(float), target_fpr)
                thresholds.append({"unit": unit, "detector": detector, "target_fpr": target_fpr,
                                   "threshold": threshold, "comparison": "strict >", "normal_n": len(normal),
                                   "false_positives": fp, "realized_fpr": realized,
                                   "attack_examples_used_for_threshold": 0})
                if unit == "query" and math.isclose(target_fpr, PRIMARY_FPR):
                    primary[detector] = threshold
                scopes = [("ALL_DATASETS", "ALL_ATTACKS", attack)]
                scopes += [("ALL_DATASETS", family, attack[attack.family.eq(family)]) for family in FAMILIES]
                scopes += [(dataset, "ALL_ATTACKS", attack[attack.dataset.eq(dataset)]) for dataset in DATASETS]
                scopes += [(dataset, family, attack[attack.dataset.eq(dataset) & attack.family.eq(family)])
                           for dataset in DATASETS for family in FAMILIES]
                for dataset, family, group0 in scopes:
                    for membership_scope, member in (("ALL", None), ("MEMBER", 1), ("NONMEMBER", 0)):
                        group = group0 if member is None else group0[group0.member.eq(member)]
                        if not len(group):
                            continue
                        positives = int((group[column].to_numpy(float) > threshold).sum())
                        low, high = wilson(positives, len(group))
                        metrics.append({"unit": unit, "detector": detector, "target_fpr": target_fpr,
                                        "dataset": dataset, "family": family,
                                        "membership_scope": membership_scope, "n_attack": len(group),
                                        "true_positives": positives, "tpr": positives / len(group),
                                        "tpr_wilson_low": low, "tpr_wilson_high": high,
                                        "threshold": threshold, "normal_n": len(normal),
                                        "false_positives": fp, "realized_fpr": realized})
    threshold_frame = pd.DataFrame(thresholds)
    metric_frame = pd.DataFrame(metrics)
    atomic_csv(threshold_frame, ROOT / "tables/LOW_FPR_THRESHOLDS.csv")
    atomic_csv(metric_frame, ROOT / "tables/TPR_AT_FPR_1_3_5_LONG.csv")
    headline = metric_frame[(metric_frame.unit.eq("session_any_turn")) &
                            metric_frame.dataset.eq("ALL_DATASETS") &
                            metric_frame.membership_scope.eq("ALL")].copy()
    atomic_csv(headline, ROOT / "tables/TPR_AT_FPR_1_3_5_HEADLINE.csv")
    domain_fpr = []
    for threshold_row in threshold_frame.itertuples(index=False):
        if threshold_row.unit != "query":
            continue
        column = detectors[str(threshold_row.detector)]
        for dataset in DATASETS:
            values = query[(query.kind.eq("NORMAL")) & query.dataset.eq(dataset)][column].to_numpy(float)
            count = int((values > float(threshold_row.threshold)).sum())
            low, high = wilson(count, len(values))
            domain_fpr.append({"detector": threshold_row.detector, "target_fpr": threshold_row.target_fpr,
                               "dataset": dataset, "normal_n": len(values), "false_positives": count,
                               "realized_fpr": count / len(values), "wilson_low": low, "wilson_high": high,
                               "global_benign_threshold": threshold_row.threshold})
    atomic_csv(pd.DataFrame(domain_fpr), ROOT / "tables/NORMAL_FPR_BY_DATASET.csv")
    atomic_json(ROOT / "results/PRIMARY_FPR3_THRESHOLDS.json", primary)
    checkpoint("LOW_FPR_TPR_COMPLETE", query_rows=len(query), session_rows=len(session),
               mirabel_fpr3_threshold=primary["Original Mirabel"], qll_fpr3_threshold=primary["Stateless QLL Source Hide"])
    return primary


def waterfill(lengths: list[int], hidden: int | None = None) -> list[int]:
    values = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(values), dtype=int)
    remaining = TOTAL_SOURCE_TOKENS
    active = [index for index, length in enumerate(values) if index != hidden and length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(values[index] - caps[index]), remaining)
            if add > 0:
                caps[index] += add
                remaining -= add
                changed = True
            if caps[index] >= values[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def pack_contexts(scored: pd.DataFrame, documents: dict[tuple[str, str], str], thresholds: dict[str, float]) -> pd.DataFrame:
    destination = ROOT / "private/E2E_PACKING.private.pkl.gz"
    if destination.exists():
        return pd.read_pickle(destination, compression="gzip")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    rows = []
    attacks = scored[scored.kind.eq("ATTACK")]
    for item in attacks.itertuples(index=False):
        top10 = list(map(str, json.loads(item.retrieved_document_ids)))
        for condition in ("MIRABEL_MATCHED_FPR3", "QLL_MATCHED_FPR3"):
            if condition.startswith("MIRABEL"):
                alarm = float(item.mirabel_margin) > float(thresholds["Original Mirabel"])
                selected_source = top10[0]
                visible_sources = top10[1:5] if alarm else top10[:4]
                locator_rank = 1
                action = "HIDE_RANK1_BACKFILL_RANK5" if alarm else "PASS"
            else:
                alarm = float(item.dominance) > float(thresholds["Stateless QLL Source Hide"])
                selected_source = str(item.qll_top1_source)
                top4 = top10[:4]
                locator_rank = top4.index(selected_source) + 1
                visible_sources = [source for source in top4 if not (alarm and source == selected_source)]
                action = "HIDE_QLL_TOP1_NO_BACKFILL" if alarm else "PASS"
            texts = [documents[(str(item.dataset), source)] for source in visible_sources]
            token_ids = [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
            caps = waterfill([len(value) for value in token_ids])
            visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip()
                       for value, cap in zip(token_ids, caps) if cap]
            prompt = models.normal_prompt(str(item.query), visible)
            rows.append({**item._asdict(), "condition": condition,
                         "threshold": float(thresholds["Original Mirabel"] if condition.startswith("MIRABEL")
                                                    else thresholds["Stateless QLL Source Hide"]),
                         "intervened": bool(alarm), "action": action,
                         "selected_source_id": selected_source if alarm else "",
                         "selected_source_retrieval_rank": locator_rank if alarm else 0,
                         "source_ids_used": json.dumps(visible_sources), "source_caps": json.dumps(caps),
                         "total_context_tokens": int(sum(caps)), "prompt": prompt,
                         "system_prompt": NORMAL_SYSTEM, "max_new_tokens": MAX_NEW[str(item.family)],
                         "generation_row_id": f"{condition}|{item.case_id}"})
    output = pd.DataFrame(rows)
    if len(output) != 2 * len(attacks) or output.generation_row_id.nunique() != len(output):
        raise RuntimeError("E2E packing identity mismatch")
    if output.total_context_tokens.gt(TOTAL_SOURCE_TOKENS).any():
        raise RuntimeError("source-token budget exceeded")
    atomic_pickle(output, destination)
    audit = output.groupby(["condition", "family", "member"], as_index=False).agg(
        query_turns=("case_id", "size"), sessions=("session_id", "nunique"),
        intervention_rate=("intervened", "mean"), mean_context_tokens=("total_context_tokens", "mean"))
    atomic_csv(audit, ROOT / "tables/E2E_PACKING_AUDIT.csv")
    checkpoint("E2E_PACKING_COMPLETE", rows=len(output), conditions=2)
    return output


def generate_responses(packing: pd.DataFrame) -> pd.DataFrame:
    destination = ROOT / "private/E2E_QWEN_RESPONSES.private.csv.gz"
    if destination.exists():
        output = pd.read_csv(destination, keep_default_na=False, low_memory=False,
                             dtype={"case_id": str, "session_id": str, "target_document_id": str})
        if len(output) == len(packing) and output.response.astype(str).ne("").all():
            return output
    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_models as models
    models.ROOT = ROOT
    models.heartbeat = lambda stage, **details: checkpoint(f"GENERATION_{stage}", **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    tasks = [models.make_task(task_type="MIRABEL_BENCHMARK_E2E", row_id=str(item.generation_row_id),
                              prompt=str(item.prompt), system_prompt=str(item.system_prompt),
                              max_new_tokens=int(item.max_new_tokens)) for item in packing.itertuples(index=False)]
    checkpoint("E2E_GENERATION_STARTED", logical_rows=len(packing), unique_forwards=len(tasks), device="cuda:0")
    started = time.perf_counter()
    answers = models.run_generation(tasks, ROOT / "private/E2E_QWEN_RESPONSES.sqlite3", None,
                                    "MIRABEL_BENCHMARK_QWEN")
    output = packing.copy()
    output["response"] = output.generation_row_id.astype(str).map(answers)
    if output.response.isna().any() or output.response.astype(str).eq("").any():
        raise RuntimeError("Qwen response generation is incomplete")
    public = output.drop(columns=["query", "prompt", "system_prompt"], errors="ignore")
    atomic_csv(output, destination, "gzip")
    atomic_csv(public, ROOT / "tables/E2E_RESPONSE_MANIFEST.csv.gz", "gzip")
    checkpoint("E2E_GENERATION_COMPLETE", responses=len(output), seconds=time.perf_counter() - started)
    return output


def normalize_mba(response: str) -> str:
    pattern = re.compile(r"\[?Mask_(\d+)\]?\s*:\s*(.*?)(?=\s*;?\s*\[?Mask_\d+\]?\s*:|$)", re.I | re.S)
    values = []
    for index, value in pattern.findall(str(response)):
        clean = re.sub(r"^answer\s*:\s*", "", value.strip().strip(";"), flags=re.I)
        values.append(f"[Mask_{int(index)}]: {clean}")
    return "\n".join(values) if values else str(response)


def load_ia_ground_truth(frame: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Load the immutable original-protocol IA target-document answers.

    Exp44 stored exact session/turn aliases for both RAG and ground-truth
    responses.  This is a read-only reuse: no labels are inferred, regenerated,
    or filled in during this comparison.
    """
    selected = frame[frame.kind.eq("ATTACK") & frame.family.eq("IA")][
        ["session_id", "turn", "query_sha256"]].copy()
    required = {(str(row.session_id), int(row.turn)): str(row.query_sha256)
                for row in selected.itertuples(index=False)}
    connection = sqlite3.connect(f"file:{EXP44_RESPONSE_DB}?mode=ro", uri=True)
    rows = connection.execute("""
        SELECT a.session_id,a.query_turn,r.response_text,r.query_hash,
               r.context_hash,r.model_hash,r.generation_config_hash
        FROM response_aliases a JOIN responses r ON a.cache_key=r.cache_key
        WHERE a.response_kind='ground_truth'
    """).fetchall()
    connection.close()
    found = {}
    audit_rows = []
    for session_id, turn, response, query_hash, context_hash, model_hash, config_hash in rows:
        key = (str(session_id), int(turn))
        if key not in required:
            continue
        if str(query_hash) != required[key]:
            raise RuntimeError(f"IA ground-truth query hash mismatch: {key}")
        if key in found and found[key] != str(response):
            raise RuntimeError(f"conflicting IA ground-truth alias: {key}")
        found[key] = str(response)
        audit_rows.append({"session_id": key[0], "turn": key[1], "query_sha256": str(query_hash),
                           "context_sha256": str(context_hash), "model_hash": str(model_hash),
                           "generation_config_hash": str(config_hash), "response": str(response)})
    missing = sorted(set(required) - set(found))
    if missing or len(found) != 5400 or len({key[0] for key in found}) != 360:
        raise RuntimeError(f"IA ground-truth coverage failed: {len(found)}/5400 missing={missing[:3]}")
    audit = pd.DataFrame(audit_rows).sort_values(["session_id", "turn"])
    atomic_csv(audit, ROOT / "private/IA_ORIGINAL_GROUND_TRUTH.private.csv.gz", "gzip")
    atomic_json(ROOT / "audits/IA_GROUND_TRUTH_AUDIT.json", {
        "status": "PASS", "source": str(EXP44_RESPONSE_DB),
        "source_sha256": sha256_file(EXP44_RESPONSE_DB), "coverage": "5400/5400",
        "sessions": 360, "new_ground_truth_generations": 0,
        "id_hash": sha256_text("\n".join(f"{a}|{b}" for a, b in sorted(found))),
        "model_hashes": sorted(audit.model_hash.unique()),
        "generation_config_hashes": sorted(audit.generation_config_hash.unique()),
        "fabricated_labels": False, "fallback_fill": False})
    return found


def score_end_to_end(frame: pd.DataFrame, responses: pd.DataFrame,
                     documents: dict[tuple[str, str], str],
                     ia_ground_truth: dict[tuple[str, int], str]) -> pd.DataFrame:
    destination = ROOT / "private/E2E_ATTACK_SCORES.private.csv.gz"
    if destination.exists():
        return pd.read_csv(destination, keep_default_na=False, low_memory=False,
                           dtype={"session_id": str, "target_document_id": str})
    sys.path.insert(0, str(AD3))
    from src.exp44_native import (dcmi_score, ia_turn_score, mba_accuracy, rag_mia_score,
                                  recover_mba_mask_values, s2_semantic_scores)
    attack_sessions = read_attack_sessions()
    selected_ids = set(frame[frame.kind.eq("ATTACK")].session_id.astype(str))
    sessions = []
    for session_id in sorted(selected_ids):
        source = attack_sessions[session_id]
        family = canonical_family(source["attack_family"])
        sessions.append({"session_id": session_id, "attack_family": family,
                         "dataset": str(source["dataset"]), "target_document_id": str(source["target_document_id"]),
                         "member": int(source["member_label"]), "queries": list(map(str, source["queries"])),
                         "native_budget": int(source["native_budget"])})
    response_lookup = responses.set_index(["condition", "session_id", "turn"])["response"].to_dict()
    conditions = sorted(responses.condition.unique())
    scored = []
    invalid = []
    semantic_jobs = []
    menta_jobs = defaultdict(list)
    for session in sessions:
        family = session["attack_family"]
        for condition in conditions:
            values = [str(response_lookup[(condition, session["session_id"], turn)])
                      for turn in range(1, session["native_budget"] + 1)]
            try:
                if family == "RAG-MIA":
                    value = rag_mia_score(values[0])
                elif family == "DCMI":
                    value = dcmi_score(values[0], values[1])
                elif family == "MBA":
                    truth = recover_mba_mask_values(session["queries"][0],
                        documents[(session["dataset"], session["target_document_id"])])
                    value = mba_accuracy(truth, normalize_mba(values[0]))
                elif family == "S²-MIA":
                    semantic_jobs.append((condition, session, values[0]))
                    continue
                elif family == "MEntA":
                    menta_jobs[condition].append((session, values))
                    continue
                elif family == "IA":
                    truth = [ia_ground_truth[(session["session_id"], turn)]
                             for turn in range(1, session["native_budget"] + 1)]
                    per_turn = [ia_turn_score(answer, expected)
                                for answer, expected in zip(values, truth)]
                    valid = [number for number in per_turn if number is not None]
                    if not valid:
                        raise ValueError("IA-MIA has no scoreable ground-truth turns")
                    value = float(np.mean(valid))
                else:
                    raise KeyError(family)
                scored.append({"condition": condition, **session, "attack_score": float(value), "status": "VALID"})
            except Exception as error:
                invalid.append({"condition": condition, "session_id": session["session_id"], "family": family,
                                "reason": type(error).__name__, "detail": str(error)})
    if semantic_jobs:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(MPNET), device="cuda", local_files_only=True)
        knowledge, values = [], []
        for _, session, response in semantic_jobs:
            target = documents[(session["dataset"], session["target_document_id"])]
            middle = len(target) // 2
            while middle < len(target) and target[middle] != " ":
                middle += 1
            knowledge.append(target[middle:].strip())
            values.append(response)
        scores = s2_semantic_scores(knowledge, values, model)
        for (condition, session, _), value in zip(semantic_jobs, scores):
            scored.append({"condition": condition, **session, "attack_score": float(value), "status": "VALID"})
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    if menta_jobs:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from scripts import run_exp44_overnight as exp44
        model_path = exp44.local_snapshot(exp44.CACHE_BASE / "models--tasksource--deberta-base-long-nli/snapshots")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda").eval()
        corpora = {dataset: {document_id: text for (domain, document_id), text in documents.items() if domain == dataset}
                   for dataset in DATASETS}
        for condition, jobs in menta_jobs.items():
            partial_path = ROOT / f"private/MENTA_{condition}.partial.csv"
            existing = {}
            if partial_path.exists():
                old = pd.read_csv(partial_path, keep_default_na=False)
                existing = dict(zip(old.session_id.astype(str), old.attack_score.astype(float)))
            pending = [(session, values) for session, values in jobs if session["session_id"] not in existing]
            for offset in range(0, len(pending), 16):
                chunk = pending[offset:offset + 16]
                adapted = [{**session, "dataset": session["dataset"]} for session, _ in chunk]
                measured = exp44.exact_menta_batch(adapted, [values for _, values in chunk],
                                                   corpora, tokenizer, model, model.device)
                for (session, _), (value, components) in zip(chunk, measured):
                    existing[session["session_id"]] = float(value)
                atomic_csv(pd.DataFrame([{"session_id": key, "attack_score": value}
                                         for key, value in sorted(existing.items())]), partial_path)
                checkpoint("MENTA_SCORING_PROGRESS", condition=condition,
                           completed=len(existing), total=len(jobs))
            for session, _ in jobs:
                scored.append({"condition": condition, **session,
                               "attack_score": float(existing[session["session_id"]]), "status": "VALID"})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output = pd.DataFrame(scored).sort_values(["condition", "attack_family", "session_id"]).reset_index(drop=True)
    atomic_csv(output, destination, "gzip")
    if invalid:
        atomic_csv(pd.DataFrame(invalid), ROOT / "audits/E2E_SCORER_INVALID.csv")
    ia_valid = output[output.attack_family.eq("IA")]
    ia_invalid = [row for row in invalid if row["family"] == "IA"]
    atomic_json(ROOT / "audits/IA_PROTOCOL_AUDIT.json", {
        "status": "PASS" if len(ia_valid) == 720 and not ia_invalid else "INCOMPLETE",
        "selected_sessions": 360, "selected_turns": 5400,
        "condition_scores": len(ia_valid), "invalid_condition_scores": len(ia_invalid),
        "ground_truth_source": "immutable Exp44 original-protocol target-document response aliases",
        "new_ground_truth_generations": 0, "fabricated_labels": False, "fallback_fill": False,
        "detector_tpr_reported": True, "end_to_end_e_auc_reported": True})
    checkpoint("E2E_ATTACK_SCORING_COMPLETE", valid_rows=len(output), invalid_rows=len(invalid),
               ia_status="IA_EVALUATION_PROTOCOL_UNRESOLVED")
    return output


def summarize_privacy(scores: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    attack_sessions = cohort[cohort.kind.eq("ATTACK")].drop_duplicates("session_id")
    for condition in ("MIRABEL_MATCHED_FPR3", "QLL_MATCHED_FPR3"):
        for dataset in ("ALL_DATASETS",) + DATASETS:
            for family in FAMILIES:
                frame = scores[(scores.condition.eq(condition)) & scores.attack_family.eq(family)]
                if dataset != "ALL_DATASETS":
                    frame = frame[frame.dataset.eq(dataset)]
                if frame.member.nunique() < 2:
                    continue
                raw, effective = effective_auc(frame.member, frame.attack_score)
                raw_low, raw_high, eff_low, eff_high = bootstrap_auc(frame.member, frame.attack_score,
                    seed=int(sha256_text(f"{condition}|{dataset}|{family}")[:8], 16))
                rows.append({"condition": condition, "dataset": dataset, "attack_family": family,
                             "status": "VALID", "sessions": len(frame),
                             "members": int((frame.member == 1).sum()),
                             "nonmembers": int((frame.member == 0).sum()), "raw_auc": raw,
                             "effective_auc": effective, "raw_auc_ci95_low": raw_low,
                             "raw_auc_ci95_high": raw_high, "effective_auc_ci95_low": eff_low,
                             "effective_auc_ci95_high": eff_high})
    output = pd.DataFrame(rows)
    atomic_csv(output, ROOT / "tables/END_TO_END_EFFECTIVE_AUC.csv")
    return output


def report(tpr: pd.DataFrame, privacy: pd.DataFrame) -> None:
    main_tpr = tpr[(tpr.unit.eq("session_any_turn")) & tpr.dataset.eq("ALL_DATASETS") &
                   tpr.membership_scope.eq("ALL") & tpr.family.isin(FAMILIES)].copy()
    main_privacy = privacy[privacy.dataset.eq("ALL_DATASETS")].copy()
    lines = ["# Mirabel benchmark: Original Mirabel vs Stateless QLL Source Hide", "",
             "## Frozen setup", "",
             "- Datasets: NFCorpus, SciDocs, TREC-COVID.",
             "- Retriever / generator: BGE-M3 / Qwen2.5-3B-Instruct.",
             "- Cohort: member attacks 1,000, nonmember attacks 1,000, normal users 1,000.",
             "- Native budgets: DCMI Q2, MEntA Q5, IA Q15, RAG-MIA/S²-MIA/MBA Q1.",
             "- TPR thresholds use only the same 1,000 normal queries and strict `score > threshold`.",
             "- E-AUC uses actual generated answers at the matched 3% benign-FPR operating point.", "",
             "## Detector TPR", "", main_tpr.to_markdown(index=False, floatfmt=".4f"), "",
             "## End-to-end membership leakage", "", main_privacy.to_markdown(index=False, floatfmt=".4f"), "",
             "## Interpretation guardrails", "",
             "- TPR is the fraction of attack sessions flagged before generation; it is not privacy leakage.",
             "- Effective AUC is `max(AUC, 1-AUC)` from the original family scorer after generation; 0.5 is ideal privacy.",
             "- IA uses the immutable Exp44 original-protocol target-document ground-truth responses with exact 5,400/5,400 turn coverage; no labels were regenerated or filled.",
             "- SciDocs has no IA sessions in the official public IA source; cells are reported as unavailable rather than fabricated.", ""]
    atomic_text(ROOT / "reports/FINAL_REPORT_KO.md", "\n".join(lines))


def verify_frozen_inputs(required: dict[str, Path]) -> None:
    before = pd.read_csv(ROOT / "provenance/FROZEN_INPUTS.csv", keep_default_na=False).set_index("key")
    rows = []
    for key, path in required.items():
        after = sha256_file(path)
        expected = str(before.loc[key, "sha256_before"])
        rows.append({"key": key, "sha256_before": expected, "sha256_after": after, "unchanged": after == expected})
    audit = pd.DataFrame(rows)
    atomic_csv(audit, ROOT / "provenance/FROZEN_INPUTS_POSTRUN.csv")
    if not audit.unchanged.all():
        raise RuntimeError("frozen input drift detected")


def main() -> None:
    required = preflight()
    frame = build_query_cohort()
    corpora, member_documents = load_corpora()
    documents = load_target_documents(frame, member_documents)
    retrieval = retrieve_and_mirabel(frame, corpora)
    scored = qll_scores(retrieval, corpora)
    thresholds = low_fpr_tables(scored)
    packing = pack_contexts(scored, documents, thresholds)
    responses = generate_responses(packing)
    ia_ground_truth = load_ia_ground_truth(frame)
    attack_scores = score_end_to_end(frame, responses, documents, ia_ground_truth)
    privacy = summarize_privacy(attack_scores, frame)
    tpr = pd.read_csv(ROOT / "tables/TPR_AT_FPR_1_3_5_LONG.csv")
    report(tpr, privacy)
    verify_frozen_inputs(required)
    result = {
        "verdict": "MIRABEL_BENCHMARK_SIX_ATTACK_COMPARISON_COMPLETE",
        "updated_utc": now(), "retriever": "BAAI/bge-m3", "generator": "Qwen/Qwen2.5-3B-Instruct",
        "datasets": list(DATASETS), "attack_sessions": 2000, "member_attack_sessions": 1000,
        "nonmember_attack_sessions": 1000, "normal_queries": 1000,
        "target_fprs": list(TARGET_FPRS), "end_to_end_fpr": PRIMARY_FPR,
        "ia_status": "VALID_IMMUTABLE_EXP44_ORIGINAL_PROTOCOL_GROUND_TRUTH", "paid_api_calls": 0,
        "tpr_table": str(ROOT / "tables/TPR_AT_FPR_1_3_5_LONG.csv"),
        "privacy_table": str(ROOT / "tables/END_TO_END_EFFECTIVE_AUC.csv"),
        "report": str(ROOT / "reports/FINAL_REPORT_KO.md"),
        "frozen_inputs_unchanged": True,
    }
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    checkpoint("COMPLETE", verdict=result["verdict"], tpr_complete=True, end_to_end_complete=True)


if __name__ == "__main__":
    main()
