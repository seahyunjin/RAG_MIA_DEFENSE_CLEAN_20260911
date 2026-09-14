#!/usr/bin/env python3
"""Exp197 RA-Cap64-BF preparation and frozen-Qwen generation.

RA-Cap64 is mechanism-identical to the frozen ALL_SOURCE_STABLE_PREFIX64
implementation and is therefore reused.  RA-Cap64-BF preserves each frozen
case's top-4 and appends the remaining frozen MPNet order, excluding those
top-4 IDs.  Every source contributes at most 64 Qwen tokens until T=2048.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp197_rank_agnostic_cap64_backfill_20260829"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP160E = PROJECT / "exp160e_global_cap64_external_attacks_exploratory_20260826"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP173 = PROJECT / "exp173_all_source_stable_prefix64_20260828"
EXP174 = PROJECT / "exp174_stable_prefix64_native_sessions_20260828"
EXP175 = PROJECT / "exp175_stable_prefix64_external_attacks_20260828"
EXP176 = PROJECT / "exp176_prefix64_mirabel_external_attacks_20260828"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP195 = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
EXP196 = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
AD3 = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EMBED_ROOT = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM2/outputs/exp1_mirabel_repro")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")

ATTACK_CASES = EXP193 / "private/EXP193_CASES.private.pkl.gz"
NORMAL_CASES = EXP195 / "private/EXP195_NORMAL_CASES.private.pkl.gz"
DEEP = ROOT / "private/EXP197_DEEP_RETRIEVAL.private.pkl.gz"
PACKING = ROOT / "private/EXP197_RA_CAP64_BF_PACKING.private.pkl.gz"
RESPONSES = ROOT / "private/EXP197_RA_CAP64_BF_RESPONSES.private.csv.gz"
GEN_DB = ROOT / "private/EXP197_QWEN_RESPONSES.sqlite3"
CAP = 64
TOTAL = 2048
# Some TREC-COVID corpus rows have an empty body (the frozen Prefix64
# convention intentionally ignores titles), so a query may need to scan well
# past rank 128 before accumulating 2,048 non-empty body tokens.  Depth 512 is
# a retrieval-availability bound, not a tuned defense parameter.
RETRIEVAL_DEPTH = 512


def now(): return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def sha256_text(value): return hashlib.sha256(str(value).encode()).hexdigest()


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def atomic_csv(frame, path, compression=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent); os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_pickle(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".pkl.gz", dir=path.parent); os.close(fd)
    try:
        frame.to_pickle(temporary, compression="gzip"); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checkpoint(stage, **details):
    payload = {"experiment": "Exp197", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "candidate": "RA_CAP64_BF", "cap": CAP, "total_context_budget": TOTAL,
               "attack_specific_tuning": 0, "rank_specific_tuning": 0,
               "learned_parameters": 0, "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    atomic_text(ROOT / "STATUS.md", "\n".join(["# Exp197 Status", "", f"- Stage: **{stage}**",
                f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{payload['pid']}`"] +
                [f"- {key}: `{value}`" for key, value in details.items()]) + "\n")


def preflight():
    required = {
        "attack_cases": ATTACK_CASES, "normal_cases": NORMAL_CASES,
        "prefix64_code": EXP175 / "code/run_exp175.py",
        # Exp175's retained read-only generation artifact is preserved in the
        # Exp176 lineage package after storage cleanup.
        "prefix64_external": EXP176 / "private/EXP175_GENERATIONS.private.csv.gz",
        "prefix64_native": EXP174 / "private/EXP174_NATIVE_RESPONSES.private.csv.gz",
        "prefix64_q1": EXP173 / "private/EXP173_RESPONSES.private.csv.gz",
        "prefix64_mirabel_external": EXP176 / "private/EXP176_GENERATIONS.private.csv.gz",
        "normal_retrieval": EXP166 / "private/TOPIOCQA_MPNET_RETRIEVAL.private.csv.gz",
        "normal_corpus": EXP166 / "private/TOPIOCQA_CORPUS.private.csv.gz",
        "normal_embeddings": EXP166 / "private/TOPIOCQA_MPNET_CORPUS_EMBEDDINGS.npy",
        "qwen": QWEN / "config.json", "mpnet": MPNET / "config.json",
        "exp196_final": EXP196 / "FINAL_RESULT.json",
    }
    for domain in ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"):
        required[f"{domain}_corpus"] = AD3 / f"data/beir/{domain}/corpus_member.jsonl"
        required[f"{domain}_embeddings"] = EMBED_ROOT / domain / "corpus_embeddings.npy"
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise RuntimeError(f"missing frozen inputs: {missing}")
    code = required["prefix64_code"].read_text()
    collision = all(value in code for value in (
        "def collapse_prefix64", "input_ids[:64]", "per_source_cap\": 64", "retrieved_top4_ids"))
    if not collision: raise RuntimeError("Prefix64 collision could not be established")
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path), "access": "READ_ONLY"}
                             for key, path in required.items()])
    atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    rows = [
        {"mechanism": "GlobalCap64", "restricted_ranks": "rank1", "per_source_cap": "rank1=64",
         "total_context_cap": 2048, "backfill": "redistribute within ranks2-4", "additional_retrieval": False,
         "rank_specific_rule": True, "collision_with_RA_Cap64": False},
        {"mechanism": "Prefix64 / ALL_SOURCE_STABLE_PREFIX64", "restricted_ranks": "rank1-4",
         "per_source_cap": "all=64", "total_context_cap": 256, "backfill": False,
         "additional_retrieval": False, "rank_specific_rule": False, "collision_with_RA_Cap64": True},
        {"mechanism": "Prefix64+Mirabel", "restricted_ranks": "rank1-4 then Mirabel hide",
         "per_source_cap": "all=64", "total_context_cap": "<=256", "backfill": False,
         "additional_retrieval": False, "rank_specific_rule": "Mirabel locator action",
         "collision_with_RA_Cap64": False},
        {"mechanism": "RA-Cap64-BF", "restricted_ranks": "all retrieved sources",
         "per_source_cap": "all<=64", "total_context_cap": 2048, "backfill": True,
         "additional_retrieval": True, "rank_specific_rule": False, "collision_with_RA_Cap64": False},
    ]
    atomic_csv(pd.DataFrame(rows), ROOT / "audits/COLLISION_AUDIT.csv")
    atomic_json(ROOT / "audits/COLLISION_VERDICT.json", {
        "verdict": "EXP197_DUPLICATES_PREFIX64_FOR_VARIANT_A",
        "ra_cap64_regeneration": False, "ra_cap64_frozen_reuse": True,
        "ra_cap64_bf_is_new": True,
        "evidence": str(required["prefix64_code"]), "evidence_sha256": sha256_file(required["prefix64_code"]),
    })
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(manifest),
               collision_verdict="EXP197_DUPLICATES_PREFIX64_FOR_VARIANT_A",
               ra_cap64_regeneration=False, ra_cap64_bf_generation=True)


def attack_corpus(domain):
    rows = [json.loads(line) for line in (AD3 / f"data/beir/{domain}/corpus_member.jsonl").open()
            if line.strip()]
    ids = np.asarray([str(row.get("_id", row.get("id"))) for row in rows])
    texts = {str(row.get("_id", row.get("id"))): str(row.get("text", "")) for row in rows}
    matrix = np.load(EMBED_ROOT / domain / "corpus_embeddings.npy").astype(np.float32, copy=False)[:len(ids)]
    if len(ids) != 1000 or matrix.shape != (1000, 768): raise RuntimeError(f"corpus mismatch {domain}")
    return ids, texts, matrix


def build_deep_retrieval(attacks, normal):
    if DEEP.exists():
        frame = pd.read_pickle(DEEP, compression="gzip")
        if "retrieval_key" in frame and frame.retrieval_key.nunique() == len(frame): return frame
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(str(MPNET), device=device, local_files_only=True)
    records = []
    unique = attacks[["domain", "query"]].drop_duplicates(["domain", "query"]).reset_index(drop=True)
    key_to_deep = {}
    for domain, cell in unique.groupby("domain", sort=True):
        ids, _, matrix = attack_corpus(str(domain))
        queries = encoder.encode(cell["query"].astype(str).tolist(), normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=128, show_progress_bar=False)
        for offset in range(0, len(cell), 256):
            block = queries[offset:offset+256].astype(np.float32) @ matrix.T
            indices = np.argpartition(-block, kth=RETRIEVAL_DEPTH-1, axis=1)[:, :RETRIEVAL_DEPTH]
            indices = np.take_along_axis(indices,
                np.argsort(-np.take_along_axis(block, indices, axis=1), axis=1), axis=1)
            for local, (_, row) in enumerate(cell.iloc[offset:offset+len(indices)].iterrows()):
                ranked = ids[indices[local]].astype(str).tolist()
                key = sha256_text(f"{domain}\0{row['query']}")
                key_to_deep[(str(domain), str(row["query"]))] = (key, ranked)
        checkpoint("DEEP_RETRIEVAL_DOMAIN", domain=domain, unique_queries=len(cell), device=device)
    for (domain, query), (key, ranked) in key_to_deep.items():
        records.append({"retrieval_key": key, "domain": domain, "query": query,
                        "ranked_source_ids": ranked, "scope": "ATTACK"})
    corpus = pd.read_csv(EXP166 / "private/TOPIOCQA_CORPUS.private.csv.gz", keep_default_na=False,
                         dtype={"document_id": str})
    ids = corpus.document_id.astype(str).to_numpy()
    matrix = np.load(EXP166 / "private/TOPIOCQA_MPNET_CORPUS_EMBEDDINGS.npy").astype(np.float32, copy=False)
    queries = encoder.encode(normal["query"].astype(str).tolist(), normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    stored = pd.read_csv(EXP166 / "private/TOPIOCQA_MPNET_RETRIEVAL.private.csv.gz",
                         keep_default_na=False, dtype={"row_id": str}).set_index("row_id")
    normal_direct = 0
    for offset in range(0, len(normal), 256):
        block = queries[offset:offset+256].astype(np.float32) @ matrix.T
        indices = np.argpartition(-block, kth=RETRIEVAL_DEPTH-1, axis=1)[:, :RETRIEVAL_DEPTH]
        indices = np.take_along_axis(indices,
            np.argsort(-np.take_along_axis(block, indices, axis=1), axis=1), axis=1)
        for local, row in enumerate(normal.iloc[offset:offset+len(indices)].itertuples(index=False)):
            ranked = ids[indices[local]].astype(str).tolist()
            frozen10 = list(map(str, json.loads(stored.loc[str(row.row_id), "retrieved_document_ids"])))
            normal_direct += int(ranked[:10] == frozen10)
            key = sha256_text(f"TopiOCQA\0{row.query}")
            records.append({"retrieval_key": key, "domain": "TopiOCQA", "query": str(row.query),
                            "ranked_source_ids": ranked, "scope": "NORMAL"})
    output = pd.DataFrame(records)
    if output.retrieval_key.duplicated().any(): raise RuntimeError("retrieval key collision")
    ranked_map = output.set_index("retrieval_key").ranked_source_ids.to_dict()
    direct_total = direct_match = external_total = external_match = 0
    for row in attacks.itertuples(index=False):
        ranked = ranked_map[sha256_text(f"{row.domain}\0{row.query}")]
        same = ranked[:4] == list(map(str, row.source_ids))
        direct_total += 1; direct_match += int(same)
        if str(row.panel) == "EXTERNAL": external_total += 1; external_match += int(same)
    atomic_pickle(output, DEEP)
    audit = {"attack_unique_queries": len(unique), "attack_direct_top4_matches": direct_match,
             "attack_direct_top4_total": direct_total, "attack_direct_top4_rate": direct_match/direct_total,
             "external_direct_top4_matches": external_match, "external_direct_top4_total": external_total,
             "external_direct_top4_rate": external_match/external_total,
             "native_top4_preserved_then_mpnet_remaining_order": True,
             "normal_stored_top10_matches": normal_direct, "normal_total": len(normal),
             "normal_top10_match_rate": normal_direct/len(normal), "retrieval_depth": RETRIEVAL_DEPTH,
             "policy_uses_labels": False, "policy_uses_attack_family": False}
    atomic_json(ROOT / "audits/DEEP_RETRIEVAL_AUDIT.json", audit)
    checkpoint("DEEP_RETRIEVAL_COMPLETE", rows=len(output), **audit)
    del encoder, queries
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output


def all_documents():
    output = {}
    for domain in ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"):
        _, texts, _ = attack_corpus(domain)
        output.update({(domain, source): text for source, text in texts.items()})
    corpus = pd.read_csv(EXP166 / "private/TOPIOCQA_CORPUS.private.csv.gz", keep_default_na=False,
                         dtype={"document_id": str})
    output.update({("TopiOCQA", str(row.document_id)): f"{row.title}\n{row.text}"
                   for row in corpus.itertuples(index=False)})
    return output


def build_tasks(attacks, normal, deep):
    if PACKING.exists():
        packing = pd.read_pickle(PACKING, compression="gzip")
        if len(packing) == len(attacks)+len(normal): return packing
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    documents = all_documents(); prefix_cache = {}
    for key, text in documents.items():
        token_ids = tokenizer(str(text), add_special_tokens=False).input_ids[:CAP]
        prefix_cache[key] = (tokenizer.decode(token_ids, skip_special_tokens=True).strip(), len(token_ids))
    cases = pd.concat([attacks.assign(split="ATTACK"), normal.assign(split="NORMAL")],
                      ignore_index=True, sort=False)
    dmap = deep.set_index("retrieval_key").ranked_source_ids.to_dict(); rows = []
    for index, row in enumerate(cases.itertuples(index=False), 1):
        ranked = list(map(str, dmap[sha256_text(f"{row.domain}\0{row.query}")]))
        frozen = list(map(str, row.source_ids))
        ids = frozen + [source for source in ranked if source not in set(frozen)]
        contexts=[]; used=[]; chosen=[]; total=0
        for source in ids:
            text, count = prefix_cache[(str(row.domain), source)]
            if count == 0: continue
            remaining = TOTAL-total
            if remaining <= 0: break
            if count > remaining:
                original = tokenizer(str(documents[(str(row.domain), source)]), add_special_tokens=False).input_ids[:remaining]
                text = tokenizer.decode(original, skip_special_tokens=True).strip(); count = len(original)
            contexts.append(text); used.append(count); chosen.append(source); total += count
            if total >= TOTAL: break
        if total < int(.95*TOTAL): raise RuntimeError(f"insufficient backfill for {row.case_id}: {total}")
        prompt = models.normal_prompt(str(row.query), contexts)
        maximum = int(row.max_new_tokens) if hasattr(row, "max_new_tokens") and not pd.isna(row.max_new_tokens) else 96
        task = models.make_task(task_type="EXP197_RA_CAP64_BF", row_id=str(row.case_id), prompt=prompt,
                                system_prompt=str(row.system_prompt), max_new_tokens=maximum)
        rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                     "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                     "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                     "target_rank": int(row.target_rank), "query": str(row.query),
                     "source_ids_used": chosen, "tokens_used": used, "source_count": len(chosen),
                     "rank5plus_used": len(chosen)>4, "total_context_tokens": total,
                     "context_budget_preservation": total/TOTAL, "prompt": prompt,
                     "prompt_hash": task["prompt_hash"], "task_key": task["task_key"],
                     "system_prompt": str(row.system_prompt), "max_new_tokens": maximum})
        if index % 4000 == 0: checkpoint("PACKING_PROGRESS", completed=index, total=len(cases))
    output = pd.DataFrame(rows); atomic_pickle(output, PACKING)
    summary = pd.DataFrame([{"condition":"RA_CAP64_BF", "queries":len(output),
        "mean_source_count":output.source_count.mean(), "median_source_count":output.source_count.median(),
        "rank5plus_use_rate":output.rank5plus_used.mean(), "mean_context_tokens":output.total_context_tokens.mean(),
        "median_context_tokens":output.total_context_tokens.median(),
        "mean_budget_preservation":output.context_budget_preservation.mean(),
        "minimum_context_tokens":output.total_context_tokens.min(), "maximum_source_tokens":max(map(max,output.tokens_used))}])
    atomic_csv(summary, ROOT / "tables/TABLE_197_01_BACKFILL_DIAGNOSTIC.csv")
    checkpoint("PACKING_COMPLETE", queries=len(output), mean_sources=float(output.source_count.mean()),
               rank5plus_use_rate=float(output.rank5plus_used.mean()),
               mean_context_tokens=float(output.total_context_tokens.mean()))
    return output


def generate(packing):
    if RESPONSES.exists():
        old = pd.read_csv(RESPONSES, keep_default_na=False, dtype={"case_id": str})
        if len(old)==len(packing) and old.case_id.nunique()==len(packing): return old
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    tasks=[]
    for row in packing.itertuples(index=False):
        tasks.append(models.make_task(task_type="EXP197_RA_CAP64_BF", row_id=str(row.case_id),
            prompt=str(row.prompt), system_prompt=str(row.system_prompt), max_new_tokens=int(row.max_new_tokens)))
    checkpoint("GENERATION_STARTED", tasks=len(tasks), generator="Qwen2.5-3B-Instruct",
               generations_per_query=1, extra_scoring_forward_passes=0, device="cuda:0")
    started=time.perf_counter(); answers=models.run_generation(tasks, GEN_DB, None, "GENERATION_PROGRESS")
    output=packing.drop(columns=["prompt"]).copy(); output["condition"]="RA_CAP64_BF"
    output["response"]=output.case_id.map(answers); output["response_sha256"]=output.response.map(sha256_text)
    if output.response.isna().any() or output.case_id.nunique()!=len(output): raise RuntimeError("generation incomplete")
    atomic_csv(output, RESPONSES, "gzip")
    elapsed=time.perf_counter()-started
    atomic_csv(pd.DataFrame([{"condition":"RA_CAP64_BF","queries":len(output),"wall_seconds":elapsed,
        "throughput_queries_per_second":len(output)/elapsed,"trainable_parameters":0,
        "generations_per_query":1,"extra_scoring_forward_passes":0,"session_state_scalars":0,
        "retrieval_depth":RETRIEVAL_DEPTH,"peak_vram":"NOT_ISOLATED"}]), ROOT/"tables/TABLE_197_02_EFFICIENCY.csv")
    checkpoint("GENERATION_COMPLETE", responses=len(output), wall_seconds=elapsed,
               throughput=len(output)/elapsed, database_sha256=sha256_file(GEN_DB))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");args=parser.parse_args()
    preflight(); attacks=pd.read_pickle(ATTACK_CASES,compression="gzip"); normal=pd.read_pickle(NORMAL_CASES,compression="gzip")
    deep=build_deep_retrieval(attacks,normal); packing=build_tasks(attacks,normal,deep)
    if args.prepare_only:
        checkpoint("PREPARE_COMPLETE", rows=len(packing)); return
    generate(packing); checkpoint("GENERATION_STAGE_COMPLETE", next_stage="evaluate_exp197.py")


if __name__ == "__main__": main()
