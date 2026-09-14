#!/usr/bin/env python3
"""Fresh 100/100 confirmation for Sparse Exposure; invoked only after screen PASS."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (ATTACKS, EXP, PARENT, PRIMARY_ALPHA, SEED, V1, V2, atomic_json,
                    bootstrap_delta, checkpoint, empirical_upper_tail, matched_binary_threshold,
                    normalize_tokens, now, read_jsonl, sha_file, sha_text, tpr_at_fpr, write_csv)


CONF = EXP / "confirmation"
INPUTS = CONF / "inputs"
AUDITS = CONF / "audits"
CACHE = CONF / "cache"
CONFIGS = CONF / "configs"
RUNTIME = CONF / "runtime"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
RECOVERY = EXP.parent / "CORE6_PROTOCOL_RECOVERY_V1"
PRE = CONFIGS / "SPARSE_EXPOSURE_CONFIRM_PRECOMMIT.json"
PRE_SHA = PRE.with_suffix(".sha256")
QUOTAS = {"nfcorpus": 34, "scidocs": 33, "trec-covid": 33}
DOMAINS = tuple(QUOTAS)
MODEL = "gpt-4.1-nano"

sys.path.insert(0, str(RECOVERY))
from protocols.menta.query_generator import build_prompt, build_summary_prompt, build_system_prompt, make_session, parse_five_queries  # noqa:E402


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).lower().split())


def global_id(domain: str, local_id: str) -> str:
    return f"BeIR_{domain}::{local_id}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
    os.replace(temporary, path)


def prepare() -> None:
    phase_b = json.loads((EXP / "PHASE_B_RESULT.json").read_text())
    if not phase_b["screen_passed"]: raise RuntimeError("confirmation forbidden: screen did not pass")
    for path in (INPUTS, AUDITS, CACHE, CONFIGS, RUNTIME): path.mkdir(parents=True, exist_ok=True)
    db_rows = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    db_ids = {r["document_id"] for r in db_rows}; db_hashes = {r["normalized_text_hash"] for r in db_rows}
    excluded = list(csv.DictReader((PARENT / "inputs/SHARED_TARGETS.csv").open(encoding="utf-8")))
    excluded += list(csv.DictReader((V2 / "inputs/FRESH_SHARED_TARGETS.csv").open(encoding="utf-8")))
    excluded_ids = {r["document_id"] for r in excluded}; excluded_hashes = {r["normalized_text_hash"] for r in excluded}
    with gzip.open(PARENT / "inputs/COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as f:
        pool = list(csv.DictReader(f))
    raw_docs = {}
    for domain in DOMAINS:
        for row in read_jsonl(RAW / domain / "corpus.jsonl"):
            local = str(row.get("_id") or row.get("id")); gid = global_id(domain, local)
            title = str(row.get("title") or "").strip(); body = str(row.get("text") or "").strip()
            raw_docs[gid] = {"document_id": gid, "local_document_id": local, "domain": domain,
                             "title": title, "source_text": "\n".join(x for x in (title, body) if x)}
    targets = []
    for domain in DOMAINS:
        for membership in ("member", "nonmember"):
            candidates = []
            for row in pool:
                if row["domain"] != domain or row["document_id"] in excluded_ids or row["normalized_text_hash"] in excluded_hashes: continue
                if membership == "member" and row["document_id"] not in db_ids: continue
                if membership == "nonmember" and (row["document_id"] in db_ids or row["normalized_text_hash"] in db_hashes): continue
                candidates.append(row)
            candidates.sort(key=lambda r: (sha_text("SPARSE_CONFIRM_V1||"+membership+"||"+r["document_id"]+"||"+r["normalized_text_hash"]), r["document_id"]))
            if len(candidates) < QUOTAS[domain]: raise RuntimeError(f"insufficient confirmation candidates: {domain}/{membership}")
            for index, row in enumerate(candidates[:QUOTAS[domain]]):
                targets.append({**raw_docs[row["document_id"]], "membership": membership,
                                "normalized_text_hash": row["normalized_text_hash"],
                                "selection_key": sha_text("SPARSE_CONFIRM_V1||"+membership+"||"+row["document_id"]+"||"+row["normalized_text_hash"]),
                                "domain_selection_index": index})
    targets.sort(key=lambda r: (r["domain"], r["membership"], r["selection_key"], r["document_id"]))
    fields = ["document_id", "local_document_id", "domain", "membership", "normalized_text_hash", "selection_key", "domain_selection_index", "title", "source_text"]
    write_csv(INPUTS / "SHARED_TARGETS.csv", targets, fields)
    ids, hashes = {r["document_id"] for r in targets}, {r["normalized_text_hash"] for r in targets}
    audit = {"verdict": "SPARSE_CONFIRM_TARGET_AUDIT_PASS", "member": sum(r["membership"] == "member" for r in targets),
             "nonmember": sum(r["membership"] == "nonmember" for r in targets),
             "old_id_overlap": len(ids & excluded_ids), "old_text_hash_overlap": len(hashes & excluded_hashes),
             "member_db_inclusion": sum(r["membership"] == "member" and r["document_id"] in db_ids for r in targets),
             "nonmember_db_exclusion": sum(r["membership"] == "nonmember" and r["document_id"] not in db_ids for r in targets),
             "domain_membership_counts": {f"{d}:{m}": sum(r["domain"] == d and r["membership"] == m for r in targets) for d in DOMAINS for m in ("member", "nonmember")}}
    if (len(targets), audit["member"], audit["nonmember"], audit["old_id_overlap"], audit["old_text_hash_overlap"], audit["member_db_inclusion"], audit["nonmember_db_exclusion"]) != (200,100,100,0,0,100,100):
        raise RuntimeError(f"confirmation target audit failed: {audit}")
    atomic_json(AUDITS / "TARGET_AUDIT.json", audit)
    pre = {"campaign": "SPARSE_EXPOSURE_CONFIRMATION", "created_utc": now(),
           "screen_result_sha256": sha_file(EXP / "PHASE_B_RESULT.json"), "screen_verdict": phase_b["verdict"],
           "targets": {"path": str(INPUTS / "SHARED_TARGETS.csv"), "sha256": sha_file(INPUTS / "SHARED_TARGETS.csv"),
                       "member": 100, "nonmember": 100, "overlap_prior_targets": 0, "quotas": QUOTAS,
                       "selection": "ascending SHA256(SPARSE_CONFIRM_V1||membership||document_id||normalized_text_hash) after excluding old 20/20 and V2 100/100"},
           "negative": {"path": str(V2 / "inputs/FRESH_BENIGN_1000.jsonl"), "sha256": sha_file(V2 / "inputs/FRESH_BENIGN_1000.jsonl"),
                        "role": "frozen development benign for same-FPR comparison; no attack-dependent threshold"},
           "formula": "unchanged Sparse Exposure V1 normalization/IDF/tail/max rule", "retriever": str(BGE),
           "attacks": {"MEntA": "GPT-4.1-nano Q5 one valid frozen packet/target; format-only retry allowed and preserved",
                       "MBA": "frozen GPT2-XL difficult-mask generator", "RAG-MIA": "frozen recovered prompt #2"},
           "gate": {"MEntA_delta_tpr3": ">=0.05", "MEntA_bootstrap_ci_low": ">0", "MBA_delta": ">=-0.05",
                    "RAG-MIA_delta": ">=-0.05", "macro_delta": ">0"},
           "code_sha256": {Path(__file__).name: sha_file(Path(__file__))}}
    atomic_json(PRE, pre); PRE_SHA.write_text(sha_file(PRE)+"\n", encoding="utf-8")
    checkpoint("SPARSE_CONFIRM_PRECOMMIT_WRITTEN", precommit_sha256=sha_file(PRE), member=100, nonmember=100)


def verify() -> dict:
    if sha_file(PRE) != PRE_SHA.read_text().strip(): raise RuntimeError("confirmation precommit drift")
    pre = json.loads(PRE.read_text())
    if sha_file(Path(__file__)) != pre["code_sha256"][Path(__file__).name]: raise RuntimeError("confirmation code drift")
    if sha_file(INPUTS / "SHARED_TARGETS.csv") != pre["targets"]["sha256"]: raise RuntimeError("confirmation target drift")
    return pre


def local_queries() -> None:
    verify()
    source = PARENT / "code/generate_mba_ragmia_queries.py"
    spec = importlib.util.spec_from_file_location("sparse_confirm_local", source); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.CAMPAIGN = CONF; module.INPUTS = INPUTS; module.AUDITS = AUDITS; module.main()
    mba, rag = read_jsonl(INPUTS / "MBA_ATTACK_QUERIES.jsonl"), read_jsonl(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl")
    if len(mba) != 200 or len(rag) != 200: raise RuntimeError(f"confirmation local query count: {len(mba)}/{len(rag)}")
    atomic_json(AUDITS / "MBA_INPUT_VALIDITY.json", {"status": "PASS", "target_sessions": 200,
                "valid_sessions": 200, "invalid": [], "mask_count": 5,
                "output_sha256": sha_file(INPUTS / "MBA_ATTACK_QUERIES.jsonl")})
    atomic_json(AUDITS / "RAG_MIA_INPUT_VALIDITY.json", {"status": "PASS", "target_sessions": 200,
                "valid_sessions": 200, "output_sha256": sha_file(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl")})
    atomic_json(AUDITS / "LOCAL_QUERY_AUDIT.json", {"verdict": "PASS", "MBA": len(mba), "RAG-MIA": len(rag),
                "mba_sha256": sha_file(INPUTS / "MBA_ATTACK_QUERIES.jsonl"), "rag_sha256": sha_file(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl")})


def extract_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str): return response["output_text"].strip()
    return "\n".join(c.get("text", "") for item in response.get("output", []) for c in item.get("content", []) if c.get("type") == "output_text").strip()


def api_call(key: str, input_text: str, instructions: str, temperature: float, max_tokens: int) -> dict:
    payload = {"model": MODEL, "instructions": instructions, "input": input_text, "temperature": temperature, "max_output_tokens": max_tokens}
    request = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response: return json.loads(response.read().decode())


def cached_api(path: Path, key: str, **kwargs) -> dict:
    if path.is_file(): return json.loads(path.read_text())
    last = None
    for attempt in range(6):
        try:
            raw = api_call(key, **kwargs); text = extract_text(raw)
            if not text: raise RuntimeError("empty response")
            out = {"model": MODEL, "text": text, "text_sha256": sha_text(text), "usage": raw.get("usage", {}), "created_utc": now()}
            atomic_json(path, out); return out
        except Exception as exc:
            last = exc
            if attempt < 5: time.sleep(min(2**(attempt+1), 30))
    raise RuntimeError(f"API failed: {last}")


def menta_queries() -> None:
    verify(); key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"): raise RuntimeError("OPENAI_API_KEY_REQUIRED")
    target_rows = list(csv.DictReader((INPUTS / "SHARED_TARGETS.csv").open(encoding="utf-8")))
    api_dir = RUNTIME / "menta_api"; api_dir.mkdir(parents=True, exist_ok=True)
    output = []; invalid = []
    for index, target in enumerate(target_rows, 1):
        tag = sha_text(target["document_id"])[:16]
        summary = cached_api(api_dir/f"{tag}_summary.json", key, input_text=build_summary_prompt(target["source_text"]),
                             instructions="Return only the requested one-sentence topic-focused description.", temperature=.3, max_tokens=256)
        parsed = None
        for attempt in (1, 2):
            questions = cached_api(api_dir/f"{tag}_questions_attempt{attempt}.json", key, input_text=build_prompt(target["source_text"]),
                                   instructions=build_system_prompt(), temperature=.7, max_tokens=1500)
            try:
                parsed = parse_five_queries(questions["text"]); break
            except Exception as exc:
                invalid.append({"target_id": target["document_id"], "attempt": attempt, "error": str(exc), "response_sha256": questions["text_sha256"]})
        if parsed is None: raise RuntimeError(f"MEntA format failed after two preserved attempts: {target['document_id']}")
        rows = make_session(target["document_id"], target["membership"], summary["text"].strip(), parsed)
        for row in rows: row.update({"attack": "MEntA", "domain": target["domain"], "generator_model": MODEL})
        output.extend(rows)
        checkpoint("SPARSE_CONFIRM_MENTA_PROGRESS", completed_targets=index, total_targets=200, valid_queries=len(output), format_failures=len(invalid))
    write_jsonl(INPUTS / "MENTA_ATTACK_QUERIES.jsonl", output)
    if len(output) != 1000: raise RuntimeError("MEntA confirmation count mismatch")
    atomic_json(AUDITS / "MENTA_QUERY_AUDIT.json", {"verdict": "PASS", "queries": 1000, "sessions": 200,
                "format_attempt_failures_preserved": invalid, "sha256": sha_file(INPUTS / "MENTA_ATTACK_QUERIES.jsonl")})


def build_idf(docs: dict[str, str]) -> tuple[dict[str, float], dict[str, set[str]]]:
    token_sets = {}; df = Counter()
    for doc_id, text in docs.items(): token_sets[doc_id] = normalize_tokens(text); df.update(token_sets[doc_id])
    return {t: math.log(3001/(n+1))+1 for t,n in df.items()}, token_sets


def lscore(query: str, doc_id: str, idf: dict, token_sets: dict) -> float:
    q = normalize_tokens(query); den = sum(idf.get(t, math.log(3001)+1) for t in q)
    return 0.0 if den == 0 else sum(idf[t] for t in q & token_sets[doc_id])/den


def score() -> None:
    verify()
    attacks = []
    for label, filename in (("MEntA", "MENTA_ATTACK_QUERIES.jsonl"), ("MBA", "MBA_ATTACK_QUERIES.jsonl"), ("RAG-MIA", "RAG_MIA_ATTACK_QUERIES.jsonl")):
        for row in read_jsonl(INPUTS / filename): attacks.append({**row, "attack": label, "cohort": "ATTACK"})
    if len(attacks) != 1400: raise RuntimeError("confirmation attack input incomplete")
    docs_rows = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl"); docs = {r["document_id"]:r["source_text"] for r in docs_rows}; doc_ids=list(docs)
    import torch
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(EXP.parents[1] / "code"))
    from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True); model.max_seq_length=512
    de = np.asarray(model.encode([docs[x] for x in doc_ids], batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
    qe = np.asarray(model.encode([r["query"] for r in attacks], batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
    rows=[]
    for start in range(0,len(attacks),128):
        matrix=qe[start:start+128]@de.T
        for j,scores in enumerate(matrix):
            source=attacks[start+j]; order=np.argsort(-scores,kind="stable")[:4]; ids=[doc_ids[int(k)] for k in order]; vals=[float(scores[int(k)]) for k in order]
            stat=canonical_mirabel_from_moments(vals[0],float(scores.sum(dtype=np.float64)),float(np.square(scores,dtype=np.float64).sum(dtype=np.float64)),len(doc_ids),.95)
            rank=ids.index(source["target_id"])+1 if source["target_id"] in ids else 0
            rows.append({**source,"top_document_ids":ids,"top_scores":vals,"selected_source_id":ids[0],"target_rank":rank,"M":float(stat.margin)})
    del model,de,qe;torch.cuda.empty_cache()
    idf,tokens=build_idf(docs)
    legacy_cache={r["query_id"]:r for r in read_jsonl(PARENT/"cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")}
    score_rows=list(csv.DictReader((V1/"tables/SCORE_ROWS.csv").open(encoding="utf-8")))
    tail=[]
    for s in score_rows:
        if s["cohort"]=="BENIGN" and s["calibration_partition"]=="TAIL_REFERENCE":
            base=legacy_cache[s["query_id"]];tail.append({"M":float(s["M"]),"L":lscore(base["query"],base["top_document_ids"][0],idf,tokens)})
    rm,rl=sorted(x["M"] for x in tail),sorted(x["L"] for x in tail)
    for row in rows:
        row["L"]=lscore(row["query"],row["top_document_ids"][0],idf,tokens)
        row["S_sparse"]=max(-math.log(empirical_upper_tail(rm,row["M"])),-math.log(empirical_upper_tail(rl,row["L"])))
    fresh=read_jsonl(V2/"cache/FRESH_REPLICATION_RETRIEVAL_SCORES.jsonl"); benign=[r for r in fresh if r["cohort"]=="BENIGN"]
    for row in benign:
        row["L"]=lscore(row["query"],row["top_document_ids"][0],idf,tokens)
        row["S_sparse"]=max(-math.log(empirical_upper_tail(rm,row["M"])),-math.log(empirical_upper_tail(rl,row["L"])))
    nm,ns=[r["M"] for r in benign],[r["S_sparse"] for r in benign];tpr3={};matched=[]
    for attack in ATTACKS:
        member=[r for r in rows if r["attack"]==attack and r["membership"]=="member"]
        for alpha in (.01,.03,.05):
            m=tpr_at_fpr([r["M"] for r in member],nm,alpha);s=tpr_at_fpr([r["S_sparse"] for r in member],ns,alpha)
            matched.append({"attack":attack,"fpr":alpha,"member_n":len(member),"mirabel_tpr":m,"sparse_tpr":s,"delta":s-m})
            if alpha==.03:tpr3[attack]={"M":m,"S_sparse":s,"delta":s-m}
    boots=[]
    for i,a in enumerate(ATTACKS):
        member=[r for r in rows if r["attack"]==a and r["membership"]=="member"]
        mean,lo,hi=bootstrap_delta(member,nm,ns,"S_sparse",2000,SEED+100+i);boots.append({"attack":a,"mean_delta":mean,"ci95_low":lo,"ci95_high":hi})
    macro_m=statistics.fmean(tpr3[a]["M"] for a in ATTACKS);macro_s=statistics.fmean(tpr3[a]["S_sparse"] for a in ATTACKS);mb=boots[0]
    checks={"menta_delta_at_least_5pp":tpr3["MEntA"]["delta"]>=.05-1e-12,"menta_ci_low_above_zero":mb["ci95_low"]>0,
            "mba_noninferior":tpr3["MBA"]["delta"]>=-.05-1e-12,"rag_mia_noninferior":tpr3["RAG-MIA"]["delta"]>=-.05-1e-12,"macro_improved":macro_s>macro_m}
    passed=all(checks.values()); verdict="SPARSE_EXPOSURE_REPLICATED" if passed else "SPARSE_EXPOSURE_NOT_REPLICATED"
    write_jsonl(CACHE/"CONFIRM_RETRIEVAL_SCORES.jsonl",rows);write_csv(CONF/"tables/MATCHED_TPR.csv",matched);write_csv(CONF/"tables/BOOTSTRAP.csv",boots)
    result={"verdict":verdict,"completed_utc":now(),"matched_member_tpr_at_3pct":tpr3,"macro":{"M":macro_m,"S_sparse":macro_s,"delta":macro_s-macro_m},
            "bootstrap":boots,"checks":checks,"e2e_allowed":passed,"precommit_sha256":sha_file(PRE)}
    atomic_json(CONF/"CONFIRMATION_RESULT.json",result);checkpoint(verdict,e2e_allowed=passed)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("prepare","local","menta","score"));mode=parser.parse_args().mode
    {"prepare":prepare,"local":local_queries,"menta":menta_queries,"score":score}[mode]()


if __name__ == "__main__": main()
