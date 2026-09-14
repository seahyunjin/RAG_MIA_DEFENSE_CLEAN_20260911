#!/usr/bin/env python3
"""Exp198 SE-Mirabel: threshold-free soft source-exposure allocation."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT=PROJECT/"exp198_se_mirabel_soft_exposure_20260830"
EXP87=PROJECT/"exp87_native_scorer_stable_generation_20260821"
EXP166=PROJECT/"exp166_topiocqa_gold_utility_20260827"
EXP169=PROJECT/"exp169_k10_mechanism_validation_20260827"
EXP176=PROJECT/"exp176_prefix64_mirabel_external_attacks_20260828"
EXP193=PROJECT/"exp193_top4_context_rebase_20260829"
EXP195=PROJECT/"exp195_minimal_qll_exposure_guard_20260829"
EXP196=PROJECT/"exp196_selective_rank_agnostic_globalcap64_20260829"
EXP197=PROJECT/"exp197_rank_agnostic_cap64_backfill_20260829"
AD3=Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EMBED_ROOT=Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM2/outputs/exp1_mirabel_repro")
QWEN=Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET=Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
PRECOMMIT=ROOT/"configs/PRECOMMIT.json"
ATTACK_CASES=EXP193/"private/EXP193_CASES.private.pkl.gz"
NORMAL_CASES=EXP195/"private/EXP195_NORMAL_CASES.private.pkl.gz"
DEEP=EXP197/"private/EXP197_DEEP_RETRIEVAL.private.pkl.gz"
EXP197_PACKING=EXP197/"private/EXP197_RA_CAP64_BF_PACKING.private.pkl.gz"
RISK=ROOT/"private/EXP198_MIRABEL_RISK.private.pkl.gz"
PACKING=ROOT/"private/EXP198_SE_MIRABEL_PACKING.private.pkl.gz"
RESPONSES=ROOT/"private/EXP198_SE_MIRABEL_RESPONSES.private.csv.gz"
GEN_DB=ROOT/"private/EXP198_QWEN_RESPONSES.sqlite3"
TOP_K=10; TOTAL=2048; FLOOR=64; MAX_RENDERED=3072


def now():return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def sha256_text(value):return hashlib.sha256(str(value).encode()).hexdigest()


def atomic_text(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_json(path,value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,
        default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")


def atomic_csv(frame,path,compression=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.stem}.",suffix=".csv.gz" if compression=="gzip" else ".csv",dir=path.parent);os.close(fd)
    try:frame.to_csv(tmp,index=False,compression=compression);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_pickle(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.stem}.",suffix=".pkl.gz",dir=path.parent);os.close(fd)
    try:frame.to_pickle(tmp,compression="gzip");os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def checkpoint(stage,**details):
    payload={"experiment":"Exp198","stage":stage,"updated_utc":now(),"pid":os.getpid(),
             "candidate":"SE_MIRABEL","binary_alarm":0,"learned_parameters":0,
             "attack_specific_tuning":0,"rank_specific_tuning":0,"paid_api_calls":0,**details}
    atomic_json(ROOT/"HEARTBEAT.json",payload);atomic_json(ROOT/"checkpoints"/f"{stage}.json",payload)
    (ROOT/"logs").mkdir(parents=True,exist_ok=True)
    with (ROOT/"logs/pipeline.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(payload,default=str)+"\n")
    atomic_text(ROOT/"STATUS.md","\n".join(["# Exp198 Status","",f"- Stage: **{stage}**",f"- Updated UTC: `{payload['updated_utc']}`",f"- PID: `{payload['pid']}`"]+[f"- {k}: `{v}`" for k,v in details.items()])+"\n")


def attack_corpus(domain):
    rows=[json.loads(line) for line in (AD3/f"data/beir/{domain}/corpus_member.jsonl").open() if line.strip()]
    ids=np.asarray([str(row.get("_id",row.get("id"))) for row in rows])
    texts={str(row.get("_id",row.get("id"))):str(row.get("text","")) for row in rows}
    matrix=np.load(EMBED_ROOT/domain/"corpus_embeddings.npy").astype(np.float32,copy=False)[:len(ids)]
    return ids,texts,matrix


def all_documents():
    output={}
    for domain in ("BeIR_nfcorpus","BeIR_scidocs","BeIR_trec-covid"):
        _,texts,_=attack_corpus(domain);output.update({(domain,k):v for k,v in texts.items()})
    corpus=pd.read_csv(EXP166/"private/TOPIOCQA_CORPUS.private.csv.gz",keep_default_na=False,dtype={"document_id":str})
    output.update({("TopiOCQA",str(row.document_id)):f"{row.title}\n{row.text}" for row in corpus.itertuples(index=False)})
    return output


def preflight():
    config=json.loads(PRECOMMIT.read_text())
    required={"precommit":PRECOMMIT,"attack_cases":ATTACK_CASES,"normal_cases":NORMAL_CASES,
      "deep_retrieval":DEEP,"exp197_packing":EXP197_PACKING,"exp197_final":EXP197/"FINAL_RESULT.json",
      "exp196_responses":EXP196/"private/EXP196_ALL_RESPONSES.private.csv.gz",
      "exp169_packing_code":EXP169/"code/packing.py","qwen":QWEN/"config.json","mpnet":MPNET/"config.json",
      "normal_corpus":EXP166/"private/TOPIOCQA_CORPUS.private.csv.gz",
      "normal_embeddings":EXP166/"private/TOPIOCQA_MPNET_CORPUS_EMBEDDINGS.npy"}
    for domain in ("BeIR_nfcorpus","BeIR_scidocs","BeIR_trec-covid"):
        required[f"{domain}_corpus"]=AD3/f"data/beir/{domain}/corpus_member.jsonl"
        required[f"{domain}_embeddings"]=EMBED_ROOT/domain/"corpus_embeddings.npy"
    missing=[str(x) for x in required.values() if not x.exists()]
    if missing:raise RuntimeError(f"missing frozen input: {missing}")
    old=json.loads((EXP197/"FINAL_RESULT.json").read_text())
    if old["verdict"]!="RANK_AGNOSTIC_CAP64_FAILED":raise RuntimeError("Exp197 failure branch not satisfied")
    manifest=pd.DataFrame([{"key":k,"path":str(v),"bytes":v.stat().st_size,"sha256":sha256_file(v),"access":"READ_ONLY"} for k,v in required.items()])
    atomic_csv(manifest,ROOT/"provenance/FROZEN_INPUTS.csv")
    atomic_json(ROOT/"audits/NO_HEURISTIC_AUDIT.json",{
      "binary_threshold":False,"benign_calibration":False,"attack_labels_used_by_policy":False,
      "membership_labels_used_by_policy":False,"rank_values_used_by_policy":False,
      "learned_parameters":0,"parameter_sweep":False,"rank11plus_backfill":False})
    checkpoint("PREFLIGHT_COMPLETE",frozen_inputs=len(manifest),exp197_verdict=old["verdict"],precommit_sha256=sha256_file(PRECOMMIT))
    return config


def mirabel_probabilities(source_scores,mean,std,n):
    source_scores=np.asarray(source_scores,dtype=float)
    if len(source_scores)!=TOP_K or not np.isfinite(source_scores).all():raise ValueError("ten finite source scores required")
    root=math.sqrt(2*math.log(int(n)));g=-math.log(-math.log(.95));tau=float(mean)+float(std)*root+g*float(std)/root
    scale=max(float(std),np.finfo(float).eps);z=(source_scores-tau)/scale;z-=z.max()
    p=np.exp(z);p/=p.sum();return p,tau


def largest_remainder_caps(probabilities,total=TOTAL,floor=FLOOR):
    p=np.asarray(probabilities,dtype=float)
    if len(p)!=TOP_K or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError("probability simplex required")
    remaining=int(total)-len(p)*int(floor);raw=floor+remaining*(1-p)/(len(p)-1)
    base=np.floor(raw).astype(int);left=int(total-base.sum())
    order=sorted(range(len(p)),key=lambda i:(-(raw[i]-base[i]),i))
    for i in order[:left]:base[i]+=1
    if base.sum()!=total or base.min()<floor or base.max()>221:raise RuntimeError("allocation contract failed")
    return base,raw


def build_risk(attacks,normal):
    if RISK.exists():
        old=pd.read_pickle(RISK,compression="gzip")
        if len(old)==len(attacks)+len(normal):return old
    import torch
    from sentence_transformers import SentenceTransformer
    device="cuda" if torch.cuda.is_available() else "cpu";encoder=SentenceTransformer(str(MPNET),device=device,local_files_only=True)
    exp197=pd.read_pickle(EXP197_PACKING,compression="gzip")[["case_id","source_ids_used"]]
    all_cases=pd.concat([attacks.assign(split="ATTACK"),normal.assign(split="NORMAL")],ignore_index=True,sort=False).merge(exp197,on="case_id",validate="one_to_one")
    records=[]
    for domain,cell in all_cases.groupby("domain",sort=True):
        if domain=="TopiOCQA":
            corpus=pd.read_csv(EXP166/"private/TOPIOCQA_CORPUS.private.csv.gz",keep_default_na=False,dtype={"document_id":str})
            ids=corpus.document_id.astype(str).to_numpy();matrix=np.load(EXP166/"private/TOPIOCQA_MPNET_CORPUS_EMBEDDINGS.npy").astype(np.float32,copy=False)
        else:ids,_,matrix=attack_corpus(str(domain))
        index={str(v):i for i,v in enumerate(ids)}
        groups={str(q):part for q,part in cell.groupby("query",sort=False)}
        unique=list(groups);qemb=encoder.encode(unique,normalize_embeddings=True,convert_to_numpy=True,
            batch_size=128,show_progress_bar=False).astype(np.float32)
        # Full-corpus moments are evaluated in bounded batches.  Repeated
        # BudgetLeak output budgets share the exact same query computation.
        block_size=64 if len(ids)>5000 else 256
        for offset in range(0,len(unique),block_size):
            block=qemb[offset:offset+block_size]@matrix.T
            for local,query in enumerate(unique[offset:offset+len(block)]):
                all_scores=block[local];mx=float(all_scores.max());count=len(all_scores)-1
                mean=float((all_scores.sum(dtype=np.float64)-mx)/count)
                variance=float((np.square(all_scores,dtype=np.float64).sum()-mx*mx)/count-mean*mean)
                std=math.sqrt(max(0.0,variance))
                for row in groups[query].itertuples(index=False):
                    sources=list(map(str,row.source_ids_used[:TOP_K]))
                    if len(sources)!=TOP_K:raise RuntimeError(f"top10 unavailable {row.case_id}")
                    scores=np.asarray([all_scores[index[s]] for s in sources],float)
                    prob,tau=mirabel_probabilities(scores,mean,std,len(all_scores));caps,raw=largest_remainder_caps(prob)
                    records.append({"case_id":str(row.case_id),"domain":str(domain),"source_ids":sources,
                      "source_scores":scores.tolist(),"mirabel_background_mean":mean,"mirabel_background_std":std,
                      "mirabel_threshold":tau,"risk_probabilities":prob.tolist(),"raw_caps":raw.tolist(),"allocated_caps":caps.tolist(),
                      "max_risk":float(prob.max()),"risk_entropy":float(-(prob*np.log(np.maximum(prob,1e-300))).sum())})
        checkpoint("RISK_DOMAIN_COMPLETE",domain=str(domain),cases=len(cell),device=device)
    output=pd.DataFrame(records);atomic_pickle(output,RISK)
    del encoder;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    checkpoint("RISK_COMPLETE",cases=len(output),mean_max_risk=float(output.max_risk.mean()),mean_entropy=float(output.risk_entropy.mean()))
    return output


def render_length(tokenizer,models,query,context):
    prompt=models.normal_prompt(str(query),list(context));messages=[{"role":"system","content":models.SYSTEM_PROMPT},{"role":"user","content":prompt}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return prompt,len(tokenizer(rendered,add_special_tokens=False).input_ids)


def pack_one(tokenizer,models,query,texts,caps):
    tokenized=[tokenizer(str(v),add_special_tokens=False).input_ids for v in texts]
    caps=np.minimum(np.asarray(caps,int),np.asarray([len(ids) for ids in tokenized],int))
    def materialize():
        context=[];used=[]
        for ids,cap in zip(tokenized,caps):
            chosen=ids[:max(0,int(cap))];used.append(len(chosen));value=tokenizer.decode(chosen,skip_special_tokens=True).strip()
            if value:context.append(value)
        prompt,length=render_length(tokenizer,models,query,context);return prompt,context,used,length
    prompt,context,used,length=materialize();raw=length;removed=0
    empty_prompt,empty_length=render_length(tokenizer,models,query,[])
    # One frozen RAGLeak source row contains a 61,802-character query whose
    # query alone exceeds the generator contract.  Prior baselines use the
    # frozen adapter's deterministic left truncation for this same row.  We
    # retain cohort symmetry, expose no document tokens, and audit it rather
    # than silently treating document packing as the cause.
    if empty_length>MAX_RENDERED:
        return empty_prompt,[],[0]*len(caps),[0]*len(caps),raw,empty_length,int(caps.sum()),True
    while length>MAX_RENDERED:
        overflow=length-MAX_RENDERED;extra=np.maximum(0,caps-FLOOR)
        if extra.sum()==0:extra=caps.copy()
        if extra.sum()==0:raise RuntimeError("rendered prompt overflow with empty context")
        decrement=np.floor(overflow*extra/extra.sum()).astype(int);decrement=np.minimum(decrement,extra);decrement[decrement<0]=0
        if decrement.sum()==0:decrement[int(np.argmax(extra))]=1
        caps-=decrement;removed+=int(decrement.sum());prompt,context,used,length=materialize()
    return prompt,context,list(map(int,caps)),used,raw,length,removed,False


def build_tasks(attacks,normal,risk):
    if PACKING.exists():
        old=pd.read_pickle(PACKING,compression="gzip")
        if len(old)==len(attacks)+len(normal):return old
    from transformers import AutoTokenizer
    sys.path.insert(0,str(EXP87/"code"));import exp87_models as models
    tokenizer=AutoTokenizer.from_pretrained(QWEN,local_files_only=True);documents=all_documents();risk=risk.set_index("case_id")
    cases=pd.concat([attacks.assign(split="ATTACK"),normal.assign(split="NORMAL")],ignore_index=True,sort=False);rows=[]
    for idx,row in enumerate(cases.itertuples(index=False),1):
        rr=risk.loc[str(row.case_id)];sources=list(map(str,rr.source_ids));caps=list(map(int,rr.allocated_caps));texts=[documents[(str(row.domain),s)] for s in sources]
        prompt,context,final_caps,used,raw,final,removed,query_only_overflow=pack_one(tokenizer,models,str(row.query),texts,caps)
        maximum=int(row.max_new_tokens) if hasattr(row,"max_new_tokens") and not pd.isna(row.max_new_tokens) else 96
        task=models.make_task(task_type="EXP198_SE_MIRABEL",row_id=str(row.case_id),prompt=prompt,system_prompt=str(row.system_prompt),max_new_tokens=maximum)
        rows.append({"case_id":str(row.case_id),"row_id":str(row.row_id),"panel":str(row.panel),"family":str(row.family),
          "member":int(row.member),"domain":str(row.domain),"session_id":str(row.session_id),"turn_order":int(row.turn_order),
          "target_rank":int(row.target_rank),"query":str(row.query),"source_ids_used":sources,"tokens_used":used,
          "allocated_caps":caps,"final_caps":final_caps,"total_context_tokens":sum(used),"max_source_tokens":max(used),
          "risk_probabilities":rr.risk_probabilities,"mirabel_threshold":float(rr.mirabel_threshold),
          "max_risk":float(rr.max_risk),"risk_entropy":float(rr.risk_entropy),"raw_rendered_tokens":raw,
          "final_rendered_tokens":final,"overflow_tokens_removed":removed,"query_only_overflow":query_only_overflow,
          "prompt":prompt,"prompt_hash":task["prompt_hash"],
          "task_key":task["task_key"],"system_prompt":str(row.system_prompt),"max_new_tokens":maximum})
        if idx%4000==0:checkpoint("PACKING_PROGRESS",completed=idx,total=len(cases))
    output=pd.DataFrame(rows);atomic_pickle(output,PACKING)
    summary=pd.DataFrame([{"condition":"SE_MIRABEL","queries":len(output),"top_k":TOP_K,
      "mean_context_tokens":output.total_context_tokens.mean(),"median_context_tokens":output.total_context_tokens.median(),
      "mean_max_source_tokens":output.max_source_tokens.mean(),"maximum_source_tokens":output.max_source_tokens.max(),
      "mean_max_risk":output.max_risk.mean(),"mean_risk_entropy":output.risk_entropy.mean(),
      "overflow_adjustment_rate":output.overflow_tokens_removed.gt(0).mean(),
      "query_only_overflow_rows":int(output.query_only_overflow.sum()),"rank11plus_backfill_rate":0.0}])
    atomic_csv(summary,ROOT/"tables/TABLE_198_01_PACKING.csv")
    # Mechanism collision is defined by the frozen nominal allocation, not by name.
    b64=np.asarray([64]+[221]*8+[220]);same=output.allocated_caps.map(lambda x:np.array_equal(np.asarray(x),b64))
    atomic_json(ROOT/"audits/COLLISION_AUDIT.json",{"duplicate_of_K10_B64":bool(same.all()),
      "exact_nominal_allocation_collision_rate":float(same.mean()),"duplicate_of_Exp197":False,
      "new_mechanism":"continuous canonical-Mirabel risk allocation over top10"})
    overflow=output[output.query_only_overflow][["case_id","row_id","panel","family","raw_rendered_tokens","final_rendered_tokens"]]
    atomic_csv(overflow,ROOT/"audits/QUERY_ONLY_OVERFLOW.csv")
    checkpoint("PACKING_COMPLETE",queries=len(output),mean_context_tokens=float(output.total_context_tokens.mean()),
      maximum_source_tokens=int(output.max_source_tokens.max()),k10_b64_collision_rate=float(same.mean()))
    return output


def generate(packing):
    if RESPONSES.exists():
        old=pd.read_csv(RESPONSES,keep_default_na=False,dtype={"case_id":str})
        if len(old)==len(packing):return old
    sys.path.insert(0,str(EXP87/"code"));import exp87_models as models
    models.ROOT=ROOT;models.heartbeat=lambda stage,**details:checkpoint(stage,**details);models.GENERATION_CONFIG["batch_size"]=16
    tasks=[models.make_task(task_type="EXP198_SE_MIRABEL",row_id=str(r.case_id),prompt=str(r.prompt),
      system_prompt=str(r.system_prompt),max_new_tokens=int(r.max_new_tokens)) for r in packing.itertuples(index=False)]
    checkpoint("GENERATION_STARTED",tasks=len(tasks),device="cuda:0",generations_per_query=1,extra_scoring_forward_passes=0)
    started=time.perf_counter();answers=models.run_generation(tasks,GEN_DB,None,"GENERATION_PROGRESS")
    output=packing.drop(columns=["prompt"]).copy();output["condition"]="SE_MIRABEL";output["response"]=output.case_id.map(answers)
    output["response_sha256"]=output.response.map(sha256_text)
    if output.response.isna().any():raise RuntimeError("generation incomplete")
    atomic_csv(output,RESPONSES,"gzip");elapsed=time.perf_counter()-started
    atomic_csv(pd.DataFrame([{"condition":"SE_MIRABEL","queries":len(output),"wall_seconds":elapsed,
      "throughput_queries_per_second":len(output)/elapsed,"trainable_parameters":0,"generations_per_query":1,
      "extra_scoring_forward_passes":0,"session_state":0,"retrieval_top_k":TOP_K,"peak_vram":"NOT_ISOLATED"}]),
      ROOT/"tables/TABLE_198_02_EFFICIENCY.csv")
    checkpoint("GENERATION_COMPLETE",responses=len(output),wall_seconds=elapsed,throughput=len(output)/elapsed,database_sha256=sha256_file(GEN_DB))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");args=parser.parse_args()
    preflight();attacks=pd.read_pickle(ATTACK_CASES,compression="gzip");normal=pd.read_pickle(NORMAL_CASES,compression="gzip")
    risk=build_risk(attacks,normal);packing=build_tasks(attacks,normal,risk)
    if args.prepare_only:checkpoint("PREPARE_COMPLETE",rows=len(packing));return
    generate(packing);checkpoint("GENERATION_STAGE_COMPLETE",next_stage="evaluate_exp198.py")


if __name__=="__main__":main()
