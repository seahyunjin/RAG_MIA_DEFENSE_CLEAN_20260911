#!/usr/bin/env python3
"""Frozen Final-LC TopiOCQA gold correctness and false-positive-harm evaluation."""
from __future__ import annotations

import bisect
import json
import math
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import (BGE, EXP, LC, QWEN, ROOT, atomic_json, checkpoint, now,
                    read_jsonl, sha_file, sha_text, verify_hashed_json, write_csv, write_jsonl)

sys.path.insert(0,str(ROOT/"code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments

NLI=__import__("pathlib").Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots")
NLI=next(NLI.iterdir())
SYSTEM_PROMPT=("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
               "If the context is insufficient, answer exactly: I don't know.")


def normalize_answer(text: str) -> str:
    text=text.casefold(); text=re.sub(r"\b(a|an|the)\b"," ",text); text=re.sub(r"[^a-z0-9\s]"," ",text)
    return " ".join(text.split())


def f1(answer: str, truth: str) -> float:
    a=normalize_answer(answer).split(); b=normalize_answer(truth).split()
    if not a or not b: return float(a==b)
    overlap=sum((Counter(a)&Counter(b)).values())
    if not overlap:return 0.
    p=overlap/len(a); r=overlap/len(b); return 2*p*r/(p+r)


def refusal(value: str) -> bool:
    value=" ".join(value.casefold().split())
    return any(x in value for x in ("i don't know","i do not know","cannot determine","insufficient information","not enough information"))


def word_count(value: str) -> int:
    """Count lexical tokens for sentence-level support auditing."""
    return len(re.findall(r"\b\w+\b", value, flags=re.UNICODE))


def waterfill(lengths,total=2048):
    caps=np.zeros(len(lengths),dtype=int); remaining=total; active=[i for i,x in enumerate(lengths) if x>0]
    while remaining and active:
        share=max(1,remaining//len(active)); changed=False
        for i in list(active):
            add=min(share,lengths[i]-int(caps[i]),remaining); caps[i]+=add; remaining-=add; changed|=bool(add)
            if caps[i]>=lengths[i]:active.remove(i)
            if not remaining:break
        if not changed:break
    return caps.tolist()


def build_prompt(tok,row,docs,hidden=None):
    kept=[x for x in row["top_document_ids"] if x!=hidden]; encoded=[tok(docs[x],add_special_tokens=False).input_ids for x in kept]
    caps=waterfill([len(x) for x in encoded]); visible=[tok.decode(x[:c],skip_special_tokens=True).strip() for x,c in zip(encoded,caps)]
    context="\n\n".join(f"[Document {i}]\n{x}" for i,x in enumerate(visible,1))
    user=f"Retrieved context:\n{context}\n\nUser query:\n{row['query']}"
    rendered=tok.apply_chat_template([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],tokenize=False,add_generation_prompt=True)
    ids=tok(rendered,add_special_tokens=False).input_ids
    if len(ids)>3072:ids=ids[-3072:];rendered=tok.decode(ids,skip_special_tokens=False)
    return rendered,{"context":context,"context_sha256":sha_text(context),"prompt_sha256":sha_text(json.dumps(ids,separators=(",",":"))),
                     "source_ids":kept,"source_caps":caps,"hidden_source_id":hidden}


def score_retrieval(pre):
    cache=EXP/"cache"/"GOLD_RETRIEVAL_AND_DETECTION.jsonl"
    if cache.is_file():return read_jsonl(cache)
    docs=read_jsonl(__import__("pathlib").Path(pre["corpus"]["path"])); queries=read_jsonl(__import__("pathlib").Path(pre["evaluation"]["path"]))
    prior=read_jsonl(LC/"cache"/"LARGE_DETECTION_SCORES.jsonl"); prior_emb=np.asarray(np.load(LC/"cache"/"QUERY_EMBEDDINGS.float16.npy"),dtype=np.float32)
    ref_idx=[i for i,r in enumerate(prior) if r["split"]=="REFERENCE"]; ref_emb=prior_emb[ref_idx]; ref_m=[float(prior[i]["M"]) for i in ref_idx]
    checkpoint("PHASE_C_BGE_LOADING",documents=len(docs),queries=len(queries))
    model=SentenceTransformer(str(BGE),device="cuda",local_files_only=True);model.max_seq_length=512
    doc_emb=np.asarray(model.encode([r["source_text"] for r in docs],batch_size=32,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32)
    query_emb=np.asarray(model.encode([r["query"] for r in queries],batch_size=48,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32)
    del model;torch.cuda.empty_cache(); doc_ids=[r["document_id"] for r in docs]; output=[]; global_sorted=sorted(ref_m)
    for start in range(0,len(queries),32):
        scores_b=query_emb[start:start+32]@doc_emb.T; local_b=query_emb[start:start+32]@ref_emb.T
        for off,scores in enumerate(scores_b):
            row=queries[start+off]; idx=np.argpartition(-scores,4)[:4];idx=idx[np.argsort(-scores[idx],kind="stable")]
            ids=[doc_ids[int(i)] for i in idx]; sims=[float(scores[int(i)]) for i in idx]
            stat=canonical_mirabel_from_moments(top1=sims[0],sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores,dtype=np.float64).sum(dtype=np.float64)),corpus_size=len(docs),confidence=.95)
            ls=local_b[off]; neighbors=np.argpartition(-ls,200)[:200];neighbors=neighbors[np.argsort(-ls[neighbors],kind="stable")]
            local_sorted=sorted(ref_m[int(i)] for i in neighbors)
            pl=(1+len(local_sorted)-bisect.bisect_left(local_sorted,float(stat.margin)))/201
            pg=(1+len(global_sorted)-bisect.bisect_left(global_sorted,float(stat.margin)))/1001
            output.append({**row,"top_document_ids":ids,"top_scores":sims,"selected_source_id":ids[0],
                           "gold_rank":ids.index(row["gold_document_id"])+1 if row["gold_document_id"] in ids else 0,
                           "M":float(stat.margin),"R_LC":-math.log(pl),"R_GLOBAL":-math.log(pg)})
        checkpoint("PHASE_C_RETRIEVAL_PROGRESS",completed=min(start+32,len(queries)),total=len(queries))
    write_jsonl(cache,output);atomic_json(EXP/"cache"/"GOLD_RETRIEVAL_MANIFEST.json",{"rows":len(output),"sha256":sha_file(cache)})
    return output


def generate_answers(pre,rows):
    docs={r["document_id"]:r["source_text"] for r in read_jsonl(__import__("pathlib").Path(pre["corpus"]["path"]))}
    con=sqlite3.connect(EXP/"runtime"/"gold_generation.sqlite3",timeout=120);con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS answer(query_id TEXT,condition TEXT,answer TEXT,context TEXT,source_ids_json TEXT,hidden_source_id TEXT,prompt_sha256 TEXT,wall_seconds REAL,PRIMARY KEY(query_id,condition))");con.commit()
    tok=AutoTokenizer.from_pretrained(QWEN,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(QWEN,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    rules={"NO_DEFENSE":lambda r:False,"ORIGINAL_MIRABEL_FIXED":lambda r:r["M"]>0,
           "GLOBAL_BC_FIXED":lambda r:r["R_GLOBAL"]>3.428580914764567,"FINAL_LC_FIXED":lambda r:r["R_LC"]>3.80543877128208}
    existing={(a,b) for a,b in con.execute("SELECT query_id,condition FROM answer")}; tasks=[]
    for row in rows:
        for condition,rule in rules.items():
            if (row["query_id"],condition) in existing:continue
            hidden=row["selected_source_id"] if rule(row) else None;prompt,p=build_prompt(tok,row,docs,hidden)
            tasks.append((row,condition,prompt,p))
    for start in range(0,len(tasks),16):
        batch=tasks[start:start+16];enc=tok([x[2] for x in batch],padding=True,truncation=True,max_length=3072,add_special_tokens=False,return_tensors="pt").to("cuda");before=time.perf_counter()
        with torch.inference_mode():out=model.generate(**enc,max_new_tokens=160,do_sample=False,num_beams=1,pad_token_id=tok.eos_token_id)
        elapsed=time.perf_counter()-before; answers=tok.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
        for (row,condition,_,p),answer in zip(batch,answers):
            con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?)",(row["query_id"],condition,answer.strip(),p["context"],json.dumps(p["source_ids"]),p["hidden_source_id"],p["prompt_sha256"],elapsed/len(batch)))
        con.commit();checkpoint("PHASE_C_GENERATION_PROGRESS",completed=len(existing)+min(start+16,len(tasks)),total=len(existing)+len(tasks))
    columns=[x[0] for x in con.execute("SELECT * FROM answer LIMIT 1").description]; data=[dict(zip(columns,x)) for x in con.execute("SELECT * FROM answer ORDER BY query_id,condition")]
    write_jsonl(EXP/"runtime"/"GOLD_GENERATED_ANSWERS.jsonl",data);return data


def factuality(rows):
    cases=[]
    for row in rows:
        if refusal(row["answer"]):continue
        for sentence in re.split(r"(?<=[.!?])\s+",row["answer"]):
            if word_count(sentence)>=3:cases.append((row,sentence))
    if not cases:return []
    checkpoint("PHASE_C_NLI_LOADING",sentences=len(cases))
    tok=AutoTokenizer.from_pretrained(NLI,local_files_only=True,use_fast=False);model=AutoModelForSequenceClassification.from_pretrained(NLI,local_files_only=True).to("cuda").eval()
    results=[]
    for start in range(0,len(cases),32):
        batch=cases[start:start+32];enc=tok([x[0]["context"] for x in batch],[x[1] for x in batch],padding=True,truncation=True,max_length=2048,return_tensors="pt").to("cuda")
        with torch.inference_mode():prob=torch.softmax(model(**enc).logits,dim=-1).float().cpu().numpy()
        labels={str(v).casefold():int(k) for k,v in model.config.id2label.items()}; ent=next((v for k,v in labels.items() if "entail" in k),0); con=next((v for k,v in labels.items() if "contr" in k),2)
        for (row,sentence),p in zip(batch,prob):results.append({"query_id":row["query_id"],"condition":row["condition"],"sentence":sentence,
            "entailed":int(np.argmax(p)==ent),"contradicted":int(np.argmax(p)==con),"entailment_probability":float(p[ent]),"contradiction_probability":float(p[con])})
        checkpoint("PHASE_C_NLI_PROGRESS",completed=min(start+32,len(cases)),total=len(cases))
    return results


def main():
    pre=verify_hashed_json(EXP/"configs"/"GOLD_QA_PRECOMMIT.json")
    for rel,digest in pre["code_sha256"].items():
        if sha_file(ROOT/rel)!=digest:raise RuntimeError(f"Phase-C code drift {rel}")
    rows=score_retrieval(pre);answers=generate_answers(pre,rows);query={r["query_id"]:r for r in read_jsonl(__import__("pathlib").Path(pre["evaluation"]["path"]))}
    detail=[]
    for row in answers:
        gold=query[row["query_id"]]["gold_answers"]; row["gold_f1"]=max(f1(row["answer"],x) for x in gold); row["gold_em"]=max(normalize_answer(row["answer"])==normalize_answer(x) for x in gold);row["refusal"]=refusal(row["answer"]);detail.append(row)
    nli=factuality(detail);write_csv(EXP/"tables"/"GOLD_FACTUALITY_SENTENCES.csv",nli); nli_group={(q,c):[] for q,c in [(r["query_id"],r["condition"]) for r in detail]}
    for r in nli:nli_group[(r["query_id"],r["condition"])].append(r)
    base={r["query_id"]:r for r in detail if r["condition"]=="NO_DEFENSE"};summary=[];fp=[]
    for condition in pre["conditions"]:
        subset=[r for r in detail if r["condition"]==condition]; intervened=[r for r in subset if r["hidden_source_id"]]
        sentence=[x for r in subset for x in nli_group[(r["query_id"],condition)]]
        summary.append({"condition":condition,"n":len(subset),"intervention_rate":len(intervened)/len(subset),"gold_f1":statistics.fmean(r["gold_f1"] for r in subset),
            "gold_em":statistics.fmean(r["gold_em"] for r in subset),"new_refusal":statistics.fmean(r["refusal"] and not base[r["query_id"]]["refusal"] for r in subset),
            "source_support_rate":statistics.fmean(r["entailed"] for r in sentence) if sentence else math.nan,
            "unsupported_sentence_rate":statistics.fmean(not r["entailed"] for r in sentence) if sentence else math.nan,
            "contradiction_proxy_rate":statistics.fmean(r["contradicted"] for r in sentence) if sentence else math.nan})
        for r in intervened:fp.append({"query_id":r["query_id"],"condition":condition,"question":query[r["query_id"]]["raw_question"],"gold_answers":json.dumps(query[r["query_id"]]["gold_answers"]),
            "no_defense_answer":base[r["query_id"]]["answer"],"defended_answer":r["answer"],"no_defense_f1":base[r["query_id"]]["gold_f1"],"defended_f1":r["gold_f1"],"new_refusal":r["refusal"] and not base[r["query_id"]]["refusal"]})
    write_csv(EXP/"tables"/"GOLD_QA_SUMMARY.csv",summary);write_csv(EXP/"tables"/"GOLD_FALSE_POSITIVE_HARM.csv",fp);write_jsonl(EXP/"runtime"/"GOLD_QA_DETAIL.jsonl",detail)
    by={r["condition"]:r for r in summary}; nd=by["NO_DEFENSE"];lc=by["FINAL_LC_FIXED"];orig=by["ORIGINAL_MIRABEL_FIXED"]
    checks={"f1_drop_at_most_2pp":nd["gold_f1"]-lc["gold_f1"]<=.02+1e-12,"new_refusal_at_most_1pp":lc["new_refusal"]<=.01+1e-12,
            "utility_retention_better_than_original":nd["gold_f1"]-lc["gold_f1"]<nd["gold_f1"]-orig["gold_f1"]}
    passed=all(checks.values());verdict="GOLD_QA_UTILITY_PASS" if passed else "GOLD_QA_UTILITY_FAILED"
    result={"campaign":EXP.name,"phase":"C_GOLD_QA","verdict":verdict,"completed_utc":now(),"summary":summary,"checks":checks,
            "factuality_claim_boundary":"sentence-level frozen-NLI proxy; not hallucination-free claim","next_stage":"PHASE_D_NEW_DOMAIN" if passed else "STOP_NO_DOMAIN"}
    atomic_json(EXP/"PHASE_C_RESULT.json",result);checkpoint(verdict,gold_f1=lc["gold_f1"],f1_drop=nd["gold_f1"]-lc["gold_f1"],new_refusal=lc["new_refusal"],next_stage=result["next_stage"])


if __name__=="__main__":main()
