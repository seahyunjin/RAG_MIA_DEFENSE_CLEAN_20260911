#!/usr/bin/env python3
"""Exp212: retriever transfer of frozen Stateless QLL Source Hide."""
from __future__ import annotations

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
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT=PROJECT/"exp212_retriever_transfer_stateless_qll_20260831"
AD_ROOT=Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EXP210=PROJECT/"exp210_stateless_qll_source_hide_recovery_20260831"
EXP211=PROJECT/"exp211_untouched_blind_stateless_qll_20260831"
EXP68=AD_ROOT/"artifacts/exp68_true_fresh_blind_final"
EXP166=PROJECT/"exp166_topiocqa_gold_utility_20260827"
EXP195=PROJECT/"exp195_minimal_qll_exposure_guard_20260829"
EXP160E=PROJECT/"exp160e_global_cap64_external_attacks_exploratory_20260826"
EXP179_CODE=PROJECT/"exp179_cross_family_qwen_source_influence_20260828/code/run_exp179.py"
EXP211_CODE=EXP211/"code/run_exp211.py"
EXP87=PROJECT/"exp87_native_scorer_stable_generation_20260821"
STAGE204_CODE=PROJECT/"qll_source_hide_final_validation_20260830/stage_204_budgetleak_stress/code/run_stage_204.py"
QWEN=Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET=Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
RETRIEVERS={
    "GTE":Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--thenlper--gte-large/snapshots/4bef63f39fcc5e2d6b0aae83089f307af4970164"),
    "BGE":Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"),
}
PRECOMMIT=ROOT/"configs/PRECOMMIT.json"
THRESHOLD=0.5300846414247485
TOP_K=4
TOTAL_SOURCE_TOKENS=2048
BUDGETS=tuple(range(10,271,20))
STAGE_A_NATIVE=("DCMI","S²-MIA")
STAGE_B_NATIVE=("MEntA","RAG-MIA","MBA")
ALL_NATIVE=STAGE_A_NATIVE+STAGE_B_NATIVE
TOKEN_RE=re.compile(r"[A-Za-z0-9]+")
NORMAL_SYSTEM=("Answer the user's question using only the retrieved context. Follow any required output "
               "format exactly. If the context is insufficient, answer exactly: I don't know.")
DEFAULT_SYSTEM="Please answer the question based on the provided context."
MAX_NEW={"RAG-MIA":12,"S²-MIA":96,"MBA":160,"DCMI":12,"MEntA":96,
         "RAGLeak":128,"BudgetLeak-Z":128,"BENIGN":96}


def now():return datetime.now(timezone.utc).isoformat()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(module);return module


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def sha256_text(value):return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def atomic_text(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            handle.write(value);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def atomic_json(path,value):atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")


def atomic_csv(frame,path,compression=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    suffix=".csv.gz" if str(path).endswith(".gz") else ".csv"
    fd,temporary=tempfile.mkstemp(prefix=f".{path.name}.",suffix=suffix,dir=path.parent);os.close(fd)
    try:frame.to_csv(temporary,index=False,compression=compression);os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def checkpoint(stage,**details):
    payload={"stage":stage,"updated_utc":now(),"pid":os.getpid(),**details}
    atomic_json(ROOT/"checkpoints"/f"{stage}.json",payload)
    atomic_json(ROOT/"HEARTBEAT.json",payload)
    lines=["# Exp212 status","",f"- Stage: **{stage}**",f"- Updated UTC: `{payload['updated_utc']}`",f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key,value in details.items())
    atomic_text(ROOT/"STATUS.md","\n".join(lines)+"\n")
    with (ROOT/"logs/pipeline.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(payload,ensure_ascii=False,sort_keys=True)+"\n")


def refusal(value):
    text=str(value).casefold()
    return any(x in text for x in ("i don't know","i do not know","cannot determine","insufficient context","not enough information"))


def token_f1(left,right):
    from collections import Counter
    a=TOKEN_RE.findall(str(left).casefold());b=TOKEN_RE.findall(str(right).casefold())
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    common=sum((Counter(a)&Counter(b)).values());return 2*common/(len(a)+len(b))


def effective_auc(labels,scores):
    raw=float(roc_auc_score(np.asarray(labels,int),np.asarray(scores,float)));return raw,max(raw,1-raw)


def preflight():
    required={
      "precommit":PRECOMMIT,
      "exp210_model":EXP210/"models/STATELESS_QLL_SOURCE_HIDE_FROZEN.json",
      "exp210_code":EXP210/"code/run_exp210.py",
      "exp210_result":EXP210/"FINAL_RESULT.json",
      "exp211_result":EXP211/"FINAL_RESULT.json",
      "fiqa_source":EXP68/"private/fresh_source.private.json.gz",
      "fiqa_sessions":EXP68/"private/fresh_attack_sessions.private.jsonl.gz",
      "normal_cases":EXP195/"private/EXP195_NORMAL_CASES.private.pkl.gz",
      "normal_corpus":EXP166/"private/TOPIOCQA_CORPUS.private.csv.gz",
      "external_cohort":EXP160E/"private/EXP160E_ATTACK_COHORT.private.csv.gz",
      "qwen_config":QWEN/"config.json","mpnet_scorer_config":MPNET/"config.json",
      "gte_config":RETRIEVERS["GTE"]/"config.json","bge_config":RETRIEVERS["BGE"]/"config.json",
      "qll_code":EXP179_CODE,"evaluation_code":EXP211_CODE,
      "budget_scorer":STAGE204_CODE,"generator_adapter":EXP87/"code/exp87_models.py",
    }
    for domain in ("BeIR_nfcorpus","BeIR_scidocs","BeIR_trec-covid"):
        required[f"{domain}_member_corpus"]=AD_ROOT/f"data/beir/{domain}/corpus_member.jsonl"
    missing=[str(path) for path in required.values() if not path.exists()]
    if missing:raise RuntimeError(f"missing frozen inputs: {missing}")
    expected={
      "exp210_model":"9c0da63340dd642846cf81fc325c8bd188aed4bf47079b98cfcf9a88eb2f29c8",
      "exp210_code":"aba8dd61648eaae7d705b3487d8b0b4b2dc2df1dab42691a9a40fe27c9231a4b",
      "exp210_result":"74bfa284bde4a70d73dd201285ac16ccf590fbd02422e6a0a11915f23869c587",
      "gte_config":"42a037b389d02db73d1d5bd0d049d3269e3617e368f86992474a32c42ffbd859",
      "bge_config":"26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    }
    for key,digest in expected.items():
        if sha256_file(required[key])!=digest:raise RuntimeError(f"frozen hash drift: {key}")
    exp210=json.loads(required["exp210_result"].read_text())
    exp211=json.loads(required["exp211_result"].read_text())
    if exp210.get("verdict")!="STATELESS_QLL_SOURCE_HIDE_RECOVERED" or not exp210.get("passed"):
        raise RuntimeError("Exp210 frozen parent is not recovered")
    if exp211.get("verdict")!="EXP211_INPUT_INSUFFICIENT":raise RuntimeError("unexpected Exp211 lineage")
    rows=pd.DataFrame([{"key":key,"path":str(path),"bytes":path.stat().st_size,
                       "sha256_before":sha256_file(path),"access":"READ_ONLY"} for key,path in required.items()])
    atomic_csv(rows,ROOT/"provenance/FROZEN_INPUTS.csv")
    atomic_text(ROOT/"configs/PRECOMMIT.sha256",f"{sha256_file(PRECOMMIT)}  PRECOMMIT.json\n")
    checkpoint("PREFLIGHT_COMPLETE",frozen_inputs=len(rows),retrievers="GTE,BGE",model_changed=False)
    return rows


def load_source_sessions_external():
    with gzip.open(EXP68/"private/fresh_source.private.json.gz","rt",encoding="utf-8") as handle:source=json.load(handle)
    with gzip.open(EXP68/"private/fresh_attack_sessions.private.jsonl.gz","rt",encoding="utf-8") as handle:sessions=[json.loads(line) for line in handle]
    external=pd.read_csv(EXP160E/"private/EXP160E_ATTACK_COHORT.private.csv.gz",keep_default_na=False,
                         dtype={"row_id":str,"target_document_id":str})
    return source,sessions,external


def build_query_frame(source,sessions,external):
    path=ROOT/"private/FROZEN_QUERY_FRAME.private.csv.gz"
    if path.exists():return pd.read_csv(path,keep_default_na=False,low_memory=False,
                                        dtype={"row_id":str,"case_id":str,"session_id":str,"target_document_id":str})
    normal=pd.read_pickle(EXP195/"private/EXP195_NORMAL_CASES.private.pkl.gz",compression="gzip")
    rows=[]
    for x in normal.itertuples(index=False):
        rows.append({"case_id":f"BENIGN|{x.case_id}","row_id":str(x.row_id),"kind":"BENIGN",
          "attack_family":"BENIGN","session_id":str(x.session_id),"turn":1,"query":str(x.query),
          "member":-1,"target_document_id":str(x.target_document_id),"dataset":"TopiOCQA",
          "domain":"TopiOCQA","system_prompt":str(x.system_prompt),"reference":""})
    for session in sessions:
        family=str(session["attack_family"])
        if family=="IA":continue
        for turn,query in enumerate(session["queries"],1):
            rows.append({"case_id":f"NATIVE|{session['session_id']}|Q{turn}","row_id":f"{session['session_id']}|Q{turn}",
              "kind":"ATTACK","attack_family":family,"session_id":str(session["session_id"]),"turn":turn,
              "query":str(query),"member":int(session["member_label"]),
              "target_document_id":str(session["target_document_id"]),"dataset":"FiQA-2018","domain":"FiQA-2018",
              "system_prompt":NORMAL_SYSTEM,"reference":""})
    for x in external.itertuples(index=False):
        rows.append({"case_id":f"RAGLEAK|{x.row_id}","row_id":str(x.row_id),"kind":"EXTERNAL",
          "attack_family":"RAGLeak","session_id":str(x.row_id),"turn":1,"query":str(x.ragleak_query),
          "member":int(x.member),"target_document_id":str(x.target_document_id),"dataset":str(x.domain),
          "domain":str(x.domain),"system_prompt":DEFAULT_SYSTEM,"reference":str(x.ragleak_reference)})
        rows.append({"case_id":f"BUDGET|{x.row_id}","row_id":str(x.row_id),"kind":"EXTERNAL",
          "attack_family":"BudgetLeak-Z","session_id":str(x.row_id),"turn":1,"query":str(x.budgetleak_query),
          "member":int(x.member),"target_document_id":str(x.target_document_id),"dataset":str(x.domain),
          "domain":str(x.domain),"system_prompt":DEFAULT_SYSTEM,"reference":str(x.budgetleak_reference)})
    frame=pd.DataFrame(rows)
    if len(frame)!=5200 or frame.case_id.nunique()!=5200:raise RuntimeError(f"frozen query frame mismatch: {frame.shape}")
    native=frame[frame.kind.eq("ATTACK")].groupby(["attack_family","member"]).session_id.nunique().unstack(fill_value=0)
    if set(native.index)!=set(ALL_NATIVE) or not (native[0].eq(30)&native[1].eq(30)).all():raise RuntimeError(f"native balance failure\n{native}")
    if len(frame[frame.kind.eq("BENIGN")])!=1000:raise RuntimeError("benign cohort is not 1000")
    ext=frame[frame.kind.eq("EXTERNAL")].drop_duplicates(["attack_family","session_id"])
    if not all(ext[ext.attack_family.eq(f)].member.value_counts().to_dict()=={0:900,1:900} for f in ("RAGLeak","BudgetLeak-Z")):
        raise RuntimeError("external membership balance failure")
    atomic_csv(frame,path,"gzip")
    atomic_json(ROOT/"audits/COHORT_AUDIT.json",{
      "queries":len(frame),"benign":1000,"native_query_turns":600,"external_queries":3600,
      "native_sessions":300,"external_sessions_per_family":1800,"attack_calibration_examples":0,
      "query_id_hash":sha256_text("\n".join(sorted(frame.case_id))),"membership_or_query_changed":False})
    checkpoint("COHORT_FROZEN",queries=len(frame),benign=1000,native_turns=600,external=3600)
    return frame


def load_corpora(source,sessions):
    corpora={}
    topi=pd.read_csv(EXP166/"private/TOPIOCQA_CORPUS.private.csv.gz",keep_default_na=False,dtype={"document_id":str})
    corpora["TopiOCQA"]=(topi.document_id.astype(str).tolist(),(topi.title.astype(str)+"\n"+topi.text.astype(str)).tolist())
    nonmember={str(x["target_document_id"]) for x in sessions if int(x["member_label"])==0}
    ids=[];texts=[]
    for document_id,document in source["documents"].items():
        if str(document_id) in nonmember:continue
        ids.append(str(document_id));texts.append("\n".join(v for v in (str(document.get("title","")),str(document.get("text",""))) if v))
    corpora["FiQA-2018"]=(ids,texts)
    for domain in ("BeIR_nfcorpus","BeIR_scidocs","BeIR_trec-covid"):
        ids=[];texts=[]
        with (AD_ROOT/f"data/beir/{domain}/corpus_member.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():continue
                item=json.loads(line);ids.append(str(item.get("_id",item.get("id"))));texts.append(str(item.get("text","")))
        if len(ids)!=1000 or len(set(ids))!=1000:raise RuntimeError(f"{domain} member corpus contract failure")
        corpora[domain]=(ids,texts)
    atomic_json(ROOT/"audits/CORPUS_PROVENANCE.json",{
      name:{"documents":len(ids),"id_hash":sha256_text("\n".join(sorted(ids)))} for name,(ids,_) in corpora.items()})
    return corpora


def retrieve(retriever,frame,corpora):
    path=ROOT/f"private/{retriever}_RETRIEVAL.private.csv.gz"
    if path.exists():return pd.read_csv(path,keep_default_na=False,low_memory=False,
                                        dtype={"row_id":str,"case_id":str,"session_id":str,"target_document_id":str})
    import torch
    from sentence_transformers import SentenceTransformer
    started=time.perf_counter();model=SentenceTransformer(str(RETRIEVERS[retriever]),device="cuda",local_files_only=True);model.max_seq_length=512
    rows=[];corpus_seconds=0.0;query_seconds=0.0
    for dataset,(ids,texts) in corpora.items():
        subset=frame[frame.dataset.eq(dataset)].reset_index(drop=True)
        if subset.empty:continue
        emb_path=ROOT/f"private/{retriever}_{dataset.replace('/','_')}_CORPUS.float16.npy"
        stamp=time.perf_counter()
        if emb_path.exists():emb=np.asarray(np.load(emb_path),np.float32);emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
        else:
            checkpoint(f"{retriever}_CORPUS_EMBEDDING",dataset=dataset,documents=len(ids))
            batch=16 if retriever=="BGE" else 96
            emb=model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,batch_size=batch,show_progress_bar=True).astype(np.float32)
            np.save(emb_path,emb.astype(np.float16))
        corpus_seconds+=time.perf_counter()-stamp
        stamp=time.perf_counter();qemb=model.encode(subset["query"].astype(str).tolist(),normalize_embeddings=True,
          convert_to_numpy=True,batch_size=64 if retriever=="BGE" else 128,show_progress_bar=False).astype(np.float32)
        query_seconds+=time.perf_counter()-stamp
        corpus_tensor=torch.as_tensor(emb,device="cuda",dtype=torch.float32);lookup={str(value):index for index,value in enumerate(ids)}
        for start in range(0,len(subset),64):
            scores=torch.as_tensor(qemb[start:start+64],device="cuda")@corpus_tensor.T
            values,indices=torch.topk(scores,k=TOP_K,dim=1)
            for offset in range(len(values)):
                item=subset.iloc[start+offset];top=indices[offset].cpu().tolist();target=lookup.get(str(item.target_document_id))
                rank=0 if target is None else int((scores[offset]>scores[offset,target]).sum().item()+1)
                rows.append({**item.to_dict(),"retriever":retriever,
                  "retrieved_document_ids":json.dumps([ids[index] for index in top]),
                  "retrieved_scores":json.dumps([float(value) for value in values[offset].cpu().tolist()]),"target_rank":rank})
            checkpoint(f"{retriever}_RETRIEVAL_PROGRESS",dataset=dataset,completed=min(start+64,len(subset)),total=len(subset))
        del corpus_tensor,qemb,emb;gc.collect();torch.cuda.empty_cache()
    del model;gc.collect();torch.cuda.empty_cache()
    output=pd.DataFrame(rows)
    if len(output)!=len(frame) or output.case_id.nunique()!=len(frame):raise RuntimeError(f"{retriever} retrieval row mismatch")
    atomic_csv(output,path,"gzip")
    target=output[output.member.eq(1)]
    audit={"retriever":retriever,"checkpoint":str(RETRIEVERS[retriever]),"queries":len(output),
      "top_k":TOP_K,"similarity":"cosine","target_top4_rate":float(target.target_rank.between(1,4).mean()),
      "corpus_embedding_seconds":corpus_seconds,"query_and_search_seconds":query_seconds,"total_seconds":time.perf_counter()-started}
    atomic_json(ROOT/f"audits/{retriever}_RETRIEVAL_AUDIT.json",audit)
    checkpoint(f"{retriever}_RETRIEVAL_COMPLETE",queries=len(output),target_top4_rate=audit["target_top4_rate"])
    return output


def qll_scores(retriever,retrieval,corpora):
    output_path=ROOT/f"private/{retriever}_QLL_CASES.private.csv.gz"
    if output_path.exists():return pd.read_csv(output_path,keep_default_na=False,low_memory=False,
                                               dtype={"row_id":str,"case_id":str,"session_id":str,"target_document_id":str})
    qll=load_module(f"exp212_qll_{retriever}",EXP179_CODE)
    cell=ROOT/retriever.lower();cell.mkdir(exist_ok=True)
    qll.ROOT=cell;qll.DB=ROOT/f"private/{retriever}_QLL.sqlite3";qll.PRIOR_DB=ROOT/"private/NO_PRIOR_QLL.sqlite3";qll.QWEN=QWEN
    qll.checkpoint=lambda stage,**details:checkpoint(f"{retriever}_QLL_{stage}",**details)
    documents={}
    for dataset,(ids,texts) in corpora.items():documents.update({(dataset,str(did)):text for did,text in zip(ids,texts)})
    rows=[]
    for item in retrieval.itertuples(index=False):
        query_hash=sha256_text(item.query);source_ids=list(map(str,json.loads(item.retrieved_document_ids)))
        for rank,source_id in enumerate(source_ids,1):
            rows.append({"task_id":sha256_text(f"{retriever}\0{item.case_id}\0{source_id}\0{query_hash}"),
              "case_id":str(item.case_id),"attack_family":str(item.attack_family),"cohort":str(item.kind),
              "dataset":str(item.dataset),"member":int(item.member),"query":str(item.query),"query_sha256":query_hash,
              "source_id":source_id,"source_rank":rank,"target_document_id":str(item.target_document_id),
              "target_rank":int(item.target_rank),"is_labeled_source":source_id==str(item.target_document_id)})
    tasks=pd.DataFrame(rows)
    if len(tasks)!=4*len(retrieval) or tasks.task_id.nunique()!=len(tasks):raise RuntimeError(f"{retriever} QLL task identity failure")
    started=time.perf_counter();scored=qll.score_tasks(tasks,documents);source_scores,case_scores=qll.case_scores(scored)
    atomic_csv(source_scores,ROOT/f"private/{retriever}_QLL_SOURCE_SCORES.private.csv.gz","gzip")
    merged=retrieval.merge(case_scores[["case_id","qll_top1_source","qll_top2_source","margin","dominance","entropy","labeled_source_qll_rank"]],on="case_id",validate="one_to_one")
    normal=merged[merged.kind.eq("BENIGN")].dominance.to_numpy(float)
    protocol=float(np.quantile(normal,.95,method="higher"))
    thresholds={"retriever":retriever,"strict_numeric_threshold":THRESHOLD,
      "benign_only_protocol_threshold":protocol,"quantile":.95,"method":"higher","comparison":"strict greater than",
      "benign_examples":len(normal),"attack_examples":0,"strict_benign_exceed_rate":float(np.mean(normal>THRESHOLD)),
      "protocol_benign_exceed_rate":float(np.mean(normal>protocol)),"frozen_before_attack_response_generation":True}
    atomic_json(ROOT/f"audits/{retriever}_THRESHOLD_FREEZE.json",thresholds)
    atomic_csv(merged,output_path,"gzip")
    atomic_json(ROOT/f"efficiency/{retriever}_QLL.json",{"pairs":len(tasks),"seconds":time.perf_counter()-started,
      "pairs_per_second":len(tasks)/max(time.perf_counter()-started,1e-9)})
    checkpoint(f"{retriever}_QLL_COMPLETE",cases=len(merged),pairs=len(tasks),protocol_threshold=protocol,
               attack_calibration_examples=0)
    return merged


def waterfill(lengths,hidden=None):
    lengths=np.asarray(lengths,int);caps=np.zeros(len(lengths),int);remaining=TOTAL_SOURCE_TOKENS
    active=[index for index in range(len(lengths)) if index!=hidden and lengths[index]>0]
    while remaining>0 and active:
        share=max(1,remaining//len(active));changed=False
        for index in list(active):
            add=min(share,int(lengths[index]-caps[index]),remaining)
            if add>0:caps[index]+=add;remaining-=add;changed=True
            if caps[index]>=lengths[index]:active.remove(index)
            if remaining<=0:break
        if not changed:break
    return caps.tolist()


def build_packing(retriever,scored,corpora):
    path=ROOT/f"private/{retriever}_PACKING.private.pkl.gz"
    if path.exists():return pd.read_pickle(path,compression="gzip")
    from transformers import AutoTokenizer
    sys.path.insert(0,str(EXP87/"code"));import exp87_models as models
    tokenizer=AutoTokenizer.from_pretrained(QWEN,local_files_only=True)
    documents={}
    for dataset,(ids,texts) in corpora.items():documents.update({(dataset,str(did)):text for did,text in zip(ids,texts)})
    threshold_info=json.loads((ROOT/f"audits/{retriever}_THRESHOLD_FREEZE.json").read_text())
    conditions={"STRICT":THRESHOLD,"PROTOCOL":float(threshold_info["benign_only_protocol_threshold"])}
    rows=[]
    def add(item,condition,threshold,baseline=False):
        sources=list(map(str,json.loads(item.retrieved_document_ids)));texts=[documents[(str(item.dataset),source)] for source in sources]
        lengths=[len(tokenizer(text,add_special_tokens=False).input_ids) for text in texts]
        trigger=False if baseline else float(item.dominance)>float(threshold)
        selected="" if baseline else str(item.qll_top1_source)
        hidden=sources.index(selected) if trigger and selected in sources else None
        caps=waterfill(lengths,hidden);visible=[];used=[]
        for text,cap in zip(texts,caps):
            ids=tokenizer(text,add_special_tokens=False).input_ids[:int(cap)];used.append(len(ids))
            if ids:visible.append(tokenizer.decode(ids,skip_special_tokens=True).strip())
        prompt=models.normal_prompt(str(item.query),visible)
        rows.append({**item._asdict(),"condition":condition,"threshold":float(threshold) if threshold is not None else math.nan,
          "trigger":trigger,"intervened":hidden is not None,"selected_source_id":selected if hidden is not None else "",
          "selected_source_rank":hidden+1 if hidden is not None else 0,"ledger_active":False,"ledger_source_id":"",
          "source_ids_used":json.dumps([source for index,source in enumerate(sources) if index!=hidden]),
          "caps":json.dumps(caps),"tokens_used":json.dumps(used),"total_context_tokens":int(sum(used)),
          "prompt":prompt,"max_new_tokens":MAX_NEW[str(item.attack_family)],
          "generation_row_id":f"{retriever}|{condition}|{item.case_id}"})
    for item in scored.itertuples(index=False):
        for condition,threshold in conditions.items():add(item,condition,threshold)
        if str(item.kind)=="BENIGN":add(item,"NO_DEFENSE",None,True)
    output=pd.DataFrame(rows)
    if output.total_context_tokens.gt(TOTAL_SOURCE_TOKENS).any():raise RuntimeError("source token budget exceeded")
    for condition,threshold in conditions.items():
        cell=output[output.condition.eq(condition)]
        if not cell.intervened.eq(cell.dominance.astype(float).gt(threshold)).all():raise RuntimeError(f"{retriever} {condition} strict policy mismatch")
    output.to_pickle(path,compression="gzip")
    public=output.drop(columns=["query","prompt"]);atomic_csv(public,ROOT/f"tables/{retriever}_PACKING_AUDIT.csv","gzip")
    diagnostics=[]
    for condition,cell in output[output.condition.ne("NO_DEFENSE")].groupby("condition"):
        diagnostics.append({"condition":condition,"threshold":float(cell.threshold.iloc[0]),"hide_rate":float(cell.intervened.mean()),
          "normal_hide_rate":float(cell[cell.kind.eq("BENIGN")].intervened.mean()),"qll_top1_target_alignment":float(cell[cell.member.eq(1)].qll_top1_source.astype(str).eq(cell[cell.member.eq(1)].target_document_id.astype(str)).mean())})
    atomic_csv(pd.DataFrame(diagnostics),ROOT/f"tables/{retriever}_QLL_MECHANISM.csv")
    checkpoint(f"{retriever}_PACKING_COMPLETE",rows=len(output),candidate_rows=2*len(scored),normal_baselines=1000)
    return output


def generator_module():
    sys.path.insert(0,str(EXP87/"code"));import exp87_models as models
    models.ROOT=ROOT;models.heartbeat=lambda stage,**details:checkpoint(stage,**details);models.GENERATION_CONFIG["batch_size"]=16
    return models


def generate_rows(retriever,work,path,label,expand_budget=False):
    if path.exists():return pd.read_csv(path,keep_default_na=False,low_memory=False,
                                        dtype={"row_id":str,"case_id":str,"session_id":str,"target_document_id":str})
    models=generator_module();rows=[]
    if expand_budget:
        for item in work.itertuples(index=False):
            for budget in BUDGETS:rows.append({**item._asdict(),"output_budget":budget,"requested_max_new_tokens":budget})
    else:
        rows=[{**item._asdict(),"output_budget":int(item.max_new_tokens),"requested_max_new_tokens":int(item.max_new_tokens)} for item in work.itertuples(index=False)]
    output=pd.DataFrame(rows)
    output["task_signature"]=[sha256_text(f"{system}\0{prompt}\0{budget}") for system,prompt,budget in zip(output.system_prompt,output.prompt,output.output_budget)]
    unique=output.drop_duplicates("task_signature").sort_values("task_signature")
    tasks=[models.make_task(task_type=f"EXP212_{retriever}_{label}",row_id=str(item.task_signature),prompt=str(item.prompt),
                            system_prompt=str(item.system_prompt),max_new_tokens=int(item.output_budget)) for item in unique.itertuples(index=False)]
    checkpoint(f"{retriever}_{label}_GENERATION_STARTED",logical_rows=len(output),unique_prompts=len(tasks),device="cuda:0")
    started=time.perf_counter();answers=models.run_generation(tasks,ROOT/f"private/{retriever}_QWEN_RESPONSES.sqlite3",None,
                                                               f"{retriever}_{label}_GENERATION_PROGRESS")
    elapsed=time.perf_counter()-started;output["response"]=output.task_signature.map(answers)
    if output.response.isna().any() or output.response.eq("").any():raise RuntimeError(f"{retriever} {label} generation incomplete")
    atomic_csv(output,path,"gzip");atomic_json(ROOT/f"efficiency/{retriever}_{label}_GENERATION.json",
      {"logical_rows":len(output),"unique_forwards":len(tasks),"seconds":elapsed,"forwards_per_second":len(tasks)/max(elapsed,1e-9)})
    checkpoint(f"{retriever}_{label}_GENERATION_COMPLETE",logical_rows=len(output),unique_forwards=len(tasks),seconds=elapsed)
    return output


def generate_stage_a(retriever,packing):
    path=ROOT/f"private/{retriever}_STAGE_A_RESPONSES.private.csv.gz"
    if path.exists():return pd.read_csv(path,keep_default_na=False,low_memory=False,
                                        dtype={"row_id":str,"case_id":str,"session_id":str,"target_document_id":str})
    regular=packing[((packing.kind.eq("BENIGN"))) | ((packing.attack_family.isin(STAGE_A_NATIVE+("RAGLeak",))))].copy()
    budget=packing[packing.attack_family.eq("BudgetLeak-Z") & packing.condition.ne("NO_DEFENSE")].copy()
    regular_answers=generate_rows(retriever,regular,ROOT/f"private/{retriever}_STAGE_A_REGULAR.private.csv.gz","STAGE_A_REGULAR")
    budget_answers=generate_rows(retriever,budget,ROOT/f"private/{retriever}_STAGE_A_BUDGET.private.csv.gz","STAGE_A_BUDGET",True)
    output=pd.concat([regular_answers,budget_answers],ignore_index=True);atomic_csv(output,path,"gzip")
    return output


def native_scores_cell(retriever,condition,answers,source,sessions,families,cell_root):
    evaluator=load_module(f"exp212_eval_{retriever}_{condition}_{'_'.join(families)}",EXP211_CODE)
    evaluator.ROOT=cell_root;evaluator.DATASET="FiQA-2018";evaluator.MPNET=MPNET
    evaluator.checkpoint=lambda stage,**details:checkpoint(f"{retriever}_{condition}_{stage}",**details)
    return evaluator.native_scores(answers,source,sessions,families,cell_root/"private/NATIVE_SCORES.private.csv.gz")


def privacy_metric_rows(scores,packing):
    rows=[]
    for family,group in scores.groupby("attack_family",sort=True):
        valid=group.dropna(subset=["attack_score"]);raw,effective=effective_auc(valid.member,valid.attack_score)
        actions=packing[packing.attack_family.eq(family)].groupby("session_id").agg(member=("member","first"),hide=("intervened","mean"))
        rates=actions.groupby("member").hide.mean().to_dict()
        rows.append({"Attack":family,"Raw AUC":raw,"E-AUC":effective,"N member":int((valid.member==1).sum()),
          "N nonmember":int((valid.member==0).sum()),"Hide rate member":float(rates.get(1,math.nan)),
          "Hide rate nonmember":float(rates.get(0,math.nan)),"Verdict":"PASS" if len(valid)==len(group) and effective<=.65 else "FAIL"})
    return rows


def ragleak_score_cell(answers,cell_root):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(str(MPNET),device="cuda",local_files_only=True)
    left=model.encode(answers.response.astype(str).tolist(),normalize_embeddings=True,convert_to_numpy=True,batch_size=128,show_progress_bar=False)
    right=model.encode(answers.reference.astype(str).tolist(),normalize_embeddings=True,convert_to_numpy=True,batch_size=128,show_progress_bar=False)
    scores=answers[["session_id","member"]].copy();scores["attack_score"]=np.sum(left*right,axis=1)
    raw,effective=effective_auc(scores.member,scores.attack_score);rates=answers.groupby("member").intervened.mean().to_dict()
    atomic_csv(scores,cell_root/"tables/RAGLEAK_SCORES.csv")
    del model,left,right;gc.collect()
    return {"Attack":"RAGLeak","Raw AUC":raw,"E-AUC":effective,"N member":int((scores.member==1).sum()),
      "N nonmember":int((scores.member==0).sum()),"Hide rate member":float(rates.get(1,math.nan)),
      "Hide rate nonmember":float(rates.get(0,math.nan)),"Verdict":"PASS" if effective<=.65 else "FAIL"}


def budget_task_ids(row_ids,output_budgets):
    return [f"{row_id}|B{int(budget)}" for row_id,budget in zip(row_ids,output_budgets)]


def budget_score_cell(retriever,condition,answers,before,cell_root):
    # Stage204's immutable scorer identifies every logical output-budget row as
    # ``<row_id>|B<budget>``. Exp212 deduplicates generator forwards with a
    # separate task_signature, so reconstruct only the scorer-facing identity.
    answers=answers.copy()
    answers["task_id"]=budget_task_ids(answers.row_id,answers.output_budget)
    if answers.task_id.nunique()!=len(answers):raise RuntimeError("BudgetLeak scorer task identity is not unique")
    rank_counts=answers.drop_duplicates("row_id").query("member == 1").target_rank.value_counts().to_dict()
    coverage=all(int(rank_counts.get(rank,0))>0 for rank in (1,2,3,4))
    atomic_json(cell_root/"audits/BUDGETLEAK_RANK_COVERAGE.json",{"member_rank_counts":{str(k):int(v) for k,v in rank_counts.items()},
      "required_ranks":[1,2,3,4],"passed":coverage})
    if not coverage:
        return {"verdict":"BUDGETLEAK_RANK_COVERAGE_INSUFFICIENT","passed":False,"overall_raw_auc":math.nan,
          "overall_effective_auc":math.nan,"rank_effective_auc":{},"worst_rank_effective_auc":math.nan,
          "worst_output_budget_effective_auc":math.nan,"worst_output_budget":None,"member_rank_counts":rank_counts}
    scorer=load_module(f"exp212_budget_{retriever}_{condition}",STAGE204_CODE)
    root=cell_root/"budgetleak";[(root/name).mkdir(parents=True,exist_ok=True) for name in ("private","tables","audits","figures","provenance","reports","checkpoints")]
    scorer.ROOT=root;scorer.METRICS=root/"private/BUDGETLEAK_METRICS.private.csv.gz";scorer.SCORES=root/"private/BUDGETLEAK_SCORES.private.csv.gz"
    scorer.THRESHOLD=float(answers.threshold.iloc[0]);scorer.checkpoint=lambda stage,**details:checkpoint(f"{retriever}_{condition}_BUDGET_{stage}",**details)
    metrics=scorer.response_metrics(answers);result=scorer.score_and_evaluate(metrics,answers,before)
    ranks=pd.read_csv(root/"tables/BUDGETLEAK_BY_TARGET_RANK.csv");budgets=pd.read_csv(root/"tables/BUDGETLEAK_BY_OUTPUT_BUDGET.csv")
    result["rank_effective_auc"]={str(int(row.target_rank)):float(row.effective_auc) for row in ranks.itertuples(index=False)}
    result["worst_output_budget_effective_auc"]=float(budgets.effective_auc.max())
    result["member_rank_counts"]={str(k):int(v) for k,v in rank_counts.items()};return result


def budget_privacy_row(result,answers):
    rates=answers.drop_duplicates("row_id").groupby("member").intervened.mean().to_dict()
    effective=float(result.get("overall_effective_auc",math.nan));raw=float(result.get("overall_raw_auc",math.nan))
    return {"Attack":"BudgetLeak-Z","Raw AUC":raw,"E-AUC":effective,"N member":900,"N nonmember":900,
      "Hide rate member":float(rates.get(1,math.nan)),"Hide rate nonmember":float(rates.get(0,math.nan)),
      "Verdict":"PASS" if bool(result.get("passed",False)) else "FAIL"}


def benign_utility(retriever,condition,answers,cell_root):
    from sentence_transformers import SentenceTransformer
    candidate=answers[answers.condition.eq(condition)].set_index("case_id").sort_index()
    baseline=answers[answers.condition.eq("NO_DEFENSE")].set_index("case_id").sort_index()
    if len(candidate)!=1000 or len(baseline)!=1000 or not candidate.index.equals(baseline.index):raise RuntimeError("paired benign alignment failure")
    cand=candidate.response.astype(str);base=baseline.response.astype(str);hidden=candidate.intervened.astype(bool).to_numpy()
    f1=np.asarray([token_f1(a,b) for a,b in zip(cand,base)]);exact=cand.eq(base).to_numpy()
    new_ref=np.asarray([refusal(a) and not refusal(b) for a,b in zip(cand,base)])
    model=SentenceTransformer(str(MPNET),device="cuda",local_files_only=True)
    a=model.encode(cand.tolist(),normalize_embeddings=True,convert_to_numpy=True,batch_size=128,show_progress_bar=False)
    b=model.encode(base.tolist(),normalize_embeddings=True,convert_to_numpy=True,batch_size=128,show_progress_bar=False)
    semantic=np.sum(a*b,axis=1);del model,a,b;gc.collect()
    detail=pd.DataFrame({"case_id":candidate.index,"intervened":hidden,"token_f1":f1,"semantic_similarity":semantic,
      "exact_match":exact,"new_refusal":new_ref,"answer_changed":~exact})
    atomic_csv(detail,cell_root/"tables/BENIGN_PAIRED_UTILITY.csv")
    result={"queries":1000,"relative_token_f1":float(f1.mean()),"exact_match":float(exact.mean()),
      "semantic_preservation":float(semantic.mean()),"answer_change_rate":float((~exact).mean()),
      "hide_rate":float(hidden.mean()),"new_refusal_rate":float(new_ref.mean()),"hidden_queries":int(hidden.sum()),
      "hidden_token_f1":float(f1[hidden].mean()) if hidden.any() else 1.0,
      "hidden_semantic_similarity":float(semantic[hidden].mean()) if hidden.any() else 1.0,
      "hidden_new_refusal_rate":float(new_ref[hidden].mean()) if hidden.any() else 0.0}
    result["passed"]=bool(result["relative_token_f1"]>=.95 and result["hide_rate"]<=.10 and result["new_refusal_rate"]<=.05)
    atomic_json(cell_root/"audits/BENIGN_UTILITY.json",result);return result


def observable_sidechannel(native,rag,budget,cell_root):
    rows=[]
    def record(attack,feature,labels,values):
        raw,effective=effective_auc(labels,values);rows.append({"attack":attack,"feature":feature,"raw_auc":raw,"effective_auc":effective,"n":len(labels)})
    def features(frame):
        output=frame.copy();output["length"]=output.response.astype(str).map(lambda text:len(TOKEN_RE.findall(text)))
        output["refusal"]=output.response.map(refusal).astype(int);output["empty"]=output.response.astype(str).str.strip().eq("").astype(int);return output
    native=features(native)
    for family,group in native.groupby("attack_family"):
        session=group.sort_values("turn").groupby("session_id").agg(member=("member","first"),mean_length=("length","mean"),
          first_length=("length","first"),last_length=("length","last"),refusal_count=("refusal","sum"),empty_count=("empty","sum"))
        session["length_delta"]=session.last_length-session.first_length
        for feature in ("mean_length","first_length","last_length","length_delta","refusal_count","empty_count"):record(family,feature,session.member,session[feature])
    rag=features(rag)
    for feature in ("length","refusal","empty"):record("RAGLeak",feature,rag.member,rag[feature])
    budget=features(budget)
    for output_budget,group in budget.groupby("output_budget"):
        for feature in ("length","refusal","empty"):record(f"BudgetLeak-B{output_budget}",feature,group.member,group[feature])
    table=pd.DataFrame(rows);atomic_csv(table,cell_root/"tables/OBSERVABLE_SIDECHANNEL.csv")
    return {"worst_effective_auc":float(table.effective_auc.max()),"worst_attack":str(table.loc[table.effective_auc.idxmax(),"attack"]),
      "worst_feature":str(table.loc[table.effective_auc.idxmax(),"feature"]),"diagnostic_pass_at_0_65":bool(table.effective_auc.max()<=.65)}


def evaluate_stage_a(retriever,condition,responses,source,sessions,before):
    cell_root=ROOT/retriever.lower()/condition.lower();[(cell_root/name).mkdir(parents=True,exist_ok=True) for name in ("private","tables","audits","reports")]
    candidate=responses[responses.condition.eq(condition)].copy();baseline=responses[responses.condition.eq("NO_DEFENSE")].copy()
    native=candidate[candidate.attack_family.isin(STAGE_A_NATIVE)].copy();rag=candidate[candidate.attack_family.eq("RAGLeak")].copy();budget=candidate[candidate.attack_family.eq("BudgetLeak-Z")].copy()
    scores=native_scores_cell(retriever,condition,native,source,sessions,STAGE_A_NATIVE,cell_root)
    privacy=privacy_metric_rows(scores,native);privacy.append(ragleak_score_cell(rag,cell_root))
    budget_result=budget_score_cell(retriever,condition,budget,before,cell_root);privacy.append(budget_privacy_row(budget_result,budget))
    utility=benign_utility(retriever,condition,pd.concat([candidate[candidate.kind.eq("BENIGN")],baseline],ignore_index=True),cell_root)
    observable=observable_sidechannel(native,rag,budget,cell_root)
    privacy_pass=all(row["Verdict"]=="PASS" for row in privacy)
    passed=bool(privacy_pass and utility["passed"])
    result={"retriever":retriever,"condition":condition,"threshold":float(candidate.threshold.iloc[0]),"privacy":privacy,
      "budgetleak":budget_result,"benign":utility,"observable":observable,"privacy_pass":privacy_pass,"utility_pass":utility["passed"],
      "hard_screen_pass":passed,"attack_calibration_examples":0}
    atomic_csv(pd.DataFrame(privacy),cell_root/"tables/STAGE_A_PRIVACY.csv");atomic_json(cell_root/"STAGE_A_RESULT.json",result)
    checkpoint(f"{retriever}_{condition}_STAGE_A_COMPLETE",privacy_pass=privacy_pass,utility_pass=utility["passed"],hard_screen_pass=passed)
    return result


def generate_and_evaluate_stage_b(retriever,conditions,packing,source,sessions):
    if not conditions:return {}
    path=ROOT/f"private/{retriever}_STAGE_B_RESPONSES.private.csv.gz"
    work=packing[packing.condition.isin(conditions)&packing.attack_family.isin(STAGE_B_NATIVE)].copy()
    responses=generate_rows(retriever,work,path,"STAGE_B")
    results={}
    for condition in conditions:
        cell_root=ROOT/retriever.lower()/condition.lower();answers=responses[responses.condition.eq(condition)].copy()
        scores=native_scores_cell(retriever,condition,answers,source,sessions,STAGE_B_NATIVE,cell_root)
        privacy=privacy_metric_rows(scores,answers);passed=all(row["Verdict"]=="PASS" for row in privacy)
        atomic_csv(pd.DataFrame(privacy),cell_root/"tables/STAGE_B_PRIVACY.csv")
        results[condition]={"privacy":privacy,"passed":passed}
        checkpoint(f"{retriever}_{condition}_STAGE_B_COMPLETE",attacks=len(privacy),passed=passed)
    return results


def diagnostics(retriever,packing):
    base=packing[packing.condition.eq("STRICT")].copy();members=base[(base.member.eq(1))&base.kind.ne("BENIGN")]
    rows=[]
    for family,group in members.groupby("attack_family",sort=True):
        rows.append({"attack_family":family,"member_query_rows":len(group),"target_top4_rate":float(group.target_rank.between(1,4).mean()),
          "target_rank1_rate":float(group.target_rank.eq(1).mean()),"target_rank2_rate":float(group.target_rank.eq(2).mean()),
          "target_rank3_rate":float(group.target_rank.eq(3).mean()),"target_rank4_rate":float(group.target_rank.eq(4).mean()),
          "qll_top1_target_alignment":float(group.qll_top1_source.astype(str).eq(group.target_document_id.astype(str)).mean()),
          "dominance_mean":float(group.dominance.mean()),"dominance_std":float(group.dominance.std(ddof=1))})
    table=pd.DataFrame(rows);atomic_csv(table,ROOT/f"tables/{retriever}_SOURCE_POSITION_AUDIT.csv")
    normal=base[base.kind.eq("BENIGN")]
    attack=base[base.kind.ne("BENIGN")]
    mechanism={"benign_dominance_mean":float(normal.dominance.mean()),"benign_dominance_std":float(normal.dominance.std(ddof=1)),
      "benign_dominance_q95":float(np.quantile(normal.dominance,.95,method="higher")),
      "attack_member_dominance_mean":float(attack[attack.member.eq(1)].dominance.mean()),
      "attack_nonmember_dominance_mean":float(attack[attack.member.eq(0)].dominance.mean()),
      "member_target_top4_rate":float(members.target_rank.between(1,4).mean()),
      "qll_top1_target_alignment":float(members.qll_top1_source.astype(str).eq(members.target_document_id.astype(str)).mean())}
    atomic_json(ROOT/f"audits/{retriever}_MECHANISM_DIAGNOSTICS.json",mechanism);return mechanism


def run_retriever(retriever,frame,corpora,source,sessions,before):
    retrieval=retrieve(retriever,frame,corpora);scored=qll_scores(retriever,retrieval,corpora);packing=build_packing(retriever,scored,corpora)
    mechanism=diagnostics(retriever,packing);responses=generate_stage_a(retriever,packing)
    stage_a={condition:evaluate_stage_a(retriever,condition,responses,source,sessions,before) for condition in ("STRICT","PROTOCOL")}
    opened=[condition for condition in ("STRICT","PROTOCOL") if stage_a[condition]["hard_screen_pass"]]
    stage_b=generate_and_evaluate_stage_b(retriever,opened,packing,source,sessions)
    condition_results={}
    for condition in ("STRICT","PROTOCOL"):
        full_rows=list(stage_a[condition]["privacy"])
        stage_b_pass=False
        if condition in stage_b:
            full_rows.extend(stage_b[condition]["privacy"]);stage_b_pass=bool(stage_b[condition]["passed"])
        final_pass=bool(stage_a[condition]["hard_screen_pass"] and stage_b_pass)
        numeric=[float(row["E-AUC"]) for row in full_rows if math.isfinite(float(row["E-AUC"]))]
        condition_results[condition]={"screen":stage_a[condition],"stage_b":stage_b.get(condition),"full_privacy":full_rows,
          "mean_effective_auc":float(np.mean(numeric)) if numeric else math.nan,"worst_effective_auc":max(numeric) if numeric else math.nan,
          "final_pass":final_pass,"verdict":f"{condition}_TRANSFER_PASS" if final_pass else f"{condition}_TRANSFER_FAIL"}
        atomic_csv(pd.DataFrame(full_rows),ROOT/retriever.lower()/condition.lower()/"tables/FULL_PRIVACY.csv")
    if condition_results["STRICT"]["final_pass"]:verdict=f"EXP212_{retriever}_STRICT_TRANSFER_PASS";level="STRICT"
    elif condition_results["PROTOCOL"]["final_pass"]:verdict=f"EXP212_{retriever}_PROTOCOL_TRANSFER_PASS";level="PROTOCOL"
    else:verdict=f"EXP212_{retriever}_TRANSFER_FAILED";level="FAILED"
    retrieval_audit=json.loads((ROOT/f"audits/{retriever}_RETRIEVAL_AUDIT.json").read_text())
    result={"retriever":retriever,"checkpoint":str(RETRIEVERS[retriever]),"verdict":verdict,"transfer_level":level,
      "strict_numeric_threshold":THRESHOLD,"benign_95_threshold":json.loads((ROOT/f"audits/{retriever}_THRESHOLD_FREEZE.json").read_text())["benign_only_protocol_threshold"],
      "attack_examples_used_for_calibration":0,"conditions":condition_results,"mechanism":mechanism,
      "retrieval":retrieval_audit,"ia_status":"IA_EVALUATION_PROTOCOL_UNRESOLVED","new_trainable_parameters":0}
    atomic_json(ROOT/f"{retriever}_FINAL_RESULT.json",result);checkpoint(f"{retriever}_COMPLETE",verdict=verdict,transfer_level=level)
    return result


def frozen_postrun(before):
    output=before.copy();output["sha256_after"]=[sha256_file(path) for path in output.path];output["unchanged"]=output.sha256_before.eq(output.sha256_after)
    atomic_csv(output,ROOT/"provenance/FROZEN_INPUTS_POSTRUN.csv")
    if not output.unchanged.all():raise RuntimeError("frozen input mutation")
    return True


def efficiency_summary():
    rows=[]
    for retriever in RETRIEVERS:
        retrieval=json.loads((ROOT/f"audits/{retriever}_RETRIEVAL_AUDIT.json").read_text())
        qll=json.loads((ROOT/f"efficiency/{retriever}_QLL.json").read_text());generation=[]
        for path in (ROOT/"efficiency").glob(f"{retriever}_*_GENERATION.json"):generation.append(json.loads(path.read_text()))
        forwards=sum(int(item["unique_forwards"]) for item in generation);seconds=sum(float(item["seconds"]) for item in generation)
        rows.append({"retriever":retriever,"retrieval_seconds":retrieval["total_seconds"],"qll_seconds":qll["seconds"],
          "generation_seconds":seconds,"unique_generation_forwards":forwards,"generation_throughput":forwards/max(seconds,1e-9),
          "qll_forwards_per_query":4,"generation_per_runtime_query":1,"trainable_parameters":0,"session_state":0})
    table=pd.DataFrame(rows);atomic_csv(table,ROOT/"tables/EFFICIENCY.csv");return table


def final_report(results,campaign,efficiency):
    lines=["# Exp212 Retriever Transfer 최종 결과","",f"- Campaign verdict: **{campaign}**",
      f"- Strict numeric threshold: **{THRESHOLD}**","- Attack examples used for calibration: **0**",
      "- New trainable parameters: **0**","- Session state: **0**","- IA: **IA_EVALUATION_PROTOCOL_UNRESOLVED**",""]
    for retriever,result in results.items():
        selected="STRICT" if result["transfer_level"]=="STRICT" else "PROTOCOL"
        lines.extend([f"## {retriever}","",f"- Verdict: **{result['verdict']}**",f"- Exact checkpoint: `{result['checkpoint']}`",
          f"- Benign-only 95% threshold: **{result['benign_95_threshold']:.10f}**",
          f"- Member target Top-4 retrieval: **{100*result['mechanism']['member_target_top4_rate']:.2f}%**",
          f"- QLL top1-target alignment: **{100*result['mechanism']['qll_top1_target_alignment']:.2f}%**",""])
        for condition in ("STRICT","PROTOCOL"):
            cell=result["conditions"][condition];screen=cell["screen"];benign=screen["benign"];budget=screen["budgetleak"]
            privacy={row["Attack"]:row for row in cell["full_privacy"]}
            ranks=budget.get("rank_effective_auc",{})
            lines.extend([f"### {condition}","",f"- Final pass: **{cell['final_pass']}**",
              f"- DCMI / RAGLeak / S²-MIA: **{privacy.get('DCMI',{}).get('E-AUC',math.nan):.4f} / {privacy.get('RAGLeak',{}).get('E-AUC',math.nan):.4f} / {privacy.get('S²-MIA',{}).get('E-AUC',math.nan):.4f}**",
              f"- BudgetLeak overall: **{privacy.get('BudgetLeak-Z',{}).get('E-AUC',math.nan):.4f}**",
              f"- BudgetLeak rank 1–4: **{ranks.get('1',math.nan):.4f} / {ranks.get('2',math.nan):.4f} / {ranks.get('3',math.nan):.4f} / {ranks.get('4',math.nan):.4f}**",
              f"- MEntA / RAG-MIA / MBA: **{privacy.get('MEntA',{}).get('E-AUC',math.nan):.4f} / {privacy.get('RAG-MIA',{}).get('E-AUC',math.nan):.4f} / {privacy.get('MBA',{}).get('E-AUC',math.nan):.4f}**",
              f"- Mean / worst E-AUC: **{cell['mean_effective_auc']:.4f} / {cell['worst_effective_auc']:.4f}**",
              f"- Benign hide / relative Token-F1 / new refusal: **{100*benign['hide_rate']:.2f}% / {benign['relative_token_f1']:.4f} / {100*benign['new_refusal_rate']:.2f}%**",
              f"- Hidden-benign Token-F1 / semantic / new refusal: **{benign['hidden_token_f1']:.4f} / {benign['hidden_semantic_similarity']:.4f} / {100*benign['hidden_new_refusal_rate']:.2f}%**",
              f"- Observable-only worst E-AUC: **{screen['observable']['worst_effective_auc']:.4f}**",""])
    lines.extend(["## Efficiency","",efficiency.to_markdown(index=False),"",f"- Exp213 may open: **{all(result['transfer_level']!='FAILED' for result in results.values())}**",
      "","Strict와 benign-only protocol 결과를 분리했다. 공격 예시는 threshold 계산에 사용하지 않았고, retriever 결과를 본 뒤 QLL·Top-K·hide·water-fill을 수정하지 않았다."])
    atomic_text(ROOT/"reports/EXP212_FINAL_REPORT_KO.md","\n".join(lines)+"\n")


def main():
    for name in ("code","configs","tests","scripts","logs","checkpoints","provenance","audits","reports","tables","private","efficiency","gte","bge"):(ROOT/name).mkdir(parents=True,exist_ok=True)
    try:
        before=preflight();source,sessions,external=load_source_sessions_external();frame=build_query_frame(source,sessions,external);corpora=load_corpora(source,sessions)
        results={}
        for retriever in ("GTE","BGE"):results[retriever]=run_retriever(retriever,frame,corpora,source,sessions,before)
        levels={key:value["transfer_level"] for key,value in results.items()}
        if all(value=="STRICT" for value in levels.values()):campaign="QLL_SOURCE_HIDE_CROSS_RETRIEVER_STRICT_GENERALIZATION"
        elif set(levels.values())=={"STRICT","PROTOCOL"}:campaign="QLL_SOURCE_HIDE_CROSS_RETRIEVER_PROTOCOL_GENERALIZATION"
        elif all(value in {"STRICT","PROTOCOL"} for value in levels.values()):campaign="QLL_SOURCE_HIDE_CALIBRATION_PROTOCOL_GENERALIZES"
        else:campaign="QLL_SOURCE_HIDE_RETRIEVER_DEPENDENCE_OBSERVED"
        unchanged=frozen_postrun(before);efficiency=efficiency_summary();exp213=all(value!="FAILED" for value in levels.values())
        final={"campaign_verdict":campaign,"retrievers":results,"strict_numeric_threshold":THRESHOLD,
          "attack_examples_used_for_calibration":0,"frozen_inputs_unchanged":unchanged,"new_trainable_parameters":0,
          "attack_specific_thresholds":0,"rank_specific_thresholds":0,"family_specific_rules":0,"score_fusion":0,
          "learned_detector":0,"top_k_search":0,"hide_count_search":0,"ia_status":"IA_EVALUATION_PROTOCOL_UNRESOLVED",
          "exp213_may_open":exp213,"efficiency":efficiency.to_dict("records")}
        atomic_json(ROOT/"FINAL_RESULT.json",final);final_report(results,campaign,efficiency)
        checkpoint("COMPLETE_PASS" if exp213 else "COMPLETE_FAIL",campaign_verdict=campaign,exp213_may_open=exp213)
    except Exception as error:
        checkpoint("FAILED_EXCEPTION",error_type=type(error).__name__,error=str(error));raise


if __name__=="__main__":main()
