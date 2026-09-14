#!/usr/bin/env python3
"""Frozen strict and benign-refresh FinQA score-only transfer evaluation."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import BGE, EXP, K_LOCAL, LC, ROOT, atomic_json, checkpoint, read_jsonl, sha_file, sha_text, verify_hashed_json, write_csv, write_jsonl

sys.path.insert(0,str(ROOT/"code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments


def threshold(values,alpha=.025):
    ordered=sorted(map(float,values),reverse=True);allowance=math.floor(alpha*len(ordered)+1e-12)
    value=ordered[allowance] if allowance<len(ordered) else -math.inf
    alarms=sum(x>value for x in ordered)
    if alarms>allowance:raise RuntimeError("threshold exceeds benign alarm budget")
    return value,alarms


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--scope",choices=("screen","all"),required=True);args=parser.parse_args()
    pre=verify_hashed_json(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json")
    for relative,digest in pre["code_sha256"].items():
        if sha_file(ROOT/relative)!=digest:raise RuntimeError(f"Phase-D code drift {relative}")
    queries=read_jsonl(EXP/"inputs"/f"FINQA_{args.scope.upper()}_ATTACK_QUERIES.jsonl")
    docs=read_jsonl(EXP/"inputs"/"FINQA_PROTECTED_DB.jsonl");reference=read_jsonl(EXP/"inputs"/"FINQA_BENIGN_REFERENCE_1000.jsonl");holdout=read_jsonl(EXP/"inputs"/"FINQA_BENIGN_HOLDOUT_1000.jsonl")
    old=read_jsonl(LC/"cache"/"LARGE_DETECTION_SCORES.jsonl");old_emb=np.asarray(np.load(LC/"cache"/"QUERY_EMBEDDINGS.float16.npy"),dtype=np.float32)
    old_idx=[i for i,row in enumerate(old) if row["split"]=="REFERENCE"]
    if len(old_idx)!=1000:raise RuntimeError("old benign reference drift")
    old_ref_emb=old_emb[old_idx];old_m=[float(old[i]["M"]) for i in old_idx];old_sorted=sorted(old_m)
    checkpoint("PHASE_D_BGE_LOADING",scope=args.scope,documents=len(docs),attack_queries=len(queries))
    model=SentenceTransformer(str(BGE),device="cuda",local_files_only=True);model.max_seq_length=512
    doc_emb=np.asarray(model.encode([row["source_text"] for row in docs],batch_size=32,normalize_embeddings=True,show_progress_bar=True),dtype=np.float32)
    ref_emb=np.asarray(model.encode([row["query"] for row in reference],batch_size=48,normalize_embeddings=True,show_progress_bar=True),dtype=np.float32)
    hold_emb=np.asarray(model.encode([row["query"] for row in holdout],batch_size=48,normalize_embeddings=True,show_progress_bar=True),dtype=np.float32)
    attack_emb=np.asarray(model.encode([row["query"] for row in queries],batch_size=48,normalize_embeddings=True,show_progress_bar=True),dtype=np.float32)
    del model;torch.cuda.empty_cache();doc_ids=[row["document_id"] for row in docs]

    def raw(rows,embeddings,label):
        output=[]
        for start in range(0,len(rows),64):
            scores_batch=embeddings[start:start+64]@doc_emb.T
            old_local=embeddings[start:start+64]@old_ref_emb.T
            for offset,scores in enumerate(scores_batch):
                row=rows[start+offset];idx=np.argpartition(-scores,4)[:4];idx=idx[np.argsort(-scores[idx],kind="stable")]
                ids=[doc_ids[int(i)] for i in idx];top=[float(scores[int(i)]) for i in idx]
                stat=canonical_mirabel_from_moments(top1=top[0],sum_all=float(scores.sum(dtype=np.float64)),sumsq_all=float(np.square(scores,dtype=np.float64).sum(dtype=np.float64)),corpus_size=len(docs),confidence=.95)
                sim=old_local[offset];neighbors=np.argpartition(-sim,K_LOCAL)[:K_LOCAL];local_sorted=sorted(old_m[int(i)] for i in neighbors)
                p=(1+len(local_sorted)-bisect.bisect_left(local_sorted,float(stat.margin)))/(K_LOCAL+1)
                target=row.get("target_id");rank=ids.index(target)+1 if target in ids else 0
                output.append({**row,"attack":row.get("attack","BENIGN"),"cohort":label,"top_document_ids":ids,"top_scores":top,"selected_source_id":ids[0],"target_rank":rank,
                    "M":float(stat.margin),"R_STRICT":-math.log(p),"strict_local_neighbor_hash":sha_text("\n".join(old[old_idx[int(i)]]["query_id"] for i in neighbors))})
            checkpoint("PHASE_D_RAW_SCORE_PROGRESS",scope=args.scope,cohort=label,completed=min(start+64,len(rows)),total=len(rows))
        return output
    ref_rows=raw(reference,ref_emb,"BENIGN_REFERENCE");hold_rows=raw(holdout,hold_emb,"BENIGN_HOLDOUT");attack_rows=raw(queries,attack_emb,"ATTACK")
    ref_m=[float(row["M"]) for row in ref_rows]
    for rows,embeddings in ((ref_rows,ref_emb),(hold_rows,hold_emb),(attack_rows,attack_emb)):
        similarity=embeddings@ref_emb.T
        for index,row in enumerate(rows):
            nearest=np.argpartition(-similarity[index],K_LOCAL)[:K_LOCAL]
            # For reference rows, exclude self and pull the next neighbor to avoid self-conditioning.
            if row["cohort"]=="BENIGN_REFERENCE":
                nearest=np.argpartition(-similarity[index],K_LOCAL+1)[:K_LOCAL+1]
                nearest=[int(i) for i in nearest if int(i)!=index][:K_LOCAL]
            local_sorted=sorted(ref_m[int(i)] for i in nearest)
            p=(1+len(local_sorted)-bisect.bisect_left(local_sorted,float(row["M"])))/(len(local_sorted)+1)
            row["R_REFRESH"]=-math.log(p);row["refresh_local_neighbor_hash"]=sha_text("\n".join(reference[int(i)]["query_id"] for i in nearest))
    strict_m=float(pre["strict_thresholds"]["MIRABEL@0.025"]["threshold"]);strict_lc=float(pre["strict_thresholds"]["Final LC@0.025"]["threshold"])
    refresh_m,m_count=threshold([row["M"] for row in hold_rows]);refresh_lc,lc_count=threshold([row["R_REFRESH"] for row in hold_rows])
    for row in hold_rows+attack_rows:
        row.update({"strict_mirabel_alarm":row["M"]>strict_m,"strict_lc_alarm":row["R_STRICT"]>strict_lc,
            "refresh_mirabel_alarm":row["M"]>refresh_m,"refresh_lc_alarm":row["R_REFRESH"]>refresh_lc})
    cache=EXP/"cache"/f"FINQA_{args.scope.upper()}_RETRIEVAL_AND_DETECTION.jsonl";write_jsonl(cache,ref_rows+hold_rows+attack_rows)
    summary=[]
    for mode,mfield,lfield in (("STRICT","strict_mirabel_alarm","strict_lc_alarm"),("BENIGN_REFRESH","refresh_mirabel_alarm","refresh_lc_alarm")):
        for attack in ("MEntA","RAG-MIA","IA-Std-Q15"):
            subset=[row for row in attack_rows if row["attack"]==attack and row["membership"]=="member"]
            mir=statistics.fmean(row[mfield] for row in subset);lc=statistics.fmean(row[lfield] for row in subset)
            summary.append({"scope":args.scope,"mode":mode,"attack":attack,"member_queries":len(subset),"mirabel_tpr":mir,"final_lc_tpr":lc,"delta":lc-mir,
                "target_retrieval_at4":statistics.fmean(row["target_rank"]>0 for row in subset),"locator_hit_at1":statistics.fmean(row["target_rank"]==1 for row in subset)})
    write_csv(EXP/"tables"/f"FINQA_{args.scope.upper()}_DETECTION.csv",summary)
    refresh=[row for row in summary if row["mode"]=="BENIGN_REFRESH"]
    checks={"macro_delta_nonnegative":statistics.fmean(row["delta"] for row in refresh)>=-1e-12,
        "no_attack_drop_over_3pp":min(row["delta"] for row in refresh)>=-.03-1e-12,
        "no_catastrophic_tpr":min(row["final_lc_tpr"] for row in refresh)>=.05}
    passed=all(checks.values());verdict=f"PHASE_D_{args.scope.upper()}_DETECTION_PASS" if passed else f"PHASE_D_{args.scope.upper()}_DETECTION_FAILED"
    result={"verdict":verdict,"scope":args.scope,"cache":{"path":str(cache),"sha256":sha_file(cache)},"summary":summary,"checks":checks,
        "strict":{"mirabel_threshold":strict_m,"final_lc_threshold":strict_lc,"benign_mirabel_fpr":statistics.fmean(row["strict_mirabel_alarm"] for row in hold_rows),"benign_final_lc_fpr":statistics.fmean(row["strict_lc_alarm"] for row in hold_rows)},
        "benign_refresh":{"mirabel_threshold":refresh_m,"final_lc_threshold":refresh_lc,"mirabel_alarm_count":m_count,"final_lc_alarm_count":lc_count,"benign_mirabel_fpr":m_count/1000,"benign_final_lc_fpr":lc_count/1000}}
    atomic_json(EXP/f"PHASE_D_{args.scope.upper()}_DETECTION_RESULT.json",result);checkpoint(verdict,scope=args.scope,macro_delta=round(statistics.fmean(row["delta"] for row in refresh),6),cache_sha256=sha_file(cache))


if __name__=="__main__":main()
