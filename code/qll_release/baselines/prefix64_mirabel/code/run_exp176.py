#!/usr/bin/env python3
"""Exp176: external attacks for the frozen prefix64 + Mirabel-to-hide policy."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT=PROJECT/"exp176_prefix64_mirabel_external_attacks_20260828"
EXP160E=PROJECT/"exp160e_global_cap64_external_attacks_exploratory_20260826"
EXP174R=PROJECT/"exp174r_stable_prefix64_session_repair_20260828"
EXP174S=PROJECT/"exp174s_prefix64_mirabel_clean3k_20260828"
EXP174T=PROJECT/"exp174t_prefix64_mirabel_menta5_20260828"
EXP175=PROJECT/"exp175_stable_prefix64_external_attacks_20260828"
PRECOMMIT=ROOT/"configs/PRECOMMIT.json"
COHORT=EXP160E/"private/EXP160E_ATTACK_COHORT.private.csv.gz"
RETRIEVAL=EXP160E/"private/EXP160_RETRIEVAL.private.csv.gz"
OLD_GENERATIONS=EXP160E/"private/EXP160_GENERATIONS.private.csv.gz"
OLD_SUMMARY=EXP160E/"tables/TABLE_160_01_ATTACK_PERFORMANCE.csv"


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value); return value


runner=load_module("exp175_base_for_exp176",EXP175/"code/run_exp175.py")
runner.ROOT=ROOT; runner.base.ROOT=ROOT


def checkpoint(stage,**values):
    payload={"experiment":"Exp176","stage":stage,"updated_utc":datetime.now(timezone.utc).isoformat(),
             "pid":os.getpid(),"candidate":"PREFIX64_MIRABEL_TO_HIDE","paid_api_calls":0,**values}
    runner.base.atomic_json(ROOT/"HEARTBEAT.json",payload)
    runner.base.atomic_json(ROOT/"checkpoints"/f"{stage}.json",payload)
    with (ROOT/"logs/pipeline.log").open("a") as handle: handle.write(json.dumps(payload)+"\n")
    runner.base.atomic_text(ROOT/"STATUS.md","\n".join(["# Exp176 status","",f"- Stage: **{stage}**",
      f"- Updated UTC: {payload['updated_utc']}",f"- PID: `{os.getpid()}`",*[f"- {k}: {v}" for k,v in values.items()],""]))


runner.checkpoint=checkpoint


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()


def initialize(config):
    predecessors={"Exp174R":EXP174R,"Exp174S":EXP174S,"Exp174T":EXP174T}
    observed={name:json.loads((path/"FINAL_RESULT.json").read_text())["verdict"] for name,path in predecessors.items()}
    if observed!=config["required_predecessors"]: raise RuntimeError(f"predecessor mismatch: {observed}")
    paths=[PRECOMMIT,COHORT,RETRIEVAL,OLD_GENERATIONS,OLD_SUMMARY,EXP175/"code/run_exp175.py",
           *[path/"FINAL_RESULT.json" for path in predecessors.values()]]
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise RuntimeError(f"missing: {missing}")
    runner.base.atomic_csv(pd.DataFrame([{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p),
                                          "access":"READ_ONLY"} for p in paths]),ROOT/"provenance/FROZEN_INPUTS.csv")
    cohort=pd.read_csv(COHORT,keep_default_na=False,dtype={"row_id":str})
    retrieval=pd.read_csv(RETRIEVAL,keep_default_na=False,dtype={"row_id":str})
    if len(cohort)!=1800 or cohort.member.value_counts().to_dict()!={1:900,0:900}: raise RuntimeError("cohort mismatch")
    if len(retrieval)!=3600: raise RuntimeError("retrieval mismatch")
    checkpoint("INPUTS_FROZEN",member=900,nonmember=900,retrieval_rows=3600)
    return cohort,retrieval


def compose(candidate,retrieval):
    old=pd.read_csv(OLD_GENERATIONS,keep_default_na=False,dtype={"row_id":str})
    hidden=old[old.condition.eq("ORIGINAL_MIRABEL")].set_index(["row_id","attack","budget"]).response
    alarm=retrieval.set_index(["row_id","attack"]).mirabel_alarm.map(lambda x:str(x).lower()=="true")
    rows=[]
    for item in candidate.itertuples(index=False):
        key=(str(item.row_id),str(item.attack),int(item.budget)); fired=bool(alarm[(key[0],key[1])])
        rows.append({"composite_id":f"{key[1]}|PREFIX64_MIRABEL_TO_HIDE|B{key[2]}|{key[0]}",
          "row_id":key[0],"member":int(item.member),"domain":str(item.domain),"attack":key[1],
          "condition":"PREFIX64_MIRABEL_TO_HIDE","budget":key[2],"reference":str(item.reference),
          "response":str(hidden[key]) if fired else str(item.response),"mirabel_alarm":fired})
    hybrid=pd.DataFrame(rows)
    candidate_copy=candidate.copy(); candidate_copy["mirabel_alarm"]=candidate_copy.apply(
      lambda row:bool(alarm[(str(row.row_id),str(row.attack))]),axis=1)
    combined=pd.concat([candidate_copy,hybrid],ignore_index=True)
    runner.base.atomic_csv(combined,ROOT/"private/EXP176_GENERATIONS.private.csv.gz","gzip")
    checkpoint("RESPONSES_COMPOSED",candidate_rows=len(candidate),hybrid_rows=len(hybrid),
               alarm_rate=float(hybrid.mirabel_alarm.mean()))
    return combined


def rank_table(scores,retrieval):
    candidate=scores[(scores.attack.eq("BudgetLeak-Z"))&(scores.condition.eq("PREFIX64_MIRABEL_TO_HIDE"))].merge(
      retrieval[retrieval.attack.eq("BudgetLeak-Z")][["row_id","member","target_retrieval_rank"]],
      on=["row_id","member"],validate="one_to_one")
    candidate["target_rank_group"]=candidate.target_retrieval_rank.astype(int).map(
      lambda v:str(v) if v in (1,2,3,4) else "not_retrieved_top4")
    negatives=candidate[candidate.member.eq(0)].attack_score.to_numpy(float); rows=[]
    for group in ["1","2","3","4","not_retrieved_top4"]:
      members=candidate[candidate.member.eq(1)&candidate.target_rank_group.eq(group)].attack_score.to_numpy(float)
      labels=np.r_[np.ones(len(members)),np.zeros(len(negatives))]; values=np.r_[members,negatives]
      raw=float(roc_auc_score(labels,values)); rows.append({"condition":"PREFIX64_MIRABEL_TO_HIDE",
        "target_rank_group":group,"member_n":len(members),"nonmember_reference_n":len(negatives),
        "raw_auc":raw,"effective_auc":max(raw,1-raw)})
    output=pd.DataFrame(rows); runner.base.atomic_csv(output,ROOT/"tables/TABLE_176_04_BUDGETLEAK_BY_RANK.csv")


def finalize(summary,scores,retrieval,config):
    old=pd.read_csv(OLD_SUMMARY)
    comparison=pd.concat([old[old.condition.isin(["NO_DEFENSE","ORIGINAL_MIRABEL","GLOBAL_CAP64"])],summary],ignore_index=True)
    runner.base.atomic_csv(comparison,ROOT/"tables/TABLE_176_01_ATTACK_PERFORMANCE.csv")
    runner.bootstrap_scores(scores,config["bootstrap"]["resamples"],config["bootstrap"]["seed"])
    rank_table(scores,retrieval)
    candidate=comparison[comparison.condition.eq(config["candidate"])].set_index("attack")
    mirabel=comparison[comparison.condition.eq("ORIGINAL_MIRABEL")].set_index("attack")
    rows=[]
    for attack in ("RAGLeak","BudgetLeak-Z"):
      value=float(candidate.loc[attack,"effective_auc"]); baseline=float(mirabel.loc[attack,"effective_auc"])
      rows.append({"attack":attack,"candidate_effective_auc":value,"original_mirabel_effective_auc":baseline,
        "absolute_gate":value<=config["gates"]["each_effective_auc_max"],
        "mirabel_noninferiority_gate":value<=baseline+config["gates"]["original_mirabel_noninferiority_margin"]})
    gates=pd.DataFrame(rows); gates["attack_gate"]=gates.absolute_gate&gates.mirabel_noninferiority_gate
    runner.base.atomic_csv(gates,ROOT/"tables/TABLE_176_03_GATES.csv")
    passed=bool(gates.attack_gate.all()); verdict="PREFIX64_MIRABEL_EXTERNAL_PASS" if passed else "PREFIX64_MIRABEL_EXTERNAL_FAIL"
    result={"experiment":"Exp176","verdict":verdict,"gate":passed,"candidate":config["candidate"],
      "member":900,"nonmember":900,"paid_api_calls":0,"fresh_blind":False,
      "completed_utc":datetime.now(timezone.utc).isoformat(),"claim_boundary":config["claim_boundary"]}
    runner.base.atomic_json(ROOT/"FINAL_RESULT.json",result)
    runner.base.atomic_text(ROOT/"reports/EXP176_FINAL_REPORT_KO.md","\n".join([
      "# Exp176 — Prefix64 + Mirabel 외부 공격", "",f"- 판정: **{verdict}**","- 900 member / 900 nonmember","",
      "## 성능","",comparison.to_markdown(index=False),"","## Gate","",gates.to_markdown(index=False),"",
      config["claim_boundary"],""]))
    checkpoint("COMPLETE",verdict=verdict,gate=passed)


def main():
    config=json.loads(PRECOMMIT.read_text()); cohort,retrieval=initialize(config)
    documents=runner.base.all_documents(retrieval)
    p64=runner.generate(cohort,retrieval,documents,config)
    combined=compose(p64,retrieval)
    metrics=runner.base.response_metrics(combined); summary,scores=runner.base.attack_scores(metrics)
    finalize(summary,scores,retrieval,config)


if __name__=="__main__":
  try: main()
  except Exception as error:
    runner.base.atomic_json(ROOT/"RUNTIME_ERROR.json",{"experiment":"Exp176","error_type":type(error).__name__,
      "error":str(error),"updated_utc":datetime.now(timezone.utc).isoformat()})
    checkpoint("FAILED",error_type=type(error).__name__,error=str(error)); raise
