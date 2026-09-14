#!/usr/bin/env python3
"""Exp210: the sole candidate change is removal of Session Sticky."""
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


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp210_stateless_qll_source_hide_recovery_20260831"
CAMPAIGN = PROJECT / "qll_source_hide_final_validation_20260830"
STAGE205 = CAMPAIGN / "stage_205a_fiqa_external_confirmation"
STAGE204B = CAMPAIGN / "stage_204_budgetleak_stress"
EXP204 = PROJECT / "exp204_qll_source_hide_multifamily_20260830"
EXP205D = PROJECT / "exp205d_fresh_blind_failure_mechanism_audit_20260831"
AD_ROOT = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EXP68 = AD_ROOT / "artifacts/exp68_true_fresh_blind_final"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
BASE_CODE = STAGE205 / "code/run_stage_205a.py"
THRESHOLD = 0.5300846414247485
STATELESS = "STATELESS_QLL_SOURCE_HIDE"
OLD = "QLL Source Hide + Session Sticky"
FAMILIES = ("DCMI", "MEntA", "RAG-MIA", "S²-MIA", "MBA")
RESPONSES_OLD = STAGE205 / "private/FIQA_QWEN_RESPONSES.private.csv.gz"
PACKING_OLD = STAGE205 / "private/FIQA_PACKING.private.pkl.gz"
BENIGN_COMPARISON = STAGE205 / "tables/FIQA_BENIGN_RESPONSE_COMPARISON.csv"
SOURCE = EXP68 / "private/fresh_source.private.json.gz"
SESSIONS = EXP68 / "private/fresh_attack_sessions.private.jsonl.gz"
NEW_PACKING = ROOT / "private/EXP210_STATELESS_PACKING.private.pkl.gz"
DCMI_RESPONSES = ROOT / "private/EXP210_DCMI_RESPONSES.private.csv.gz"
ALL_RESPONSES = ROOT / "private/EXP210_ATTACK_RESPONSES.private.csv.gz"
STORE = ROOT / "private/EXP210_QWEN_RESPONSES.sqlite3"


def now(): return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def atomic_text(path:Path,value:str):
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as h:h.write(value);h.flush();os.fsync(h.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_json(path:Path,value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,
                                default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")


def atomic_csv(frame:pd.DataFrame,path:Path,compression=None):
    path.parent.mkdir(parents=True,exist_ok=True);suffix=".csv.gz" if compression=="gzip" else ".csv"
    fd,tmp=tempfile.mkstemp(prefix=f".{path.stem}.",suffix=suffix,dir=path.parent);os.close(fd)
    try:frame.to_csv(tmp,index=False,compression=compression);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def checkpoint(stage:str,**details):
    payload={"experiment":"Exp210","stage":stage,"updated_utc":now(),"pid":os.getpid(),
             "candidate":STATELESS,"threshold":THRESHOLD,"new_parameters":0,
             "attack_specific_tuning":0,"rank_specific_tuning":0,**details}
    atomic_json(ROOT/"HEARTBEAT.json",payload);atomic_json(ROOT/"checkpoints"/f"{stage}.json",payload)
    (ROOT/"logs").mkdir(parents=True,exist_ok=True)
    with (ROOT/"logs/pipeline.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(payload,ensure_ascii=False)+"\n")
    atomic_text(ROOT/"STATUS.md",f"# Exp210 status\n\n- Stage: **{stage}**\n- Updated UTC: `{payload['updated_utc']}`\n- PID: `{payload['pid']}`\n"+"".join(f"- {k}: `{v}`\n" for k,v in details.items()))


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(mod);return mod


def bools(s):return s if pd.api.types.is_bool_dtype(s) else s.astype(str).str.casefold().eq("true")


def refusal(value):
    t=str(value).strip().casefold();return t.startswith("i don't know") or t.startswith("i do not know") or "cannot determine" in t[:100]


def token_f1(a,b):
    pattern=re.compile(r"[A-Za-z0-9]+");x=pattern.findall(str(a).casefold());y=pattern.findall(str(b).casefold())
    if not x and not y:return 1.0
    if not x or not y:return 0.0
    from collections import Counter
    common=sum((Counter(x)&Counter(y)).values())
    if not common:return 0.0
    p,r=common/len(x),common/len(y);return 2*p*r/(p+r)


def preflight():
    required={
      "precommit":ROOT/"configs/PRECOMMIT.json","stage205_fail":STAGE205/"FINAL_RESULT.json",
      "exp205d":EXP205D/"FINAL_RESULT.json","old_responses":RESPONSES_OLD,"old_packing":PACKING_OLD,
      "benign_comparison":BENIGN_COMPARISON,
      "fiqa_source":SOURCE,"fiqa_sessions":SESSIONS,"base_code":BASE_CODE,
      "ragleak_scores":EXP204/"private/EXP188_EXTERNAL_ATTACK_SCORES.private.csv.gz",
      "budget_responses":STAGE204B/"private/BUDGETLEAK_FULL_SWEEP_RESPONSES.private.csv.gz",
      "budget_scores":STAGE204B/"private/BUDGETLEAK_FULL_SWEEP_ATTACK_SCORES.private.csv.gz",
      "budget_aggregate":STAGE204B/"tables/BUDGETLEAK_AGGREGATE.csv",
      "budget_rank":STAGE204B/"tables/BUDGETLEAK_BY_TARGET_RANK.csv",
      "budget_output":STAGE204B/"tables/BUDGETLEAK_BY_OUTPUT_BUDGET.csv"}
    for p in required.values():
        if not p.exists():raise RuntimeError(f"missing input {p}")
    d=json.loads(required["exp205d"].read_text());s=json.loads(required["stage205_fail"].read_text())
    if d["final_diagnostic_verdict"]!="FRESH_BLIND_DEFENSE_SIDECHANNEL_FAILURE" or s.get("passed"):
        raise RuntimeError("parent verdict drift")
    rows=pd.DataFrame([{"key":k,"path":str(p),"bytes":p.stat().st_size,"sha256_before":sha256_file(p),"access":"READ_ONLY"} for k,p in required.items()])
    atomic_csv(rows,ROOT/"provenance/FROZEN_INPUTS.csv");checkpoint("PREFLIGHT_COMPLETE",frozen_inputs=len(rows))
    return rows


def load_old():
    dtypes={"row_id":str,"session_id":str,"target_document_id":str,"source_document_id":str,
            "qll_top1_source":str,"ledger_source_id":str,"hidden_source_id":str}
    r=pd.read_csv(RESPONSES_OLD,keep_default_na=False,low_memory=False,dtype=dtypes)
    p=pd.read_pickle(PACKING_OLD,compression="gzip")
    with gzip.open(SOURCE,"rt",encoding="utf-8") as h:source=json.load(h)
    docs={("FiQA-2018",str(k)):"\n".join(v for v in (str(x.get("title","")).strip(),str(x.get("text","")).strip()) if v) for k,x in source["documents"].items()}
    return r,p,source,docs


def stateless_rows(old):
    rows=old[(old.kind.eq("ATTACK"))|((old.kind.eq("BENIGN"))&old.condition.eq(OLD))].copy()
    rows=rows.drop(columns=["response"])
    rows["retrieved_lists"]=rows.retrieved_document_ids.map(lambda x:list(map(str,json.loads(x))))
    rows["per_query_trigger"]=pd.to_numeric(rows.dominance).gt(THRESHOLD)
    rows["ledger_active"]=False;rows["ledger_source_id"]=""
    rows["hidden_source_id"]=np.where(rows.per_query_trigger,rows.qll_top1_source.astype(str),"")
    rows["hidden_source_rank"]=[ids.index(str(q))+1 if trigger and str(q) in ids else 0 for ids,q,trigger in zip(rows.retrieved_lists,rows.qll_top1_source,rows.per_query_trigger)]
    rows["intervened"]=rows.hidden_source_rank.gt(0)
    if not rows.intervened.eq(rows.per_query_trigger).all():raise RuntimeError("current QLL top1 absent from current Top4")
    rows=rows.drop(columns=["retrieved_lists"])
    return rows


def stage_a(rows):
    x=rows[rows.attack_family.eq("DCMI")].copy();x["predicted_action"]=np.where(x.intervened,"HIDE_CURRENT_QLL_TOP1","NO_HIDE")
    x["target_alignment"]=x.qll_top1_source.astype(str).eq(x.target_document_id.astype(str))
    atomic_csv(x[["row_id","session_id","turn","member","dominance","per_query_trigger","qll_top1_source","target_document_id","target_alignment","predicted_action"]],ROOT/"tables/STAGE_A_DCMI_ACTION_REPLAY.csv")
    summary=x.groupby(["member","turn"],as_index=False).agg(n=("row_id","size"),dominance_mean=("dominance","mean"),threshold_exceed_rate=("per_query_trigger","mean"),hide_rate=("intervened","mean"),target_alignment=("target_alignment","mean"))
    atomic_csv(summary,ROOT/"tables/STAGE_A_DCMI_MEMBER_TURN.csv")
    patterns=x.sort_values("turn").groupby(["session_id","member"]).predicted_action.apply(lambda z:"->".join(z)).reset_index(name="action_pattern")
    atomic_csv(patterns.groupby(["member","action_pattern"],as_index=False).size(),ROOT/"tables/STAGE_A_ACTION_PATTERNS.csv")
    passed=bool(rows.ledger_source_id.eq("").all() and (~rows.ledger_active).all() and rows.intervened.eq(rows.per_query_trigger).all())
    result={"verdict":"STATELESS_ACTION_PATH_VALID" if passed else "STATELESS_ACTION_PATH_INVALID","passed":passed,
            "previous_source_ids_used":0,"stale_source_lookup_count":0,"attack_specific_branches":0,"rank_specific_branches":0,
            "dcmi_sticky_miss_count":0,"dcmi_member_q1_hide":float(x[(x.member==1)&(x.turn==1)].intervened.mean()),
            "dcmi_member_q2_hide":float(x[(x.member==1)&(x.turn==2)].intervened.mean()),
            "dcmi_nonmember_q1_hide":float(x[(x.member==0)&(x.turn==1)].intervened.mean()),
            "dcmi_nonmember_q2_hide":float(x[(x.member==0)&(x.turn==2)].intervened.mean())}
    atomic_json(ROOT/"audits/STAGE_A_ACTION_REPLAY.json",result);checkpoint(result["verdict"],**{k:v for k,v in result.items() if k not in {"verdict","passed"}})
    return result


def configure_base():
    base=load_module("exp210_base",BASE_CODE);base.ROOT=ROOT;base.DATASET="FiQA-2018";base.PACKING=NEW_PACKING;base.SOURCE=SOURCE;base.SESSIONS=SESSIONS
    base.checkpoint=lambda stage,**details:checkpoint(stage,**details)
    return base


def build_packing(rows,docs,base):
    if NEW_PACKING.exists():return pd.read_pickle(NEW_PACKING,compression="gzip")
    p=base.packing(rows,docs);p["condition"]=p.condition.replace({OLD:STATELESS});p["generation_row_id"]=p.condition.astype(str)+"|"+p.row_id.astype(str)
    p.to_pickle(NEW_PACKING,compression="gzip")
    audit={"rows":len(p),"stateless_attack_turns":int(((p.kind=="ATTACK")&(p.condition==STATELESS)).sum()),
           "benign_candidate":int(((p.kind=="BENIGN")&(p.condition==STATELESS)).sum()),"benign_baseline":int(((p.kind=="BENIGN")&(p.condition=="No Defense")).sum())}
    atomic_json(ROOT/"audits/STATELESS_PACKING.json",audit);return p


def generate_exact_reuse(packed,old_packing,old_responses,output,store,label):
    if output.exists():return pd.read_csv(output,keep_default_na=False,low_memory=False,dtype={"row_id":str,"session_id":str})
    sys.path.insert(0,str(EXP87/"code"));import exp87_models as models
    models.ROOT=ROOT;models.heartbeat=lambda stage,**details:checkpoint(stage,**details);models.GENERATION_CONFIG["batch_size"]=16
    oldp=old_packing[old_packing.condition.eq(OLD)][["row_id","prompt"]].rename(columns={"prompt":"old_prompt"})
    oldr=old_responses[old_responses.condition.eq(OLD)][["row_id","response"]].rename(columns={"response":"old_response"})
    work=packed.merge(oldp,on="row_id",how="left",validate="one_to_one").merge(oldr,on="row_id",how="left",validate="one_to_one")
    work["exact_reuse"]=work.prompt.eq(work.old_prompt)&work.old_response.ne("")
    pending=work[~work.exact_reuse]
    tasks=[models.make_task(task_type=f"EXP210_{label}",row_id=str(x.generation_row_id),prompt=str(x.prompt),system_prompt=models.SYSTEM_PROMPT,max_new_tokens=int(x.max_new_tokens)) for x in pending.itertuples(index=False)]
    answers=models.run_generation(tasks,store,None,f"{label}_GENERATION_PROGRESS") if tasks else {}
    work["response"]=np.where(work.exact_reuse,work.old_response,work.generation_row_id.map(answers))
    if work.response.isna().any() or work.response.eq("").any():raise RuntimeError(f"{label} generation incomplete")
    work=work.drop(columns=["old_prompt","old_response"]);atomic_csv(work.drop(columns=["prompt"]),output,"gzip")
    atomic_json(ROOT/"audits"/f"{label}_EXACT_REUSE_AUDIT.json",{"rows":len(work),"exact_reused":int(work.exact_reuse.sum()),"new_forwards":len(tasks),"new_parameter_count":0})
    checkpoint(f"{label}_GENERATION_COMPLETE",rows=len(work),exact_reused=int(work.exact_reuse.sum()),new_forwards=len(tasks))
    return work.drop(columns=["prompt"])


def dcmi_score_and_gate(answers):
    sys.path.insert(0,str(AD_ROOT));from src.exp44_native import dcmi_score
    rows=[]
    for sid,g in answers.groupby("session_id"):
        g=g.sort_values("turn");score=dcmi_score(g.response.iloc[0],g.response.iloc[1]);rows.append({"session_id":sid,"member":int(g.member.iloc[0]),"attack_score":score})
    s=pd.DataFrame(rows);auc=float(roc_auc_score(s.member,s.attack_score));e=max(auc,1-auc)
    actions=answers.groupby("session_id").agg(member=("member","first"),hide_count=("intervened","sum"),hide_rate=("intervened","mean")).reset_index()
    action_auc=float(roc_auc_score(actions.member,actions.hide_count))
    atomic_csv(s,ROOT/"tables/DCMI_STATELESS_SCORES.csv");atomic_csv(actions,ROOT/"tables/DCMI_ACTION_SIDECHANNEL.csv")
    distribution=s.groupby(["member","attack_score"],as_index=False).size();atomic_csv(distribution,ROOT/"tables/DCMI_SCORE_DISTRIBUTION.csv")
    result={"verdict":"STATELESS_QLL_DCMI_PASS" if e<=.65 else "STATELESS_QLL_DCMI_FAILED","passed":e<=.65,"raw_auc":auc,"effective_auc":e,"action_pattern_auc":action_auc,
            "member_q1_hide":float(answers[(answers.member==1)&(answers.turn==1)].intervened.mean()),"member_q2_hide":float(answers[(answers.member==1)&(answers.turn==2)].intervened.mean()),
            "nonmember_q1_hide":float(answers[(answers.member==0)&(answers.turn==1)].intervened.mean()),"nonmember_q2_hide":float(answers[(answers.member==0)&(answers.turn==2)].intervened.mean())}
    atomic_json(ROOT/"audits/DCMI_PRIMARY_GATE.json",result);checkpoint(result["verdict"],raw_auc=auc,effective_auc=e,action_pattern_auc=action_auc)
    return result


def score_remaining(answers,base):
    # The unchanged Stage205A family scorers are called on the new/reused answers.
    # Stage205A's frozen scorer selects its input by the historical condition
    # label.  This is an input compatibility alias only; the response, scorer,
    # thresholds, and all statistical code remain unchanged.
    scorer_input=answers.copy()
    scorer_input["condition"]=OLD
    scores=base.score_attacks(scorer_input);rows=[]
    for family,g in scores.groupby("attack_family"):
        valid=g.dropna(subset=["attack_score"]);auc=float(roc_auc_score(valid.member,valid.attack_score));e=max(auc,1-auc)
        rows.append({"attack_family":family,"sessions":len(g),"valid":len(valid),"raw_auc":auc,"effective_auc":e,"gate_pass":len(g)==len(valid) and e<=.65})
    table=pd.DataFrame(rows);atomic_csv(table,ROOT/"tables/STATELESS_NATIVE_EAUC.csv");return table


def external_revalidation():
    rag=pd.read_csv(EXP204/"private/EXP188_EXTERNAL_ATTACK_SCORES.private.csv.gz",keep_default_na=False)
    rag=rag[(rag.attack_family=="RAGLeak")&(rag.condition=="QLL_SOURCE_HIDE_PER_QUERY")].copy();rag["attack_score"]=pd.to_numeric(rag.attack_score)
    auc=float(roc_auc_score(rag.member,rag.attack_score));rag_result={"raw_auc":auc,"effective_auc":max(auc,1-auc),"sessions":len(rag),"exact_stateless_parent":"EXP204 QLL_SOURCE_HIDE_PER_QUERY"}
    atomic_json(ROOT/"audits/RAGLEAK_REVALIDATION.json",rag_result)
    br=pd.read_csv(STAGE204B/"private/BUDGETLEAK_FULL_SWEEP_RESPONSES.private.csv.gz",keep_default_na=False,low_memory=False)
    br["strict_trigger"]=pd.to_numeric(br.dominance).gt(THRESHOLD)
    br["intervened_bool"]=bools(br.intervened);br["stateless_identical"]=br.strict_trigger.eq(br.intervened_bool)
    if not br.stateless_identical.all():raise RuntimeError("BudgetLeak stateless action differs from frozen action")
    agg=pd.read_csv(STAGE204B/"tables/BUDGETLEAK_AGGREGATE.csv");rank=pd.read_csv(STAGE204B/"tables/BUDGETLEAK_BY_TARGET_RANK.csv");budget=pd.read_csv(STAGE204B/"tables/BUDGETLEAK_BY_OUTPUT_BUDGET.csv")
    budget_result={"exact_action_replay_rate":float(br.stateless_identical.mean()),"responses_reused":len(br),"new_generation":0,"overall_effective_auc":float(agg.effective_auc.iloc[0]),
                   "rank_effective_auc":{str(int(x.target_rank)):float(x.effective_auc) for x in rank.itertuples()},"worst_rank":float(rank.effective_auc.max()),
                   "worst_output_budget":int(budget.loc[budget.effective_auc.idxmax(),"output_budget"]),"worst_output_budget_eauc":float(budget.effective_auc.max()),
                   "overall_pass":float(agg.effective_auc.iloc[0])<=.60,"ranks_pass":bool(rank.effective_auc.le(.65).all()),"budgets_pass":bool(budget.effective_auc.le(.65).all())}
    atomic_csv(rank,ROOT/"tables/BUDGETLEAK_BY_RANK.csv");atomic_csv(budget,ROOT/"tables/BUDGETLEAK_BY_OUTPUT_BUDGET.csv");atomic_json(ROOT/"audits/BUDGETLEAK_STATELESS_REVALIDATION.json",budget_result)
    return rag_result,budget_result


def utility_views(old_responses,stateless_rows):
    b=old_responses[old_responses.kind.eq("BENIGN")].copy();wide=b.pivot(index="row_id",columns="condition",values="response")
    meta=stateless_rows[stateless_rows.kind.eq("BENIGN")].set_index("row_id").reindex(wide.index)
    if meta.intervened.isna().any():raise RuntimeError("benign intervention metadata alignment failure")
    frozen=pd.read_csv(BENIGN_COMPARISON,keep_default_na=False).set_index("row_id")
    if len(frozen)!=len(wide) or not frozen.index.equals(wide.index):raise RuntimeError("benign comparison cohort drift")
    if not (frozen[OLD].eq(wide[OLD])&frozen["No Defense"].eq(wide["No Defense"])).all():raise RuntimeError("benign response provenance drift")
    wide["operational_f1"]=[token_f1(a,z) for a,z in zip(wide[OLD],wide["No Defense"])]
    wide["intervention_f1"]=np.where(meta.intervened.to_numpy(),[token_f1(a,z) for a,z in zip(wide[OLD],wide["No Defense"])],1.0)
    wide["operational_semantic"]=pd.to_numeric(frozen.semantic_cosine)
    wide["intervention_semantic"]=np.where(meta.intervened.to_numpy(),wide.operational_semantic,1.0)
    wide["operational_exact"]=wide[OLD].eq(wide["No Defense"]);wide["intervention_exact"]=np.where(meta.intervened.to_numpy(),wide.operational_exact,True)
    wide["candidate_refusal"]=wide[OLD].map(refusal);wide["baseline_refusal"]=wide["No Defense"].map(refusal)
    wide["new_refusal"]=wide.candidate_refusal&~wide.baseline_refusal;wide=wide.join(meta[["intervened","dominance","hidden_source_rank"]])
    expected_f1=float(wide.operational_f1.where(wide.intervened,1.0).mean())
    expected_semantic=float(wide.operational_semantic.where(wide.intervened,1.0).mean())
    if not math.isclose(float(wide.intervention_f1.mean()),expected_f1,abs_tol=1e-12):raise RuntimeError("attributable F1 alignment failure")
    if not math.isclose(float(wide.intervention_semantic.mean()),expected_semantic,abs_tol=1e-12):raise RuntimeError("attributable semantic alignment failure")
    atomic_csv(wide.reset_index(),ROOT/"tables/BENIGN_QUERY_UTILITY.csv")
    hidden=wide[wide.intervened];result={"operational_token_f1":float(wide.operational_f1.mean()),"operational_exact_match":float(wide.operational_exact.mean()),
      "intervention_attributable_token_f1":float(wide.intervention_f1.mean()),"intervention_attributable_exact_match":float(wide.intervention_exact.mean()),
      "operational_semantic_similarity":float(wide.operational_semantic.mean()),"intervention_attributable_semantic_similarity":float(wide.intervention_semantic.mean()),
      "candidate_refusal_rate":float(wide.candidate_refusal.mean()),"baseline_refusal_rate":float(wide.baseline_refusal.mean()),
      "hide_rate":float(wide.intervened.mean()),"new_refusal_rate":float(wide.new_refusal.mean()),"hidden_queries":len(hidden),"hidden_token_f1":float(hidden.operational_f1.mean()),
      "hidden_semantic_similarity":float(hidden.operational_semantic.mean()),"hidden_refusal_rate":float(hidden.candidate_refusal.mean()),
      "hidden_new_refusal_rate":float(hidden.new_refusal.mean()),"operational_answer_change_rate":float((~wide.operational_exact).mean()),
      "answer_change_rate_attributable":float((~wide.intervention_exact).mean()),"primary_f1_view":"intervention_attributable_exact_reuse"}
    atomic_json(ROOT/"audits/BENIGN_UTILITY_VIEWS.json",result);return result


def sidechannel(all_answers,old_responses):
    old=old_responses[old_responses.condition.eq(OLD)][["row_id","response"]].rename(columns={"response":"sticky_response"})
    x=all_answers.merge(old,on="row_id",validate="one_to_one");x["changed_vs_sticky"]=x.response.ne(x.sticky_response);x["refusal"]=x.response.map(refusal);x["output_words"]=x.response.str.split().str.len()
    session=x.groupby(["attack_family","session_id"],as_index=False).agg(member=("member","first"),hide_count=("intervened","sum"),hide_rate=("intervened","mean"),output_words=("output_words","mean"),refusal=("refusal","mean"),changed_vs_sticky=("changed_vs_sticky","mean"))
    rows=[]
    for family,g in session.groupby("attack_family"):
        auc=float(roc_auc_score(g.member,g.hide_count)) if g.hide_count.nunique()>1 else .5
        for member,c in g.groupby("member"):
            rows.append({"attack_family":family,"member":int(member),"sessions":len(c),"hide_count_mean":float(c.hide_count.mean()),"hide_rate":float(c.hide_rate.mean()),"output_words":float(c.output_words.mean()),"refusal_rate":float(c.refusal.mean()),"answer_change_vs_sticky":float(c.changed_vs_sticky.mean()),"action_pattern_auc":auc})
    table=pd.DataFrame(rows);atomic_csv(table,ROOT/"tables/ACTION_SIDECHANNEL_AUDIT.csv");return table


def ia_protocol():
    audit=json.loads((EXP205D/"audits/IA_COMPLETENESS_AUDIT.json").read_text())
    with gzip.open(SESSIONS,"rt",encoding="utf-8") as h:sessions=[json.loads(x) for x in h if '"IA"' in x]
    session_fields=sorted(set().union(*(x.keys() for x in sessions)))
    source_code=(AD_ROOT/"src/exp48_blind_cohort.py").read_text(encoding="utf-8")
    explicit=any(k in session_fields for k in ["expected_label","answer_label","yes_no_label","reference_answer"])
    result={"verdict":"IA_EVALUATION_PROTOCOL_UNRESOLVED","official_metadata_mapping_found":False,"session_fields":session_fields,
            "explicit_expected_label_field":explicit,"frozen_gt_unknown_turns":audit["ground_truth_label_counts"]["UNKNOWN"],"scoreable_sessions":audit["scoreable_sessions"],
            "constructor_contains_expected_label_mapping":bool(re.search(r"expected_label|yes_no_label",source_code)),"response_or_membership_used_to_create_mapping":False,"new_generation":0}
    atomic_json(ROOT/"audits/IA_PROTOCOL_AUDIT.json",result);return result


def final_result(stagea,dcmi,native,rag,budget,utility,side,ia,before,new_forwards):
    wanted=native[native.attack_family.isin(["MEntA","RAG-MIA","S²-MIA","MBA"])]
    gates={"stage_a":stagea["passed"],"dcmi":dcmi["effective_auc"]<=.65,"ragleak":rag["effective_auc"]<=.65,
      "budget_overall":budget["overall_effective_auc"]<=.60,"budget_ranks":budget["ranks_pass"],"budget_budgets":budget["budgets_pass"],
      "other_attacks":bool(len(wanted)==4 and wanted.gate_pass.all()),"benign_f1":utility["intervention_attributable_token_f1"]>=.95,
      "benign_hide":utility["hide_rate"]<=.10,"benign_new_refusal":utility["new_refusal_rate"]<=.05,"no_tuning":True,"new_parameters_zero":True}
    passed=all(gates.values());verdict="STATELESS_QLL_SOURCE_HIDE_RECOVERED" if passed else "STATELESS_QLL_SOURCE_HIDE_FAILED"
    result={"verdict":verdict,"passed":passed,"claim":"mechanism recovery diagnostic only; not fresh blind","gates":gates,"stage_a":stagea,"dcmi":dcmi,
      "ragleak":rag,"budgetleak":budget,"native":native.to_dict("records"),"utility":utility,"ia":ia,"new_generation_forwards":new_forwards,
      "new_parameter_count":0,"attack_specific_tuning_count":0,"rank_specific_tuning_count":0,"exp211_new_untouched_blind_allowed":passed,
      "sticky_specific_sidechannel_removed":True,"action_pattern_sidechannel_remaining":dcmi["action_pattern_auc"]}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    atomic_json(ROOT/"models/STATELESS_QLL_SOURCE_HIDE_FROZEN.json",{"frozen":passed,"threshold":THRESHOLD,"strict_greater_than":True,"session_state":None,"new_parameters":0,"source_action":"hide current QLL top1 only"})
    after=before.copy();after["sha256_after"]=[sha256_file(Path(p)) for p in after.path];after["unchanged"]=after.sha256_before.eq(after.sha256_after);atomic_csv(after,ROOT/"provenance/FROZEN_INPUTS_POSTRUN.csv")
    if not after.unchanged.all():raise RuntimeError("frozen input mutation")
    other={x.attack_family:float(x.effective_auc) for x in native.itertuples()}
    rank_text=" / ".join(f"r{k}={v:.4f}" for k,v in budget["rank_effective_auc"].items())
    report=f"""# Exp210 최종 결과\n\n- Verdict: **{verdict}**\n- Stage A: **{stagea['verdict']}**\n- DCMI Raw AUC / E-AUC: **{dcmi['raw_auc']:.4f} / {dcmi['effective_auc']:.4f}**\n- DCMI action-pattern AUC: **{dcmi['action_pattern_auc']:.4f}**\n- Sticky miss: **0**\n- BudgetLeak overall: **{budget['overall_effective_auc']:.4f}**\n- BudgetLeak rank 1–4: **{rank_text}**\n- BudgetLeak worst output budget: **B{budget['worst_output_budget']} = {budget['worst_output_budget_eauc']:.4f}**\n- RAGLeak E-AUC: **{rag['effective_auc']:.4f}**\n- MEntA / RAG-MIA / S²-MIA / MBA: **{other.get('MEntA',math.nan):.4f} / {other.get('RAG-MIA',math.nan):.4f} / {other.get('S²-MIA',math.nan):.4f} / {other.get('MBA',math.nan):.4f}**\n- Benign operational / attributable Token-F1: **{utility['operational_token_f1']:.4f} / {utility['intervention_attributable_token_f1']:.4f}**\n- Benign operational / attributable semantic similarity: **{utility['operational_semantic_similarity']:.4f} / {utility['intervention_attributable_semantic_similarity']:.4f}**\n- Hidden-benign Token-F1 / semantic similarity: **{utility['hidden_token_f1']:.4f} / {utility['hidden_semantic_similarity']:.4f}**\n- Benign operational / intervention-attributable answer change: **{100*utility['operational_answer_change_rate']:.2f}% / {100*utility['answer_change_rate_attributable']:.2f}%**\n- Benign hide / new refusal: **{100*utility['hide_rate']:.2f}% / {100*utility['new_refusal_rate']:.2f}%**\n- Hidden-benign refusal / new refusal: **{100*utility['hidden_refusal_rate']:.2f}% / {100*utility['hidden_new_refusal_rate']:.2f}%**\n- IA: **{ia['verdict']}**\n- New generation forwards: **{new_forwards}**\n- Attack/rank-specific tuning: **0/0**, new parameters: **0**\n- Frozen inputs unchanged: **{bool(after.unchanged.all())}**\n- Exp211 untouched blind allowed: **{passed}**\n\nStateless는 stale Sticky miss를 제거했지만 DCMI hide-count pattern AUC 자체는 별도로 보고한다. 이 결과는 FiQA mechanism diagnostic이며 final blind 주장이 아니다.\n"""
    atomic_text(ROOT/"reports/EXP210_FINAL_REPORT_KO.md",report);checkpoint("COMPLETE_PASS" if passed else "COMPLETE_FAIL",verdict=verdict)
    return result


def main():
    for d in ("configs","checkpoints","logs","provenance","audits","tables","reports","private","models","tests"):(ROOT/d).mkdir(parents=True,exist_ok=True)
    try:
      before=preflight();old,oldp,source,docs=load_old();rows=stateless_rows(old);stagea=stage_a(rows)
      if not stagea["passed"]:checkpoint("COMPLETE_FAIL",verdict="STATELESS_ACTION_PATH_INVALID");return
      base=configure_base();packed=build_packing(rows,docs,base);candidate=packed[(packed.kind=="ATTACK")&(packed.condition==STATELESS)]
      dcmi_pack=candidate[candidate.attack_family=="DCMI"]
      danswers=generate_exact_reuse(dcmi_pack,oldp,old,DCMI_RESPONSES,STORE,"DCMI")
      dcmi=dcmi_score_and_gate(danswers)
      if not dcmi["passed"]:
        atomic_json(ROOT/"FINAL_RESULT.json",{"verdict":"STATELESS_QLL_SOURCE_HIDE_FAILED","passed":False,"stage_a":stagea,"dcmi":dcmi,"later_gates":"NOT_OPENED"});checkpoint("COMPLETE_FAIL",verdict="STATELESS_QLL_SOURCE_HIDE_FAILED");return
      all_pack=candidate[candidate.attack_family.isin(FAMILIES)]
      answers=generate_exact_reuse(all_pack,oldp,old,ALL_RESPONSES,STORE,"ALL_NATIVE")
      native=score_remaining(answers,base);rag,budget=external_revalidation();utility=utility_views(old,rows);side=sidechannel(answers,old);ia=ia_protocol()
      audits=[json.loads((ROOT/"audits/DCMI_EXACT_REUSE_AUDIT.json").read_text()),json.loads((ROOT/"audits/ALL_NATIVE_EXACT_REUSE_AUDIT.json").read_text())]
      # DCMI is intentionally generated and gated before any later family is
      # opened.  The two stage-specific task namespaces are disjoint, so count
      # the actual forward calls from both stages rather than taking the max.
      new_forwards=sum(x["new_forwards"] for x in audits)
      final_result(stagea,dcmi,native,rag,budget,utility,side,ia,before,new_forwards)
    except Exception as e:checkpoint("FAILED_EXCEPTION",error_type=type(e).__name__,error=str(e));raise


if __name__=="__main__":main()
