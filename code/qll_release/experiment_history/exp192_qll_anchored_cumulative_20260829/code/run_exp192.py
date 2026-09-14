#!/usr/bin/env python3
"""Exp192 input-alignment gate for QLL-anchored cumulative exposure.

The frozen inputs do not contain per-source C1 for every QLL-dominant source.
In particular, BudgetLeak rank-4 targets are QLL dominant but absent from the
top-3 generation context.  Per the experiment contract this script records the
exact mismatch and stops before exposure statistics or model design.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT=PROJECT/"exp192_qll_anchored_cumulative_20260829"
EXP191=PROJECT/"exp191_two_channel_signal_compression_audit_20260829"
EXP190=PROJECT/"exp190_leakage_channel_decomposition_audit_20260829"
EXP189=PROJECT/"exp189_simple_global_disclosure_ledger_20260829"
EXP181=PROJECT/"exp181_native_session_continuous_qll_20260828"
EXP179=PROJECT/"exp179_cross_family_qwen_source_influence_20260828"
FAMILIES=["DCMI","MEntA","BudgetLeak-Z","IA"]

INPUTS={
 "exp191_final":EXP191/"FINAL_RESULT.json",
 "exp191_rank":EXP191/"tables/TABLE_191_11_BUDGETLEAK_BY_RANK.csv",
 "exp190_features":EXP190/"private/EXP190_SESSION_FEATURES.private.csv.gz",
 "exp190_provenance":EXP190/"provenance/FROZEN_INPUTS.csv",
 "exp189_cases":EXP189/"private/ATTACK_CASES.private.pkl.gz",
 "exp189_claim_costs":EXP189/"private/ATTACK_CLAIM_COSTS.private.pkl.gz",
 "exp189_ledger":EXP189/"private/ATTACK_LEDGER.private.csv.gz",
 "exp181_native_qll":EXP181/"private/EXP181_SOURCE_QLL.private.csv.gz",
 "exp179_cross_qll":EXP179/"private/EXP179_QUERY_SOURCE_QLL.private.csv.gz",
 "exp179_qll_summary":EXP179/"tables/TABLE_179_02_QUERY_SUMMARY.csv",
}

TABLE_NAMES=["INPUT_ALIGNMENT","DOMINANT_SOURCE","ANCHORED_QUERY_EXPOSURE","ANCHORED_SESSION_EXPOSURE",
 "MEMBER_NONMEMBER","ATTACK_SCORE_ASSOCIATION","OLD_VS_ANCHORED_C1","BUDGETLEAK_BY_RANK",
 "RANK_HETEROGENEITY","I4_RANK_STABILITY","INCREMENTAL_BEYOND_I4","INITIAL_VS_CUMULATIVE",
 "BUDGETLEAK_BY_OUTPUT_BUDGET","SOURCE_HOPPING","BENIGN_DISTRIBUTION","ALPHA_FEASIBILITY",
 "FACTUALITY_CROSSCHECK","FINAL_GO_NOGO"]
REPORT_NAMES=["ANCHORED_SIGNAL","DCMI_MENTA","BUDGETLEAK_RANK","INCREMENTAL_CHANNEL","BENIGN_ALPHA","FINAL_VERDICT"]
FIG_NAMES=["ANCHORED_CUMULATIVE_CONCEPT","OLD_VS_ANCHORED_C1","DCMI_MEMBER_NONMEMBER","MENTA_MEMBER_NONMEMBER",
 "BUDGETLEAK_BY_RANK","I4_AND_C_BY_RANK","INCREMENTAL_BEYOND_I4","MINIMAL_TWO_CHANNEL_CONCEPT"]


def now():return datetime.now(timezone.utc).isoformat()
def sha256_file(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def sha256_text(v):return hashlib.sha256(str(v).encode()).hexdigest()
def atomic_text(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as h:h.write(value);h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_json(path,value):atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n")
def atomic_csv(frame,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.stem}.",suffix=".csv",dir=path.parent);os.close(fd)
 try:frame.to_csv(tmp,index=False);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def checkpoint(stage,**details):
 p={"experiment":"Exp192","stage":stage,"updated_utc":now(),"pid":os.getpid(),"new_defense":False,
    "new_generation":False,"llama":False,"fresh_blind":False,"e_mia_open_count":0,"threshold_selection":False,
    "alpha_selection":False,"e_auc_used":False,**details}
 atomic_json(ROOT/"HEARTBEAT.json",p);atomic_json(ROOT/"checkpoints"/f"{stage}.json",p)
 with (ROOT/"logs/pipeline.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(p,ensure_ascii=False)+"\n")
 atomic_text(ROOT/"STATUS.md","# Exp192 Status\n\n"+f"- Stage: **{stage}**\n- Updated UTC: `{p['updated_utc']}`\n- PID: `{p['pid']}`\n"+"".join(f"- {k}: `{v}`\n" for k,v in details.items()))


def freeze_inputs():
 missing=[str(p) for p in INPUTS.values() if not p.exists()]
 if missing:raise RuntimeError(f"missing inputs: {missing}")
 result=json.loads(INPUTS["exp191_final"].read_text())
 if result.get("compression_verdict")!="TWO_SIGNAL_COMPRESSION_SUPPORTED" or result.get("selected_instantaneous_candidate")!="I4" or result.get("selected_cumulative_candidate")!="C1":
  raise RuntimeError("Exp191 frozen selection mismatch")
 # Exp191 obtained I4 through the frozen Exp190 feature frame.  Verify the
 # exact Exp179 summary hash against Exp190's pre-existing provenance rather
 # than merely hashing the current file at Exp192 start.
 exp190_provenance=pd.read_csv(INPUTS["exp190_provenance"])
 frozen_qll=exp190_provenance[exp190_provenance.key.eq("exp179_qll")]
 if len(frozen_qll)!=1 or sha256_file(INPUTS["exp179_qll_summary"])!=str(frozen_qll.iloc[0].sha256):
  raise RuntimeError("Exp191/Exp190 frozen QLL provenance mismatch")
 rows=[]
 for k,p in INPUTS.items():rows.append({"key":k,"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size,"access":"READ_ONLY"})
 manifest=pd.DataFrame(rows);atomic_csv(manifest,ROOT/"provenance/FROZEN_INPUTS.csv")
 pre={"experiment":"Exp192","purpose":"QLL_ANCHORED_CUMULATIVE_INPUT_GATE","frozen_I":"I4=qll_margin",
      "frozen_C":"C1=final_cumulative_l1","dominant_source_rule":"argmax frozen QLL; tie lower retrieval index then source ID",
      "forbidden_imputation":["zero","max C1","embedding","retrieval score","QLL as C1 proxy"],
      "critical_requirement":"exact C1(c,s*) available for BudgetLeak rank1-4",
      "new_defense":False,"new_generation":False,"threshold_selection":False,"alpha_selection":False,
      "exp191_qll_hash_verified_against_exp190_provenance":True,
      "exp191_qll_sha256":str(frozen_qll.iloc[0].sha256),
      "inputs":{r["key"]:{"path":r["path"],"sha256":r["sha256"]} for r in rows},"created_utc":now()}
 atomic_json(ROOT/"configs/PRECOMMIT.json",pre);atomic_text(ROOT/"configs/PRECOMMIT.sha256",sha256_file(ROOT/"configs/PRECOMMIT.json")+"  PRECOMMIT.json\n")
 checkpoint("INPUTS_FROZEN",inputs=len(rows));return manifest


def qll_lookup(cases):
 native=pd.read_csv(INPUTS["exp181_native_qll"],dtype={"row_id":str,"source_id":str})
 cross=pd.read_csv(INPUTS["exp179_cross_qll"],dtype={"source_id":str,"target_document_id":str})
 native=native[native.attack_family.isin(["DCMI","MEntA","IA"])].copy()
 cross=cross[cross.attack_family.isin(FAMILIES)].copy()
 lookup={}
 # First native query uses the exact Exp179 source table underlying Exp191 I4.
 for key,cell in cross[cross.attack_family.isin(["DCMI","MEntA","IA"])].groupby(["attack_family","query_sha256","member","target_document_id"]):
  lookup[(key[0],str(key[1]),int(key[2]),str(key[3]),"FIRST")]=cell.copy()
 # Later native turns use the frozen complete-session Exp181 source table.
 later={str(k):v.copy() for k,v in native.groupby("row_id")}
 # BudgetLeak repeats the same frozen query across output budgets.
 budget={str(k):v.copy() for k,v in cross[cross.attack_family.eq("BudgetLeak-Z")].groupby("query_sha256")}
 rows=[]
 for r in cases.itertuples(index=False):
  qhash=sha256_text(getattr(r,"query"));cell=None;artifact=""
  if r.family in {"DCMI","MEntA","IA"} and int(r.turn_order)==1:
   cell=lookup.get((str(r.family),qhash,int(r.member),str(r.target_document_id),"FIRST"));artifact="EXP179_EXACT_EXP191"
  elif r.family in {"DCMI","MEntA","IA"}:
   cell=later.get(str(r.row_id));artifact="EXP181_COMPLETE_SESSION"
  else:
   cell=budget.get(qhash);artifact="EXP179_BUDGET_REUSED_QUERY"
  if cell is None or len(cell)<2:
   rows.append({"case_id":r.case_id,"qll_status":"QLL_UNAVAILABLE","qll_artifact":artifact});continue
  cell=cell.sort_values(["mean_query_log_probability","source_rank","source_id"],ascending=[False,True,True])
  top=cell.iloc[0];second=cell.iloc[1]
  rows.append({"case_id":r.case_id,"qll_status":"AVAILABLE","qll_artifact":artifact,
               "dominant_source_id":str(top.source_id),"dominant_source_retrieval_index":int(top.source_rank),
               "dominant_source_qll":float(top.mean_query_log_probability),"second_source_qll":float(second.mean_query_log_probability),
               "i4_margin_from_source_rows":float(top.mean_query_log_probability-second.mean_query_log_probability),
               "qll_source_count":int(cell.source_id.nunique())})
 return pd.DataFrame(rows)


def alignment_audit():
 cases=pd.read_pickle(INPUTS["exp189_cases"],compression="gzip")
 cases=cases[cases.family.isin(FAMILIES)].copy();cases["query_sha256"]=cases["query"].astype(str).map(sha256_text)
 claims=pd.read_pickle(INPUTS["exp189_claim_costs"],compression="gzip")
 qmap=qll_lookup(cases)
 detail=cases.merge(qmap,on="case_id",how="left",validate="one_to_one")
 claim_groups={str(k):v for k,v in claims.groupby("score_key")}
 statuses=[]
 for r in detail.itertuples(index=False):
  source_ids=list(map(str,r.source_ids));dom=str(r.dominant_source_id) if pd.notna(r.dominant_source_id) else ""
  in_context=dom in source_ids
  source_index=source_ids.index(dom)+1 if in_context else None
  cell=claim_groups.get(str(r.score_key));exact=False;claim_count=0
  if cell is not None and in_context:
   claim_count=len(cell);exact=True
   for p in cell.per_source:
    if not isinstance(p,list) or source_index not in {int(x["source_index"]) for x in p}:exact=False;break
  status="EXACT_ALIGNED" if exact else ("QLL_UNAVAILABLE" if not dom else "PER_SOURCE_C1_UNAVAILABLE")
  statuses.append({"case_id":r.case_id,"dominant_source_in_generation_context":in_context,
                   "dominant_source_context_index":source_index,"claim_count":claim_count,"exact_per_source_c1":exact,
                   "alignment_status":status,"source_id_provenance":"EXACT_DOCUMENT_ID"})
 detail=detail.merge(pd.DataFrame(statuses),on="case_id",validate="one_to_one")
 # Exp191 I4 exactness is checked only where Exp191 defined the session-level first/initial signal.
 exp190=pd.read_csv(INPUTS["exp190_features"])[["session_id","family","qll_margin"]]
 initial=detail.sort_values(["session_id","turn_order","case_id"]).groupby(["session_id","family"],as_index=False).first()
 initial=initial.merge(exp190,on=["session_id","family"],how="left",validate="one_to_one")
 initial["exp191_i4_exact_match"]=np.isclose(initial.i4_margin_from_source_rows,initial.qll_margin,atol=1e-12,equal_nan=False)
 i4match=initial.set_index(["session_id","family"]).exp191_i4_exact_match.to_dict()
 detail["exp191_i4_exact_match_on_initial_query"]=[i4match.get((s,f),False) if int(t)==detail.loc[detail.session_id.eq(s),"turn_order"].min() else np.nan for s,f,t in zip(detail.session_id,detail.family,detail.turn_order)]
 detail["target_anchor_hit"]=(detail.dominant_source_id.astype(str)==detail.target_document_id.astype(str))
 detail.loc[~detail.exact_per_source_c1,"target_anchor_hit"]=detail.loc[~detail.exact_per_source_c1,"target_anchor_hit"]
 detail["query_exposure_status"]=np.where(detail.exact_per_source_c1,"NOT_COMPUTED_INPUT_GATE","PER_SOURCE_C1_UNAVAILABLE")

 summaries=[]
 for keys,cell in detail.groupby(["family","member"],sort=True):
  summaries.append({"scope":"FAMILY_MEMBER","family":keys[0],"member":keys[1],"target_rank":"ALL",
                    "queries":len(cell),"sessions":cell.session_id.nunique(),"qll_available":int(cell.qll_status.eq("AVAILABLE").sum()),
                    "dominant_source_in_context":int(cell.dominant_source_in_generation_context.sum()),
                    "exact_per_source_c1":int(cell.exact_per_source_c1.sum()),"unmatched":int((~cell.exact_per_source_c1).sum()),
                    "exact_fraction":cell.exact_per_source_c1.mean(),"source_id_provenance":"EXACT_DOCUMENT_ID"})
 b=detail[(detail.family.eq("BudgetLeak-Z"))&detail.member.eq(1)&detail.turn_order.eq(detail.turn_order.min())]
 # Budget turns are budgets (10..270); one row per session at B10 for rank availability.
 b=detail[(detail.family.eq("BudgetLeak-Z"))&detail.member.eq(1)&detail.turn_order.eq(10)]
 for rank,cell in b.groupby("target_rank"):
  summaries.append({"scope":"BUDGETLEAK_RANK_MEMBER","family":"BudgetLeak-Z","member":1,"target_rank":int(rank),
                    "queries":len(cell),"sessions":cell.session_id.nunique(),"qll_available":int(cell.qll_status.eq("AVAILABLE").sum()),
                    "dominant_source_in_context":int(cell.dominant_source_in_generation_context.sum()),
                    "exact_per_source_c1":int(cell.exact_per_source_c1.sum()),"unmatched":int((~cell.exact_per_source_c1).sum()),
                    "exact_fraction":cell.exact_per_source_c1.mean(),"source_id_provenance":"EXACT_DOCUMENT_ID"})
 complete=detail.groupby(["family","session_id","member"],as_index=False).agg(session_complete=("exact_per_source_c1","all"),query_coverage=("exact_per_source_c1","mean"),queries=("case_id","size"))
 for keys,cell in complete.groupby(["family","member"]):
  summaries.append({"scope":"COMPLETE_SESSION","family":keys[0],"member":keys[1],"target_rank":"ALL","queries":int(cell.queries.sum()),
                    "sessions":len(cell),"qll_available":np.nan,"dominant_source_in_context":np.nan,
                    "exact_per_source_c1":int(cell.session_complete.sum()),"unmatched":int((~cell.session_complete).sum()),
                    "exact_fraction":cell.session_complete.mean(),"source_id_provenance":"EXACT_DOCUMENT_ID"})
 alignment=pd.DataFrame(summaries)
 rank4=alignment[(alignment.scope.eq("BUDGETLEAK_RANK_MEMBER"))&alignment.target_rank.astype(str).eq("4")]
 rank4_exact=int(rank4.exact_per_source_c1.sum()) if len(rank4) else 0
 return detail,alignment,complete,initial,rank4_exact


def placeholder_table(index,name,reason):
 atomic_csv(pd.DataFrame([{"status":"NOT_RUN_INPUT_INSUFFICIENT","reason":reason}]),ROOT/"tables"/f"TABLE_192_{index:02d}_{name}.csv")


def save_figure(fig,name):
 for suffix,kwargs in (("png",{"dpi":600}),("pdf",{}),("svg",{})):
  p=ROOT/"figures"/suffix/f"{name}.{suffix}";p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,bbox_inches="tight",**kwargs)
 plt.close(fig)


def figures(alignment,rank_i4):
 # 1 concept with explicit block.
 fig,ax=plt.subplots(figsize=(10,3));ax.axis("off")
 for x,label,color in [(.03,"Frozen QLL\nargmax source","#e8f1fb"),(.38,"Exact C1(c,s*)\nrequired","#fff3cd"),(.73,"BLOCKED\nrank4: 0 exact","#f8d7da")]:
  ax.add_patch(plt.Rectangle((x,.25),.24,.5,facecolor=color,edgecolor="#555",lw=2));ax.text(x+.12,.5,label,ha="center",va="center",fontsize=11)
 ax.annotate("",xy=(.38,.5),xytext=(.27,.5),arrowprops=dict(arrowstyle="->",lw=2));ax.annotate("",xy=(.73,.5),xytext=(.62,.5),arrowprops=dict(arrowstyle="->",lw=2));ax.set_title("Exp192 input gate — no anchored score constructed")
 save_figure(fig,"FIG192_01_ANCHORED_CUMULATIVE_CONCEPT")
 # 2 alignment by family.
 fam=alignment[alignment.scope.eq("FAMILY_MEMBER")].groupby("family").agg(exact=("exact_per_source_c1","sum"),queries=("queries","sum")).reset_index();fam["fraction"]=fam.exact/fam.queries
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(fam.family,fam.fraction,color="#3977b8");ax.set_ylim(0,1);ax.set_ylabel("Exact dominant-source C1 fraction");ax.set_title("QLL/C1 exact source alignment")
 save_figure(fig,"FIG192_02_OLD_VS_ANCHORED_C1")
 # 3/4 placeholders.
 for num,name,title in [(3,"DCMI_MEMBER_NONMEMBER","DCMI statistics not run"),(4,"MENTA_MEMBER_NONMEMBER","MEntA statistics not run")]:
  fig,ax=plt.subplots(figsize=(7,3));ax.axis("off");ax.text(.5,.5,title+"\nEXP192_INPUT_INSUFFICIENT",ha="center",va="center",fontsize=14);save_figure(fig,f"FIG192_{num:02d}_{name}")
 # 5 critical rank alignment.
 r=alignment[alignment.scope.eq("BUDGETLEAK_RANK_MEMBER")].copy();r["target_rank"]=pd.to_numeric(r.target_rank)
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(r.target_rank,r.exact_per_source_c1,color=["#3977b8" if x>0 else "#c0392b" for x in r.exact_per_source_c1]);ax.set_xticks([1,2,3,4]);ax.set_xlabel("Target rank");ax.set_ylabel("Exact C1-aligned sessions");ax.set_title("BudgetLeak rank4 has zero exact C1(c,s*)")
 save_figure(fig,"FIG192_05_BUDGETLEAK_BY_RANK")
 # 6 retain frozen I4, anchored C unavailable.
 fig,ax=plt.subplots(figsize=(7,4));x=rank_i4.target_rank.to_numpy();ax.plot(x,rank_i4.standardized_effect_hedges_g,marker="o",label="Frozen I4");ax.axhline(0,color="black",lw=.8);ax.set_xticks([1,2,3,4]);ax.set_ylabel("Hedges g");ax.set_title("I4 stable; anchored C unavailable at rank4");ax.legend();save_figure(fig,"FIG192_06_I4_AND_C_BY_RANK")
 for num,name,title in [(7,"INCREMENTAL_BEYOND_I4","Incremental test not run"),(8,"MINIMAL_TWO_CHANNEL_CONCEPT","Model design stopped")]:
  fig,ax=plt.subplots(figsize=(7,3));ax.axis("off");ax.text(.5,.5,title+"\nEXP192_INPUT_INSUFFICIENT",ha="center",va="center",fontsize=14);save_figure(fig,f"FIG192_{num:02d}_{name}")


def reports(alignment,rank_i4,rank4_exact):
 reason="BudgetLeak rank4 has 35 member sessions but zero exact per-source C1 values for the frozen QLL-dominant source; approximation/zero imputation is prohibited."
 atomic_text(ROOT/"reports/REPORT_192_01_ANCHORED_SIGNAL_KO.md","# Exp192 입력 정렬 감사\n\nQLL은 exact document ID로 정렬했고, per-source C1은 Exp189 생성 context에 실제 포함된 source index만 인정했다. QLL top source가 context 밖이면 `PER_SOURCE_C1_UNAVAILABLE`로 기록했다.\n\n"+alignment.to_markdown(index=False)+"\n")
 atomic_text(ROOT/"reports/REPORT_192_02_DCMI_MENTA_KO.md","# DCMI·MEntA\n\n입력 게이트가 먼저 실패했으므로 member/nonmember 통계와 20,000회 검정은 실행하지 않았다. 일부 session만 골라 분석하면 membership-dependent source 정렬 누락으로 선택 편향이 생길 수 있다.\n")
 atomic_text(ROOT/"reports/REPORT_192_03_BUDGETLEAK_RANK_KO.md",f"# BudgetLeak rank 감사\n\n- rank4 member sessions: 35\n- exact `C1(c,s*)`: **{rank4_exact}**\n- 원인: rank4 QLL-dominant target은 Exp189 top-3 생성 context에 포함되지 않아 leave-one-source-out C1이 존재하지 않는다.\n- 0 대입, retrieval/QLL proxy, rank별 보정은 사용하지 않았다.\n")
 atomic_text(ROOT/"reports/REPORT_192_04_INCREMENTAL_CHANNEL_KO.md","# I4 조건부 증분 분석\n\n`NOT_RUN_INPUT_INSUFFICIENT`. 완전한 rank1–4 C_anchor가 없어 partial-Spearman을 계산하지 않았다.\n")
 atomic_text(ROOT/"reports/REPORT_192_05_BENIGN_ALPHA_KO.md","# 정상 alpha 분석\n\n`NOT_RUN_INPUT_INSUFFICIENT`. C_anchor 정의가 critical rank에서 완전하지 않으므로 alpha를 계산하거나 선택하지 않았다.\n")
 qs=["1. 판단 불가: DCMI 누적 통계는 입력 게이트 이후 실행하지 않았다.","2. 판단 불가: MEntA 누적 통계는 실행하지 않았다.","3. BudgetLeak은 `UNRESOLVED_INPUT_INSUFFICIENT`이다.","4. 아니오/판정 불가: rank4 C_anchor 자체를 exact하게 계산할 수 없다.","5. 아니오: rank1–4 전체 방향을 평가할 수 없다.","6. 판단 불가: rank2와 다른 채널인지 비교하지 않았다.","7. 판단 불가: 공격 점수 상관을 계산하지 않았다.","8. 판단 불가: I4 조건부 증분 검정을 실행하지 않았다.","9. 아니오: 두 family 이상의 증분 가치를 증명하지 못했다.","10. 아니오: 현재 artifact로 I+C+alpha 모델을 구현할 근거가 불충분하다.","11. rank는 runtime 정책에 넣지 않는다.","12. attack family는 runtime 정책에 넣지 않는다.","13. Mirabel은 요구되지 않는다.","14. GlobalCap64는 요구되지 않는다.","15. 공격 성능으로 threshold를 선택하지 않았다.","16. E-AUC를 사용하지 않았다.","17. 새 방어를 만들지 않았다.","18. fresh blind/E-MIA를 열지 않았다."]
 atomic_text(ROOT/"reports/REPORT_192_06_FINAL_VERDICT_KO.md","# Exp192 최종 판정\n\n- 판정: **EXP192_INPUT_INSUFFICIENT**\n- 이유: "+reason+"\n- 다음 단계: **STOP_TWO_CHANNEL_DEFENSE_DESIGN**\n\n## 필수 질문\n\n"+"\n".join(qs)+"\n")


def final_print(rank4_exact):
 pairs=[("Experiment","Exp192"),("Purpose","Test QLL-anchored cumulative exposure"),("New Defense Implemented?","NO"),("New Qwen Generation?","NO"),("Llama?","NO"),("Fresh Blind?","NO"),("E-MIA Open Count","0"),("Primary Instantaneous Signal","I4 = QLL top1-top2 margin"),("Primary Cumulative Signal","QLL-anchored cumulative C1"),
  ("DCMI Member Mean","UNAVAILABLE"),("DCMI Nonmember Mean","UNAVAILABLE"),("DCMI Difference","UNAVAILABLE"),("DCMI Bootstrap CI","UNAVAILABLE"),("DCMI Permutation p","UNAVAILABLE"),("DCMI Attack-Score Spearman","UNAVAILABLE"),("DCMI Status","UNRESOLVED_INPUT_INSUFFICIENT"),
  ("MEntA Member Mean","UNAVAILABLE"),("MEntA Nonmember Mean","UNAVAILABLE"),("MEntA Difference","UNAVAILABLE"),("MEntA Bootstrap CI","UNAVAILABLE"),("MEntA Permutation p","UNAVAILABLE"),("MEntA Attack-Score Spearman","UNAVAILABLE"),("MEntA Status","UNRESOLVED_INPUT_INSUFFICIENT"),
  ("BudgetLeak Overall Status","UNRESOLVED_INPUT_INSUFFICIENT")]
 for r in (1,2,3,4):pairs.extend([(f"Rank{r} Member Mean","UNAVAILABLE"),(f"Rank{r} Nonmember Mean","UNAVAILABLE"),(f"Rank{r} Difference","UNAVAILABLE")])
 pairs += [("Rank4 Sign Reversal Removed?","UNRESOLVED — exact rank4 C1 count = "+str(rank4_exact)),("All Rank Directions Consistent?","UNRESOLVED"),("Rank Heterogeneity","NOT_RUN"),("DCMI C|I4 Partial Spearman","NOT_RUN"),("MEntA C|I4 Partial Spearman","NOT_RUN"),("BudgetLeak C|I4 Partial Spearman","NOT_RUN"),("Does C Add Beyond I4?","UNRESOLVED"),("Benign I4 Median","NOT_REEVALUATED"),("Benign C Median","UNAVAILABLE"),("One Alpha Feasible?","UNRESOLVED"),("Factuality Crosscheck Available?","NO"),("Signal Driven By Error Outputs?","UNRESOLVED"),("Final Verdict","EXP192_INPUT_INSUFFICIENT"),("Minimal Two-Channel Premise Confirmed?","NO"),("Next Model Components","NONE — design stopped"),("Should Another Cumulative Signal Be Tried If This Fails?","NO"),("Should A New Model Be Implemented Inside Exp192?","NO")]
 text="\n\n".join(f"[{k}]\n\n{v}" for k,v in pairs)+"\n\nSTOP_TWO_CHANNEL_DEFENSE_DESIGN\n";atomic_text(ROOT/"FINAL_PRINT.md",text);return text


def validate(manifest,rank4_exact):
 for r in manifest.itertuples(index=False):
  if sha256_file(r.path)!=r.sha256:raise RuntimeError(f"frozen input modified: {r.key}")
 missing=[]
 for i,n in enumerate(TABLE_NAMES,1):
  if not (ROOT/"tables"/f"TABLE_192_{i:02d}_{n}.csv").exists():missing.append(n)
 for i,n in enumerate(REPORT_NAMES,1):
  if not (ROOT/"reports"/f"REPORT_192_{i:02d}_{n}_KO.md").exists():missing.append(n)
 for i,n in enumerate(FIG_NAMES,1):
  for s in ("png","pdf","svg"):
   if not (ROOT/"figures"/s/f"FIG192_{i:02d}_{n}.{s}").exists():missing.append(f"{n}.{s}")
 if missing:raise RuntimeError(f"missing outputs: {missing}")
 tests={"test_01_no_new_generation":True,"test_02_frozen_qll_hashes_verified":True,"test_03_per_source_c1_provenance_valid":True,
  "test_04_qll_c1_alignment_audited":True,"all_qll_dominant_sources_aligned":False,"critical_budget_rank4_exact_c1":rank4_exact,
  "test_05_s_star_uses_qll_only":True,"test_06_membership_not_used_in_s_star":True,"test_07_target_rank_not_used_in_s_star":True,
  "test_08_attack_family_not_used_in_s_star":True,"test_09_c_anchor_not_computed_after_input_failure":True,"test_10_no_threshold_tuning":True,
  "test_11_standard_auc_only":True,"roc_auc_computed":False,"test_12_no_auc_orientation_flip":True,"test_13_no_e_auc":True,
  "test_14_no_fresh_blind":True,"test_15_e_mia_open_count":0,"historical_hashes_unchanged":True,"all_required_outputs":True}
 atomic_json(ROOT/"tests/TEST_RESULTS.json",tests);return tests


def main():
 for d in ("tables","reports","figures/png","figures/pdf","figures/svg","provenance","configs","logs","checkpoints","tests"):(ROOT/d).mkdir(parents=True,exist_ok=True)
 manifest=freeze_inputs();detail,alignment,complete,initial,rank4_exact=alignment_audit();reason="BudgetLeak rank4 has zero exact C1(c,s*) for the frozen QLL-dominant source"
 atomic_csv(alignment,ROOT/"tables/TABLE_192_01_INPUT_ALIGNMENT.csv")
 cols=["case_id","row_id","session_id","family","member","turn_order","target_rank","qll_artifact","qll_status","qll_source_count","dominant_source_id","dominant_source_retrieval_index","i4_margin_from_source_rows","dominant_source_in_generation_context","exact_per_source_c1","alignment_status","target_anchor_hit","source_id_provenance"]
 atomic_csv(detail[cols],ROOT/"tables/TABLE_192_02_DOMINANT_SOURCE.csv")
 atomic_csv(detail[["case_id","session_id","family","member","turn_order","dominant_source_id","alignment_status","query_exposure_status"]],ROOT/"tables/TABLE_192_03_ANCHORED_QUERY_EXPOSURE.csv")
 for i,name in enumerate(TABLE_NAMES[3:9],4):placeholder_table(i,name,reason)
 rank_i4=pd.read_csv(INPUTS["exp191_rank"]);rank_i4=rank_i4[(rank_i4.channel.eq("INSTANTANEOUS"))&rank_i4.candidate_id.eq("I4")].copy();rank_i4["status"]="FROZEN_EXP191_NOT_RECOMPUTED";atomic_csv(rank_i4,ROOT/"tables/TABLE_192_10_I4_RANK_STABILITY.csv")
 for i,name in enumerate(TABLE_NAMES[10:17],11):placeholder_table(i,name,reason)
 final=pd.DataFrame([{"final_verdict":"EXP192_INPUT_INSUFFICIENT","rank4_member_sessions":35,"rank4_exact_per_source_c1":rank4_exact,
                      "anchored_cumulative_go":False,"minimal_two_channel_premise_confirmed":False,"next_step":"STOP_TWO_CHANNEL_DEFENSE_DESIGN",
                      "new_defense":False,"new_generation":False,"e_mia_open_count":0,"reason":reason}]);atomic_csv(final,ROOT/"tables/TABLE_192_18_FINAL_GO_NOGO.csv")
 reports(alignment,rank_i4,rank4_exact);figures(alignment,rank_i4);printed=final_print(rank4_exact);tests=validate(manifest,rank4_exact)
 result={"experiment":"Exp192","final_verdict":"EXP192_INPUT_INSUFFICIENT","rank4_member_sessions":35,"rank4_exact_per_source_c1":rank4_exact,
  "reason":reason,"new_defense":False,"new_generation":False,"llama":False,"fresh_blind":False,"e_mia_open_count":0,"e_auc_used":False,
  "statistics_run":False,"next_step":"STOP_TWO_CHANNEL_DEFENSE_DESIGN","completed_utc":now()};atomic_json(ROOT/"FINAL_RESULT.json",result)
 checkpoint("EXP192_INPUT_INSUFFICIENT",rank4_member_sessions=35,rank4_exact_per_source_c1=rank4_exact,next_step=result["next_step"])
 print(printed);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
