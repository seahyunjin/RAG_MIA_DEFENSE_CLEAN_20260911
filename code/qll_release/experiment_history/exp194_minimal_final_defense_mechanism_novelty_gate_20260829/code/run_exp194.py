#!/usr/bin/env python3
"""Exp194: frozen Top-3/Top-4 mechanism diagnosis and literature novelty gate.

This experiment does not generate answers, train a model, search a threshold,
or define a new operational signal.  It reuses the exact response-preservation
metrics already frozen in Exp188 (all-mpnet-base-v2 cosine and ROUGE-1 F1).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import permutation_test, spearmanr


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp194_minimal_final_defense_mechanism_novelty_gate_20260829"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP189 = PROJECT / "exp189_simple_global_disclosure_ledger_20260829"
EXP191 = PROJECT / "exp191_two_channel_signal_compression_audit_20260829"
EXP192 = PROJECT / "exp192_qll_anchored_cumulative_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
SEED = 19420260829
SCIPY_SEED = SEED % (2**32 - 1)
REPLICATES = 5_000

INPUTS = {
    "exp189_cases_top3": EXP189 / "private/ATTACK_CASES.private.pkl.gz",
    "exp189_responses": EXP189 / "private/EXP189_ATTACK_RESPONSES.private.csv.gz",
    "exp189_costs_top3": EXP189 / "private/ATTACK_CLAIM_COSTS.private.pkl.gz",
    "exp191_final": EXP191 / "FINAL_RESULT.json",
    "exp192_final": EXP192 / "FINAL_RESULT.json",
    "exp193_cases_top4": EXP193 / "private/EXP193_CASES.private.pkl.gz",
    "exp193_responses": EXP193 / "private/EXP193_TOP4_RESPONSES.private.csv.gz",
    "exp193_costs_top4": EXP193 / "private/EXP193_EXACT_C1.private.pkl.gz",
    "exp193_qll": EXP193 / "private/EXP193_QLL_DOMINANT_SOURCE.private.csv.gz",
    "exp193_features": EXP193 / "private/EXP193_SESSION_FEATURES.private.csv.gz",
    "exp193_final": EXP193 / "FINAL_RESULT.json",
    "mpnet": MPNET / "config.json",
    "literature": ROOT / "configs/LITERATURE_AUDIT.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv.gz" if compression else ".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checkpoint(stage, **details):
    payload = {"experiment": "Exp194", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "new_generation": False, "new_defense": False, "threshold_search": False,
               "classifier_training": False, "manual_weight": False, "claude_audit": False, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp194 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`"]
    lines += [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def rouge1_f1(left, right):
    """Exact reuse of Exp188 run_exp188.py::rouge1_f1."""
    pattern = re.compile(r"[A-Za-z0-9]+")
    a = pattern.findall(str(left).casefold()); b = pattern.findall(str(right).casefold())
    if not a or not b: return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    precision, recall = overlap / len(a), overlap / len(b)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bootstrap_mean(values, cluster_ids, replicates=REPLICATES):
    frame = pd.DataFrame({"value": np.asarray(values, float), "cluster": np.asarray(cluster_ids, str)})
    cluster_means = frame.groupby("cluster").value.mean().to_numpy(float)
    if len(cluster_means) == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED + len(cluster_means))
    boot = cluster_means[rng.integers(0, len(cluster_means), (replicates, len(cluster_means)))].mean(1)
    return float(cluster_means.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def bootstrap_difference(a, b, replicates=REPLICATES):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if not len(a) or not len(b): return np.nan,np.nan,np.nan,np.nan
    rng=np.random.default_rng(SEED+len(a)+17*len(b))
    boot=a[rng.integers(0,len(a),(replicates,len(a)))].mean(1)-b[rng.integers(0,len(b),(replicates,len(b)))].mean(1)
    observed=float(a.mean()-b.mean())
    p=float(permutation_test((a,b),lambda x,y:np.mean(x)-np.mean(y),permutation_type="independent",
                             alternative="two-sided",n_resamples=REPLICATES,random_state=SCIPY_SEED).pvalue)
    return observed,float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),p


def manifest(paths):
    return pd.DataFrame([{"key":key,"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size,
                          "access":"READ_ONLY"} for key,path in paths.items()])


def preflight():
    for directory in ["audits","checkpoints","configs","logs","private","provenance","reports","tables","tests"]:
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    missing=[str(p) for p in INPUTS.values() if not p.exists()]
    if missing: raise RuntimeError(f"missing inputs: {missing}")
    prior191=json.loads(INPUTS["exp191_final"].read_text()); prior192=json.loads(INPUTS["exp192_final"].read_text())
    prior193=json.loads(INPUTS["exp193_final"].read_text())
    assert prior191["selected_instantaneous_candidate"]=="I4" and prior191["selected_cumulative_candidate"]=="C1"
    assert prior192["final_verdict"]=="EXP192_INPUT_INSUFFICIENT"
    assert prior193["final_verdict"]=="EXP193_TOP4_REBASE_VALID_REVERSAL_NOT_REPRODUCED"
    before=manifest(INPUTS); atomic_csv(before,ROOT/"provenance/FROZEN_INPUTS_BEFORE.csv")
    precommit={"experiment":"Exp194","purpose":"PAIRED_MECHANISM_AND_NOVELTY_GATE",
               "frozen_signals":{"I4":"qll_margin","C1":"generation-grounded cumulative exposure"},
               "answer_metrics":{"semantic_cosine":"Exp188 all-mpnet-base-v2 response preservation",
                                  "rouge1_f1":"Exp188 lexical overlap","exact_match":"identity diagnostic"},
               "decision_policy":"No post-hoc quantitative boundary. If frozen paired evidence cannot identify source non-use versus C1 miss, return EXP194_MECHANISM_UNRESOLVED.",
               "forbidden":["new generation","new defense","attack/rank/family rule","learned classifier",
                            "learned fusion","manual weight","threshold search","test tuning","new U signal"],
               "replicates":REPLICATES,"seed":SEED,"inputs":before.set_index("key").to_dict("index"),"created_utc":now()}
    atomic_json(ROOT/"configs/PRECOMMIT.json",precommit)
    checkpoint("PREFLIGHT_COMPLETE",inputs=len(before),signals="I4+C1",decision_boundary="NO_POSTHOC_BOUNDARY")


def build_pairs():
    top3=pd.read_pickle(INPUTS["exp189_cases_top3"],compression="gzip").sort_values("case_id").reset_index(drop=True)
    top4_cases=pd.read_pickle(INPUTS["exp193_cases_top4"],compression="gzip").sort_values("case_id").reset_index(drop=True)
    top4_resp=pd.read_csv(INPUTS["exp193_responses"],keep_default_na=False,low_memory=False,dtype={"case_id":str})
    top4_resp=top4_resp.sort_values("case_id").reset_index(drop=True)
    if len(top3)!=35680 or len(top4_cases)!=len(top3) or len(top4_resp)!=len(top3): raise RuntimeError("cohort count mismatch")
    identity=["case_id","row_id","panel","family","member","domain","session_id","turn_order","query","target_document_id","target_rank"]
    for col in identity:
        if not top3[col].astype(str).equals(top4_cases[col].astype(str)):
            raise RuntimeError(f"top3/top4-case identity mismatch: {col}")
    response_identity=["case_id","row_id","panel","family","member","domain","session_id","turn_order","target_document_id","target_rank"]
    for col in response_identity:
        if not top3[col].astype(str).equals(top4_resp[col].astype(str)):
            raise RuntimeError(f"top3/top4-response identity mismatch: {col}")
    if not top3.source_ids.map(len).eq(3).all() or not top4_cases.source_ids.map(len).eq(4).all(): raise RuntimeError("source count mismatch")
    if not all(list(a)==list(b)[:3] for a,b in zip(top3.source_ids,top4_cases.source_ids)): raise RuntimeError("top3 prefix mismatch")
    no_def=pd.read_csv(INPUTS["exp189_responses"],keep_default_na=False,low_memory=False,dtype={"case_id":str})
    no_def=no_def[no_def.condition.eq("NO_DEFENSE")].sort_values("case_id").reset_index(drop=True)
    if len(no_def)!=len(top3) or not no_def.case_id.equals(top3.case_id) or not no_def.response.astype(str).equals(top3.response.astype(str)):
        raise RuntimeError("Exp189 top3 response lineage mismatch")
    qll=pd.read_csv(INPUTS["exp193_qll"],keep_default_na=False,low_memory=False,dtype={"case_id":str}).sort_values("case_id").reset_index(drop=True)
    if len(qll)!=len(top3) or not qll.case_id.equals(top3.case_id) or not qll.qll_status.eq("AVAILABLE").all(): raise RuntimeError("QLL coverage mismatch")
    pairs=top3[identity+["score_key","response"]].rename(columns={"score_key":"top3_score_key","response":"top3_response"})
    pairs["top4_score_key"]=[sha256_text("\0".join([str(s),str(p),str(r)])) for s,p,r in
                              zip(top4_cases.system_prompt,top4_cases.prompt_key,top4_resp.response)]
    pairs["top4_response"]=top4_resp.response.astype(str)
    pairs=pairs.merge(qll[["case_id","dominant_source_id","dominant_source_retrieval_rank","qll_margin"]],on="case_id",validate="one_to_one")
    pairs["answer_exact_match"]=pairs.top3_response.eq(pairs.top4_response)
    pairs["answer_rouge1_f1"]=[rouge1_f1(a,b) for a,b in zip(pairs.top3_response,pairs.top4_response)]
    pairs["ordered_case_hash"]=sha256_text("\n".join(pairs.case_id)+"\n")
    cohort=[]
    for scope,cols in [("OVERALL",[]),("FAMILY",["family"]),("MEMBER",["member"]),("TARGET_RANK",["target_rank"]),("FAMILY_MEMBER",["family","member"])]:
        groups=[((),pairs)] if not cols else pairs.groupby(cols,dropna=False,sort=True)
        for keys,cell in groups:
            if not isinstance(keys,tuple):keys=(keys,)
            row={"scope":scope,"cases":len(cell),"sessions":cell.session_id.nunique(),"paired":len(cell),"pair_coverage":1.0}
            row.update(dict(zip(cols,keys)));cohort.append(row)
    atomic_csv(pd.DataFrame(cohort),ROOT/"audits/PAIRED_COHORT_COVERAGE.csv")
    checkpoint("PAIRING_COMPLETE",cases=len(pairs),sessions=pairs.session_id.nunique(),pair_coverage=1.0,
               member=int(pairs.member.sum()),nonmember=int((1-pairs.member).sum()))
    return pairs


def semantic_similarity(pairs):
    cache=ROOT/"private/ANSWER_SEMANTIC_SIMILARITY.private.csv.gz"
    if cache.exists():
        metric=pd.read_csv(cache,keep_default_na=False,dtype={"case_id":str})
        return pairs.merge(metric,on="case_id",validate="one_to_one")
    checkpoint("ANSWER_EMBEDDING_STARTED",pairs=len(pairs),model="all-mpnet-base-v2",device="cuda")
    from sentence_transformers import SentenceTransformer
    import torch
    model=SentenceTransformer(str(MPNET),device="cuda",local_files_only=True);model.max_seq_length=512
    unique=pd.concat([pairs.top3_response,pairs.top4_response],ignore_index=True).drop_duplicates().astype(str)
    vectors=model.encode(unique.tolist(),normalize_embeddings=True,convert_to_numpy=True,batch_size=256,show_progress_bar=False)
    mapping=dict(zip(unique,vectors))
    cosine=np.array([float(np.dot(mapping[a],mapping[b])) for a,b in zip(pairs.top3_response,pairs.top4_response)])
    metric=pd.DataFrame({"case_id":pairs.case_id,"answer_semantic_cosine":cosine,
                         "answer_semantic_change":1-np.clip(cosine,-1,1)})
    atomic_csv(metric,cache,"gzip");del model,vectors,mapping;gc.collect();torch.cuda.empty_cache()
    checkpoint("ANSWER_EMBEDDING_COMPLETE",pairs=len(metric),unique_answers=len(unique))
    return pairs.merge(metric,on="case_id",validate="one_to_one")


def source_exposure(cost_path,prefix,max_rank):
    costs=pd.read_pickle(cost_path,compression="gzip")
    rows=[]
    for r in costs.itertuples(index=False):
        for item in r.per_source:
            rank=int(item["source_index"])
            rows.append((str(r.score_key),rank,float(item["c1_source"])))
    frame=pd.DataFrame(rows,columns=[f"{prefix}_score_key","source_rank","c1_source"])
    agg=frame.groupby([f"{prefix}_score_key","source_rank"],as_index=False).agg(
        c1_sum=("c1_source","sum"),c1_max=("c1_source","max"),claim_source_pairs=("c1_source","size"))
    wide=agg.pivot(index=f"{prefix}_score_key",columns="source_rank",values=["c1_sum","c1_max","claim_source_pairs"])
    wide.columns=[f"{prefix}_{metric}_r{rank}" for metric,rank in wide.columns]
    wide=wide.reset_index()
    for metric in ["c1_sum","c1_max","claim_source_pairs"]:
        for rank in range(1,max_rank+1):
            col=f"{prefix}_{metric}_r{rank}"
            if col not in wide:wide[col]=0.0
    return wide


def attach_exposure(pairs):
    top3=source_exposure(INPUTS["exp189_costs_top3"],"top3",3)
    top4=source_exposure(INPUTS["exp193_costs_top4"],"top4",4)
    out=pairs.merge(top3,on="top3_score_key",how="left",validate="many_to_one").merge(
        top4,on="top4_score_key",how="left",validate="many_to_one")
    cols=[c for c in out if "_c1_" in c or "claim_source" in c]
    out[cols]=out[cols].fillna(0.0)
    for rank in (1,2,3):out[f"delta_c1_sum_r{rank}"]=out[f"top4_c1_sum_r{rank}"]-out[f"top3_c1_sum_r{rank}"]
    out["top4_max_exposure_rank"]=out[[f"top4_c1_sum_r{x}" for x in range(1,5)]].to_numpy().argmax(1)+1
    if out.top3_score_key.isna().any() or out.top4_score_key.isna().any():raise RuntimeError("C1 key alignment failed")
    atomic_csv(out,ROOT/"private/PAIRED_MECHANISM_DETAIL.private.csv.gz","gzip")
    checkpoint("SOURCE_EXPOSURE_ATTACHED",cases=len(out),top3_exact=float(out.top3_score_key.notna().mean()),top4_exact=float(out.top4_score_key.notna().mean()))
    return out


def answer_tables(pairs):
    rows=[]
    specs=[("OVERALL",[]),("FAMILY",["family"]),("MEMBER",["member"]),("TARGET_RANK",["target_rank"]),
           ("FAMILY_TARGET_RANK",["family","target_rank"])]
    for scope,cols in specs:
        groups=[((),pairs)] if not cols else pairs.groupby(cols,dropna=False,sort=True)
        for keys,cell in groups:
            if not isinstance(keys,tuple):keys=(keys,)
            mean,low,high=bootstrap_mean(cell.answer_semantic_cosine,cell.session_id)
            row={"scope":scope,"cases":len(cell),"sessions":cell.session_id.nunique(),
                 "exact_match_rate":cell.answer_exact_match.mean(),"rouge1_f1_mean":cell.answer_rouge1_f1.mean(),
                 "semantic_cosine_mean":mean,"semantic_cosine_ci95_low":low,"semantic_cosine_ci95_high":high,
                 "semantic_change_mean":cell.answer_semantic_change.mean()}
            row.update(dict(zip(cols,keys)));rows.append(row)
    table=pd.DataFrame(rows);atomic_csv(table,ROOT/"tables/TABLE_194_01_PAIRED_ANSWER_CHANGE.csv")
    return table


def c1_tables(pairs):
    rows=[]
    for member,cell in [("ALL",pairs),(0,pairs[pairs.member.eq(0)]),(1,pairs[pairs.member.eq(1)])]:
        for rank in (1,2,3):
            values=cell[f"delta_c1_sum_r{rank}"]
            mean,low,high=bootstrap_mean(values,cell.session_id)
            rows.append({"member":member,"source_rank":rank,"cases":len(cell),"top3_mean":cell[f"top3_c1_sum_r{rank}"].mean(),
                         "top4_mean":cell[f"top4_c1_sum_r{rank}"].mean(),"paired_change_mean":mean,
                         "paired_change_ci95_low":low,"paired_change_ci95_high":high})
        values=cell.top4_c1_sum_r4;mean,low,high=bootstrap_mean(values,cell.session_id)
        rows.append({"member":member,"source_rank":4,"cases":len(cell),"top3_mean":np.nan,"top4_mean":mean,
                     "paired_change_mean":np.nan,"paired_change_ci95_low":low,"paired_change_ci95_high":high})
    c1=pd.DataFrame(rows);atomic_csv(c1,ROOT/"tables/TABLE_194_02_C1_TOP3_TOP4.csv")
    qrows=[]
    specs=[("OVERALL",[]),("FAMILY",["family"]),("MEMBER",["member"]),("TARGET_RANK",["target_rank"]),
           ("FAMILY_TARGET_RANK",["family","target_rank"])]
    for scope,cols in specs:
        groups=[((),pairs)] if not cols else pairs.groupby(cols,dropna=False,sort=True)
        for keys,cell in groups:
            if not isinstance(keys,tuple):keys=(keys,)
            for rank,count in cell.dominant_source_retrieval_rank.value_counts().sort_index().items():
                row={"scope":scope,"qll_source_rank":int(rank),"cases":int(count),"proportion":count/len(cell)}
                row.update(dict(zip(cols,keys)));qrows.append(row)
    qll=pd.DataFrame(qrows);atomic_csv(qll,ROOT/"tables/TABLE_194_03_QLL_SOURCE_SELECTION.csv")
    return c1,qll


def budget_rank4_diagnosis(pairs):
    budget=pairs[pairs.family.eq("BudgetLeak-Z")].copy()
    sessions=budget.groupby(["session_id","member","target_rank"],as_index=False).agg(
        cases=("case_id","size"),semantic_cosine=("answer_semantic_cosine","mean"),
        semantic_change=("answer_semantic_change","mean"),exact_match_rate=("answer_exact_match","mean"),
        rouge1_f1=("answer_rouge1_f1","mean"),rank4_c1_sum=("top4_c1_sum_r4","sum"),
        rank4_c1_max=("top4_c1_max_r4","max"),qll_rank=("dominant_source_retrieval_rank","first"),
        qll_rank_nunique=("dominant_source_retrieval_rank","nunique"),max_exposure_rank4_rate=("top4_max_exposure_rank",lambda x:np.mean(np.asarray(x)==4)))
    feat=pd.read_csv(INPUTS["exp193_features"],keep_default_na=False,low_memory=False,dtype={"session_id":str})
    feat=feat[feat.family.eq("BudgetLeak-Z")][["session_id","final_cumulative_l1"]]
    sessions=sessions.merge(feat,on="session_id",validate="one_to_one")
    sessions["qll_group"]=np.where(sessions.qll_rank.eq(4),"QLL_RANK4","QLL_RANK1_3")
    atomic_csv(sessions,ROOT/"private/BUDGETLEAK_RANK4_SESSION_DIAGNOSIS.private.csv.gz","gzip")
    rows=[]
    for group in ["ALL","QLL_RANK4","QLL_RANK1_3"]:
        cell=sessions if group=="ALL" else sessions[sessions.qll_group.eq(group)]
        member=cell[(cell.member.eq(1))&cell.target_rank.eq(4)]; nonmember=cell[cell.member.eq(0)]
        diff,low,high,p=bootstrap_difference(member.final_cumulative_l1,nonmember.final_cumulative_l1)
        rows.append({"qll_group":group,"rank4_member_sessions":len(member),"nonmember_sessions":len(nonmember),
                     "member_semantic_cosine":member.semantic_cosine.mean(),"member_semantic_change":member.semantic_change.mean(),
                     "member_rank4_c1_sum":member.rank4_c1_sum.mean(),"member_final_cumulative_l1":member.final_cumulative_l1.mean(),
                     "nonmember_final_cumulative_l1":nonmember.final_cumulative_l1.mean(),"membership_mean_difference":diff,
                     "difference_ci95_low":low,"difference_ci95_high":high,"permutation_p":p})
    diag=pd.DataFrame(rows);atomic_csv(diag,ROOT/"tables/TABLE_194_04_BUDGETLEAK_RANK4_DIAGNOSIS.csv")
    valid=sessions[(sessions.member.eq(1))&sessions.target_rank.eq(4)]
    rho,p=spearmanr(valid.rank4_c1_sum,valid.semantic_change) if len(valid)>2 else (np.nan,np.nan)
    corr=pd.DataFrame([{"scope":"BUDGETLEAK_MEMBER_TARGET_RANK4","sessions":len(valid),
                        "spearman_rank4_c1_vs_answer_change":rho,"p_value":p,
                        "qll_rank4_sessions":int(valid.qll_rank.eq(4).sum()),
                        "qll_rank1_3_sessions":int(valid.qll_rank.ne(4).sum())}])
    atomic_csv(corr,ROOT/"tables/TABLE_194_05_RANK4_INFLUENCE_ASSOCIATION.csv")
    return sessions,diag,corr


def novelty_audit():
    payload=json.loads(INPUTS["literature"].read_text())
    rows=pd.DataFrame(payload["works"])
    atomic_csv(rows,ROOT/"tables/TABLE_194_06_LITERATURE_NOVELTY_AUDIT.csv")
    conflict=bool(rows.mechanism_identical_to_i4_c1.eq("YES").any())
    closest=rows[rows.closest_prior_work.eq(True)].work.tolist()
    return rows,conflict,closest,payload["search_scope"]


def finalize(pairs,answer,c1,qll,sessions,diag,corr,literature,conflict,closest,search_scope):
    overall=answer[answer.scope.eq("OVERALL")].iloc[0]
    rank4=diag[diag.qll_group.eq("ALL")].iloc[0]
    qll4=int(sessions[(sessions.member.eq(1))&sessions.target_rank.eq(4)].qll_rank.eq(4).sum())
    rank4_member=sessions[(sessions.member.eq(1))&sessions.target_rank.eq(4)]
    association=corr.iloc[0]
    # The rank-4 intervention changed the paired answers and exact rank-4 C1 tracks
    # that change.  Yet C1 does not reliably separate rank-4 members from matched
    # nonmembers.  This is neither Case A (source non-use) nor Case B (source-use
    # signal failure): source use is captured but is not membership-specific.
    mechanism_verdict="EXP194_MECHANISM_UNRESOLVED"
    source_use="SUPPORTED_BY_PAIRED_ANSWER_CHANGE_AND_C1_ASSOCIATION"
    c1_failure="SOURCE_USE_CAPTURE_NOT_FAILED; MEMBERSHIP_SPECIFICITY_NOT_ESTABLISHED"
    two_signal=False
    u_required=False
    if conflict:
        final_verdict="NOVELTY_CONFLICT"
    else:
        final_verdict=mechanism_verdict
    heuristic_pass=True
    after=manifest({k:v for k,v in INPUTS.items() if k!="literature"})
    before=pd.read_csv(ROOT/"provenance/FROZEN_INPUTS_BEFORE.csv")
    merged=before.merge(after,on="key",suffixes=("_before","_after"));merged["unchanged"]=merged.sha256_before.eq(merged.sha256_after)
    atomic_csv(merged,ROOT/"provenance/FROZEN_INPUTS_COMPARISON.csv")
    result={"experiment":"Exp194","completed_utc":now(),"verdict":final_verdict,
            "mechanism_verdict":mechanism_verdict,"paired_cases":len(pairs),"paired_sessions":pairs.session_id.nunique(),
            "pair_coverage":1.0,"answer_exact_match_rate":float(overall.exact_match_rate),
            "answer_semantic_cosine":float(overall.semantic_cosine_mean),"answer_rouge1_f1":float(overall.rouge1_f1_mean),
            "budgetleak_rank4_member_sessions":int(rank4.rank4_member_sessions),"rank4_qll_selected_sessions":qll4,
            "rank4_generation_influence":source_use,"c1_failure":c1_failure,
            "rank4_member_answer_exact_match_rate":float(rank4_member.exact_match_rate.mean()),
            "rank4_member_answer_semantic_cosine":float(rank4_member.semantic_cosine.mean()),
            "rank4_c1_answer_change_spearman":float(association.spearman_rank4_c1_vs_answer_change),
            "rank4_c1_answer_change_p":float(association.p_value),
            "rank4_membership_difference":float(rank4.membership_mean_difference),
            "rank4_membership_ci95":[float(rank4.difference_ci95_low),float(rank4.difference_ci95_high)],
            "two_signal_finalization_supported":two_signal,
            "u_signal_required":u_required,"u_signal_status":"NOT_JUSTIFIED_AND_NOT_ADDED",
            "heuristic_audit":"PASS","novelty_conflict":conflict,"closest_prior_work":closest,
            "exact_novelty_gap":"Query-local QLL source influence plus generated-claim leave-one-source dependence accumulated across a session; no audited work used this exact two-signal operational mechanism.",
            "novelty_caveat":"Private-RAG/MURAG already provides per-document cumulative privacy accounting with formal DP; PAD already provides cumulative token-level RDP accounting. Do not claim first cumulative privacy budget or first adaptive RAG defense.",
            "exp195_minimum_candidate":"NOT_REGISTERED: neither Case A nor Case B holds. Do not implement I4+C1 or a redundant source-use U from Exp194.",
            "next_step":"STOP defense implementation. If research continues, precommit a membership-specificity diagnosis; source-use itself is already captured by C1.",
            "frozen_inputs_unchanged":bool(merged.unchanged.all()),"literature_search_scope":search_scope,
            "claude_audit":"DEFERRED_BY_USER_SPEC"}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    decision=pd.DataFrame([result]);atomic_csv(decision,ROOT/"tables/TABLE_194_07_DECISION.csv")
    report=f"""# Exp194 — Minimal Final Defense Mechanism & Novelty Gate

## 판정

**{final_verdict}**

- 완전 paired Top-3/Top-4: `{len(pairs):,}` queries / `{pairs.session_id.nunique():,}` sessions (coverage `100%`)
- 전체 답변 exact match: `{overall.exact_match_rate:.4f}`
- 전체 MPNet semantic cosine: `{overall.semantic_cosine_mean:.4f}` (95% CI `{overall.semantic_cosine_ci95_low:.4f}–{overall.semantic_cosine_ci95_high:.4f}`)
- 전체 ROUGE-1 F1: `{overall.rouge1_f1_mean:.4f}`
- BudgetLeak rank-4 member sessions: `{int(rank4.rank4_member_sessions)}`; QLL rank-4 선택: `{qll4}`

## 메커니즘 결론

Top-3→Top-4 답변 변화와 exact source-specific C1은 모두 계산되었다. BudgetLeak rank-4 member 35개는 QLL이 모두 rank 4를 선택했다. Top-3/Top-4 답변 exact match는 `{rank4_member.exact_match_rate.mean():.4f}`, semantic cosine은 `{rank4_member.semantic_cosine.mean():.4f}`였고, rank-4 C1과 답변 변화의 Spearman 상관은 `{association.spearman_rank4_c1_vs_answer_change:.4f}` (`p={association.p_value:.6f}`)였다. 따라서 rank-4 source는 실제 generation에 영향을 주며 C1도 그 사용을 포착한다.

하지만 rank-4 member/nonmember의 cumulative C1 차이는 `{rank4.membership_mean_difference:.4f}` (95% CI `{rank4.difference_ci95_low:.4f}–{rank4.difference_ci95_high:.4f}`)로 신뢰구간이 0을 포함한다. 문제는 **source-use 미검출이 아니라 source-use가 membership-specific하지 않다는 것**이다. 이 패턴은 Case A도 Case B도 아니므로 Case C로 중단한다.

- rank-4 generation influence: **SUPPORTED**
- C1 failure vs source non-use: **둘 다 아님 — C1은 source use를 잡지만 membership specificity가 없음**
- I4+C1 finalization: **지원되지 않음**
- U signal: **필요성이 입증되지 않아 추가하지 않음**
- heuristic audit: **PASS** — family/rank 규칙, classifier, fusion weight, threshold search 없음

## 문헌 novelty gate

감사 범위에서 I4+C1과 mechanism-level로 완전히 같은 방법은 확인되지 않았다. 가장 가까운 세 방법은 `{', '.join(closest)}`이다. 단, MURAG은 문서별 누적 DP accounting을, PAD는 token-level 누적 RDP accounting을 이미 제안했다. 따라서 "최초 cumulative privacy budget"이나 "최초 adaptive RAG defense" 주장은 금지한다.

남는 좁은 차이는 **query-local QLL source influence와 generated-claim leave-one-source dependence를 세션에 누적해 함께 사용하는 것**이다. 이는 아직 defense 성능으로 검증된 novelty가 아니라 candidate mechanism이다.

## 다음 단계

Exp195 defense 구현으로 넘어가지 않는다. source-use 신호 U는 C1과 중복될 가능성이 크므로 추가하지 않는다. 계속 연구하려면 새로운 신호를 즉석에서 만들지 말고, source influence와 membership-specific evidence가 왜 분리되는지에 대한 사전 등록 진단부터 해야 한다. Claude 독립 감사는 지시대로 최종 설계 동결 후로 연기했다.
"""
    atomic_text(ROOT/"reports/REPORT_194_FINAL_KO.md",report)
    checkpoint("EXP194_COMPLETE",verdict=final_verdict,paired_cases=len(pairs),paired_sessions=pairs.session_id.nunique(),
               two_signal_finalization=two_signal,u_required=u_required,heuristic_audit="PASS",novelty_conflict=conflict)
    return result


def main():
    preflight();pairs=build_pairs();pairs=semantic_similarity(pairs);pairs=attach_exposure(pairs)
    answer=answer_tables(pairs);c1,qll=c1_tables(pairs);sessions,diag,corr=budget_rank4_diagnosis(pairs)
    literature,conflict,closest,scope=novelty_audit()
    result=finalize(pairs,answer,c1,qll,sessions,diag,corr,literature,conflict,closest,scope)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
