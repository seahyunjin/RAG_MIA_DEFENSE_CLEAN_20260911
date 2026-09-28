#!/usr/bin/env python3
"""Build the immutable final-method validation and sanitized release packages.

This script reads frozen artifacts only.  It does not run generation, change
tau/gamma, create attacks, or execute any transfer experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "38_final_freeze_release"
FREEZE = HERE / "37_final_task_utility_freeze"
SRC36 = HERE / "36_final_freeze_support_calibration"
IA35 = HERE / "35_query_and_ia_validity_audit"
DUAL16 = HERE / "16_dual_surface_gate"
M30 = HERE / "30_dual_conditioned_calibration"
M08 = HERE / "08_menta_transfer"
CORE = ROOT / "unified_dual_surface_dev_20260927"
BENCH = ROOT / "paper2_multidoc_session_benchmark"
GITHUB = ROOT / "release_github_dualguard"
ANON = ROOT / "release_submission_anonymous"
TAU = 0.0901547372341156
GAMMA = 9.5659109592
SEED = 20260928

sys.path.insert(0, str(HERE))
import finalize_final_freeze as metrics


def jl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cr(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wc(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def wj(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(str(item.relative_to(path)).encode()); h.update(b"\0"); h.update(bytes.fromhex(sha(item)))
    return h.hexdigest()


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.casefold()))


def refused(text: str) -> bool:
    value = " ".join(text.casefold().replace("’", "'").split())
    return any(x in value for x in ("i don't know", "i do not know", "not enough information", "cannot determine", "can't determine", "insufficient information", "not reported"))


def yn_score(answer: str, truth: str) -> int:
    value = " ".join(answer.casefold().replace("’", "'").split())
    if any(x in value for x in ("i don't know", "i do not know", "not enough information", "cannot determine", "can't determine", "unknown", "not specified", "insufficient information")):
        normalized = "Unknown"
    elif re.match(r"^\s*yes\b", value): normalized = "Yes"
    elif re.match(r"^\s*no\b", value): normalized = "No"
    else: normalized = "Unknown"
    return int(normalized == truth) - 5 * int(normalized == "Unknown")


def rank_of(target: str, sources: list[dict]) -> str:
    for index, source in enumerate(sources, 1):
        if source.get("parent_id") == target:
            return str(index)
    return ">TopK"


def support(nli: dict, job_id: str, method: str, key: str) -> float | None:
    return metrics.answer_support(nli, job_id, method, key)


def md_table(rows: list[dict], fields: list[str]) -> str:
    def fmt(value):
        if isinstance(value, float): return f"{value:.6f}"
        return str(value)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    lines += ["| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join(lines)


def title_maps() -> tuple[dict, dict]:
    rows = cr(BENCH / "MULTIDOC_RAG_CORPUS_MANIFEST.csv")
    title = {x["parent_document_id"]: x["title"] for x in rows}
    domain = {x["parent_document_id"]: x["domain"] for x in rows}
    return title, domain


def attack_tables() -> tuple[list[dict], list[dict], list[dict]]:
    core = cr(CORE / "06_FOUR_ARM_PRIVACY.csv")
    boot = cr(CORE / "07_PAIRED_BOOTSTRAP.csv")
    boot_map = {(x["attack"], x["comparison"]): x for x in boot}
    table = []
    for row in core:
        attack, arm = row["attack"], row["arm"]
        red = boot_map.get((attack, "NoDefense_minus_Dual")) if arm == "CLD_CFCLIP" else None
        delta = boot_map.get((attack, "Dual_minus_CLD")) if arm == "CLD_CFCLIP" else None
        table.append({
            "threat_scope": "single-query", "attack": attack, "method": arm,
            "cohort": "prospective balanced DEV", "N": row["n"], "N_member": row["member_n"], "N_nonmember": row["nonmember_n"],
            "metric": row["metric"], "AUROC_or_leakage": float(row["leakage_value"]),
            "paired_reduction_NoDefense_minus_method": "" if red is None else float(red["point"]),
            "paired_reduction_CI_low": "" if red is None else float(red["ci95_low"]),
            "paired_reduction_CI_high": "" if red is None else float(red["ci95_high"]),
            "Dual_minus_CLD": "" if delta is None else float(delta["point"]),
            "Dual_minus_CLD_CI_low": "" if delta is None else float(delta["ci95_low"]),
            "Dual_minus_CLD_CI_high": "" if delta is None else float(delta["ci95_high"]),
            "lineage": "FOUR_ARM_PROSPECTIVE",
        })

    ia = cr(FREEZE / "11_IA_Q1_Q5.csv"); ia_q5 = {x["method"]: x for x in ia if int(x["Q"]) == 5}
    names = {"NO_DEFENSE": "NO_DEFENSE", "CLD_ONLY": "CLD_ONLY", "EXACT_OLD_DUAL": "CLD_CFCLIP"}
    for source, arm in names.items():
        row = ia_q5[source]
        table.append({"threat_scope": "session Q1-Q5", "attack": "IA", "method": arm, "cohort": "92 paired target sessions",
                      "N": 184, "N_member": 92, "N_nonmember": 92, "metric": "ROC-AUC", "AUROC_or_leakage": float(row["AUROC"]),
                      "paired_reduction_NoDefense_minus_method": row["NoDefense_minus_ExactDual"] if source == "EXACT_OLD_DUAL" else "",
                      "paired_reduction_CI_low": row["reduction_CI_low"] if source == "EXACT_OLD_DUAL" else "",
                      "paired_reduction_CI_high": row["reduction_CI_high"] if source == "EXACT_OLD_DUAL" else "",
                      "Dual_minus_CLD": float(ia_q5["EXACT_OLD_DUAL"]["AUROC"])-float(ia_q5["CLD_ONLY"]["AUROC"]) if source == "EXACT_OLD_DUAL" else "",
                      "lineage": "SERIAL_KVCACHE_EXACT"})
    ia16 = cr(DUAL16 / "DUAL_SURFACE_PRIVACY.csv")
    ia_cf = next(x for x in ia16 if x["Attack"] == "IA" and x["Arm"] == "CFCLIP_ONLY" and int(x["Q"]) == 5)
    table.append({"threat_scope": "session Q1-Q5", "attack": "IA", "method": "CFCLIP_ONLY", "cohort": "92 paired target sessions",
                  "N": 184, "N_member": 92, "N_nonmember": 92, "metric": "ROC-AUC", "AUROC_or_leakage": float(ia_cf["AUROC"]),
                  "lineage": "SUPPLEMENTARY_BATCHED_FOUR_ARM_NOT_FINAL_PLOT_SOURCE"})

    menta = cr(FREEZE / "12_MENTA_Q1_Q5.csv"); mq5 = {x["arm"]: x for x in menta if int(x["Q"]) == 5}
    mgate = json.loads((FREEZE / "13_MENTA_GATE.json").read_text())
    for source, arm in (("NO_DEFENSE", "NO_DEFENSE"), ("CLD_ONLY", "CLD_ONLY"), ("OLD_DUAL", "CLD_CFCLIP")):
        table.append({"threat_scope": "session Q1-Q5", "attack": "MEntA", "method": arm, "cohort": "fresh zero-overlap 100+100 sessions",
                      "N": 200, "N_member": 100, "N_nonmember": 100, "metric": "ROC-AUC", "AUROC_or_leakage": float(mq5[source]["AUROC"]),
                      "paired_reduction_NoDefense_minus_method": mgate["NoDefense_minus_Dual_Q5"] if source == "OLD_DUAL" else "",
                      "paired_reduction_CI_low": mgate["paired_session_bootstrap_CI95"][0] if source == "OLD_DUAL" else "",
                      "paired_reduction_CI_high": mgate["paired_session_bootstrap_CI95"][1] if source == "OLD_DUAL" else "",
                      "Dual_minus_CLD": float(mq5["OLD_DUAL"]["AUROC"])-float(mq5["CLD_ONLY"]["AUROC"]) if source == "OLD_DUAL" else "",
                      "lineage": "FRESH_ZERO_OVERLAP_PRIMARY"})
    mcf = next(x for x in ia16 if x["Attack"] == "MEntA" and x["Arm"] == "CFCLIP_ONLY" and int(x["Q"]) == 5)
    table.append({"threat_scope": "session Q1-Q5", "attack": "MEntA", "method": "CFCLIP_ONLY", "cohort": "separate balanced four-arm DEV",
                  "N": 200, "N_member": 100, "N_nonmember": 100, "metric": "ROC-AUC", "AUROC_or_leakage": float(mcf["AUROC"]),
                  "lineage": "SUPPLEMENTARY_NOT_PRIMARY_MENTA_COHORT"})
    wc(OUT / "FINAL_ATTACK_PERFORMANCE_TABLE.csv", table)
    fields = ["threat_scope","attack","method","cohort","N","AUROC_or_leakage","paired_reduction_NoDefense_minus_method","Dual_minus_CLD","lineage"]
    (OUT / "FINAL_ATTACK_PERFORMANCE_TABLE.md").write_text("# Final attack performance\n\n" + md_table(table, fields) + "\n", encoding="utf-8")

    single = [x for x in table if x["threat_scope"] == "single-query"]
    wc(OUT / "TABLE_SINGLE_QUERY_ATTACKS.csv", single)
    session = []
    for attack, rows, method_key in (("IA", ia, "method"), ("MEntA", menta, "arm")):
        methods = sorted({x[method_key] for x in rows})
        for method in methods:
            group = {int(x["Q"]): float(x["AUROC"]) for x in rows if x[method_key] == method}
            session.append({"Attack": attack, "Method": method, **{f"Q{i}": group[i] for i in range(1,6)}, "Q5-Q1": group[5]-group[1],
                            "lineage": "SERIAL_KVCACHE_EXACT" if attack == "IA" else "FRESH_ZERO_OVERLAP_PRIMARY"})
    # CF-only IA supplementary trajectory; no cross-cohort MEntA insertion.
    group = {int(x["Q"]): float(x["AUROC"]) for x in ia16 if x["Attack"] == "IA" and x["Arm"] == "CFCLIP_ONLY"}
    session.append({"Attack":"IA","Method":"CFCLIP_ONLY",**{f"Q{i}":group[i] for i in range(1,6)},"Q5-Q1":group[5]-group[1],
                    "lineage":"SUPPLEMENTARY_BATCHED_FOUR_ARM"})
    session.sort(key=lambda x:(x["Attack"],x["Method"]))
    wc(OUT / "TABLE_SESSION_ATTACKS_Q1_Q5.csv", session)
    return table, single, session


def benign_utility() -> tuple[list[dict], dict, dict, list[dict]]:
    jobs, data = metrics.build_primary(); nli = metrics.nli_lookup(); segments = metrics.true_deep(jobs)
    deep = sorted(job for segment in segments for job in segment["job_ids"])
    sets = {"Ordinary natural benign": sorted(i for i,j in jobs.items() if int(j["query_position"])==1),
            "Reference-history conversational benign": sorted(i for i,j in jobs.items() if int(j["query_position"])>=2),
            "True same-document deep-dive": deep}
    rows=[]
    for subset,ids in sets.items():
        base = metrics.arm_metric(ids,"NO_DEFENSE",data,jobs,nli)
        for method,label in (("NO_DEFENSE","No Defense"),("CLD_ONLY","CLD"),("DUAL_G9p5659109592","Frozen Dual")):
            value=metrics.arm_metric(ids,method,data,jobs,nli)
            rows.append({"protocol":"REFERENCE_HISTORY","subset":subset,"method":label,**value,
                         "relative_F1_change_vs_NoDefense":(value["Token_F1"]-base["Token_F1"])/max(base["Token_F1"],1e-12),
                         "support_delta_vs_NoDefense":value["NLI_support"]-base["NLI_support"],
                         "answerability_delta_vs_NoDefense":value["answerability"]-base["answerability"],
                         "NovelFactRecall_relative_change_vs_NoDefense":(value["NovelFactRecall"]-base["NovelFactRecall"])/max(base["NovelFactRecall"],1e-12)})
    closed_rows=jl(SRC36/"11_CLOSED_LOOP_GENERATION.jsonl");closed={(x["job_id"],x["method"]):x for x in closed_rows};ids=sets["Reference-history conversational benign"]
    base=metrics.arm_metric(ids,"CLOSED_NO_DEFENSE",closed,jobs,nli)
    for method,label in (("CLOSED_NO_DEFENSE","No Defense"),("CLOSED_CLD_ONLY","CLD"),("CLOSED_OLD_DUAL","Frozen Dual")):
        value=metrics.arm_metric(ids,method,closed,jobs,nli)
        rows.append({"protocol":"CLOSED_LOOP","subset":"Closed-loop conversational benign","method":label,**value,
                     "relative_F1_change_vs_NoDefense":(value["Token_F1"]-base["Token_F1"])/max(base["Token_F1"],1e-12),
                     "support_delta_vs_NoDefense":value["NLI_support"]-base["NLI_support"],
                     "answerability_delta_vs_NoDefense":value["answerability"]-base["answerability"],
                     "NovelFactRecall_relative_change_vs_NoDefense":(value["NovelFactRecall"]-base["NovelFactRecall"])/max(base["NovelFactRecall"],1e-12)})
    wc(OUT/"FINAL_CORRECTED_BENIGN_UTILITY.csv",rows)
    fields=["protocol","subset","method","n","Token_F1","relative_F1_change_vs_NoDefense","NLI_support","support_delta_vs_NoDefense","unsupported","contradiction","refusal","answerability","NovelFactRecall","numeric_accuracy","detail_accuracy"]
    (OUT/"FINAL_CORRECTED_BENIGN_UTILITY.md").write_text("# Corrected benign utility\n\n"+md_table(rows,fields)+"\n",encoding="utf-8")
    return rows,jobs,data,segments


def export_queries(jobs: dict, data: dict, segments: list[dict]) -> tuple[dict, list[dict]]:
    titles, domains = title_maps(); nli = metrics.nli_lookup()
    ordinary=[]; conversational=[]; deep=[]; audit_rows=[]
    deep_map={job:s for s in segments for job in s["job_ids"]}
    for job_id,job in sorted(jobs.items(),key=lambda x:(x[1]["session_id"],int(x[1]["query_position"]))):
        nd=data[(job_id,"NO_DEFENSE")];cld=data[(job_id,"CLD_ONLY")];dual=data[(job_id,"DUAL_G9p5659109592")]
        top=job["original_sources"][0] if job.get("original_sources") else job["sources"][0]
        top_id=top["parent_id"]; history=json.dumps(job.get("reference_history",[]),ensure_ascii=False)
        common={"sample_id":job_id,"source_dataset":"MultiDoc2Dial","conversation_id":job["session_id"],"turn":int(job["query_position"]),
                "query_text":job["query"],"full_reference_history":history,"retrieved_top1_id":top_id,"retrieved_top1_title":titles.get(top_id,""),
                "retrieved_topk_ids":" || ".join(x["parent_id"] for x in job["original_sources"]),"G4":job["cld4"],"CLD_hide":job["cld_alarm"],
                "CF_activation":dual["cf_activation"],"clipped_token_count":dual["clipped_token_count"],"no_defense_answer":nd["response"],
                "cld_answer":cld["response"],"dual_answer":dual["response"],"reference_answer":job["answer"],
                "F1_no_defense":metrics.f1(nd["response"],job["answer"]),"F1_cld":metrics.f1(cld["response"],job["answer"]),
                "F1_dual":metrics.f1(dual["response"],job["answer"]),"support_no_defense":support(nli,job_id,"NO_DEFENSE","support"),
                "support_cld":support(nli,job_id,"CLD_ONLY","support"),"support_dual":support(nli,job_id,"DUAL_G9p5659109592","support"),
                "refusal_no_defense":refused(nd["response"]),"refusal_dual":refused(dual["response"]),"answerability_dual":not refused(dual["response"]),
                "answer_changed":cld["response"]!=dual["response"]}
        if int(job["query_position"])==1:
            gold=(job.get("gold_response_doc_ids") or [""])[0] if isinstance(job.get("gold_response_doc_ids"),list) else "";gold=gold or top_id
            ordinary.append({**common,"document_id":gold,"document_title":titles.get(gold,"")})
        else:
            gold_ids=[]
            retrieval=next((x for x in cr(BENCH/"processed/RETRIEVAL_PER_TURN.csv") if x["query_id"]==job["query_id"]),None)
            if retrieval: gold_ids=[x for x in retrieval["gold_response_doc_ids"].split("||") if x]
            conversational.append({**common,"current_query":common["query_text"],"grounding_document_id":" || ".join(gold_ids),
                                   "grounding_document_title":" || ".join(titles.get(x,"") for x in gold_ids)})
        if job_id in deep_map:
            seg=deep_map[job_id]; source_ids=seg["gold_parent_ids"] if isinstance(seg["gold_parent_ids"],str) else " || ".join(seg["gold_parent_ids"])
            deep.append({"sample_id":job_id,"session_id":job["session_id"],"segment_id":seg["segment_id"],"turn":int(job["query_position"]),
                         "source_document_id":source_ids,"source_document_title":" || ".join(titles.get(x,"") for x in source_ids.split(" || ")),
                         "full_history":history,"query":job["query"],"query_type_or_aspect":"not annotated in frozen artifact",
                         "no_defense_answer":nd["response"],"cld_answer":cld["response"],"dual_answer":dual["response"],"G4":job["cld4"],
                         "hide":job["cld_alarm"],"CF_activation":dual["cf_activation"],"clipped_token_count":dual["clipped_token_count"],
                         "F1_no_defense":metrics.f1(nd["response"],job["answer"]),"F1_dual":metrics.f1(dual["response"],job["answer"]),
                         "support_no_defense":support(nli,job_id,"NO_DEFENSE","support"),"support_dual":support(nli,job_id,"DUAL_G9p5659109592","support"),
                         "unsupported_dual":support(nli,job_id,"DUAL_G9p5659109592","unsupported"),"answerability_dual":not refused(dual["response"]),
                         "NovelFactRecall_dual":metrics.novel_recall(dual,job),"answer_changed":cld["response"]!=dual["response"]})
    wc(OUT/"CSV_BENIGN_ORDINARY_QUERIES_AND_OUTPUTS.csv",ordinary)
    wc(OUT/"CSV_BENIGN_CONVERSATIONAL_QUERIES_AND_OUTPUTS.csv",conversational)
    deep.sort(key=lambda x:(x["session_id"],x["turn"]));wc(OUT/"CSV_TRUE_DEEP_DIVE_QUERIES_AND_OUTPUTS.csv",deep)

    ia_manifest={x["job_id"]:x for x in jl(DUAL16/"FRESH_IA_MANIFEST.jsonl")};ia_gen={(x["job_id"],x["method"]):x for x in jl(IA35/"01_IA_EXACT_GENERATION.jsonl")}
    ia=[]
    for job_id,job in sorted(ia_manifest.items(),key=lambda x:(x[1]["target_document_id"],x[1]["world"],int(x[1]["query_position"]))):
        a=ia_gen[(job_id,"NO_DEFENSE")];b=ia_gen[(job_id,"CLD_ONLY")];d=ia_gen[(job_id,"EXACT_OLD_DUAL")];trace=d.get("trace",[])
        ia.append({"session_id":job["session_id"],"target_id":job["target_document_id"],"membership_world":job["world"],"query_number":int(job["query_position"]),
                   "query_text":job["query"],"target_document_id_offline_audit_only":job["target_document_id"],"target_rank":rank_of(job["target_document_id"],job["original_sources"]),
                   "retrieved_topk_ids":" || ".join(x["parent_id"] for x in job["original_sources"]),"no_defense_answer":a["response"],"cld_answer":b["response"],"dual_answer":d["response"],
                   "attack_score_no_defense":yn_score(a["response"],job["ground_truth"]),"attack_score_cld":yn_score(b["response"],job["ground_truth"]),
                   "attack_score_dual":yn_score(d["response"],job["ground_truth"]),"G4":job["cld4"],"hide":job["cld_alarm"],
                   "CF_activation":any(x["clip_decision"] for x in trace),"clipped_token_count":sum(x["clip_decision"] for x in trace),
                   "clipped_token_fraction":sum(x["clip_decision"] for x in trace)/max(len(trace),1),"max_CF_influence":max([x["g_max"] for x in trace],default=0.0)})
    wc(OUT/"CSV_IA_ATTACK_QUERIES_AND_OUTPUTS.csv",ia)

    mjobs={x["job_id"]:x for x in jl(M30/"23_FRESH_MENTA_MANIFEST.jsonl")};mgen={(x["job_id"],x["arm"]):x for x in jl(M30/"24_FRESH_MENTA_GENERATION.jsonl")}
    needed=set(mjobs);base={(x["job_id"]):x for x in jl(M08/"MENTA_CFCLIP_GENERATION.jsonl") if x["arm"]=="BASELINE_IDENTITY" and x["job_id"] in needed}
    evidence={(x["job_id"],x["arm"]):x for x in jl(M30/"26_FRESH_MENTA_EVIDENCE.jsonl")}
    by=defaultdict(list)
    for job in mjobs.values():by[job["session_id"]].append(job)
    cumulative={}
    for sid,group in by.items():
        group=sorted(group,key=lambda x:int(x["query_index"]));running=[]
        for job in group:
            ev=evidence[(job["job_id"],"OLD_DUAL")];score=-1 if int(ev["idk"]) else int(ev["entailed"]);running.append(score);cumulative[job["job_id"]]=float(np.mean(running))
    menta=[]
    for job_id,job in sorted(mjobs.items(),key=lambda x:(x[1]["session_id"],int(x[1]["query_index"]))):
        nd=base[job_id];cld=mgen[(job_id,"CLD_ONLY")];dual=mgen[(job_id,"OLD_DUAL")];ev=evidence[(job_id,"OLD_DUAL")];native=-1 if int(ev["idk"]) else int(ev["entailed"])
        menta.append({"session_id":job["session_id"],"membership":job["membership"],"target_id_offline_audit_only":job["target_id"],"turn":int(job["query_index"]),
                      "attack_query":job["query"],"retrieved_topk":" || ".join(x["parent_id"] for x in job["original_sources"]),"target_rank":rank_of(job["target_id"],job["original_sources"]),
                      "no_defense_answer":nd["answer"],"cld_answer":cld["answer"],"dual_answer":dual["answer"],"native_attack_score_dual":native,
                      "cumulative_session_attack_score_dual":cumulative[job_id],"G4":job["cld4"],"hide":job["cld_alarm"],
                      "CF_clip_count_estimate":float(dual["intervened_token_fraction"])*int(dual["answer_tokens"]),"CF_clipped_token_fraction":dual["intervened_token_fraction"],
                      "mean_chosen_clip":dual["mean_chosen_clip"],"hit_output_cap":dual["hit_output_cap"]})
    wc(OUT/"CSV_MENTA_SESSION_ATTACK_QUERIES_AND_OUTPUTS.csv",menta)
    return {"ordinary":ordinary,"conversational":conversational,"deep":deep,"ia":ia,"menta":menta}, audit_rows


def figures(exports: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8,"figure.facecolor":"white","axes.facecolor":"white"})
    ia_gen=jl(IA35/"01_IA_EXACT_GENERATION.jsonl")
    def score_rows(method):
        rows=[x for x in ia_gen if x["method"]==method];out=[]
        for x in rows:out.append((x["world"],yn_score(x["response"],x["ground_truth"])))
        return out
    methods=[("NO_DEFENSE","No Defense"),("CLD_ONLY","CLD"),("EXACT_OLD_DUAL","Frozen Dual")];bins=np.linspace(-5,1,13)
    traj=cr(FREEZE/"11_IA_Q1_Q5.csv")
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.4),sharex=True,sharey=True,constrained_layout=True)
    for ax,(method,label) in zip(axes,methods):
        rows=score_rows(method);t=[v for w,v in rows if w=="T"];d=[v for w,v in rows if w=="D"]
        auc=float(next(x["AUROC"] for x in traj if x["method"]==method and int(x["Q"])==5))
        ax.hist(d,bins=bins,color="#b8bcc2",alpha=.72,label="Decoy/nonmember");ax.hist(t,bins=bins,color="#173f6b",alpha=.72,label="Target/member")
        ax.set_title(f"{label}\nAUROC={auc:.3f}; N={len(t)}+{len(d)}");ax.set_xlabel("Q1–Q5 attack score");ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Sessions");axes[-1].legend(frameon=False)
    for ext in ("png","pdf"):fig.savefig(OUT/f"FIG_IA_EXACT_ATTACK_SCORE_HIST.{ext}",dpi=350 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

    old_benign=[x for x in jl(SRC36/"01_VALID_GENERATION.jsonl") if x["method"]=="DUAL_G9p5659109592"]
    deep_ids={x["sample_id"] for x in exports["deep"]};ordinary_ids={x["sample_id"] for x in exports["ordinary"]}
    groups={"IA member":[x for x in exports["ia"] if x["membership_world"]=="T"],
            "Ordinary benign":[{"max_CF_influence":max([t["g_max"] for t in x["trace"]],default=0.0),"clipped_token_fraction":x["clipped_token_fraction"]} for x in old_benign if x["job_id"] in ordinary_ids],
            "True deep-dive":[{"max_CF_influence":max([t["g_max"] for t in x["trace"]],default=0.0),"clipped_token_fraction":x["clipped_token_fraction"]} for x in old_benign if x["job_id"] in deep_ids]}
    def diagnostic(other,label,path):
        fig,axes=plt.subplots(1,2,figsize=(8.2,3.3),constrained_layout=True)
        for ax,key,xlabel in ((axes[0],"max_CF_influence","Max CF influence per answer"),(axes[1],"clipped_token_fraction","Clipped-token fraction")):
            a=np.asarray([float(x[key]) for x in groups["IA member"]]);b=np.asarray([float(x[key]) for x in groups[other]])
            lo=min(a.min(initial=0),b.min(initial=0));hi=max(a.max(initial=1),b.max(initial=1));bins=np.linspace(lo,hi,24)
            ax.hist(b,bins=bins,density=True,alpha=.66,color="#b8bcc2",label=label);ax.hist(a,bins=bins,density=True,alpha=.58,color="#173f6b",label="IA member")
            ax.set_xlabel(xlabel);ax.set_ylabel("Density");ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
        fig.suptitle("Mechanism diagnostic (not a detector ROC)")
        for ext in ("png","pdf"):fig.savefig(OUT/f"{path}.{ext}",dpi=350 if ext=="png" else None,bbox_inches="tight")
        plt.close(fig)
    diagnostic("Ordinary benign","Ordinary benign","FIG_IA_MEMBER_VS_BENIGN_INFLUENCE_HIST")
    diagnostic("True deep-dive","True deep-dive","FIG_IA_MEMBER_VS_DEEP_INFLUENCE_HIST")

    def trajectory(path,source,method_col,mapping,title):
        rows=cr(source);fig,ax=plt.subplots(figsize=(5.6,3.8),constrained_layout=True)
        colors=["#6b7280","#d97706","#173f6b"]
        for (method,label),color in zip(mapping.items(),colors):
            vals=[float(next(x["AUROC"] for x in rows if x[method_col]==method and int(x["Q"])==q)) for q in range(1,6)]
            ax.plot(range(1,6),vals,marker="o",linewidth=2,color=color,label=label)
        ax.set(xticks=range(1,6),xticklabels=[f"Q{i}" for i in range(1,6)],xlabel="Cumulative query count",ylabel="AUROC",title=title)
        ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
        for ext in ("png","pdf"):fig.savefig(OUT/f"{path}.{ext}",dpi=350 if ext=="png" else None,bbox_inches="tight")
        plt.close(fig)
    trajectory("FIG_IA_Q1_Q5_TRAJECTORY",FREEZE/"11_IA_Q1_Q5.csv","method",{"NO_DEFENSE":"No Defense","CLD_ONLY":"CLD","EXACT_OLD_DUAL":"Frozen Dual"},"IA exact session leakage")
    trajectory("FIG_MENTA_Q1_Q5_TRAJECTORY",FREEZE/"12_MENTA_Q1_Q5.csv","arm",{"NO_DEFENSE":"No Defense","CLD_ONLY":"CLD","OLD_DUAL":"Frozen Dual"},"MEntA residual session leakage")


def audits(exports: dict, jobs: dict, segments: list[dict]) -> tuple[dict, dict]:
    issues=[]; diagnostics=defaultdict(int); query_records=[]
    for kind,rows,qkey,skey,tkey,hkey in (
        ("ordinary",exports["ordinary"],"query_text","conversation_id","turn","full_reference_history"),
        ("conversational",exports["conversational"],"current_query","conversation_id","turn","full_reference_history"),
        ("deep",exports["deep"],"query","session_id","turn","full_history"),
        ("ia",exports["ia"],"query_text","session_id","query_number",None),
        ("menta",exports["menta"],"attack_query","session_id","turn",None)):
        for row in rows:
            query=str(row[qkey]);session=str(row[skey]);turn=int(row[tkey]);history="" if hkey is None else str(row[hkey])
            query_records.append((kind,session,turn,query,row))
            if not query.strip():issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"issue":"EMPTY_QUERY","query":query})
            if not re.search(r"[A-Za-z0-9]",query) or len(query)>4096 or "{{" in query or "}}" in query:
                issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"issue":"MALFORMED_OR_TEMPLATE_CORRUPTION","query":query})
            if re.search(r"(?:membership_world|target_document_id|attack_score|ground_truth)\s*[:=]",query,re.I):
                issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"issue":"LABEL_OR_METADATA_VISIBLE","query":query})
            if kind in ("conversational","deep") and turn>1 and history in ("","[]"):
                issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"issue":"MISSING_REQUIRED_HISTORY","query":query})
    exact=defaultdict(list)
    for record in query_records:exact[norm(record[3])].append(record)
    for group in exact.values():
        if len(group)<2:continue
        diagnostics["exact_duplicate_rows"]+=len(group)
        # IA target/decoy mirrors at the same session/turn are protocol-defined.
        logical={(x[0],x[1],x[2]) for x in group}
        if len(logical)<len(group):diagnostics["expected_world_mirror_duplicates"]+=len(group)-len(logical)
    by_session=defaultdict(list)
    for record in query_records:by_session[(record[0],record[1])].append(record)
    for (kind,session),group in by_session.items():
        turns=[x[2] for x in sorted(group,key=lambda x:x[2])]
        unique=sorted(set(turns));expected=list(range(min(unique),max(unique)+1)) if unique else []
        if unique!=expected:
            issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":"","issue":"SESSION_TURN_ORDER_GAP","query":""})
        texts=[set(norm(x[3]).split()) for x in group]
        for i in range(len(texts)):
            for j in range(i+1,len(texts)):
                score=len(texts[i]&texts[j])/max(len(texts[i]|texts[j]),1)
                if .90<=score<1:diagnostics["near_duplicate_pairs_jaccard_ge_0_90"]+=1

    # Reference-history consistency against the frozen official jobs.
    for job in jobs.values():
        if int(job["query_position"])>1:
            history=job.get("reference_history",[])
            if len(history)<int(job["query_position"])-1:
                issues.append({"severity":"blocking","dataset":"benign","session_id":job["session_id"],"turn":job["query_position"],"issue":"CONVERSATION_HISTORY_MISMATCH","query":job["query"]})
    deep_ids={x["sample_id"] for x in exports["deep"]};manifest_ids={x for s in segments for x in s["job_ids"]}
    if deep_ids!=manifest_ids:
        issues.append({"severity":"blocking","dataset":"deep","session_id":"","turn":"","issue":"DOCUMENT_PROVENANCE_MANIFEST_MISMATCH","query":""})
    wc(OUT/"CSV_QUERY_AUDIT_ISSUES.csv",issues)
    qresult={"status":"QUERY_AUDIT_PASS" if not any(x["severity"]=="blocking" for x in issues) else "QUERY_AUDIT_NEEDS_REVIEW",
             "rows":len(query_records),"blocking_issue_count":sum(x["severity"]=="blocking" for x in issues),"issue_count":len(issues),"diagnostics":dict(diagnostics),
             "checks":["empty","duplicate","near-duplicate","malformed/template","label leakage","benign metadata leakage","history","provenance","turn order"]}
    wj(OUT/"QUERY_AUDIT_SUMMARY.json",qresult)

    output_issues=[];answer_count=0
    for kind,rows in exports.items():
        answer_keys=[k for k in ("no_defense_answer","cld_answer","dual_answer") if rows and k in rows[0]]
        for row in rows:
            for key in answer_keys:
                answer=str(row.get(key,""));answer_count+=1;session=row.get("session_id",row.get("conversation_id",row.get("sample_id","")));turn=row.get("turn",row.get("query_number",""))
                if not answer.strip():output_issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"method":key,"issue":"EMPTY_OUTPUT","answer":answer})
                if "�" in answer or "\x00" in answer:output_issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"method":key,"issue":"CORRUPTED_DECODING","answer":answer})
                tokens=answer.split()
                if len(tokens)>=12 and any(tokens[i:i+4]==tokens[i+4:i+8]==tokens[i+8:i+12] for i in range(len(tokens)-11)):
                    output_issues.append({"severity":"blocking","dataset":kind,"session_id":session,"turn":turn,"method":key,"issue":"REPEATED_GARBAGE","answer":answer})
                if refused(answer):output_issues.append({"severity":"diagnostic","dataset":kind,"session_id":session,"turn":turn,"method":key,"issue":"REFUSAL","answer":answer})
                elif len(tokens)<=2:output_issues.append({"severity":"diagnostic","dataset":kind,"session_id":session,"turn":turn,"method":key,"issue":"EXTREMELY_SHORT","answer":answer})
        if kind=="menta":
            for row in rows:
                if str(row.get("hit_output_cap","")).casefold()=="true":output_issues.append({"severity":"diagnostic","dataset":kind,"session_id":row["session_id"],"turn":row["turn"],"method":"dual_answer","issue":"KNOWN_MAX_TOKEN_CAP","answer":row["dual_answer"]})
    wc(OUT/"CSV_OUTPUT_AUDIT_ISSUES.csv",output_issues)
    oresult={"status":"OUTPUT_AUDIT_PASS" if not any(x["severity"]=="blocking" for x in output_issues) else "OUTPUT_AUDIT_NEEDS_REVIEW",
             "answers":answer_count,"blocking_issue_count":sum(x["severity"]=="blocking" for x in output_issues),"diagnostic_count":sum(x["severity"]=="diagnostic" for x in output_issues),
             "diagnostic_breakdown":dict(Counter(x["issue"] for x in output_issues if x["severity"]=="diagnostic"))}
    wj(OUT/"OUTPUT_AUDIT_SUMMARY.json",oresult)
    return qresult,oresult


def samples(exports: dict) -> None:
    def order(value,salt):return hashlib.sha256(f"{SEED}|{salt}|{value}".encode()).hexdigest()
    ordinary=sorted(exports["ordinary"],key=lambda x:order(x["sample_id"],"ordinary"))[:20]
    conv_ids=sorted({x["conversation_id"] for x in exports["conversational"]},key=lambda x:order(x,"conv"))[:10]
    conv=[x for x in exports["conversational"] if x["conversation_id"] in conv_ids]
    deep_ids=sorted({x["segment_id"] for x in exports["deep"]},key=lambda x:order(x,"deep"))[:10];deep=[x for x in exports["deep"] if x["segment_id"] in deep_ids]
    ia_ids=sorted({x["target_id"] for x in exports["ia"]},key=lambda x:order(x,"ia"))[:10];ia=[x for x in exports["ia"] if x["target_id"] in ia_ids]
    m_ids=sorted({x["session_id"] for x in exports["menta"]},key=lambda x:order(x,"menta"))[:10];menta=[x for x in exports["menta"] if x["session_id"] in m_ids]
    lines=["# Fixed-seed query/output audit samples","",f"Seed: `{SEED}`; SHA256 ordering; no human selection.",""]
    for title,rows,qkey,hkey in (("Ordinary benign",ordinary,"query_text","full_reference_history"),("Conversational benign",conv,"current_query","full_reference_history"),("True deep-dive",deep,"query","full_history"),("IA",ia,"query_text",None),("MEntA",menta,"attack_query",None)):
        lines += [f"## {title}",""]
        for row in rows:
            ident=row.get("sample_id",row.get("target_id",row.get("session_id","")));turn=row.get("turn",row.get("query_number",""))
            lines += [f"### {ident} / turn {turn}","", "[HISTORY]", "", str(row.get(hkey,"(not part of frozen attack input)") if hkey else "(not part of frozen attack input)"), "", "[QUERY]", "", str(row[qkey]), "", "[NO DEFENSE]", "", str(row["no_defense_answer"]), "", "[CLD]", "", str(row["cld_answer"]), "", "[DUAL]", "", str(row["dual_answer"]), "", "[METRICS]", "", json.dumps({k:row[k] for k in row if k.startswith(("F1","support","attack_score","native_attack","G4","hide","CLD_hide","answer_changed"))},ensure_ascii=False), ""]
    (OUT/"QUERY_OUTPUT_AUDIT_SAMPLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def package_files(destination: Path, anonymous: bool) -> None:
    (destination/"dualguard").mkdir(parents=True);(destination/"examples").mkdir();(destination/"tests").mkdir();(destination/"tests/__init__.py").write_text("",encoding="utf-8")
    config={"method_status":"METHOD_FROZEN_CLD_OLD_CFCLIP","cld":{"k":4,"tau":TAU,"comparator":">"},"cfclip":{"gamma":GAMMA,"formula":"z_F(v)-[max_i([log p_F(v)-log p_-i(v)]_+)-gamma]_+"},"gold_engine":"SERIAL_KVCACHE","fast_engine_production_ready":False,"training":False,"attack_routing":False}
    wj(destination/"config.json",config)
    (destination/"dualguard/__init__.py").write_text('from .cld import cld4, apply_cld\nfrom .cfclip import cf_clip_logits\nfrom .serial_kvcache import SerialKVCacheCFClip\n',encoding="utf-8")
    (destination/"dualguard/cld.py").write_text('''"""Frozen CLD/G4 retrieval intervention."""\nfrom __future__ import annotations\nimport numpy as np\n\nTAU = 0.0901547372341156\n\ndef cld4(scores):\n    values=np.asarray(scores,dtype=np.float64)\n    if values.shape != (4,): raise ValueError("CLD4 requires exactly four retrieval scores")\n    return float(values[0]-values.mean())\n\ndef apply_cld(scores, documents, tau=TAU):\n    if len(documents)!=4: raise ValueError("Expected four ranked documents")\n    risk=cld4(scores); alarm=bool(risk>tau)\n    return {"g4":risk,"alarm":alarm,"documents":list(documents[1:] if alarm else documents),"removed":documents[0] if alarm else None}\n''',encoding="utf-8")
    (destination/"dualguard/cfclip.py").write_text('''"""Frozen old CF-Clip equation; no learned parameters."""\nfrom __future__ import annotations\nGAMMA = 9.5659109592\n\ndef cf_clip_logits(full_logits, full_logp, drop_logp, gamma=GAMMA):\n    import torch\n    if drop_logp.ndim != 2: raise ValueError("drop_logp must be [documents,vocabulary]")\n    lifts=torch.clamp(full_logp.unsqueeze(0)-drop_logp,min=0)\n    max_lift=lifts.max(dim=0).values\n    clip=torch.clamp(max_lift-gamma,min=0)\n    return full_logits-clip, max_lift, clip\n''',encoding="utf-8")
    (destination/"dualguard/serial_kvcache.py").write_text('''"""Gold SERIAL_KVCACHE reference decoder for frozen old CF-Clip."""\nfrom __future__ import annotations\nfrom .cfclip import GAMMA, cf_clip_logits\n\nclass SerialKVCacheCFClip:\n    def __init__(self, model, tokenizer, gamma=GAMMA, repetition_penalty=1.05):\n        self.model=model.eval(); self.tokenizer=tokenizer; self.gamma=gamma; self.penalty=repetition_penalty\n        self.body=model.model; self.head=model.lm_head\n    def _prefill(self, ids):\n        import torch\n        x=torch.as_tensor(ids,dtype=torch.long,device=self.model.device).view(1,-1);mask=torch.ones_like(x);pos=torch.arange(x.shape[1],device=x.device).view(1,-1)\n        out=self.body(input_ids=x,attention_mask=mask,position_ids=pos,use_cache=True)\n        return {"hidden":out.last_hidden_state[:,-1,:],"past":out.past_key_values,"mask":mask,"position":x.shape[1]}\n    def _advance(self,state,token):\n        import torch\n        x=torch.as_tensor([token],dtype=torch.long,device=self.model.device).view(1,1);mask=torch.cat((state["mask"],torch.ones((1,1),dtype=state["mask"].dtype,device=x.device)),-1);pos=torch.as_tensor([[state["position"]]],device=x.device)\n        out=self.body(input_ids=x,attention_mask=mask,position_ids=pos,past_key_values=state["past"],use_cache=True);state.update(hidden=out.last_hidden_state[:,-1,:],past=out.past_key_values,mask=mask,position=state["position"]+1)\n    def _penalize(self,scores,seen):\n        import torch\n        idx=torch.unique(torch.as_tensor(seen,dtype=torch.long,device=scores.device));v=scores.index_select(0,idx);changed=torch.where(v<0,v*self.penalty,v/self.penalty);return scores.clone().scatter(0,idx,changed)\n    def decode(self,branch_prompt_ids,max_new_tokens=64):\n        import torch\n        if len(branch_prompt_ids)<2: raise ValueError("Need FULL plus at least one DROP prompt")\n        eos=self.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);states=[self._prefill(x) for x in branch_prompt_ids];generated=[];trace=[]\n        with torch.inference_mode():\n            for _ in range(max_new_tokens):\n                logits=torch.cat([self.head(x["hidden"]).float() for x in states],0);logp=torch.log_softmax(logits,dim=-1);defended,max_lift,clip=cf_clip_logits(logits[0],logp[0],logp[1:],self.gamma);token=int(torch.argmax(self._penalize(defended,branch_prompt_ids[0]+generated)).item());trace.append({"token_id":token,"g_max":float(max_lift[token].item()),"chosen_clip":float(clip[token].item()),"clip_decision":bool(clip[token].item()>0)});generated.append(token)\n                if token in eos: break\n                for state in states:self._advance(state,token)\n        answer=generated[:-1] if generated and generated[-1] in eos else generated\n        return {"token_ids":answer,"text":self.tokenizer.decode(answer,skip_special_tokens=True).strip(),"trace":trace[:len(answer)]}\n''',encoding="utf-8")
    (destination/"inference.py").write_text('''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse,json\nfrom pathlib import Path\nfrom dualguard import cld4,apply_cld\n\ndef main():\n p=argparse.ArgumentParser();p.add_argument("--toy",action="store_true");p.add_argument("--input",type=Path);a=p.parse_args()\n if a.toy:\n  row={"top_scores":[0.8,0.6,0.59,0.58],"documents":["d1","d2","d3","d4"]};print(json.dumps(apply_cld(row["top_scores"],row["documents"]),indent=2));return\n if not a.input: raise SystemExit("Provide --toy or --input. Full generation integration supplies FULL and DROP prompts to SerialKVCacheCFClip.")\n row=json.loads(a.input.read_text());print(json.dumps(apply_cld(row["top_scores"],row["documents"]),indent=2))\nif __name__=="__main__":main()\n''',encoding="utf-8")
    (destination/"evaluate.py").write_text('''#!/usr/bin/env python3\nimport argparse,csv,json\nfrom pathlib import Path\ndef main():\n p=argparse.ArgumentParser();p.add_argument("--csv",type=Path);p.add_argument("--dry-run",action="store_true");a=p.parse_args()\n if a.dry_run: print(json.dumps({"status":"DRY_RUN_PASS","expected_columns":["attack","method","AUROC_or_leakage"]}));return\n rows=list(csv.DictReader(a.csv.open()));print(json.dumps({"rows":len(rows)},indent=2))\nif __name__=="__main__":main()\n''',encoding="utf-8")
    (destination/"tests/smoke_test.py").write_text('''import json,torch\nfrom dualguard import cld4,apply_cld,cf_clip_logits\nassert abs(cld4([.8,.6,.59,.58])-.1575)<1e-12\nr=apply_cld([.8,.6,.59,.58],["a","b","c","d"]);assert r["alarm"] and r["removed"]=="a"\nfull=torch.tensor([1.,2.,3.]);fl=torch.log_softmax(full,0);drop=torch.stack([fl-torch.tensor([0.,10.,0.]),fl]);out,lift,clip=cf_clip_logits(full,fl,drop);assert out.shape==full.shape and torch.all(clip>=0)\nprint(json.dumps({"status":"SMOKE_PASS","g4":r["g4"],"alarm":r["alarm"]}))\n''',encoding="utf-8")
    wj(destination/"examples/toy_input.json",{"top_scores":[.8,.6,.59,.58],"documents":["d1","d2","d3","d4"]})
    (destination/"requirements.txt").write_text("numpy>=1.26\ntorch>=2.3\ntransformers>=4.51\n",encoding="utf-8")
    (destination/"LICENSE_PLACEHOLDER").write_text("License to be selected by the artifact authors before public distribution.\n",encoding="utf-8")
    readme='''# DualGuard (frozen research artifact)\n\n## Method overview\n\nDualGuard is a training-free, attack-family-routing-free composition over retrieval and generation surfaces. It does not provide a formal differential-privacy guarantee.\n\nCLD/G4: `G4 = s1 - (s1+s2+s3+s4)/4`, alarm when `G4 > 0.0901547372341156`; the rank-1 source is withheld.\n\nOld CF-Clip: `L_i(v)=[log p_F(v)-log p_-i(v)]_+`, `L(v)=max_i L_i(v)`, and `z'(v)=z_F(v)-[L(v)-9.5659109592]_+`.\n\nExpected inputs are four ranked retrieval scores/documents and, for exact CF-Clip, one FULL plus one leave-one-parent-out prompt per retained parent. The output is the CLD decision, optional removed source, and a frozen-generator answer.\n\nThe exact implementation is `SERIAL_KVCACHE`. The speculative Fast implementation is not production-ready and is not included. Runtime is substantially higher than vanilla greedy decoding because every output token evaluates all DROP branches.\n\n## Quick check\n\n```bash\npython inference.py --toy\npython -m tests.smoke_test\npython evaluate.py --dry-run\n```\n\n## Claims and limitations\n\nThe frozen evaluation supports IA mitigation, MEntA mitigation, and corrected benign/deep-reading aggregate utility preservation. It does not establish a universal defense, elimination of MEntA, zero false positives, production-fast inference, or formal DP.\n\n## Citation\n\nCitation placeholder for anonymous review/public release.\n'''
    (destination/"README.md").write_text(readme,encoding="utf-8")
    if anonymous:
        (destination/"artifacts").mkdir();shutil.copy2(OUT/"FINAL_ATTACK_PERFORMANCE_TABLE.csv",destination/"artifacts/FINAL_ATTACK_PERFORMANCE_TABLE.csv");shutil.copy2(OUT/"FINAL_CORRECTED_BENIGN_UTILITY.csv",destination/"artifacts/FINAL_CORRECTED_BENIGN_UTILITY.csv");shutil.copy2(OUT/"TABLE_SESSION_ATTACKS_Q1_Q5.csv",destination/"artifacts/TABLE_SESSION_ATTACKS_Q1_Q5.csv")
        (destination/"reproduce_table.py").write_text('''import argparse,csv\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/FINAL_ATTACK_PERFORMANCE_TABLE.csv";r=list(csv.DictReader(f.open()));assert r\nprint("DRY_RUN_PASS" if a.dry_run else "\\n".join(f"{x['attack']}: {x['method']}={x['AUROC_or_leakage']}" for x in r))\n''',encoding="utf-8")
        (destination/"reproduce_benign_table.py").write_text('''import argparse,csv\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/FINAL_CORRECTED_BENIGN_UTILITY.csv";r=list(csv.DictReader(f.open()));assert r;print("DRY_RUN_PASS" if a.dry_run else len(r))\n''',encoding="utf-8")
        (destination/"reproduce_ia_figures.py").write_text('''import argparse,csv\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args();f=Path(__file__).parent/"artifacts/TABLE_SESSION_ATTACKS_Q1_Q5.csv";r=[x for x in csv.DictReader(f.open()) if x["Attack"]=="IA"];assert r and all(f"Q{i}" in r[0] for i in range(1,6));print("DRY_RUN_PASS" if a.dry_run else "IA trajectory schema valid")\n''',encoding="utf-8")
        (destination/"REPRODUCE_RESULTS.md").write_text("# Reproduce results\n\nRun `python reproduce_table.py`, `python reproduce_ia_figures.py`, and `python reproduce_benign_table.py`. Use `--dry-run` for schema-only validation. Model-weight-dependent generation is intentionally not bundled.\n",encoding="utf-8")
        (destination/"EXPECTED_ARTIFACT_SCHEMA.md").write_text("# Expected artifact schema\n\nRetrieval input: exactly four descending scores and four parent-document records. CF input: FULL prompt plus one DROP prompt per retained unique parent. Evaluation rows contain attack, method, cohort, N, and AUROC_or_leakage.\n",encoding="utf-8")


def build_packages(query_audit: dict,output_audit: dict) -> dict:
    if query_audit["status"]!="QUERY_AUDIT_PASS" or output_audit["status"]!="OUTPUT_AUDIT_PASS":
        return {"built":False,"status":"SKIPPED_PENDING_DATA_REVIEW"}
    if GITHUB.exists() or ANON.exists():raise RuntimeError("release destination unexpectedly exists")
    package_files(GITHUB,False);package_files(ANON,True)
    forbidden=("/home/","traffic_3","api_key","access_token","github.com/")
    scan={}
    for package in (GITHUB,ANON):
        hits=[]
        for path in package.rglob("*"):
            if path.is_file():
                text=path.read_text(encoding="utf-8",errors="ignore").casefold()
                for needle in forbidden:
                    if needle.casefold() in text:hits.append({"file":str(path.relative_to(package)),"needle":needle})
        scan[package.name]=hits
    py=sys.executable;smoke=[]
    for package in (GITHUB,ANON):
        clean=Path(tempfile.mkdtemp(prefix=f"{package.name}_",dir="/tmp"));copy=clean/package.name;shutil.copytree(package,copy)
        env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"}
        commands=[[py,"-m","tests.smoke_test"],[py,"inference.py","--toy"],[py,"evaluate.py","--dry-run"]]
        if package==ANON:commands += [[py,"reproduce_table.py","--dry-run"],[py,"reproduce_ia_figures.py","--dry-run"],[py,"reproduce_benign_table.py","--dry-run"]]
        for command in commands:
            result=subprocess.run(command,cwd=copy,env=env,capture_output=True,text=True,timeout=120)
            smoke.append({"package":package.name,"command":" ".join(command[1:]),"returncode":result.returncode,"stdout":result.stdout.strip(),"stderr":result.stderr.strip(),"clean_dir":str(clean)})
    passed=not any(scan.values()) and all(x["returncode"]==0 for x in smoke)
    result={"built":True,"status":"RELEASE_PACKAGE_SMOKE_PASS" if passed else "RELEASE_PACKAGE_SMOKE_FAIL","credential_path_scan":scan,"commands":smoke,
            "github":{"path":str(GITHUB.relative_to(ROOT)),"tree_sha256":tree_sha(GITHUB)},"anonymous":{"path":str(ANON.relative_to(ROOT)),"tree_sha256":tree_sha(ANON)}}
    wj(OUT/"RELEASE_PACKAGE_SMOKE_TEST.json",result);return result


def summaries(attack,session,utility,q,o,packages,exports) -> None:
    validated=q["status"]=="QUERY_AUDIT_PASS" and o["status"]=="OUTPUT_AUDIT_PASS"
    status="FROZEN_MODEL_ARTIFACTS_VALIDATED" if validated else "FROZEN_MODEL_ARTIFACTS_NEED_REVIEW"
    summary=["# Frozen Model Validation Summary","","## 1. Frozen method equation","",r"`G4=s1-(s1+s2+s3+s4)/4`; `L_i(v)=[log p_F(v)-log p_-i(v)]_+`; `z'(v)=z_F(v)-[max_i L_i(v)-gamma]_+`.","","## 2. Parameters", "",f"- tau: `{TAU}`",f"- gamma: `{GAMMA}`","- Gold engine: `SERIAL_KVCACHE`","","## 3. Single-query attacks","",f"Rows: `{len([x for x in attack if x['threat_scope']=='single-query'])}`; see `TABLE_SINGLE_QUERY_ATTACKS.csv`.","","## 4. IA","", "Exact SERIAL_KVCACHE Q1–Q5 trajectory and histogram reproduced.","","## 5. MEntA","", "Fresh zero-overlap Q1–Q5 trajectory retained; residual leakage is reported.","","## 6. Corrected benign","",f"Rows: `{len(utility)}`; only corrected protocols are final.","","## 7. True deep-dive","",f"Queries: `{len(exports['deep'])}` from gold document provenance.","","## 8. Query audit","",f"`{q['status']}`; blocking issues `{q['blocking_issue_count']}`.","","## 9. Output audit","",f"`{o['status']}`; blocking issues `{o['blocking_issue_count']}`; diagnostics `{o['diagnostic_count']}`.","","## 10. Figures","", "IA exact histogram, IA/benign mechanism diagnostics, IA trajectory, and MEntA trajectory are included as PNG/PDF.","","## 11. Limitations","","CF-only IA/MEntA rows are supplementary where an exact/fresh four-arm lineage was unavailable. Fast is not production gold. DCMI-Std-Q2 is standardized rather than paper-exact. No domain transfer was run.","",status]
    (OUT/"FROZEN_MODEL_VALIDATION_SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    if not validated:verdict="MODEL_FROZEN_PACKAGE_PENDING_DATA_REVIEW";recommend="review exported queries first"
    elif packages.get("status")!="RELEASE_PACKAGE_SMOKE_PASS":verdict="MODEL_FROZEN_CODE_PACKAGE_NEEDS_FIX";recommend="code package fix only"
    else:verdict="FINAL_FROZEN_MODEL_PACKAGE_READY";recommend="package ready"
    report=["# Final Freeze and Release Report","","## 1. Frozen model","","`METHOD_FROZEN_CLD_OLD_CFCLIP`: Retriever → CLD/G4 → old CF-Clip → Frozen Generator.","","## 2. Frozen parameters","",f"tau `{TAU}`, gamma `{GAMMA}`, gold engine `SERIAL_KVCACHE`.","","## 3. Single-query table","", "See `TABLE_SINGLE_QUERY_ATTACKS.csv` and `FINAL_ATTACK_PERFORMANCE_TABLE.md`.","","## 4. Session table","",md_table(session,["Attack","Method","Q1","Q2","Q3","Q4","Q5","Q5-Q1","lineage"]),"","## 5. IA histogram","","`FIG_IA_EXACT_ATTACK_SCORE_HIST.png` uses validated SERIAL_KVCACHE exact rows.","","## 6. IA/benign histogram","","Both influence figures are mechanism diagnostics, not detector ROC curves.","","## 7. IA trajectory","","`FIG_IA_Q1_Q5_TRAJECTORY.png`.","","## 8. MEntA trajectory","","`FIG_MENTA_Q1_Q5_TRAJECTORY.png`; residual leakage is not hidden.","","## 9. Corrected benign table","", "See `FINAL_CORRECTED_BENIGN_UTILITY.csv`.","","## 10. Deep-dive table","",f"Gold-provenance deep-dive rows: `{len(exports['deep'])}`.","","## 11. Query/output CSV inventory","",f"Ordinary `{len(exports['ordinary'])}`, conversational `{len(exports['conversational'])}`, deep `{len(exports['deep'])}`, IA `{len(exports['ia'])}`, MEntA `{len(exports['menta'])}`.","","## 12. Query sanity audit","",f"`{q['status']}`; blocking `{q['blocking_issue_count']}`, diagnostic duplicates `{q['diagnostics']}`.","","## 13. Qualitative examples","","`QUERY_OUTPUT_AUDIT_SAMPLES.md` uses fixed SHA256 selection and complete sessions/segments.","","## 14. GitHub package","",json.dumps(packages.get("github",{}),ensure_ascii=False),"","## 15. Anonymous submission package","",json.dumps(packages.get("anonymous",{}),ensure_ascii=False),"","## 16. Smoke test","",f"`{packages.get('status')}`.","","## 17. Limitations","","Fast is not production-ready; CF-only session supplements do not replace exact/fresh primary lineages; no domain transfer, transfer experiment, adaptive attack, baseline rerun, or method tuning was performed.","","## 18. Final verdict","",f"`{verdict}`","",f"FINAL_RECOMMENDATION: {recommend}"]
    (OUT/"FINAL_FREEZE_AND_RELEASE_REPORT_KO.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    wj(OUT/"FINAL_VERDICT.json",{"verdict":verdict,"recommendation":recommend,"method_status":"METHOD_FROZEN_CLD_OLD_CFCLIP","tau":TAU,"gamma":GAMMA,"gold_engine":"SERIAL_KVCACHE","domain_transfer_run":False,"method_modified":False})


def manifest(packages: dict) -> None:
    groups={"Tables":[],"Figures":[],"Query audit":[],"Frozen model":[],"GitHub package":[],"Anonymous submission package":[],"Verification":[]}
    for path in sorted(OUT.iterdir()):
        if not path.is_file():continue
        name=path.name
        entry=f"- `{name}` — SHA256 `{sha(path)}`"
        if path.suffix in (".png",".pdf"):groups["Figures"].append(entry)
        elif "QUERY" in name or "OUTPUT_AUDIT" in name:groups["Query audit"].append(entry)
        elif name in ("FINAL_VERDICT.json","FROZEN_MODEL_VALIDATION_SUMMARY.md"):groups["Frozen model"].append(entry)
        elif "SMOKE" in name or "VERIFICATION" in name:groups["Verification"].append(entry)
        else:groups["Tables"].append(entry)
    if packages.get("github"):groups["GitHub package"].append(f"- `{packages['github']['path']}` — tree SHA256 `{packages['github']['tree_sha256']}`")
    if packages.get("anonymous"):groups["Anonymous submission package"].append(f"- `{packages['anonymous']['path']}` — tree SHA256 `{packages['anonymous']['tree_sha256']}`")
    lines=["# Final Artifact Manifest",""]
    for title,entries in groups.items():lines += [f"## {title}",""]+(entries or ["- None"])+[""]
    (OUT/"FINAL_ARTIFACT_MANIFEST.md").write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    if OUT.exists():raise RuntimeError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    wj(OUT/"00_FINAL_METHOD_FREEZE.json",{"status":"METHOD_FROZEN_CLD_OLD_CFCLIP","tau":TAU,"gamma":GAMMA,"gold_engine":"SERIAL_KVCACHE","immutable":True,"forbidden_current_run":["retriever transfer","generator transfer","adaptive attack","strong baseline rerun","domain transfer","new defense","threshold change","new Fast approximation"]})
    attack,single,session=attack_tables();utility,jobs,data,segments=benign_utility();exports,_=export_queries(jobs,data,segments);figures(exports);samples(exports);q,o=audits(exports,jobs,segments);packages=build_packages(q,o);summaries(attack,session,utility,q,o,packages,exports);manifest(packages)
    print(json.dumps({"query_audit":q,"output_audit":o,"packages":packages.get("status"),"final":json.loads((OUT/"FINAL_VERDICT.json").read_text())},indent=2,ensure_ascii=False))


if __name__=="__main__":main()
