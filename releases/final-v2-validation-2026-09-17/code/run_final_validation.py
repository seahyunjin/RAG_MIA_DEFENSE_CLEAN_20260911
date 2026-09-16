#!/usr/bin/env python3
"""Fail-closed frozen Final-LC-V2 validation, visualization, utility and support audit."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from lineage_guard import require_same_retrieval_db

ROOT = Path(__file__).resolve().parents[3]
EXPS = ROOT / "experiments"
EXP = EXPS / "FINAL_LC_V2_FINAL_FREEZE_V1"
CODE = Path(__file__).resolve()
PRE = EXP / "configs" / "PRECOMMIT.json"
SIDE = EXPS / "FINAL_LC_DETECTOR_VISUAL_SIDECAR_V2"
F8 = EXPS / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
C6 = EXPS / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
IAR = EXPS / "FINAL_LC_IA_RESOLUTION_V1"
FPR3 = IAR / "fpr3_sensitivity"
IAB = EXPS / "FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1" / "IA_STD_Q15_API1"
GOLD = EXPS / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
LC = EXPS / "LC_MIRABEL_LARGE_V1"
CORE3 = EXPS / "CLEAN_CORE3_DEV_V1"
NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots/04dcf11f844b07bc57015169fca2b7d6df8299d5")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
SEED = 20260914
BOOT = 2000
ATTACKS = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")
METHODS = ("MIRABEL", "Final LC V1", "Final LC Q1 V2")
ALPHAS = (.01, .025, .03, .05)
COLORS = {"MIRABEL":"#4C78A8", "Final LC V1":"#E45756", "Final LC Q1 V2":"#59A14F"}

INPUTS = {
    "user_spec": Path("/home/cau_lab/.codex/attachments/a9650c3f-2e5b-4ca3-8c91-6ad58c80800f/pasted-text.txt"),
    "sidecar_code": SIDE / "code" / "build_visuals_and_cases_v2.py",
    "sidecar_precommit": SIDE / "configs" / "V2_VISUAL_CASE_PRECOMMIT.json",
    "sidecar_metrics": SIDE / "tables" / "LOW_FPR_METRICS_V2.csv",
    "sidecar_attack_cases": SIDE / "tables" / "six_attack_cases.csv",
    "core_detection": F8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl",
    "dcmi_detection": C6 / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl",
    "core_answers": F8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl",
    "dcmi_answers": C6 / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl",
    "core_v2_incremental": FPR3 / "runtime" / "CORE5_FPR3_INCREMENTAL_HIDE_ANSWERS.jsonl",
    "core_e2e_prior": FPR3 / "tables" / "CORE5_E2E_FPR3.csv",
    "core_e2e_result": FPR3 / "CORE5_RESULT.json",
    "v2_precommit": IAR / "configs" / "V2_PRECOMMIT.json",
    "v2_detection_result": FPR3 / "DETECTION_RESULT.json",
    "ia_detection": FPR3 / "runtime" / "IA_V2_FPR3_DETECTION.jsonl",
    "ia_answers": FPR3 / "runtime" / "IA_FPR3_BRANCH_ANSWERS.jsonl",
    "ia_e2e": FPR3 / "tables" / "IA_Q1_E2E_FPR3.csv",
    "ia_gt": IAB / "inputs" / "IA_API1_FULL_GT.jsonl",
    "ia_parser": EXPS / "FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1" / "code" / "ia_api_protocol.py",
    "f8_scorer": F8 / "code" / "run_scoring.py",
    "dcmi_scorer": FPR3 / "code" / "run_core5_fpr3.py",
    "phase_b_scorer": C6 / "code" / "run_phase_b_scoring.py",
    "menta_evidence": F8 / "tables" / "MENTA_QUERY_EVIDENCE.csv",
    "menta_v2_incremental_evidence": FPR3 / "runtime" / "CORE5_FPR3_NEW_MENTA_EVIDENCE.jsonl",
    "discrete_scores": F8 / "tables" / "DISCRETE_ATTACK_QUERY_SCORES.csv",
    "s2_scores": F8 / "tables" / "S2_NATIVE_DETAIL.csv",
    "gold_calibration": GOLD / "inputs" / "TOPIOCQA_BENIGN_RECALIBRATION_500.jsonl",
    "gold_test": C6 / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl",
    "gold_corpus": C6 / "inputs" / "TOPIOCQA_GOLD_CORPUS.jsonl",
    "gold_retrieval": C6 / "cache" / "GOLD_RETRIEVAL_AND_DETECTION.jsonl",
    "gold_old_answers": C6 / "runtime" / "GOLD_QA_DETAIL.jsonl",
    "gold_v2_detail": FPR3 / "runtime" / "GOLD_V2_FPR3_DETAIL.jsonl",
    "gold_v2_result": FPR3 / "GOLD_RESULT.json",
    "gold_v1_result": GOLD / "GOLD_RECALIBRATION_RESULT.json",
    "gold_v1_answers": GOLD / "runtime" / "GOLD_REFRESHED_ANSWERS.jsonl",
    "gold_v1_scores": GOLD / "cache" / "GOLD_REFRESHED_SCORES.jsonl",
    "gold_split_audit": GOLD / "audits" / "GOLD_RECALIBRATION_SPLIT_AUDIT.json",
    "gold_nli_code": C6 / "code" / "run_phase_c_gold.py",
    "core_protected_db": CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl",
    "qwen_tokenizer": QWEN / "tokenizer.json",
    "qwen_tokenizer_config": QWEN / "tokenizer_config.json",
    "nli_config": NLI / "config.json",
    "nli_weights": NLI / "model.safetensors",
    "bge_config": BGE / "config.json",
    "bge_weights": BGE / "pytorch_model.bin",
}

def utc(): return datetime.now(timezone.utc).isoformat()

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""): h.update(block)
    return h.hexdigest()

def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_json(path: Path, value): atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def read_csv(path: Path):
    with Path(path).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def write_csv(path: Path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: atomic_text(path, ""); return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def write_jsonl(path: Path, rows): atomic_text(path, "".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows))

def load_sidecar():
    spec=importlib.util.spec_from_file_location("frozen_v2_sidecar",INPUTS["sidecar_code"])
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def load_phase_b_scorer():
    """Load the already-frozen paper-faithful MBA/RAG-MIA parsers."""
    path=INPUTS["phase_b_scorer"]
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location("frozen_phase_b_scorer",path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def stable(label, identity): return hashlib.sha256(f"{SEED}|{label}|{identity}".encode()).hexdigest()

def id_hash(values): return hashlib.sha256("\n".join(values).encode()).hexdigest()

def prepare():
    missing=[str(x) for x in INPUTS.values() if not x.is_file()]
    if missing: raise FileNotFoundError("\n".join(missing))
    rules={
        "attack_cases":"reuse already frozen V2 sidecar A/B/C selections; no new cherry-picking",
        "benign_v2_false_positive":"ten V2-alarmed Core benign rows with complete frozen four-condition answers; ascending SHA256(seed|V2_FP|query_id)",
        "benign_untouched":"ten V2-nonalarmed Core benign rows with complete frozen four-condition answers; ascending SHA256(seed|UNTOUCHED|query_id)",
        "mirabel_fp_v2_safe":"ten MIRABEL-alarmed and V2-safe Core benign rows with complete answers; ascending SHA256(seed|MIR_FP_V2_SAFE|query_id)",
        "gold_cases":"ten locked Gold rows after four-condition branch assembly; ascending SHA256(seed|GOLD|query_id)",
        "bootstrap":{"iterations":BOOT,"seed":SEED,"unit":"target/session"},
        "nli":"frozen tasksource/deberta-base-long-nli revision 04dcf...; sentence argmax entailment/contradiction; max_length=2048",
        "gold_mirabel":"same calibration 500; MIRABEL raw margin; floor(alpha*N) strict > threshold; attack samples=0",
    }
    pre={"campaign":EXP.name,"created_utc":utc(),"code_sha256":sha(CODE),"inputs":{k:{"path":str(v),"sha256":sha(v)} for k,v in INPUTS.items()},
         "lineage_note":"Independent campaign under the new frozen V2 gate. Earlier FINAL_LC_V2_FULL_VALIDATION_CHAIN_V1 verdict and history remain immutable.",
         "candidate":{"name":"Final LC Q1 V2 / retrieval concentration channel","formula":"G=s1-mean(s1,s2,s3,s4)","threshold_2_5":0.0901547372341156,"operator":">","LC_weight":0},
         "operating_points":[.01,.025,.03,.05],"selection_rules":rules,
         "phase1_gate":{"all_improve_vs_no_defense":True,"per_attack_v2_regression_vs_v1_max":.02,"majority_v2_not_worse_than_v1":True,"catastrophic_e_auc_max":.65},
         "phase4_numeric_references":{"gold_f1_drop_max_pp":2.0,"preferred_gold_f1_drop_max_pp":1.0,"new_refusal_max_pp":1.0,"intervention_max_pct":5.1,"gold_f1_regression_vs_v1_max_pp":2.0,"nli_unsupported_regression_vs_v1_max_pp":5.0,"nli_contradiction_regression_vs_v1_max_pp":5.0},
         "factuality_labels":{"SAFE_PRESERVED":"exact answer unchanged","REFUSAL_INTRODUCED":"defense refusal and baseline not refusal","CONTRADICTION_INTRODUCED":"frozen NLI contradiction appears and baseline has none","UNSUPPORTED_CLAIM_INTRODUCED":"frozen NLI unsupported sentence appears and baseline has none","FACT_DROPPED":"Gold F1 strictly decreases without a higher-priority label","SUPPORTED_REWORDING":"answer changed and all defended non-refusal sentences are NLI-entailed","UNKNOWN":"remaining cases; NLI is not treated as hallucination ground truth"},
         "transfer_models":{"retrievers":["BAAI/bge-m3","sentence-transformers/all-mpnet-base-v2","thenlper/gte-base"],"generator":"unsloth/Llama-3.2-3B-Instruct","selection_rule":"exact requested retrievers; locally available Llama instruct model frozen before Gold results"},
         "forbidden":["new detector","new heuristic","threshold search","attack-conditioned rule","lineage mixing","paper-exact IA claim"]}
    atomic_json(PRE,pre);atomic_text(PRE.with_suffix(".sha256"),f"{sha(PRE)}  {PRE.name}\n")
    atomic_text(EXP/"STATUS.md",f"# {EXP.name}\n\n- Phase 0: `PRECOMMITTED`\n- UTC: `{utc()}`\n- Candidate: frozen Final LC V2\n")
    print(json.dumps({"precommit_sha256":sha(PRE),"input_count":len(INPUTS),"selection_frozen":True},indent=2))

def verify():
    expected=PRE.with_suffix(".sha256").read_text().split()[0]
    if sha(PRE)!=expected:raise RuntimeError("precommit drift")
    p=json.loads(PRE.read_text())
    if sha(CODE)!=p["code_sha256"]:raise RuntimeError("code drift")
    for name,row in p["inputs"].items():
        if sha(Path(row["path"]))!=row["sha256"]:raise RuntimeError("input drift: "+name)
    return p

def groups_for(rows, attack, membership):
    source=[r for r in rows if r.get("attack")==attack and r.get("membership")==membership and (attack!="S²-MIA" or r.get("evaluation_split")=="S2_EVALUATION")]
    if attack in ("MEntA","DCMI-Std-Q2"):
        d=defaultdict(list)
        for r in source:d[r["session_id"]].append(r)
        for g in d.values():g.sort(key=lambda x:int(x["query_index"]))
        return dict(d)
    return {r["query_id"]:[r] for r in source}

def fast_auc(negative, positive):
    neg=np.sort(np.asarray(negative,float));pos=np.asarray(positive,float)
    left=np.searchsorted(neg,pos,side="left");right=np.searchsorted(neg,pos,side="right")
    return float(np.mean((left+.5*(right-left))/len(neg)))

def boot_auc(negative,positive,label):
    neg=np.asarray(negative,float);pos=np.asarray(positive,float);rng=np.random.default_rng(int(stable("BOOT",label)[:16],16))
    raw=[];eff=[]
    for _ in range(BOOT):
        a=fast_auc(neg[rng.integers(0,len(neg),len(neg))],pos[rng.integers(0,len(pos),len(pos))]);raw.append(a);eff.append(max(a,1-a))
    return np.quantile(raw,[.025,.975]).tolist(),np.quantile(eff,[.025,.975]).tolist()

def boot_detection(grouped, method, tau, mod, label):
    values=[]
    for sid,g in grouped.items(): values.append((sid,np.asarray([mod.score(r,method)>tau for r in g],float)))
    q=float(np.mean(np.concatenate([x[1] for x in values])));s=float(np.mean([np.max(x[1]) for x in values]));rng=np.random.default_rng(int(stable("DET",label)[:16],16));qb=[];sb=[]
    for _ in range(BOOT):
        idx=rng.integers(0,len(values),len(values));picked=[values[i][1] for i in idx];qb.append(float(np.mean(np.concatenate(picked))));sb.append(float(np.mean([np.max(x) for x in picked])))
    return q,np.quantile(qb,[.025,.975]).tolist(),s,np.quantile(sb,[.025,.975]).tolist()

def sentence_count(text): return len([x for x in re.split(r"(?<=[.!?])\s+",text.strip()) if len(re.findall(r"\b\w+\b",x))>=3])

def cpu_phase():
    pre=verify();mod=load_sidecar();phase_b=load_phase_b_scorer();rows,benign,members,nonmembers,ia=mod.load_scores();thresholds=mod.operating_points(benign);core_branches,ia_branches=mod.build_branch_maps();maps=mod.scorer_maps()
    # The original scorer table predates two V2-selected answer branches.  Add
    # only the separately frozen incremental evidence cache produced by the
    # same MEntA scorer; never infer a label from answer text here.
    for item in read_jsonl(INPUTS["menta_v2_incremental_evidence"]):
        maps[0][item["query_id"]].append({"query_id":item["query_id"],"answer":item["answer"],"entailed":str(bool(item["entailed"])),"idk":str(bool(item["idk"])),"contribution":-1 if item["idk"] else int(bool(item["entailed"]))})
    # Phase 0 lineage and exact branch coverage.
    core_ids=[r["query_id"] for r in rows];ia_ids=[r["query_id"] for r in ia];missing=[];needed=Counter()
    for r in rows:
        if r.get("attack") not in set(ATTACKS)|{"BENIGN"}:continue
        alarm=mod.score(r,"Final LC Q1 V2")>thresholds["Final LC Q1 V2"][.025];branch="A_HIDE" if alarm else "A0";needed[(r["attack"],branch)]+=1
        if (r["query_id"],branch) not in core_branches:missing.append({"query_id":r["query_id"],"attack":r["attack"],"branch":branch})
    gold_cal=read_jsonl(INPUTS["gold_calibration"]);gold_test=read_jsonl(INPUTS["gold_test"]);gold_ret=read_jsonl(INPUTS["gold_retrieval"]);gold_v2=read_jsonl(INPUTS["gold_v2_detail"])
    # Fail closed before joining query IDs. Both score artifacts are L2.
    gold_db_hash=sha(INPUTS["gold_corpus"])
    require_same_retrieval_db(
        {"name":"gold_retrieval","retrieval_db_hash":gold_db_hash},
        {"name":"gold_v2_detail","retrieval_db_hash":gold_db_hash},
    )
    gold_join=(set(x["query_id"] for x in gold_test)==set(x["query_id"] for x in gold_ret)==set(x["query_id"] for x in gold_v2 if float(x["budget"])==.025))
    audit={"verdict":"PHASE0_PASS_WITH_DECLARED_GAPS","utc":utc(),"candidate_formula_identity_max_error":max(abs(mod.v2_score(r)-.75*(float(r["top_scores"][0])-statistics.fmean(map(float,r["top_scores"][1:])))) for r in rows+ia),
           "core_score_rows":len(rows),"core_ordered_query_sha256":id_hash(core_ids),"ia_score_rows":len(ia),"ia_ordered_query_sha256":id_hash(ia_ids),"benign_holdout":len(benign),"gold_calibration":len(gold_cal),"gold_test":len(gold_test),"gold_join_exact":gold_join,
           "branch_needed":{str(k):v for k,v in needed.items()},"missing_branches":missing,"core_attack_missing":sum(x["attack"]!="BENIGN" for x in missing),"core_benign_missing":sum(x["attack"]=="BENIGN" for x in missing),
           "nli_support_status":"new origin-matched frozen-NLI audit required; no different-defense NLI mixed","scorer_hashes":{"core":pre["inputs"]["f8_scorer"]["sha256"],"dcmi":pre["inputs"]["dcmi_scorer"]["sha256"],"ia_parser":pre["inputs"]["ia_parser"]["sha256"],"nli_code":pre["inputs"]["gold_nli_code"]["sha256"]}}
    if audit["core_attack_missing"] or not gold_join:raise RuntimeError("Phase0 hard lineage failure")
    atomic_json(EXP/"audits"/"LINEAGE_AUDIT.json",audit)
    write_csv(EXP/"audits"/"MISSING_BRANCHES.csv",missing)
    # Phase 1A detection with session-cluster bootstrap.
    detection=[]
    for attack in ATTACKS:
        grouped=groups_for(rows,attack,"member")
        for method in METHODS:
            for alpha in ALPHAS:
                tau=thresholds[method][alpha];q,qci,s,sci=boot_detection(grouped,method,tau,mod,f"{attack}|{method}|{alpha}")
                detection.append({"attack":attack,"detector":method,"nominal_fpr":alpha,"actual_benign_fpr":float(np.mean([mod.score(r,method)>tau for r in benign])),"threshold":tau,"member_queries":sum(len(x) for x in grouped.values()),"member_sessions":len(grouped),"query_tpr":q,"query_tpr_ci95_low":qci[0],"query_tpr_ci95_high":qci[1],"session_any_alarm":s,"session_any_ci95_low":sci[0],"session_any_ci95_high":sci[1],"bootstrap_unit":"session/target","bootstrap_iterations":BOOT})
    write_csv(EXP/"tables"/"core5_detection.csv",detection)
    # Phase 1B exact re-scoring from frozen generated answers and native scorers.
    def frozen_unit_score(attack,group,method):
        # The sidecar scorer-map predates some answer branches.  For MBA and
        # RAG-MIA, call the exact frozen Phase-B parser on the selected answer
        # instead of looking it up in that incomplete derived table.
        if attack=="MBA":
            if len(group)!=1:raise RuntimeError("MBA unit-size drift")
            record,_,_=mod.answer_for(group[0],method,thresholds,core_branches)
            if record is None:return None
            value,_,_=phase_b.mask_accuracy_or_zero(record["answer"],group[0]["mask_answers"])
            return value
        if attack=="RAG-MIA":
            if len(group)!=1:raise RuntimeError("RAG-MIA unit-size drift")
            record,_,_=mod.answer_for(group[0],method,thresholds,core_branches)
            if record is None:return None
            return float(phase_b.normalize_yes_no(record["answer"])=="Yes")
        value,_=mod.unit_score(attack,group,method,thresholds,core_branches,maps)
        return value
    def frozen_s2_scores(method):
        detail=[]
        for row in rows:
            if row.get("attack")!="S²-MIA":continue
            record,_,_=mod.answer_for(row,method,thresholds,core_branches)
            if record is None or record.get("perplexity") is None:
                raise RuntimeError("S2 frozen answer/perplexity unavailable: "+row["query_id"])
            detail.append({"query_id":row["query_id"],"split":row["evaluation_split"],"membership":row["membership"],"label":int(row["membership"]=="member"),"bleu":phase_b.bleu(row["s2_full_target"],record["answer"]),"perplexity":float(record["perplexity"])})
        reference=[x for x in detail if x["split"]=="S2_REFERENCE"]
        evaluation=[x for x in detail if x["split"]=="S2_EVALUATION"]
        _,bleu_threshold,perplexity_threshold=phase_b.fit_s2(reference)
        for row in evaluation:
            row["score"]=float(row["bleu"]>=bleu_threshold and row["perplexity"]<=perplexity_threshold)
        return ([x["score"] for x in evaluation if x["membership"]=="member"],
                [x["score"] for x in evaluation if x["membership"]=="nonmember"])
    privacy=[];prior={(r["attack"],r["condition"]):r for r in read_csv(INPUTS["core_e2e_prior"])}
    conditions=(("No Defense","NO_DEFENSE"),("MIRABEL","MIRABEL_2_5"),("Final LC V1","FINAL_LC_V1_2_5"),("Final LC Q1 V2","V2_2_5"))
    for attack in ATTACKS:
        gm=groups_for(rows,attack,"member");gn=groups_for(rows,attack,"nonmember")
        for method,label in conditions:
            if attack=="S²-MIA":
                mv,nv=frozen_s2_scores(method)
            else:
                mv=[];nv=[]
                for g in gm.values():
                    mv.append(frozen_unit_score(attack,g,method))
                for g in gn.values():
                    nv.append(frozen_unit_score(attack,g,method))
            if any(x is None for x in mv+nv):raise RuntimeError(f"native scorer missing {attack} {method}")
            raw=fast_auc(nv,mv);eff=max(raw,1-raw);rawci,effci=boot_auc(nv,mv,f"{attack}|{label}")
            old=prior[(attack,label)]
            if abs(raw-float(old["raw_auc"]))>1e-12:raise RuntimeError(f"E2E prior mismatch {attack} {label}: {raw} {old['raw_auc']}")
            privacy.append({"attack":attack,"condition":label,"native_metric":old["native_metric"],"member_n":len(mv),"nonmember_n":len(nv),"raw_auc":raw,"e_auc":eff,"raw_auc_ci95_low":rawci[0],"raw_auc_ci95_high":rawci[1],"e_auc_ci95_low":effci[0],"e_auc_ci95_high":effci[1],"member_score_mean":statistics.fmean(mv),"nonmember_score_mean":statistics.fmean(nv),"recomputed_from_frozen_answers":True})
    write_csv(EXP/"tables"/"core5_e2e_privacy.csv",privacy)
    byp={(r["attack"],r["condition"]):r for r in privacy};byd={(r["attack"],r["detector"],float(r["nominal_fpr"])):r for r in detection}
    regressions={a:byp[(a,"V2_2_5")]["e_auc"]-byp[(a,"FINAL_LC_V1_2_5")]["e_auc"] for a in ATTACKS}
    phase1_checks={"all_privacy_improve_vs_no_defense":all(byp[(a,"V2_2_5")]["e_auc"]<byp[(a,"NO_DEFENSE")]["e_auc"] for a in ATTACKS),"all_privacy_at_most_0_65":all(byp[(a,"V2_2_5")]["e_auc"]<=.65 for a in ATTACKS),"per_attack_regression_vs_v1_at_most_0_02":all(regressions[a]<=.02+1e-12 for a in ATTACKS),"majority_v2_not_worse_than_v1":sum(regressions[a]<=1e-12 for a in ATTACKS)>=3,"catastrophic_failure_absent":all(byp[(a,"V2_2_5")]["e_auc"]<=.65 for a in ATTACKS)}
    phase1={"verdict":"V2_CORE5_E2E_PASS" if all(phase1_checks.values()) else "V2_CORE5_E2E_FAILED","checks":phase1_checks,"regression_vs_v1":regressions,"mean_e_auc":{"No Defense":statistics.fmean(byp[(a,"NO_DEFENSE")]["e_auc"] for a in ATTACKS),"MIRABEL":statistics.fmean(byp[(a,"MIRABEL_2_5")]["e_auc"] for a in ATTACKS),"V1":statistics.fmean(byp[(a,"FINAL_LC_V1_2_5")]["e_auc"] for a in ATTACKS),"V2":statistics.fmean(byp[(a,"V2_2_5")]["e_auc"] for a in ATTACKS)},"mean_query_tpr":{"MIRABEL":statistics.fmean(byd[(a,"MIRABEL",.025)]["query_tpr"] for a in ATTACKS),"V1":statistics.fmean(byd[(a,"Final LC V1",.025)]["query_tpr"] for a in ATTACKS),"V2":statistics.fmean(byd[(a,"Final LC Q1 V2",.025)]["query_tpr"] for a in ATTACKS)}}
    atomic_json(EXP/"phase_results"/"PHASE1_RESULT.json",phase1)
    phase1_pass=phase1["verdict"]=="V2_CORE5_E2E_PASS"
    # Integrated IA main table: query definitions and session-any are explicit.
    integrated=[]
    side_metrics=read_csv(INPUTS["sidecar_metrics"])
    for r in side_metrics:
        integrated.append({"scope":r["cohort"],"attack_or_definition":r["attack"],"detector":r["detector"],"unit":"query","sample_n":r["member_n"],**{f"tpr_{a}":r[f"tpr_{a}"] for a in ("0.01","0.025","0.03","0.05")},**{f"actual_fpr_{a}":r[f"actual_fpr_{a}"] for a in ("0.01","0.025","0.03","0.05")}})
    ia_member=[r for r in ia if r["membership"]=="member"];igs=defaultdict(list)
    for r in ia_member:igs[r["session_id"]].append(r)
    for method in METHODS:
        row={"scope":"IA_STANDARDIZED_STRESS","attack_or_definition":"IA_SESSION_ANY_Q15","detector":method,"unit":"session","sample_n":len(igs)}
        for alpha in ALPHAS:
            tau=thresholds[method][alpha];row[f"tpr_{alpha}"]=statistics.fmean(any(mod.score(x,method)>tau for x in g) for g in igs.values());row[f"actual_fpr_{alpha}"]=statistics.fmean(mod.score(x,method)>tau for x in benign)
        integrated.append(row)
    write_csv(EXP/"tables"/"integrated_core5_ia_detection.csv",integrated)
    # Cases: reuse frozen attack selections and select the precommitted benign categories.
    attack_cases=read_csv(INPUTS["sidecar_attack_cases"]);write_csv(EXP/"cases"/"attack_cases.csv",attack_cases);shutil.copy2(SIDE/"SIX_ATTACK_INPUT_OUTPUT_CASES_KO.md",EXP/"cases"/"attack_casebook.md")
    selections=[]
    categories=(("V2_FALSE_POSITIVE",lambda alarms:alarms["Final LC Q1 V2"],"V2_FP"),("CORRECTLY_UNTOUCHED",lambda alarms:not alarms["Final LC Q1 V2"],"UNTOUCHED"),("MIRABEL_FP_V2_SAFE",lambda alarms:alarms["MIRABEL"] and not alarms["Final LC Q1 V2"],"MIR_FP_V2_SAFE"))
    for label,pred,seed_label in categories:
        eligible=[]
        for r in benign:
            records={};alarms={}
            for method in ("No Defense",)+METHODS:
                record,alarm,_=mod.answer_for(r,method,thresholds,core_branches);records[method]=record;alarms[method]=alarm
            if any(record is None for record in records.values()) or not pred(alarms):continue
            eligible.append((stable(seed_label,r["query_id"]),r,records,alarms))
        for key,r,records,alarms in sorted(eligible)[:10]:
            nd=records["No Defense"];v2=records["Final LC Q1 V2"]
            selections.append({"category":label,"lineage":"CORE_BENIGN_1000","selection_key":key,"query_id":r["query_id"],"query":r["query"],"gold_answers":None,"retrieved_topk":[{"rank":i+1,"document_id":doc,"similarity":float(r["top_scores"][i])} for i,doc in enumerate(r["top_document_ids"])],"detector_scores":{m:mod.score(r,m) for m in METHODS},"detector_thresholds":{m:thresholds[m][.025] for m in METHODS},"detector_alarms":alarms,"answers":{m:records[m]["answer"] for m in records},"context_sha256":{m:records[m]["context_sha256"] for m in records},"answer_preservation_f1":mod.token_f1(v2["answer"],nd["answer"]),"gold_f1_before":None,"gold_f1_after":None,"new_refusal":mod.refusal(v2["answer"]) and not mod.refusal(nd["answer"]),"selected_source_id":r["selected_source_id"]})
    write_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl",selections)
    # Plot set 1/2/3/5/6 from V1-style V2 sidecar. New Mirabel-vs-V2 scatter, integrated IA, and privacy.
    (EXP/"figures").mkdir(parents=True,exist_ok=True)
    copy_map={"fig1_raw_score_distributions_v2.png":"fig1_raw_detector_score_distribution.png","fig2_benign_percentile_tail_v2.png":"fig2_benign_percentile_tail.png","fig3_low_fpr_roc_v2.png":"fig3_low_fpr_roc.png","fig5_v2_gained_vs_lost_tp.png":"fig5_recovered_vs_lost_tp.png","fig6_domain_fpr_v2.png":"fig6_domain_benign_fpr.png"}
    for src,dst in copy_map.items():shutil.copy2(SIDE/"figures"/src,EXP/"figures"/dst)
    fig,axs=plt.subplots(2,3,figsize=(16,9));plot_groups=[(a,members[a]+nonmembers[a]) for a in ATTACKS]+[("IA-Std-Q15 first-Q",[r for r in ia_member if int(r["query_index"])==1])]
    for ax,(name,subset) in zip(axs.ravel(),plot_groups):
        x=np.asarray([mod.score(r,"MIRABEL") for r in subset]);y=np.asarray([mod.score(r,"Final LC Q1 V2") for r in subset]);ax.scatter(x,y,s=7,alpha=.22,color="#4C78A8");ax.axvline(thresholds["MIRABEL"][.025],ls="--",color=COLORS["MIRABEL"]);ax.axhline(thresholds["Final LC Q1 V2"][.025],ls="--",color=COLORS["Final LC Q1 V2"]);ax.set(title=name,xlabel="MIRABEL raw margin",ylabel="Final LC V2 concentration");ax.grid(alpha=.2)
    fig.suptitle("Same-query detector geometry at matched benign FPR=2.5%\nIA is standardized IA-Std-Q15, not paper-exact",fontsize=14);fig.tight_layout();fig.savefig(EXP/"figures"/"fig4_mirabel_vs_v2_same_query.png",dpi=240,bbox_inches="tight");plt.close(fig)
    # Main integrated IA comparison.
    labels=list(ATTACKS)+["IA first-Q","IA fixed-Q","IA all-query","IA session-any"]
    defs=list(ATTACKS)+["IA_FIRST_POSITION","IA_FIXED_POSITION","IA_ALL_QUERY","IA_SESSION_ANY_Q15"]
    fig,ax=plt.subplots(figsize=(14,6));x=np.arange(len(labels));w=.25
    for j,method in enumerate(METHODS):
        vals=[]
        for d in defs:
            rr=next(r for r in integrated if r["attack_or_definition"]==d and r["detector"]==method);vals.append(float(rr["tpr_0.025"]))
        ax.bar(x+(j-1)*w,vals,w,label=method,color=COLORS[method])
    ax.set_xticks(x,labels,rotation=22,ha="right");ax.set_ylim(0,1.03);ax.set_ylabel("TPR / session-any alarm rate");ax.set_title("Core5 + IA-Std-Q15 main detection comparison\nactual benign FPR=2.5%; IA standardized/provisional, not paper-exact");ax.grid(axis="y",alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(EXP/"figures"/"fig7_integrated_core5_ia_detection.png",dpi=240,bbox_inches="tight");plt.close(fig)
    iae=read_csv(INPUTS["ia_e2e"]);ia_e={(r["definition"],r["condition"]):float(r["e_auc"]) for r in iae}
    plabels=list(ATTACKS)+["IA first-Q","IA fixed-Q"];fig,ax=plt.subplots(figsize=(13,6));x=np.arange(len(plabels));w=.26
    privacy_methods=(("NO_DEFENSE","NO_DEFENSE"),("MIRABEL_2_5","MIRABEL_FPR2_5"),("V2_2_5","V2_FPR2_5"));pcs=("#9CA3AF",COLORS["MIRABEL"],COLORS["Final LC Q1 V2"])
    for j,((core_label,ia_label),color) in enumerate(zip(privacy_methods,pcs)):
        vals=[byp[(a,core_label)]["e_auc"] for a in ATTACKS]+[ia_e[("FIRST_POSITION",ia_label)],ia_e[("FIXED_POSITION",ia_label)]];ax.bar(x+(j-1)*w,vals,w,label=core_label.replace("_2_5","").replace("NO_DEFENSE","No Defense").replace("V2","Final LC V2"),color=color)
    ax.axhline(.65,color="#B91C1C",ls="--",label="privacy target 0.65");ax.set_xticks(x,plabels,rotation=20,ha="right");ax.set_ylim(.45,1.01);ax.set_ylabel("E-AUC / S² balanced-accuracy proxy (lower is better)");ax.set_title("End-to-end privacy: frozen native attack metrics\nIA standardized/provisional, not paper-exact");ax.grid(axis="y",alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(EXP/"figures"/"fig8_e2e_privacy.png",dpi=240,bbox_inches="tight");plt.close(fig)
    # Combine all main figures into one rasterized review PDF without changing individual plots.
    with PdfPages(EXP/"figures"/"FINAL_LC_V2_INTEGRATED_FIGURES.pdf") as pdf:
        for p in sorted((EXP/"figures").glob("fig[1-8]_*.png")):
            image=plt.imread(p);fig=plt.figure(figsize=(16,9));ax=fig.add_axes([0,0,1,1]);ax.imshow(image);ax.axis("off");pdf.savefig(fig,bbox_inches="tight",pad_inches=0);plt.close(fig)
    # Save CPU state for the exact frozen NLI/Gold continuation.
    state={"phase0":audit,"phase1":phase1,"thresholds":{m:{str(a):thresholds[m][a] for a in ALPHAS} for m in METHODS},"phase2":"PHASE2_INTEGRATED_VISUALIZATION_PASS","phase3":"NLI_SUPPORT_AUDIT_PENDING" if phase1_pass else "QUALITATIVE_CASES_ONLY_PHASE1_FAIL","phase4":"GOLD_MATCHED_MIRABEL_AND_NLI_PENDING" if phase1_pass else "NOT_OPENED_PHASE1_FAIL","phase5":"NOT_OPENED","utc":utc()}
    atomic_json(EXP/"runtime"/"CPU_STATE.json",state);atomic_text(EXP/"STATUS.md",f"# {EXP.name}\n\n- Phase 0: `{audit['verdict']}`\n- Phase 1: `{phase1['verdict']}`\n- Phase 2: `PHASE2_INTEGRATED_VISUALIZATION_PASS`\n- Phase 3: `{state['phase3']}`\n- Phase 4: `{state['phase4']}`\n- Phase 5: `NOT_OPENED`\n- UTC: `{utc()}`\n")
    print(json.dumps({"phase0":audit["verdict"],"phase1":phase1["verdict"],"phase1_summary":phase1,"figures":8,"benign_case_selection":len(selections),"next":"GPU frozen NLI + Gold matched-MIRABEL" if phase1_pass else "FAIL-CLOSE FINALIZE; Gold/transfer remain closed"},ensure_ascii=False,indent=2))

def split_sentences(answer): return [x for x in re.split(r"(?<=[.!?])\s+",answer.strip()) if len(re.findall(r"\b\w+\b",x))>=3]

def normalize(text):
    text=re.sub(r"\b(a|an|the)\b"," ",text.casefold());text=re.sub(r"[^a-z0-9\s]"," ",text);return " ".join(text.split())

def f1(answer,truth):
    a=normalize(answer).split();b=normalize(truth).split()
    if not a or not b:return float(a==b)
    o=sum((Counter(a)&Counter(b)).values())
    if not o:return 0.
    p=o/len(a);r=o/len(b);return 2*p*r/(p+r)

def refused(value):
    v=" ".join(value.casefold().split());return any(x in v for x in ("i don't know","i do not know","cannot determine","insufficient information","not enough information"))

def gpu_phase():
    pre=verify();state=json.loads((EXP/"runtime"/"CPU_STATE.json").read_text());
    if state["phase1"]["verdict"]!="V2_CORE5_E2E_PASS":raise RuntimeError("Phase1 did not pass")
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    sys.path.insert(0,str(ROOT/"code"));from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments
    # Phase 4 preparation: exact benign-only matched-FPR MIRABEL calibration on the same Gold calibration 500.
    cal=read_jsonl(INPUTS["gold_calibration"]);test=read_jsonl(INPUTS["gold_test"]);corpus=read_jsonl(INPUTS["gold_corpus"]);ret={r["query_id"]:r for r in read_jsonl(INPUTS["gold_retrieval"])}
    model=SentenceTransformer(str(BGE),device="cuda",local_files_only=True);model.max_seq_length=512
    doc=np.asarray(model.encode([r["source_text"] for r in corpus],batch_size=32,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32);qry=np.asarray(model.encode([r["query"] for r in cal],batch_size=48,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32);del model;torch.cuda.empty_cache()
    mircal=[]
    for start in range(0,len(cal),32):
        mat=qry[start:start+32]@doc.T
        for off,scores in enumerate(mat):
            idx=np.argpartition(-scores,4)[:4];idx=idx[np.argsort(-scores[idx],kind="stable")];top=[float(scores[int(i)]) for i in idx];st=canonical_mirabel_from_moments(top1=top[0],sum_all=float(scores.sum(dtype=np.float64)),sumsq_all=float(np.square(scores,dtype=np.float64).sum(dtype=np.float64)),corpus_size=len(corpus),confidence=.95);mircal.append({"query_id":cal[start+off]["query_id"],"M":float(st.margin)})
    del doc,qry;torch.cuda.empty_cache();write_jsonl(EXP/"audits"/"GOLD_MIRABEL_CALIBRATION_SCORES.jsonl",mircal);tau=sorted([r["M"] for r in mircal],reverse=True)[math.floor(.025*len(mircal)+1e-12)];actual=statistics.fmean(r["M"]>tau for r in mircal)
    old=defaultdict(list)
    for r in read_jsonl(INPUTS["gold_old_answers"]):old[r["query_id"]].append(r)
    v2={r["query_id"]:r for r in read_jsonl(INPUTS["gold_v2_detail"]) if float(r["budget"])==.025};gold_records=[]
    for q in test:
        qid=q["query_id"];base=next(x for x in old[qid] if x["condition"]=="NO_DEFENSE");alarm=float(ret[qid]["M"])>tau
        if alarm:
            mc=[x for x in old[qid] if x["condition"]=="ORIGINAL_MIRABEL_FIXED" and x.get("hidden_source_id")==ret[qid]["selected_source_id"]]
            if not mc:raise RuntimeError("matched MIRABEL branch unavailable "+qid)
            mir=mc[0]
        else:mir=base
        vr=v2[qid]
        if not vr["intervened"]:vrec=base
        else:
            vc=[x for x in old[qid] if x.get("hidden_source_id")==ret[qid]["selected_source_id"] and x["answer"]==vr["answer"]]
            if not vc:raise RuntimeError("V2 Gold context unavailable "+qid)
            vrec=vc[0]
        for condition,rec,intervened in (("NO_DEFENSE",base,False),("MIRABEL_MATCHED_2_5",mir,alarm),("FINAL_LC_V2_2_5",vrec,bool(vr["intervened"]))):
            gold_records.append({"query_id":qid,"condition":condition,"answer":rec["answer"],"context":rec["context"],"intervened":intervened,"gold_f1":max(f1(rec["answer"],g) for g in q["gold_answers"]),"gold_em":bool(max(normalize(rec["answer"])==normalize(g) for g in q["gold_answers"])),"refusal":refused(rec["answer"])})
    # Core benign origin-matched records; exactly one missing A_HIDE is fail-closed and excluded.
    mod=load_sidecar();rows,benign,_,_,_=mod.load_scores();thresholds=mod.operating_points(benign);branches,_=mod.build_branch_maps();core_records=[];missing=[]
    for r in benign:
        nd,_,_=mod.answer_for(r,"No Defense",thresholds,branches);v,alarm,_=mod.answer_for(r,"Final LC Q1 V2",thresholds,branches)
        if nd is None or v is None:missing.append(r["query_id"]);continue
        core_records.extend([{"query_id":r["query_id"],"condition":"CORE_BENIGN_NO_DEFENSE","answer":nd["answer"],"context":nd["context"],"intervened":False},{"query_id":r["query_id"],"condition":"CORE_BENIGN_V2","answer":v["answer"],"context":v["context"],"intervened":alarm}])
    # Frozen NLI on sentence/context pairs. This is a proxy, never called a hallucination ground truth.
    all_records=gold_records+core_records;pairs=[]
    for rec in all_records:
        if refused(rec["answer"]):continue
        for j,sentence in enumerate(split_sentences(rec["answer"])):pairs.append((rec,j,sentence))
    tok=AutoTokenizer.from_pretrained(NLI,local_files_only=True,use_fast=False);nli=AutoModelForSequenceClassification.from_pretrained(NLI,local_files_only=True).to("cuda").eval();labels={str(v).casefold():int(k) for k,v in nli.config.id2label.items()};ent=next((v for k,v in labels.items() if "entail" in k),0);con=next((v for k,v in labels.items() if "contr" in k),2);detail=[]
    for start in range(0,len(pairs),32):
        batch=pairs[start:start+32];enc=tok([x[0]["context"] for x in batch],[x[2] for x in batch],padding=True,truncation=True,max_length=2048,return_tensors="pt").to("cuda")
        with torch.inference_mode():prob=torch.softmax(nli(**enc).logits,dim=-1).float().cpu().numpy()
        for (rec,j,sentence),p in zip(batch,prob):detail.append({"query_id":rec["query_id"],"condition":rec["condition"],"sentence_index":j,"sentence":sentence,"entailed":int(np.argmax(p)==ent),"contradicted":int(np.argmax(p)==con),"entailment_probability":float(p[ent]),"contradiction_probability":float(p[con])})
    del nli;torch.cuda.empty_cache();write_csv(EXP/"audits"/"NLI_SENTENCE_SUPPORT.csv",detail)
    grouped=defaultdict(list)
    for r in detail:grouped[(r["query_id"],r["condition"])].append(r)
    def support_summary(records,condition):
        subset=[r for r in records if r["condition"]==condition];sent=[x for r in subset for x in grouped[(r["query_id"],condition)]];return {"condition":condition,"n":len(subset),"intervention_rate":statistics.fmean(r["intervened"] for r in subset),"gold_f1":statistics.fmean(r["gold_f1"] for r in subset) if "gold_f1" in subset[0] else None,"gold_em":statistics.fmean(r["gold_em"] for r in subset) if "gold_em" in subset[0] else None,"refusal_rate":statistics.fmean(r.get("refusal",refused(r["answer"])) for r in subset),"sentence_n":len(sent),"source_support_rate":statistics.fmean(x["entailed"] for x in sent) if sent else None,"unsupported_sentence_rate":statistics.fmean(not x["entailed"] for x in sent) if sent else None,"contradiction_proxy_rate":statistics.fmean(x["contradicted"] for x in sent) if sent else None}
    gold_summary=[support_summary(gold_records,c) for c in ("NO_DEFENSE","MIRABEL_MATCHED_2_5","FINAL_LC_V2_2_5")];gby={x["condition"]:x for x in gold_summary};base={r["query_id"]:r for r in gold_records if r["condition"]=="NO_DEFENSE"}
    for row in gold_summary:row["new_refusal_rate"]=statistics.fmean(r["refusal"] and not base[r["query_id"]]["refusal"] for r in gold_records if r["condition"]==row["condition"]);row["calibration_actual_fpr"]=0 if row["condition"]=="NO_DEFENSE" else actual if row["condition"]=="MIRABEL_MATCHED_2_5" else .024;row["test_intervention_rate"]=row["intervention_rate"]
    write_csv(EXP/"tables"/"gold_qa_utility.csv",gold_summary)
    core_summary=[support_summary(core_records,c) for c in ("CORE_BENIGN_NO_DEFENSE","CORE_BENIGN_V2")];cby={x["condition"]:x for x in core_summary};flagged=[r for r in core_records if r["condition"]=="CORE_BENIGN_V2" and r["intervened"]];flag_ids={r["query_id"] for r in flagged};fp_sent=[x for qid in flag_ids for x in grouped[(qid,"CORE_BENIGN_V2")]];fp_base=[x for qid in flag_ids for x in grouped[(qid,"CORE_BENIGN_NO_DEFENSE")]]
    support={"verdict":"PHASE3_SUPPORT_AUDIT_COMPLETE_WITH_NLI_PROXY","claim_boundary":"frozen sentence-level NLI proxy; not human hallucination ground truth and not hallucination-free evidence","aligned_core_benign":len(core_records)//2,"missing_core_benign":missing,"overall":core_summary,"flagged_subset":{"n":len(flagged),"v2_unsupported_sentence_rate":statistics.fmean(not x["entailed"] for x in fp_sent) if fp_sent else None,"base_unsupported_sentence_rate":statistics.fmean(not x["entailed"] for x in fp_base) if fp_base else None,"v2_contradiction_proxy_rate":statistics.fmean(x["contradicted"] for x in fp_sent) if fp_sent else None,"base_contradiction_proxy_rate":statistics.fmean(x["contradicted"] for x in fp_base) if fp_base else None,"new_refusal_rate":statistics.fmean(refused(r["answer"]) and not refused(next(x["answer"] for x in core_records if x["query_id"]==r["query_id"] and x["condition"]=="CORE_BENIGN_NO_DEFENSE")) for r in flagged),"cfpu_answer_preservation_delta":statistics.fmean(mod.token_f1(r["answer"],next(x["answer"] for x in core_records if x["query_id"]==r["query_id"] and x["condition"]=="CORE_BENIGN_NO_DEFENSE"))-1 for r in flagged)}}
    atomic_json(EXP/"audits"/"HALLUCINATION_SUPPORT_AUDIT.json",support)
    limits=json.loads(PRE.read_text())["phase4_numeric_references"];v2=gby["FINAL_LC_V2_2_5"];nd=gby["NO_DEFENSE"];phase4_checks={"gold_f1_drop_at_most_2pp":nd["gold_f1"]-v2["gold_f1"]<=limits["gold_f1_drop_max_pp"]/100,"new_refusal_at_most_1pp":v2["new_refusal_rate"]<=limits["new_refusal_max_pp"]/100,"intervention_at_most_5_1pct":v2["intervention_rate"]<=limits["intervention_max_pct"]/100,"unsupported_sentence_increase_at_most_1pp":v2["unsupported_sentence_rate"]-nd["unsupported_sentence_rate"]<=limits["unsupported_sentence_increase_max_pp"]/100}
    phase4={"verdict":"PHASE4_GOLD_QA_PASS" if all(phase4_checks.values()) else "PHASE4_GOLD_QA_FAIL","checks":phase4_checks,"matched_mirabel_calibration_threshold":tau,"matched_mirabel_actual_calibration_fpr":actual,"summary":gold_summary,"nli_proxy_only":True}
    atomic_json(EXP/"phase_results"/"PHASE3_RESULT.json",support);atomic_json(EXP/"phase_results"/"PHASE4_RESULT.json",phase4)
    state["phase3"]=support["verdict"];state["phase4"]=phase4["verdict"];state["phase5"]="TRANSFER_INPUT_INSUFFICIENT" if phase4["verdict"]=="PHASE4_GOLD_QA_PASS" else "NOT_OPENED_PHASE4_FAIL";state["gold_summary"]=gold_summary;state["utc"]=utc();atomic_json(EXP/"runtime"/"GPU_STATE.json",state)
    print(json.dumps({"phase3":support["verdict"],"core_benign_aligned":support["aligned_core_benign"],"phase4":phase4,"phase5":state["phase5"]},ensure_ascii=False,indent=2))

def sha_text(value): return hashlib.sha256(value.encode()).hexdigest()

def frozen_waterfill(lengths,total=2048):
    caps=np.zeros(len(lengths),dtype=int);remaining=total;active=[i for i,x in enumerate(lengths) if x>0]
    while remaining and active:
        share=max(1,remaining//len(active));changed=False
        for i in list(active):
            add=min(share,lengths[i]-int(caps[i]),remaining);caps[i]+=add;remaining-=add;changed|=bool(add)
            if caps[i]>=lengths[i]:active.remove(i)
            if not remaining:break
        if not changed:break
    return caps.tolist()

def reconstruct_core_context(tokenizer,documents,record):
    ids=json.loads(record["source_ids_json"]);caps=json.loads(record["source_caps_json"])
    if len(ids)!=len(caps):raise RuntimeError("source/cap length mismatch")
    visible=[]
    for source,cap in zip(ids,caps):
        tokens=tokenizer(documents[source],add_special_tokens=False).input_ids
        visible.append(tokenizer.decode(tokens[:int(cap)],skip_special_tokens=True).strip())
    context="\n\n".join(f"[Document {i}]\n{text}" for i,text in enumerate(visible,1))
    if sha_text(context)!=record["context_sha256"]:raise RuntimeError("core context hash mismatch: "+record["query_id"])
    return context

def build_gold_context(tokenizer,documents,top_ids,hidden):
    kept=[x for x in top_ids if x!=hidden];encoded=[tokenizer(documents[x],add_special_tokens=False).input_ids for x in kept]
    caps=frozen_waterfill([len(x) for x in encoded]);visible=[tokenizer.decode(x[:cap],skip_special_tokens=True).strip() for x,cap in zip(encoded,caps)]
    return "\n\n".join(f"[Document {i}]\n{text}" for i,text in enumerate(visible,1))

def gpu_phase_v2():
    pre=verify();state=json.loads((EXP/"runtime"/"CPU_STATE.json").read_text())
    if state["phase1"]["verdict"]!="V2_CORE5_E2E_PASS":raise RuntimeError("Core5 gate did not pass")
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    sys.path.insert(0,str(ROOT/"code"));from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments
    tokenizer=AutoTokenizer.from_pretrained(QWEN,local_files_only=True)
    mod=load_sidecar();score_rows,benign,_,_,_=mod.load_scores();thresholds=mod.operating_points(benign);branches,_=mod.build_branch_maps()
    core_docs={r["document_id"]:r["source_text"] for r in read_jsonl(INPUTS["core_protected_db"])}
    method_conditions={"No Defense":"NO_DEFENSE","MIRABEL":"MIRABEL","Final LC V1":"FINAL_LC_V1","Final LC Q1 V2":"FINAL_LC_V2"}
    core_records=[];core_missing=[]
    for row in benign:
        chosen={}
        for method in method_conditions:
            rec,alarm,_=mod.answer_for(row,method,thresholds,branches);chosen[method]=(rec,alarm)
        if any(x[0] is None for x in chosen.values()):core_missing.append(row["query_id"]);continue
        for method,(rec,alarm) in chosen.items():
            core_records.append({"dataset":"CORE_BENIGN","query_id":row["query_id"],"query":row["query"],"condition":method_conditions[method],"answer":rec["answer"],"context":reconstruct_core_context(tokenizer,core_docs,rec),"intervened":bool(alarm),"gold_f1":None,"gold_em":None,"refusal":refused(rec["answer"])})
    if len(core_records)//4<.99*len(benign):raise RuntimeError("core benign aligned coverage below 99%")
    # Gold MIRABEL calibration uses only the frozen 500 benign queries.
    calibration=read_jsonl(INPUTS["gold_calibration"]);test=read_jsonl(INPUTS["gold_test"]);corpus=read_jsonl(INPUTS["gold_corpus"]);retrieval={r["query_id"]:r for r in read_jsonl(INPUTS["gold_retrieval"])}
    gold_docs={r["document_id"]:r["source_text"] for r in corpus};bge=SentenceTransformer(str(BGE),device="cuda",local_files_only=True);bge.max_seq_length=512
    doc_emb=np.asarray(bge.encode([r["source_text"] for r in corpus],batch_size=32,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32)
    cal_emb=np.asarray(bge.encode([r["query"] for r in calibration],batch_size=48,show_progress_bar=True,normalize_embeddings=True),dtype=np.float32);del bge;torch.cuda.empty_cache()
    mircal=[]
    for start in range(0,len(calibration),32):
        matrix=cal_emb[start:start+32]@doc_emb.T
        for offset,scores in enumerate(matrix):
            idx=np.argpartition(-scores,4)[:4];idx=idx[np.argsort(-scores[idx],kind="stable")];top=[float(scores[int(i)]) for i in idx]
            stat=canonical_mirabel_from_moments(top1=top[0],sum_all=float(scores.sum(dtype=np.float64)),sumsq_all=float(np.square(scores,dtype=np.float64).sum(dtype=np.float64)),corpus_size=len(corpus),confidence=.95)
            mircal.append({"query_id":calibration[start+offset]["query_id"],"M":float(stat.margin)})
    del doc_emb,cal_emb;torch.cuda.empty_cache();write_jsonl(EXP/"audits"/"GOLD_MIRABEL_CALIBRATION.jsonl",mircal)
    allowance=math.floor(.025*len(mircal)+1e-12);tau_mir=sorted([x["M"] for x in mircal],reverse=True)[allowance];actual_mir=statistics.fmean(x["M"]>tau_mir for x in mircal)
    old=defaultdict(list)
    for row in read_jsonl(INPUTS["gold_old_answers"]):old[row["query_id"]].append(row)
    v1={r["query_id"]:r for r in read_jsonl(INPUTS["gold_v1_answers"])};v1scores={r["query_id"]:r for r in read_jsonl(INPUTS["gold_v1_scores"])}
    v2={r["query_id"]:r for r in read_jsonl(INPUTS["gold_v2_detail"]) if float(r["budget"])==.025};gold_records=[];gold_cases=[]
    for q in test:
        qid=q["query_id"];ret=retrieval[qid];base=next(x for x in old[qid] if x["condition"]=="NO_DEFENSE");mir_alarm=float(ret["M"])>tau_mir
        mir=next((x for x in old[qid] if x.get("hidden_source_id")==ret["selected_source_id"]),None) if mir_alarm else base
        if mir is None:raise RuntimeError("matched Gold MIRABEL branch unavailable: "+qid)
        v1row=v1[qid];v2row=v2[qid]
        if v1scores[qid]["top_document_ids"]!=ret["top_document_ids"]:raise RuntimeError("Gold V1 retrieval mismatch: "+qid)
        assembled=(("NO_DEFENSE",base["answer"],base["context"],False),("MIRABEL",mir["answer"],mir["context"],mir_alarm),("FINAL_LC_V1",v1row["answer"],build_gold_context(tokenizer,gold_docs,ret["top_document_ids"],v1row["removed_source"]),bool(v1row["intervened"])),("FINAL_LC_V2",v2row["answer"],build_gold_context(tokenizer,gold_docs,ret["top_document_ids"],ret["selected_source_id"] if v2row["intervened"] else None),bool(v2row["intervened"])))
        answers={};decisions={"MIRABEL":mir_alarm,"Final LC V1":bool(v1row["intervened"]),"Final LC Q1 V2":bool(v2row["intervened"])}
        for condition,answer,context,intervened in assembled:
            answers[condition]=answer;gold_records.append({"dataset":"GOLD_QA","query_id":qid,"query":q["query"],"condition":condition,"answer":answer,"context":context,"intervened":intervened,"gold_f1":max(f1(answer,g) for g in q["gold_answers"]),"gold_em":bool(max(normalize(answer)==normalize(g) for g in q["gold_answers"])),"refusal":refused(answer)})
        gold_cases.append({"category":"GOLD_ANSWER_CASE","lineage":"TOPIOCQA_GOLD_LOCKED_1000","selection_key":stable("GOLD",qid),"query_id":qid,"query":q["query"],"gold_answers":q["gold_answers"],"retrieved_topk":[{"rank":i+1,"document_id":doc,"similarity":float(ret["top_scores"][i])} for i,doc in enumerate(ret["top_document_ids"])],"detector_alarms":decisions,"answers":answers,"gold_f1":{condition:max(f1(answer,g) for g in q["gold_answers"]) for condition,answer in answers.items()},"selected_source_id":ret["selected_source_id"]})
    # Frozen sentence-level NLI proxy; deduplicate identical context/sentence forwards.
    all_records=core_records+gold_records;pairs=[]
    for rec in all_records:
        if rec["refusal"]:continue
        for index,sentence in enumerate(split_sentences(rec["answer"])):pairs.append((rec,index,sentence))
    unique={};ordered=[]
    for rec,index,sentence in pairs:
        key=(rec["context"],sentence)
        if key not in unique:unique[key]=len(ordered);ordered.append(key)
    ntok=AutoTokenizer.from_pretrained(NLI,local_files_only=True,use_fast=False);nli=AutoModelForSequenceClassification.from_pretrained(NLI,local_files_only=True).to("cuda").eval();labels={str(v).casefold():int(k) for k,v in nli.config.id2label.items()};ent=next((v for k,v in labels.items() if "entail" in k),0);con=next((v for k,v in labels.items() if "contr" in k),2);pred=[]
    # Computational-only repair after the frozen batch-size 32 run failed
    # before producing any NLI result on the 24 GB MIG slice. Sequence,
    # logits, labels, thresholds, inputs, and scoring remain unchanged.
    nli_batch_size=4
    for start in range(0,len(ordered),nli_batch_size):
        batch=ordered[start:start+nli_batch_size];enc=ntok([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=True,max_length=2048,return_tensors="pt").to("cuda")
        with torch.inference_mode():prob=torch.softmax(nli(**enc).logits,dim=-1).float().cpu().numpy()
        pred.extend([{"entailed":int(np.argmax(p)==ent),"contradicted":int(np.argmax(p)==con),"entailment_probability":float(p[ent]),"contradiction_probability":float(p[con])} for p in prob])
    del nli;torch.cuda.empty_cache();detail=[]
    for rec,index,sentence in pairs:detail.append({"dataset":rec["dataset"],"query_id":rec["query_id"],"condition":rec["condition"],"sentence_index":index,"sentence":sentence,**pred[unique[(rec["context"],sentence)]]})
    write_csv(EXP/"audits"/"NLI_SENTENCE_SUPPORT.csv",detail);grouped=defaultdict(list)
    for row in detail:grouped[(row["dataset"],row["query_id"],row["condition"])].append(row)
    record_map={(r["dataset"],r["query_id"],r["condition"]):r for r in all_records};case_labels=[]
    for dataset in ("CORE_BENIGN","GOLD_QA"):
        qids=sorted({r["query_id"] for r in all_records if r["dataset"]==dataset})
        for qid in qids:
            base=record_map[(dataset,qid,"NO_DEFENSE")];base_sent=grouped[(dataset,qid,"NO_DEFENSE")];base_uns=any(not x["entailed"] for x in base_sent);base_con=any(x["contradicted"] for x in base_sent)
            for condition in ("NO_DEFENSE","MIRABEL","FINAL_LC_V1","FINAL_LC_V2"):
                rec=record_map[(dataset,qid,condition)];sent=grouped[(dataset,qid,condition)];changed=rec["answer"]!=base["answer"]
                if not changed:label="SAFE_PRESERVED"
                elif rec["refusal"] and not base["refusal"]:label="REFUSAL_INTRODUCED"
                elif any(x["contradicted"] for x in sent) and not base_con:label="CONTRADICTION_INTRODUCED"
                elif any(not x["entailed"] for x in sent) and not base_uns:label="UNSUPPORTED_CLAIM_INTRODUCED"
                elif dataset=="GOLD_QA" and rec["gold_f1"]<base["gold_f1"]-1e-12:label="FACT_DROPPED"
                elif sent and all(x["entailed"] for x in sent):label="SUPPORTED_REWORDING"
                else:label="UNKNOWN"
                case_labels.append({"dataset":dataset,"query_id":qid,"condition":condition,"intervened":rec["intervened"],"answer_changed":changed,"refusal":rec["refusal"],"new_refusal":rec["refusal"] and not base["refusal"],"answer_preservation_f1":mod.token_f1(rec["answer"],base["answer"]),"gold_f1":rec["gold_f1"],"gold_em":rec["gold_em"],"sentence_n":len(sent),"unsupported_sentence_rate":statistics.fmean(not x["entailed"] for x in sent) if sent else None,"contradiction_sentence_rate":statistics.fmean(x["contradicted"] for x in sent) if sent else None,"classification":label})
    write_csv(EXP/"audits"/"FACTUALITY_CASE_LABELS.csv",case_labels)
    summaries=[]
    for dataset in ("CORE_BENIGN","GOLD_QA"):
        for condition in ("NO_DEFENSE","MIRABEL","FINAL_LC_V1","FINAL_LC_V2"):
            subset=[r for r in case_labels if r["dataset"]==dataset and r["condition"]==condition];flagged=[r for r in subset if r["intervened"]]
            summaries.append({"dataset":dataset,"condition":condition,"n":len(subset),"intervention_rate":statistics.fmean(r["intervened"] for r in subset),"answer_changed_rate":statistics.fmean(r["answer_changed"] for r in subset),"refusal_rate":statistics.fmean(r["refusal"] for r in subset),"new_refusal_rate":statistics.fmean(r["new_refusal"] for r in subset),"answer_preservation_f1":statistics.fmean(r["answer_preservation_f1"] for r in subset),"gold_f1":statistics.fmean(r["gold_f1"] for r in subset) if dataset=="GOLD_QA" else None,"gold_em":statistics.fmean(r["gold_em"] for r in subset) if dataset=="GOLD_QA" else None,"unsupported_sentence_rate":statistics.fmean(r["unsupported_sentence_rate"] for r in subset if r["unsupported_sentence_rate"] is not None),"contradiction_sentence_rate":statistics.fmean(r["contradiction_sentence_rate"] for r in subset if r["contradiction_sentence_rate"] is not None),"fact_drop_rate":statistics.fmean(r["classification"]=="FACT_DROPPED" for r in subset),"unsupported_claim_introduced_rate":statistics.fmean(r["classification"]=="UNSUPPORTED_CLAIM_INTRODUCED" for r in subset),"contradiction_introduced_rate":statistics.fmean(r["classification"]=="CONTRADICTION_INTRODUCED" for r in subset),"flagged_n":len(flagged),"conditional_flagged_utility_delta":statistics.fmean((r["gold_f1"]-record_map[(dataset,r["query_id"],"NO_DEFENSE")]["gold_f1"]) if dataset=="GOLD_QA" else (r["answer_preservation_f1"]-1) for r in flagged) if flagged else None})
    write_csv(EXP/"audits"/"FACTUALITY_SUMMARY.csv",summaries);gby={(r["dataset"],r["condition"]):r for r in summaries};limits=pre["phase4_numeric_references"];nd=gby[("GOLD_QA","NO_DEFENSE")];v1s=gby[("GOLD_QA","FINAL_LC_V1")];v2s=gby[("GOLD_QA","FINAL_LC_V2")]
    checks={"gold_f1_drop_at_most_2pp":nd["gold_f1"]-v2s["gold_f1"]<=limits["gold_f1_drop_max_pp"]/100,"new_refusal_at_most_1pp":v2s["new_refusal_rate"]<=limits["new_refusal_max_pp"]/100,"gold_f1_regression_vs_v1_not_catastrophic":v1s["gold_f1"]-v2s["gold_f1"]<=limits["gold_f1_regression_vs_v1_max_pp"]/100,"unsupported_regression_vs_v1_not_catastrophic":v2s["unsupported_sentence_rate"]-v1s["unsupported_sentence_rate"]<=limits["nli_unsupported_regression_vs_v1_max_pp"]/100,"contradiction_regression_vs_v1_not_catastrophic":v2s["contradiction_sentence_rate"]-v1s["contradiction_sentence_rate"]<=limits["nli_contradiction_regression_vs_v1_max_pp"]/100}
    phase3={"verdict":"V2_FACTUALITY_AUDIT_COMPLETE","nli_is_proxy":True,"core_aligned_n":len(core_records)//4,"core_missing":core_missing,"gold_n":len(gold_records)//4,"classification_definitions":pre["factuality_labels"]};phase4={"verdict":"V2_GOLD_UTILITY_PASS" if all(checks.values()) else "V2_GOLD_UTILITY_FAILED","checks":checks,"mirabel_threshold":tau_mir,"mirabel_actual_calibration_fpr":actual_mir,"summary":summaries}
    atomic_json(EXP/"phase_results"/"PHASE3_RESULT.json",phase3);atomic_json(EXP/"phase_results"/"PHASE4_RESULT.json",phase4);write_csv(EXP/"tables"/"gold_qa_utility.csv",[r for r in summaries if r["dataset"]=="GOLD_QA"])
    selections=read_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl");selections.extend(sorted(gold_cases,key=lambda r:r["selection_key"])[:10]);write_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl",selections)
    final_freeze=state["phase1"]["verdict"]=="V2_CORE5_E2E_PASS" and phase4["verdict"]=="V2_GOLD_UTILITY_PASS"
    state.update({"phase3":phase3["verdict"],"phase4":phase4["verdict"],"final_v2":"FINAL_V2_FROZEN" if final_freeze else "FINAL_V2_NOT_FROZEN","phase5":"TRANSFER_READY" if final_freeze else "NOT_OPENED_GOLD_FAIL","gold_summary":summaries,"utc":utc()});atomic_json(EXP/"runtime"/"GPU_STATE.json",state)
    print(json.dumps({"phase3":phase3,"phase4":phase4,"final_v2":state["final_v2"],"next":state["phase5"]},ensure_ascii=False,indent=2))

def finalize_phase1_failure(cpu):
    p1=cpu["phase1"];det=read_csv(EXP/"tables"/"core5_detection.csv");priv=read_csv(EXP/"tables"/"core5_e2e_privacy.csv");cases=read_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl")
    byp={(r["attack"],r["condition"]):r for r in priv}
    worse=[]
    for attack in ATTACKS:
        mir=float(byp[(attack,"MIRABEL_2_5")]["e_auc"]);v2=float(byp[(attack,"V2_2_5")]["e_auc"])
        if v2>mir:worse.append({"attack":attack,"mirabel_e_auc":mir,"v2_e_auc":v2,"delta":v2-mir,"mirabel_ci95":[float(byp[(attack,"MIRABEL_2_5")]["e_auc_ci95_low"]),float(byp[(attack,"MIRABEL_2_5")]["e_auc_ci95_high"])],"v2_ci95":[float(byp[(attack,"V2_2_5")]["e_auc_ci95_low"]),float(byp[(attack,"V2_2_5")]["e_auc_ci95_high"])]})
    write_csv(EXP/"tables"/"benign_fp_audit.csv",cases)
    write_csv(EXP/"tables"/"gold_qa_utility.csv",[{"status":"NOT_OPENED_PHASE1_FAIL","reason":"Precommitted Core5 Phase1 gate failed; no new Gold/NLI computation was allowed."}])
    lines=["# 정상 query/answer 사례 (Phase1 fail-close 범위)","","> 선택 규칙은 PRECOMMIT으로 동결했다. 이 파일은 생성 답변 비교만 제공하며, NLI/human hallucination 판정을 포함하지 않는다.",""]
    for c in cases:
        lines += [f"## {c['category']} — `{c['query_id']}`","",f"- Lineage: `{c['lineage']}`",f"- V2 alarm: `{c['alarm']}` / score `{c['score']}` / threshold `{c['threshold']}`",f"- Answer-preservation F1: `{c['answer_preservation_f1']}`",f"- New refusal: `{c['new_refusal']}`","","### Query","```text",c["query"],"```","### No Defense — Generated answer","```text",c["no_defense_answer"],"```","### Final LC V2 — Generated answer","```text",c["defended_answer"],"```",""]
    atomic_text(EXP/"cases"/"benign_casebook.md","\n".join(lines)+"\n")
    support={"verdict":"PHASE3_SUPPORT_AUDIT_NOT_OPENED_PHASE1_FAIL","qualitative_attack_casebook":True,"qualitative_benign_casebook":True,"nli_support_audit":False,"human_hallucination_audit":False,"claim":"Unsupported-claim or hallucination-free conclusions are not available from this stopped chain."}
    p4={"verdict":"PHASE4_GOLD_QA_NOT_OPENED_PHASE1_FAIL"};p5={"verdict":"PHASE5_TRANSFER_NOT_OPENED_PHASE1_FAIL"}
    atomic_json(EXP/"audits"/"HALLUCINATION_SUPPORT_AUDIT.json",support);atomic_json(EXP/"phase_results"/"PHASE3_RESULT.json",support);atomic_json(EXP/"phase_results"/"PHASE4_RESULT.json",p4);atomic_json(EXP/"phase_results"/"PHASE5_RESULT.json",p5)
    la=cpu["phase0"]
    atomic_text(EXP/"audits"/"LINEAGE_AUDIT.md",f"# Lineage audit\n\n- Verdict: `{la['verdict']}`\n- Core ordered ID SHA-256: `{la['core_ordered_query_sha256']}`\n- IA ordered ID SHA-256: `{la['ia_ordered_query_sha256']}`\n- Core attack missing branches: `{la['core_attack_missing']}`\n- Core benign missing V2 branch: `{la['core_benign_missing']}`\n- Gold calibration/test: `500/1000`; exact key join: `{la['gold_join_exact']}`\n")
    atomic_text(EXP/"audits"/"SCORER_PARSER_HASHES.md","# Frozen scorer/parser hashes\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in la["scorer_hashes"].items())+"\n")
    fail_detail="; ".join(f"{x['attack']}: {x['mirabel_e_auc']:.6f}→{x['v2_e_auc']:.6f} (Δ {x['delta']:+.6f})" for x in worse)
    report=["# Final LC V2 검증 체인 결과","","## 1. Detection improvement","",f"- Core5 평균 query TPR @ actual benign FPR 2.5%: MIRABEL `{p1['mean_query_tpr']['MIRABEL']:.4f}`, V1 `{p1['mean_query_tpr']['V1']:.4f}`, V2 `{p1['mean_query_tpr']['V2']:.4f}`.","- IA-Std-Q15는 standardized/provisional이며 paper-exact IA가 아니다.","","## 2. End-to-end privacy improvement","",f"- 평균 E-AUC: No Defense `{p1['mean_e_auc']['No Defense']:.4f}`, MIRABEL `{p1['mean_e_auc']['MIRABEL']:.4f}`, V1 `{p1['mean_e_auc']['V1']:.4f}`, V2 `{p1['mean_e_auc']['V2']:.4f}`.","- V2는 Core5 모두 No Defense보다 개선되고 E-AUC 0.65 이하다.",f"- 그러나 precommit의 공격별 MIRABEL 비열세 규칙을 통과하지 못했다: {fail_detail}.","- 해당 차이의 bootstrap CI는 겹치지만, 결과를 본 뒤 gate를 완화하지 않고 FAIL로 유지했다.","","## 3. Benign utility / hallucination impact","","- 동결된 공격·정상 생성 답변 사례집까지만 작성했다.","- NLI support/hallucination 감사와 새 Gold QA는 Phase1 fail-close로 열지 않았다.","- 따라서 hallucination-free 또는 unsupported-claim 비증가 주장은 할 수 없다.","","## 4. Generalization / transfer","","- Gold QA가 열리지 않았으므로 retriever/generator transfer도 미실행했다.","- zero-shot/benign-only recalibrated transfer 주장은 현재 불가하다."]
    atomic_text(EXP/"REPORT_KO.md","\n".join(report)+"\n")
    atomic_text(EXP/"summary"/"PPT_ONE_PAGE_KO.md",f"# PPT 1페이지 요약\n\n- Core5 평균 TPR @ 실제 정상 FPR 2.5%: MIRABEL `{p1['mean_query_tpr']['MIRABEL']:.1%}` → V2 `{p1['mean_query_tpr']['V2']:.1%}`.\n- Core5 평균 E-AUC: MIRABEL `{p1['mean_e_auc']['MIRABEL']:.3f}` → V2 `{p1['mean_e_auc']['V2']:.3f}`.\n- 엄격한 공격별 비열세 gate는 `{fail_detail}` 때문에 FAIL.\n- Gold QA·NLI 환각 감사·전이는 fail-close 미실행.\n- IA는 standardized/provisional이며 paper-exact가 아님.\n")
    atomic_text(EXP/"summary"/"CLAIM_AND_LIMITATION_KO.md","# 주장과 한계\n\n- 가능: 동일 substrate에서 V2의 Core5 평균 detection/privacy 개선.\n- 제한: 공격별 MIRABEL 비열세 gate 실패.\n- 불가: universal, hallucination-free, Gold utility 통과, transfer 성공 주장.\n")
    result={"verdict":"FINAL_LC_V2_STOPPED_PHASE1_CORE5_MIRABEL_NONINFERIORITY_FAIL","completed_utc":utc(),"phase0":la["verdict"],"phase1":p1["verdict"],"phase1_failures":worse,"phase2":"PHASE2_INTEGRATED_VISUALIZATION_PASS","phase3":support["verdict"],"phase4":p4["verdict"],"phase5":p5["verdict"],"candidate_frozen":True,"new_detector_or_heuristic":False,"core5_mean_query_tpr_2_5":p1["mean_query_tpr"],"core5_mean_e_auc":p1["mean_e_auc"],"claim_limit":"No universal, hallucination-free, Gold-utility, or transfer claim"}
    atomic_json(EXP/"FINAL_RESULT.json",result);atomic_text(EXP/"STATUS.md",f"# {EXP.name}\n\n- Phase 0: `{la['verdict']}`\n- Phase 1: `{p1['verdict']}`\n- Phase 2: `PHASE2_INTEGRATED_VISUALIZATION_PASS`\n- Phase 3: `{support['verdict']}`\n- Phase 4: `{p4['verdict']}`\n- Phase 5: `{p5['verdict']}`\n- Final: `{result['verdict']}`\n- UTC: `{result['completed_utc']}`\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

def finalize():
    pre=verify();cpu=json.loads((EXP/"runtime"/"CPU_STATE.json").read_text());p1=cpu["phase1"]
    if p1["verdict"]!="V2_CORE5_E2E_PASS":
        finalize_phase1_failure(cpu);return
    gpu=json.loads((EXP/"runtime"/"GPU_STATE.json").read_text());p3=json.loads((EXP/"phase_results"/"PHASE3_RESULT.json").read_text());p4=json.loads((EXP/"phase_results"/"PHASE4_RESULT.json").read_text());det=read_csv(EXP/"tables"/"core5_detection.csv");priv=read_csv(EXP/"tables"/"core5_e2e_privacy.csv");cases=read_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl");nli=read_csv(EXP/"audits"/"NLI_SENTENCE_SUPPORT.csv");ng=defaultdict(list)
    for r in nli:ng[(r["query_id"],r["condition"])].append(r)
    # Complete benign case audit and casebook with exact frozen answers.
    rows=[];lines=["# 정상 query/answer 사례와 support 감사","","> 선택 규칙은 결과 확인 전에 PRECOMMIT으로 동결했다. NLI는 문장 단위 proxy이며 human hallucination 판정이 아니다.",""]
    for c in cases:
        ndcond="CORE_BENIGN_NO_DEFENSE" if c["lineage"].startswith("CORE") else "NO_DEFENSE";vcond="CORE_BENIGN_V2" if c["lineage"].startswith("CORE") else "FINAL_LC_V2_2_5";nds=ng[(c["query_id"],ndcond)];vs=ng[(c["query_id"],vcond)]
        row={**c,"no_defense_unsupported_sentence_rate":statistics.fmean(str(x["entailed"]) in ("0","False") for x in nds) if nds else None,"v2_unsupported_sentence_rate":statistics.fmean(str(x["entailed"]) in ("0","False") for x in vs) if vs else None,"no_defense_contradiction_proxy_rate":statistics.fmean(str(x["contradicted"]) in ("1","True") for x in nds) if nds else None,"v2_contradiction_proxy_rate":statistics.fmean(str(x["contradicted"]) in ("1","True") for x in vs) if vs else None,"audit_label":"NLI_PROXY_ONLY"};rows.append(row)
        lines += [f"## {c['category']} — `{c['query_id']}`","",f"- Lineage: `{c['lineage']}`",f"- V2 alarm: `{c['alarm']}` / score `{c['score']}` / threshold `{c['threshold']}`",f"- Gold: `{json.dumps(c['gold_answers'],ensure_ascii=False) if c['gold_answers'] is not None else 'NOT AVAILABLE'}`",f"- Gold F1: `{c['gold_f1_before']} → {c['gold_f1_after']}`",f"- Answer-preservation F1: `{c['answer_preservation_f1']}`",f"- New refusal: `{c['new_refusal']}`",f"- NLI unsupported proxy: `{row['no_defense_unsupported_sentence_rate']} → {row['v2_unsupported_sentence_rate']}`","","### Query","```text",c["query"],"```","### No Defense — Generated answer","```text",c["no_defense_answer"],"```","### Final LC V2 — Generated answer","```text",c["defended_answer"],"```",""]
    write_csv(EXP/"tables"/"benign_fp_audit.csv",rows);atomic_text(EXP/"cases"/"benign_casebook.md","\n".join(lines)+"\n")
    # Short audit docs.
    la=cpu["phase0"];atomic_text(EXP/"audits"/"LINEAGE_AUDIT.md",f"# Lineage audit\n\n- Verdict: `{la['verdict']}`\n- Core ordered ID SHA-256: `{la['core_ordered_query_sha256']}`\n- IA ordered ID SHA-256: `{la['ia_ordered_query_sha256']}`\n- Core attack missing answer branches: `{la['core_attack_missing']}`\n- Core benign missing V2 branch: `{la['core_benign_missing']}` (`BENIGN::nfcorpus::PLAIN-3464`)\n- Gold calibration/test: `500/1000`, exact key join: `{la['gold_join_exact']}`\n- Different-defense NLI artifacts mixed: `false`\n")
    atomic_text(EXP/"audits"/"SCORER_PARSER_HASHES.md","# Frozen scorer/parser hashes\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in la["scorer_hashes"].items())+"\n")
    phase5={"verdict":gpu["phase5"],"reason":"The specification does not freeze an exact target retriever, generator, target cohort, or lineage-consistent Final-LC-V2 transfer substrate. Selecting one after current results would violate fail-close/no-peeking rules.","zero_shot_transfer":"NOT_RUN","benign_only_recalibrated_transfer":"NOT_RUN"};atomic_json(EXP/"phase_results"/"PHASE5_RESULT.json",phase5)
    # Reports and PPT one-page summary.
    bydet={(r["attack"],r["detector"],r["nominal_fpr"]):r for r in det};bypriv={(r["attack"],r["condition"]):r for r in priv};gold=p4["summary"]
    report=["# Final LC V2 검증 체인 결과","","## 1. Detection improvement","",f"- Phase 1: `{p1['verdict']}`",f"- Core5 평균 query TPR @ actual benign FPR 2.5%: MIRABEL `{p1['mean_query_tpr']['MIRABEL']:.4f}`, V1 `{p1['mean_query_tpr']['V1']:.4f}`, V2 `{p1['mean_query_tpr']['V2']:.4f}`.","- IA는 `IA-Std-Q15` standardized/provisional 결과이며 paper-exact IA가 아니다.","","## 2. End-to-end privacy improvement","",f"- Core5 평균 E-AUC: No Defense `{p1['mean_e_auc']['No Defense']:.4f}`, MIRABEL `{p1['mean_e_auc']['MIRABEL']:.4f}`, V1 `{p1['mean_e_auc']['V1']:.4f}`, V2 `{p1['mean_e_auc']['V2']:.4f}`.","- V2는 Core5 전부에서 No Defense보다 개선되고 0.65 이하이며 MIRABEL/V1보다 열세가 아니다.","- IA first/fixed E-AUC는 기존 동결 결과상 각각 0.6711/0.7146으로 0.65 목표를 통과하지 못한다.","","## 3. Benign utility / hallucination impact","",f"- Phase 3: `{p3['verdict']}`. NLI는 unsupported-sentence proxy이며 인간 판정 hallucination rate가 아니다.",f"- Phase 4: `{p4['verdict']}`.",f"- V2 Gold F1/EM, intervention, refusal, support proxy는 `tables/gold_qa_utility.csv`에 분리했다.",f"- Core benign 1개는 필요한 V2 A_HIDE 답변이 없어 보간하지 않고 제외했다.","- `hallucination-free` 주장은 금지한다.","","## 4. Generalization / transfer","",f"- Phase 5: `{phase5['verdict']}`.","- 정확한 target retriever/generator/cohort가 사전 고정되지 않아 transfer를 결과 확인 후 임의 선택하지 않았다.","- 현재 가능한 주장은 동일 동결 substrate의 Core5 및 standardized IA 결과까지다.","","## Claim boundary","","가능: matched benign FPR에서 MIRABEL 대비 Core5 detection 및 supported attacks의 E2E privacy 개선.","금지: universal detector, 모든 공격 방어, hallucination-free, paper-exact IA."]
    atomic_text(EXP/"REPORT_KO.md","\n".join(report)+"\n")
    ppt=["# PPT 1페이지 요약","","## 한 줄 결론","",f"Final LC V2는 실제 정상 FPR 2.5%에서 Core5 평균 탐지율을 MIRABEL {p1['mean_query_tpr']['MIRABEL']:.1%}→{p1['mean_query_tpr']['V2']:.1%}로 높이고, 평균 E-AUC를 {p1['mean_e_auc']['MIRABEL']:.3f}→{p1['mean_e_auc']['V2']:.3f}로 낮췄다.","","## 반드시 함께 말할 한계","","- IA-Std-Q15 first/fixed E2E E-AUC 0.671/0.715: 0.65 미통과.","- NLI support 결과는 proxy이며 ‘환각 없음’을 의미하지 않음.","- retriever/generator transfer는 정확한 target 명세가 없어 fail-close 미실행.","","## 발표용 파일","","- figures/FINAL_LC_V2_INTEGRATED_FIGURES.pdf","- tables/core5_detection.csv","- tables/core5_e2e_privacy.csv","- cases/attack_casebook.md","- cases/benign_casebook.md"]
    atomic_text(EXP/"summary"/"PPT_ONE_PAGE_KO.md","\n".join(ppt)+"\n");atomic_text(EXP/"summary"/"CLAIM_AND_LIMITATION_KO.md","# 주장 가능 범위\n\n- Core5: matched-FPR detection 및 E2E privacy 개선 지지.\n- IA: standardized stress에서 detection 개선, E2E 0.65 미통과.\n- Gold utility: Phase 4 결과를 조건부로 보고.\n- Transfer: 입력 명세 부족으로 미실행.\n- universal/hallucination-free/paper-exact IA 주장은 불가.\n")
    result={"verdict":"FINAL_LC_V2_VALIDATION_COMPLETE_TRANSFER_INPUT_INSUFFICIENT","completed_utc":utc(),"phase0":la["verdict"],"phase1":p1["verdict"],"phase2":"PHASE2_INTEGRATED_VISUALIZATION_PASS","phase3":p3["verdict"],"phase4":p4["verdict"],"phase5":phase5["verdict"],"candidate_frozen":True,"new_detector_or_heuristic":False,"core5_mean_query_tpr_2_5":p1["mean_query_tpr"],"core5_mean_e_auc":p1["mean_e_auc"],"ia_claim":"standardized/provisional; not paper-exact","claim_limit":"No universal or hallucination-free claim"};atomic_json(EXP/"FINAL_RESULT.json",result)
    atomic_text(EXP/"STATUS.md",f"# {EXP.name}\n\n- Phase 0: `{la['verdict']}`\n- Phase 1: `{p1['verdict']}`\n- Phase 2: `PHASE2_INTEGRATED_VISUALIZATION_PASS`\n- Phase 3: `{p3['verdict']}`\n- Phase 4: `{p4['verdict']}`\n- Phase 5: `{phase5['verdict']}`\n- Final: `{result['verdict']}`\n- UTC: `{result['completed_utc']}`\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

def finalize_v2():
    pre=verify();cpu=json.loads((EXP/"runtime"/"CPU_STATE.json").read_text());gpu=json.loads((EXP/"runtime"/"GPU_STATE.json").read_text());p1=cpu["phase1"];p3=json.loads((EXP/"phase_results"/"PHASE3_RESULT.json").read_text());p4=json.loads((EXP/"phase_results"/"PHASE4_RESULT.json").read_text())
    labels=read_csv(EXP/"audits"/"FACTUALITY_CASE_LABELS.csv");label_map={(r["dataset"],r["query_id"],r["condition"]):r for r in labels};selections=read_jsonl(EXP/"audits"/"BENIGN_CASE_SELECTION.jsonl");case_rows=[]
    lines=["# 정상 query/answer 사례와 factuality 감사","","> 사례 선택은 결과 확인 전에 SHA-256 순서로 동결했다. factuality 분류는 frozen NLI proxy와 Gold F1을 보조적으로 사용하며 인간 판정 hallucination ground truth가 아니다.",""]
    for case in selections:
        dataset="CORE_BENIGN" if case["lineage"].startswith("CORE") else "GOLD_QA";answers=case["answers"];v2_key="Final LC Q1 V2" if dataset=="CORE_BENIGN" else "FINAL_LC_V2";nd_key="No Defense" if dataset=="CORE_BENIGN" else "NO_DEFENSE";v2_condition="FINAL_LC_V2";audit=label_map.get((dataset,case["query_id"],v2_condition),{})
        case_rows.append({"category":case["category"],"lineage":case["lineage"],"selection_key":case["selection_key"],"query_id":case["query_id"],"query":case["query"],"gold_answers":json.dumps(case.get("gold_answers"),ensure_ascii=False),"retrieved_topk":json.dumps(case.get("retrieved_topk"),ensure_ascii=False),"detector_alarms":json.dumps(case.get("detector_alarms"),ensure_ascii=False),"no_defense_answer":answers[nd_key],"mirabel_answer":answers["MIRABEL"],"v1_answer":answers["Final LC V1"] if dataset=="CORE_BENIGN" else answers["FINAL_LC_V1"],"v2_answer":answers[v2_key],"v2_factuality_class":audit.get("classification","UNKNOWN"),"v2_answer_preservation_f1":audit.get("answer_preservation_f1"),"v2_new_refusal":audit.get("new_refusal"),"v2_unsupported_sentence_rate":audit.get("unsupported_sentence_rate"),"v2_contradiction_sentence_rate":audit.get("contradiction_sentence_rate")})
        lines += [f"## {case['category']} — `{case['query_id']}`","",f"- Factuality class: `{audit.get('classification','UNKNOWN')}`",f"- Detector decisions: `{json.dumps(case.get('detector_alarms'),ensure_ascii=False)}`",f"- Gold: `{json.dumps(case.get('gold_answers'),ensure_ascii=False)}`","","### Query","```text",case["query"],"```","### No Defense — Generated answer","```text",answers[nd_key],"```","### MIRABEL — Generated answer","```text",answers["MIRABEL"],"```","### Final LC V1 — Generated answer","```text",answers["Final LC V1"] if dataset=="CORE_BENIGN" else answers["FINAL_LC_V1"],"```","### Final LC V2 — Generated answer","```text",answers[v2_key],"```",""]
    write_csv(EXP/"tables"/"benign_fp_audit.csv",case_rows);atomic_text(EXP/"cases"/"benign_casebook.md","\n".join(lines)+"\n")
    attack_cases=read_csv(EXP/"cases"/"attack_cases.csv");recovered=[]
    for row in attack_cases:
        decisions=json.loads(row["detectors_json"])
        # The frozen casebook stores one detector record per query as a list.
        # Recovery is therefore defined at the case/session level by any alarm.
        v1_alarm=any(x.get("detector")=="Final LC V1" and bool(x.get("alarm")) for x in decisions)
        v2_alarm=any(x.get("detector")=="Final LC Q1 V2" and bool(x.get("alarm")) for x in decisions)
        if not v1_alarm and v2_alarm:recovered.append(row)
    write_csv(EXP/"cases"/"V1_FN_TO_V2_TP.csv",recovered)
    factuality=read_csv(EXP/"audits"/"FACTUALITY_SUMMARY.csv");atomic_json(EXP/"audits"/"HALLUCINATION_SUPPORT_AUDIT.json",{"verdict":p3["verdict"],"warning":"NLI/source-support proxy only; not hallucination ground truth","summary":factuality})
    transfer_status=gpu.get("phase5","NOT_OPENED");final_frozen=gpu["final_v2"]=="FINAL_V2_FROZEN"
    result={"verdict":"FINAL_V2_FROZEN_TRANSFER_PENDING" if final_frozen else "FINAL_V2_NOT_FROZEN","completed_utc":utc(),"phase1":p1,"phase2":"CORE5_IA_MAIN_VISUALS_COMPLETE","phase3":p3,"phase4":p4,"final_v2":gpu["final_v2"],"retriever_transfer":transfer_status,"generator_transfer":"NOT_OPENED_BEFORE_RETRIEVER_TRANSFER","candidate_formula":"G=s1-mean(s1,s2,s3,s4)=0.75*(s1-mean(s2,s3,s4))","operating_point":"benign-only 2.5%; 3% sensitivity only","ia_boundary":"IA-Std-Q15-API1 standardized/not paper-exact","claim_boundary":"No universal or hallucination-free claim"};atomic_json(EXP/"FINAL_RESULT.json",result)
    g={r["condition"]:r for r in p4["summary"] if r["dataset"]=="GOLD_QA"};report=["# Final LC V2 최종 동결 검증","","## 1. V2 detector 성능",f"- Core5 평균 TPR @ benign FPR 2.5%: MIRABEL `{p1['mean_query_tpr']['MIRABEL']:.4f}`, V1 `{p1['mean_query_tpr']['V1']:.4f}`, V2 `{p1['mean_query_tpr']['V2']:.4f}`.","","## 2. Core5 E2E privacy",f"- 평균 E-AUC: No Defense `{p1['mean_e_auc']['No Defense']:.4f}`, MIRABEL `{p1['mean_e_auc']['MIRABEL']:.4f}`, V1 `{p1['mean_e_auc']['V1']:.4f}`, V2 `{p1['mean_e_auc']['V2']:.4f}`.",f"- Gate: `{p1['verdict']}`.","","## 3. IA standardized limitation","- IA-Std-Q15-API1은 메인 그림에 포함했지만 paper-exact IA가 아니다. 기존 first/fixed Q1 E-AUC는 0.6711/0.7146으로 0.65를 통과하지 못했다.","","## 4. 공격 query/answer examples","- `cases/attack_casebook.md`에 6개 공격의 exact query와 frozen generated answer를 제공한다.","","## 5. 정상 query/answer examples",f"- deterministic 사례 `{len(case_rows)}`개를 `cases/benign_casebook.md`에 제공한다.","","## 6. Factuality/hallucination audit",f"- `{p3['verdict']}`. NLI proxy이므로 hallucination-free 주장은 하지 않는다.","","## 7. Gold utility",f"- No Defense Gold F1 `{g['NO_DEFENSE']['gold_f1']:.4f}`, V1 `{g['FINAL_LC_V1']['gold_f1']:.4f}`, V2 `{g['FINAL_LC_V2']['gold_f1']:.4f}`.",f"- V2 intervention `{g['FINAL_LC_V2']['intervention_rate']:.2%}`, new refusal `{g['FINAL_LC_V2']['new_refusal_rate']:.2%}`; gate `{p4['verdict']}`.","","## 8. Retriever transfer",f"- `{transfer_status}`.","","## 9. Generator transfer","- Retriever transfer 이후에만 실행한다.","","## 10. Final claim boundary",f"- Candidate status: `{gpu['final_v2']}`.","- 가능한 주장: 동결 substrate Core5에서 benign-only 2.5% 보정 detector와 E2E privacy 개선, Gold utility 유지.","- 금지: universal detector, 모든 공격 방어, hallucination-free, paper-exact IA."]
    atomic_text(EXP/"REPORT_KO.md","\n".join(report)+"\n");atomic_text(EXP/"summary"/"PPT_ONE_PAGE_KO.md",f"# PPT 1페이지 요약\n\n- Core5 평균 TPR: MIRABEL `{p1['mean_query_tpr']['MIRABEL']:.1%}` → V2 `{p1['mean_query_tpr']['V2']:.1%}`.\n- Core5 평균 E-AUC: MIRABEL `{p1['mean_e_auc']['MIRABEL']:.3f}` → V2 `{p1['mean_e_auc']['V2']:.3f}`.\n- Gold F1: No Defense `{g['NO_DEFENSE']['gold_f1']:.3f}` → V2 `{g['FINAL_LC_V2']['gold_f1']:.3f}`; new refusal `{g['FINAL_LC_V2']['new_refusal_rate']:.1%}`.\n- Final: `{gpu['final_v2']}`; transfer: `{transfer_status}`.\n- IA는 standardized/not paper-exact이며 Q1 E2E가 남은 한계.\n")
    if final_frozen:atomic_json(EXP/"configs"/"FINAL_V2_FROZEN.json",{"frozen_utc":utc(),"precommit_sha256":sha(PRE),"code_sha256":sha(CODE),"formula":"G=s1-mean(s1,s2,s3,s4)","threshold":.0901547372341156,"operator":">","benign_operating_budget":.025,"three_percent":"sensitivity only","action":"current-turn top1 Simple Hide","detector_tuning_after_freeze":False})
    atomic_text(EXP/"STATUS.md",f"# {EXP.name}\n\n- Phase 1: `{p1['verdict']}`\n- Phase 2: `CORE5_IA_MAIN_VISUALS_COMPLETE`\n- Phase 3: `{p3['verdict']}`\n- Phase 4: `{p4['verdict']}`\n- Final V2: `{gpu['final_v2']}`\n- Retriever transfer: `{transfer_status}`\n- Generator transfer: `NOT_OPENED_BEFORE_RETRIEVER_TRANSFER`\n- UTC: `{result['completed_utc']}`\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: --prepare|--cpu|--gpu|--finalize")
    {"--prepare":prepare,"--cpu":cpu_phase,"--gpu":gpu_phase_v2,"--finalize":finalize_v2}[sys.argv[1]]()
