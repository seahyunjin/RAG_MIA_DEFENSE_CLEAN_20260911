#!/usr/bin/env python3
"""Freeze an untouched FinQA finance substrate before any Phase-D performance is observed."""
from __future__ import annotations

import csv
import json
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

from common import (BGE, EXP, LC, PARENT, QWEN, ROOT, SEED, checkpoint, freeze_json,
                    now, read_jsonl, sha_file, sha_text, write_csv, write_jsonl)

REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
BASE = f"https://raw.githubusercontent.com/czyssrs/FinQA/{REVISION}/"
FILES = ("dataset/train.json", "dataset/dev.json", "dataset/test.json", "code/evaluate/evaluate.py")


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def document_text(row: dict) -> str:
    table = row["table"]
    table_lines = [" | ".join(str(cell) for cell in values) for values in table]
    return "\n".join([*map(str, row["pre_text"]), "Financial table:", *table_lines, *map(str, row["post_text"])]).strip()


def main() -> None:
    gate_paths = {
        "full_validity": (PARENT/"IA_STD_Q15_API1"/"FINAL_RESULT.json", "IA_STD_Q15_API1_READY"),
        "same_fpr_detection": (PARENT/"IA_STD_Q15_API1"/"post_ready"/"IA_API1_DETECTION_RESULT.json", "IA_STEALTH_DETECTION_PASS"),
        "e2e_privacy": (PARENT/"IA_STD_Q15_API1"/"post_ready"/"IA_API1_E2E_RESULT.json", "IA_STEALTH_CONFIRMATION_PASS"),
    }
    gate_audit = {}
    for name, (path, expected) in gate_paths.items():
        if not path.is_file():
            raise RuntimeError(f"untouched domain prohibited: missing {name} result")
        value = json.loads(path.read_text(encoding="utf-8"))
        actual = value.get("verdict")
        if actual != expected:
            raise RuntimeError(f"untouched domain prohibited: {name}={actual!r}, expected={expected!r}")
        gate_audit[name] = {"path": str(path), "sha256": sha_file(path), "verdict": actual}
    raw_dir=EXP/"inputs"/"finqa_official_0f16e286";raw_dir.mkdir(parents=True,exist_ok=True)
    raw_manifest={}
    for relative in FILES:
        path=raw_dir/relative.replace("/","__")
        if not path.is_file():
            with urllib.request.urlopen(BASE+relative,timeout=180) as response:
                data=response.read()
            path.write_bytes(data)
        raw_manifest[relative]={"path":str(path),"sha256":sha_file(path),"url":BASE+relative}

    rows=[]
    for split in ("train","dev","test"):
        path=Path(raw_manifest[f"dataset/{split}.json"]["path"])
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows.append((split,row))
    grouped=defaultdict(list)
    document_hash={}
    for split,row in rows:
        report_id=row["id"].rsplit("-",1)[0]
        text=document_text(row);digest=sha_text(norm(text))
        if report_id in document_hash and document_hash[report_id]!=digest:
            raise RuntimeError(f"FinQA report text drift across QA rows: {report_id}")
        document_hash[report_id]=digest;grouped[report_id].append((split,row))
    if len(grouped)!=2789: raise RuntimeError(f"FinQA report count drift: {len(grouped)}")
    ordered=sorted(grouped,key=lambda value:(sha_text(f"{SEED}|FINQA|{value}"),value))
    member_ids=set(ordered[:1000]);nonmember_ids=set(ordered[1000:2000]);neutral_ids=set(ordered[2000:])
    if len(member_ids)!=1000 or len(nonmember_ids)!=1000 or len(neutral_ids)!=789: raise RuntimeError("FinQA deterministic split drift")
    documents=[]
    for report_id in ordered:
        first=grouped[report_id][0][1]
        documents.append({"document_id":f"FinQA::{report_id}","source_id":report_id,
            "title":str(first.get("filename") or report_id),"source_text":document_text(first),
            "source_text_sha256":document_hash[report_id],
            "role":"member_target" if report_id in member_ids else "nonmember_target" if report_id in nonmember_ids else "neutral_database"})
    by_id={row["source_id"]:row for row in documents}
    protected=[row for row in documents if row["source_id"] not in nonmember_ids]
    targets=[]
    for membership,ids in (("member",member_ids),("nonmember",nonmember_ids)):
        members=sorted(ids,key=lambda value:(sha_text(f"{SEED}|{membership}|{value}"),value))
        for index,report_id in enumerate(members):
            row=by_id[report_id]
            targets.append({"target_id":row["document_id"],"document_id":row["document_id"],"local_document_id":row["document_id"],"source_id":report_id,"membership":membership,
                "domain":"finance/FinQA","source_text":row["source_text"],"source_text_sha256":row["source_text_sha256"],
                "selection_index":index,"screen":index<100})
    qas=[]
    for report_id in neutral_ids:
        for split,row in grouped[report_id]:
            qa=row["qa"]
            if "exe_ans" not in qa: continue
            qas.append({"query_id":f"FinQA::{row['id']}","query":qa["question"],
                "gold_answer":str(qa["exe_ans"]),"reported_answer":str(qa.get("answer","")),
                "program":str(qa.get("program","")),"gold_document_id":f"FinQA::{report_id}",
                "source_split":split,"selection_key":sha_text(f"{SEED}|BENIGN|{row['id']}")})
    qas.sort(key=lambda row:(row["selection_key"],row["query_id"]))
    if len(qas)<2000: raise RuntimeError(f"insufficient neutral FinQA QA rows: {len(qas)}")
    reference=qas[:1000];holdout=qas[1000:2000]

    old_targets=list(csv.DictReader((LC/"inputs"/"LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    old_text={sha_text(norm(row["source_text"])) for row in old_targets}
    new_text={row["source_text_sha256"] for row in documents}
    if old_text & new_text: raise RuntimeError("FinQA document overlaps prior attack targets")
    old_queries=set()
    for path in (LC/"inputs"/"BENIGN_REFERENCE.jsonl",LC/"inputs"/"BENIGN_DEPLOYMENT_HOLDOUT.jsonl"):
        old_queries.update(sha_text(norm(row["query"])) for row in read_jsonl(path))
    new_queries={sha_text(norm(row["query"])) for row in reference+holdout}
    if old_queries & new_queries: raise RuntimeError("FinQA benign query overlaps prior Final-LC benign data")
    protected_ids={row["document_id"] for row in protected}
    if any((row["target_id"] in protected_ids)!=(row["membership"]=="member") for row in targets):
        raise RuntimeError("FinQA membership inclusion/exclusion failure")

    write_jsonl(EXP/"inputs"/"FINQA_DOCUMENTS.jsonl",documents)
    write_jsonl(EXP/"inputs"/"FINQA_PROTECTED_DB.jsonl",protected)
    write_csv(EXP/"inputs"/"FINQA_TARGETS_1000_1000.csv",targets)
    write_jsonl(EXP/"inputs"/"FINQA_BENIGN_REFERENCE_1000.jsonl",reference)
    write_jsonl(EXP/"inputs"/"FINQA_BENIGN_HOLDOUT_1000.jsonl",holdout)
    audit={"dataset":"FinQA","official_repository":"https://github.com/czyssrs/FinQA",
        "revision":REVISION,"domain":"finance","raw_files":raw_manifest,"reports":len(documents),
        "protected_documents":len(protected),"targets":{"member":1000,"nonmember":1000,"screen_each":100},
        "neutral_reports":len(neutral_ids),"neutral_qa_available":len(qas),"benign_reference":1000,"benign_holdout":1000,
        "prior_document_overlap":0,"prior_benign_query_overlap":0,"membership_inclusion":1.0,"membership_exclusion":1.0,
        "selection":"deterministic SHA-256 ordering fixed before attack generation/performance",
        "claim_boundary":"finance domain is untouched by Final-LC development; FinQA official program evaluator is archived, while natural-answer Gold QA is reported separately as execution-value answer accuracy rather than leaderboard program accuracy"}
    freeze_json(EXP/"audits"/"FINQA_UNTOUCHED_DATASET_AUDIT.json",audit)
    threshold_path=PARENT/"IA_STD_Q15_API1"/"post_ready"/"tables"/"IA_ST1_BENIGN_THRESHOLDS.csv"
    threshold_rows=list(csv.DictReader(threshold_path.open(encoding="utf-8")))
    if not threshold_rows:
        raise RuntimeError("frozen same-FPR threshold table missing")
    code=sorted((EXP/"code").glob("*.py")) + [EXP/"code"/"run_untouched_chain.sh"]
    if not all(path.is_file() for path in code): raise RuntimeError("Phase-D code set incomplete")
    pre={"campaign":EXP.name,"phase":"D_NEW_DOMAIN_GENERALIZATION","created_utc":now(),"status":"PHASE_D_PRECOMMITTED",
        "dataset_audit":{"path":str(EXP/"audits"/"FINQA_UNTOUCHED_DATASET_AUDIT.json"),"sha256":sha_file(EXP/"audits"/"FINQA_UNTOUCHED_DATASET_AUDIT.json")},
        "inputs":{name:{"path":str(EXP/"inputs"/name),"sha256":sha_file(EXP/"inputs"/name)} for name in
            ("FINQA_DOCUMENTS.jsonl","FINQA_PROTECTED_DB.jsonl","FINQA_TARGETS_1000_1000.csv","FINQA_BENIGN_REFERENCE_1000.jsonl","FINQA_BENIGN_HOLDOUT_1000.jsonl")},
        "frozen_method":{"retriever":str(BGE),"generator":str(QWEN),"k_local":200,"locator":"canonical MIRABEL top-1","action":"Simple Hide + deterministic waterfill","training_steps":0,"trainable_parameter_updates":0},
        "parent_gate_artifacts":gate_audit,"strict_threshold_source":{"path":str(threshold_path),"sha256":sha_file(threshold_path)},"strict_thresholds":{f"{row['method']}@{row['nominal_fpr']}":row for row in threshold_rows},
        "primary_budget":.025,"screen":{"member":100,"nonmember":100},"scale":{"member":1000,"nonmember":1000},
        "attacks":["MEntA","RAG-MIA","DCMI-Std-Q2","IA-Std-Q15-API1"],
        "attack_generation":{"model":str(QWEN),"do_sample":False,"num_beams":1,"maximum_attempts":2,"query_bundle_once_only":True},
        "modes":{"STRICT":"old benign reference and thresholds unchanged","BENIGN_REFRESH":"new finance benign reference bank and holdout threshold only"},
        "screen_detection_gate":{"recalibrated_macro_delta_vs_matched_mirabel":">=0","each_attack_delta":">=-0.03","each_final_lc_tpr":">=0.05"},
        "generalization_gate":{"actual_benign_fpr":"<=0.05","privacy":"all evaluated attacks improve vs No Defense","matched_mirabel":"at least 75% at or below MIRABEL; remainder within +0.03","gold_qa_drop":"<=0.02 absolute","training_updates":0},
        "code_sha256":{str(path.relative_to(ROOT)):sha_file(path) for path in code}}
    digest=freeze_json(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json",pre)
    checkpoint("PHASE_D_PRECOMMIT_COMPLETE",dataset="FinQA",precommit_sha256=digest,reports=len(documents),member=1000,nonmember=1000,screen_each=100)


if __name__=="__main__": main()
