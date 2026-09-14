#!/usr/bin/env python3
"""Build and precommit a provenance-clean TopiOCQA gold utility evaluation."""
from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

from common import EXP, LC, ROOT, checkpoint, freeze_json, now, read_jsonl, sha_file, sha_text, write_jsonl

TOPI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/datasets--McGill-NLP--TopiOCQA/snapshots/66cd1dbf5577c653ecb99b385200f08e15e12f30/data")


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main() -> None:
    result=json.loads((EXP/"PHASE_B_RESULT.json").read_text(encoding="utf-8"))
    allowed = {"CORE6_MATCHED_BUDGET_E2E_PASS", "CORE5_MATCHED_BUDGET_E2E_PASS", "CORE5_E2E_PRIVACY_FAILED"}
    if result["verdict"] not in allowed:
        raise RuntimeError("Phase C prohibited: no authorized Core6/Core5 E2E predecessor")
    train=TOPI/"topiocqa_train.jsonl"; valid=TOPI/"topiocqa_valid.jsonl"
    if not train.is_file() or not valid.is_file(): raise RuntimeError("TopiOCQA frozen local snapshot missing")
    document_map={}; raw_valid=[]
    for path in (train,valid):
        for line in path.open(encoding="utf-8"):
            row=json.loads(line); passage=row["Gold_passage"]
            key=sha_text(json.dumps([passage["id"],passage["title"],passage["text"]],ensure_ascii=False,separators=(",",":")))
            document_map[key]={"document_id":f"TopiOCQA::{key}","source_id":str(passage["id"]),
                               "title":passage["title"],"source_text":f"{passage['title']}\n{passage['text']}"}
            if path==valid: raw_valid.append((key,row))
    corpus=sorted(document_map.values(),key=lambda row:row["document_id"])
    candidates=[]
    for key,row in raw_valid:
        answers=[row["Answer"]]+[item.get("Answer","") for item in row.get("Additional_answers",[])]
        answers=[answer for answer in answers if answer and answer!="UNANSWERABLE"]
        if not answers: continue
        history=row.get("Context") or []
        history_text="\n".join(("User: " if i%2==0 else "Assistant: ")+str(value) for i,value in enumerate(history))
        query=(f"Conversation history:\n{history_text}\n\nCurrent question: {row['Question']}" if history else row["Question"])
        row_id=f"topi-valid::{row['Conversation_no']}::{row['Turn_no']}"
        candidates.append({"query_id":row_id,"query":query,"raw_question":row["Question"],
                           "conversation_no":row["Conversation_no"],"turn_no":row["Turn_no"],
                           "gold_answers":answers,"gold_document_id":f"TopiOCQA::{key}",
                           "selection_key":sha_text(row_id),"official_metric":"SQuAD-normalized token F1 and exact match"})
    evaluation=sorted(candidates,key=lambda row:(row["selection_key"],row["query_id"]))[:1000]
    if len(evaluation)!=1000 or len({row["query_id"] for row in evaluation})!=1000: raise RuntimeError("gold evaluation count drift")
    old_targets=list(csv.DictReader((LC/"inputs"/"LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    old_text={sha_text(norm(row["source_text"])) for row in old_targets}
    corpus_text={sha_text(norm(row["source_text"])) for row in corpus}
    if old_text & corpus_text: raise RuntimeError("Topi corpus overlaps attack targets")
    old_queries=set()
    for path in (LC/"inputs"/"BENIGN_REFERENCE.jsonl",LC/"inputs"/"BENIGN_DEPLOYMENT_HOLDOUT.jsonl"):
        old_queries.update(sha_text(norm(row["query"])) for row in read_jsonl(path))
    if old_queries & {sha_text(norm(row["query"])) for row in evaluation}: raise RuntimeError("gold queries overlap Final-LC benign banks")
    write_jsonl(EXP/"inputs"/"TOPIOCQA_GOLD_CORPUS.jsonl",corpus)
    write_jsonl(EXP/"inputs"/"TOPIOCQA_GOLD_EVAL_1000.jsonl",evaluation)
    audit={"dataset":"McGill-NLP/TopiOCQA","snapshot":"66cd1dbf5577c653ecb99b385200f08e15e12f30",
           "train_sha256":sha_file(train),"valid_sha256":sha_file(valid),"corpus_documents":len(corpus),
           "evaluation_queries":len(evaluation),"attack_target_text_overlap":0,"final_lc_benign_query_overlap":0,
           "selection":"lowest SHA256(query_id) among valid rows with non-UNANSWERABLE gold answer",
           "calibration_separation":"Final LC reference/threshold remain frozen; Topi evaluation rows are not used for calibration"}
    freeze_json(EXP/"audits"/"GOLD_QA_DATASET_AUDIT.json",audit)
    code=[EXP/"code"/name for name in ("common.py","prepare_phase_c.py","run_phase_c_gold.py")]
    if not all(path.is_file() for path in code): raise RuntimeError("Phase-C code set incomplete")
    pre={"campaign":EXP.name,"phase":"C_GOLD_QA","created_utc":now(),"status":"GOLD_QA_PRECOMMITTED",
         "phase_b_result":{"path":str(EXP/"PHASE_B_RESULT.json"),"sha256":sha_file(EXP/"PHASE_B_RESULT.json")},
         "dataset_audit":{"path":str(EXP/"audits"/"GOLD_QA_DATASET_AUDIT.json"),"sha256":sha_file(EXP/"audits"/"GOLD_QA_DATASET_AUDIT.json")},
         "corpus":{"path":str(EXP/"inputs"/"TOPIOCQA_GOLD_CORPUS.jsonl"),"sha256":sha_file(EXP/"inputs"/"TOPIOCQA_GOLD_CORPUS.jsonl"),"n":len(corpus)},
         "evaluation":{"path":str(EXP/"inputs"/"TOPIOCQA_GOLD_EVAL_1000.jsonl"),"sha256":sha_file(EXP/"inputs"/"TOPIOCQA_GOLD_EVAL_1000.jsonl"),"n":1000},
         "conditions":["NO_DEFENSE","ORIGINAL_MIRABEL_FIXED","GLOBAL_BC_FIXED","FINAL_LC_FIXED"],
         "official_metrics":["SQuAD-normalized token F1","SQuAD-normalized exact match"],
         "primary":"token F1","factuality":"sentence-level frozen long-NLI source-support and contradiction proxy; not hallucination rate",
         "gate":{"final_lc_f1_drop":"<=0.02 absolute","final_lc_new_refusal":"<=0.01",
                 "final_lc_better_than_original":"Final LC F1 loss < Original MIRABEL F1 loss"},
         "code_sha256":{str(path.relative_to(ROOT)):sha_file(path) for path in code}}
    digest=freeze_json(EXP/"configs"/"GOLD_QA_PRECOMMIT.json",pre)
    checkpoint("PHASE_C_PRECOMMIT_COMPLETE",precommit_sha256=digest,dataset="TopiOCQA",corpus_documents=len(corpus),gold_queries=1000)


if __name__=="__main__": main()
