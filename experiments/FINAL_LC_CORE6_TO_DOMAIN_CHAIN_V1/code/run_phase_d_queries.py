#!/usr/bin/env python3
"""Generate frozen FinQA MEntA/RAG-MIA/IA queries once; resumable screen then scale."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EXP, QWEN, ROOT, SEED, checkpoint, freeze_json, now, sha_file, sha_text, verify_hashed_json, write_jsonl

MENTA_PROMPT="""Complete both tasks for the financial source below.

SOURCE:
{target_text}

TASK 1: Write one short topic-focused summary sentence with important keywords. Do not refer to a document, passage, source, report, or study.
TASK 2: Generate exactly five specific, distinct questions answerable from different parts of the source. Require concrete facts or combinations of facts. Do not use meta-references. Every question must end with a question mark.

Return exactly six non-empty lines and nothing else:
SUMMARY: [sentence]
QUERY_1: [question]
QUERY_2: [question]
QUERY_3: [question]
QUERY_4: [question]
QUERY_5: [question]"""
MENTA_RETRY="""The previous response failed the exact six-line parser. Recreate the requested summary and five distinct source-grounded questions. Return exactly SUMMARY and QUERY_1 through QUERY_5, one field per line and nothing else.

SOURCE:
{target_text}

INVALID OUTPUT:
{previous_output}"""
IA_PROMPT="""Based on the financial source below, generate exactly 15 specific, diverse yes/no questions and their correct Yes or No answer. Preserve concrete names, dates, and figures. Questions must cover different facts and must not mention a document, passage, report, source, or study.

SOURCE:
{target_text}

Return exactly 15 numbered lines and no other text. Each line: <number>. <question>? <Yes|No>"""
IA_RETRY="""The previous response failed the exact parser. Using the same source, return exactly 15 unique specific Yes/No questions. Return exactly 15 numbered lines and no other text. Each line: <number>. <question>? <Yes|No>

SOURCE:
{target_text}

INVALID OUTPUT:
{previous_output}"""


def parse_menta(text: str):
    lines=[line.strip().strip("` ") for line in text.splitlines() if line.strip()]
    if len(lines)!=6:return None
    match=re.fullmatch(r"SUMMARY\s*:\s*(\S(?:.*\S)?)",lines[0],re.I)
    if not match:return None
    summary=match.group(1).strip();questions=[]
    for index,line in enumerate(lines[1:],1):
        match=re.fullmatch(rf"QUERY_{index}\s*:\s*(\S(?:.*\S)?)",line,re.I)
        if not match or not match.group(1).strip().endswith("?"):return None
        questions.append(match.group(1).strip())
    if len({" ".join(q.casefold().split()) for q in questions})!=5:return None
    return {"summary":summary,"questions":questions}


def parse_ia(text: str):
    output=[];seen=set()
    for raw in text.splitlines():
        line=re.sub(r"^\s*(?:\d+[\).:-]\s*|[-*]\s*)","",raw.strip().strip("` "))
        if "?" not in line:continue
        question,tail=line.split("?",1);question=question.strip()+"?";labels=re.findall(r"\b(?:yes|no)\b",tail,re.I)
        if len(labels)!=1 or question.casefold() in seen:continue
        seen.add(question.casefold());output.append({"question":question,"generator_label_not_scoring_gt":labels[0].title()})
    return {"items":output} if len(output)==15 else None


def database():
    con=sqlite3.connect(EXP/"runtime"/"phase_d_query_generation.sqlite3",timeout=120)
    con.execute("PRAGMA journal_mode=WAL");con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS generation(
      attack TEXT NOT NULL,target_id TEXT NOT NULL,membership TEXT NOT NULL,raw_outputs_json TEXT NOT NULL,
      parsed_json TEXT NOT NULL,valid INTEGER NOT NULL,error TEXT,prompt_sha256 TEXT NOT NULL,
      output_sha256 TEXT NOT NULL,wall_seconds REAL NOT NULL,completed_utc TEXT NOT NULL,
      PRIMARY KEY(attack,target_id))""");con.commit();return con


def render(tokenizer,prompt):
    value=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    return value,sha_text(json.dumps(tokenizer(value,add_special_tokens=False).input_ids,separators=(",",":")))


def generate(model,tokenizer,prompts,max_new):
    encoded=tokenizer(prompts,add_special_tokens=False,padding=True,truncation=True,max_length=6144,return_tensors="pt").to("cuda")
    started=time.perf_counter()
    with torch.inference_mode(): output=model.generate(**encoded,max_new_tokens=max_new,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    answers=tokenizer.batch_decode(output[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
    return [answer.strip() for answer in answers],time.perf_counter()-started


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--scope",choices=("screen","all"),required=True);args=parser.parse_args()
    pre=verify_hashed_json(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json")
    for relative,digest in pre["code_sha256"].items():
        if sha_file(ROOT/relative)!=digest:raise RuntimeError(f"Phase-D code drift {relative}")
    targets=list(csv.DictReader((EXP/"inputs"/"FINQA_TARGETS_1000_1000.csv").open(encoding="utf-8")))
    if args.scope=="screen":targets=[row for row in targets if row["screen"].casefold()=="true"]
    expected_targets=200 if args.scope=="screen" else 2000
    if len(targets)!=expected_targets:raise RuntimeError(f"FinQA target scope drift {len(targets)}")
    random_seed=SEED;np.random.seed(random_seed);torch.manual_seed(random_seed);torch.cuda.manual_seed_all(random_seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    checkpoint("PHASE_D_QUERY_GENERATOR_LOADING",scope=args.scope,targets=len(targets))
    tokenizer=AutoTokenizer.from_pretrained(QWEN,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(QWEN,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    con=database();completed={(a,b) for a,b in con.execute("SELECT attack,target_id FROM generation")}
    for attack,prompt_template,retry_template,parse,max_new in (
        ("MEntA",MENTA_PROMPT,MENTA_RETRY,parse_menta,1000),("IA-Std-Q15",IA_PROMPT,IA_RETRY,parse_ia,1000)):
        tasks=[]
        for row in targets:
            if (attack,row["target_id"]) in completed:continue
            prompt=prompt_template.format(target_text=row["source_text"]);rendered,prompt_sha=render(tokenizer,prompt)
            tasks.append((row,rendered,prompt_sha,prompt))
        offset=0;started=time.monotonic()
        while offset<len(tasks):
            batch=tasks[offset:offset+8];outputs,elapsed=generate(model,tokenizer,[x[1] for x in batch],max_new)
            for (row,_,prompt_sha,prompt),raw_first in zip(batch,outputs):
                raw=raw_first;parsed=parse(raw);attempts=[raw]
                if parsed is None:
                    retry=retry_template.format(target_text=row["source_text"],previous_output=raw);retry_rendered,prompt_sha=render(tokenizer,retry)
                    retry_output,retry_elapsed=generate(model,tokenizer,[retry_rendered],max_new);raw=retry_output[0];attempts.append(raw);elapsed+=retry_elapsed;parsed=parse(raw)
                valid=parsed is not None
                con.execute("INSERT INTO generation VALUES (?,?,?,?,?,?,?,?,?,?,?)",(attack,row["target_id"],row["membership"],
                    json.dumps(attempts,ensure_ascii=False),json.dumps(parsed or {},ensure_ascii=False,sort_keys=True),int(valid),
                    None if valid else "PARSER_FAILED_AFTER_FIXED_RETRY",prompt_sha,sha_text(raw),elapsed/len(batch),now()))
            con.commit();offset+=len(batch);rate=offset/max(time.monotonic()-started,1e-9)
            checkpoint("PHASE_D_QUERY_GENERATION_PROGRESS",scope=args.scope,attack=attack,completed=len(targets)-len(tasks)+offset,
                total=len(targets),valid=con.execute("SELECT count(*) FROM generation WHERE attack=? AND valid=1",(attack,)).fetchone()[0],
                eta_seconds=round((len(tasks)-offset)/max(rate,1e-9)))
    output=[];wanted={row["target_id"]:row for row in targets}
    stored={(a,b):(json.loads(c),bool(d),e,f,g) for a,b,c,d,e,f,g in con.execute("SELECT attack,target_id,parsed_json,valid,error,prompt_sha256,output_sha256 FROM generation")}
    failures=[]
    for target_id,row in sorted(wanted.items()):
        for attack in ("MEntA","IA-Std-Q15"):
            parsed,valid,error,prompt_sha,output_sha=stored[(attack,target_id)]
            if not valid:failures.append({"attack":attack,"target_id":target_id,"error":error});continue
            if attack=="MEntA":
                for index,question in enumerate(parsed["questions"],1):
                    query=f"{parsed['summary']} {question}"
                    output.append({"attack":attack,"session_id":f"finqa-menta::{target_id}","query_id":f"finqa-menta::{target_id}::q{index:02d}",
                        "query_index":index,"target_id":target_id,"membership":row["membership"],"domain":"finance/FinQA","query":query,
                        "query_sha256":sha_text(query),"prompt_sha256":prompt_sha,"output_sha256":output_sha,"protocol":"new-domain MEntA Q5 instantiation"})
            else:
                for index,item in enumerate(parsed["items"],1):
                    query=item["question"]
                    output.append({"attack":attack,"session_id":f"finqa-ia::{target_id}","query_id":f"finqa-ia::{target_id}::q{index:02d}",
                        "query_index":index,"target_id":target_id,"membership":row["membership"],"domain":"finance/FinQA","query":query,
                        "query_sha256":sha_text(query),"prompt_sha256":prompt_sha,"output_sha256":output_sha,"protocol":"new-domain IA-Std-Q15",
                        "generator_label_not_scoring_gt":item["generator_label_not_scoring_gt"]})
        query=f'Is this: "{row["source_text"]}" right? Answer with Yes or No.'
        output.append({"attack":"RAG-MIA","session_id":f"finqa-ragmia::{target_id}","query_id":f"finqa-ragmia::{target_id}::q01",
            "query_index":1,"target_id":target_id,"membership":row["membership"],"domain":"finance/FinQA","query":query,
            "query_sha256":sha_text(query),"prompt_sha256":"deterministic-paper-prompt-2","output_sha256":sha_text(query),"protocol":"paper Prompt #2"})
    expected=expected_targets*21
    if failures or len(output)!=expected:
        write_jsonl(EXP/"audits"/f"FINQA_{args.scope.upper()}_QUERY_FAILURES.jsonl",failures)
        raise RuntimeError(f"FinQA query generation fail-close failures={len(failures)} queries={len(output)}/{expected}")
    path=EXP/"inputs"/f"FINQA_{args.scope.upper()}_ATTACK_QUERIES.jsonl";write_jsonl(path,output)
    manifest={"scope":args.scope,"targets":expected_targets,"queries":len(output),"member_queries":sum(r["membership"]=="member" for r in output),
        "nonmember_queries":sum(r["membership"]=="nonmember" for r in output),"path":str(path),"sha256":sha_file(path),
        "condition_independent_once_only":True,"completed_utc":now()}
    freeze_json(EXP/"configs"/f"FINQA_{args.scope.upper()}_QUERY_MANIFEST.json",manifest)
    checkpoint("PHASE_D_QUERY_BUNDLE_FROZEN",scope=args.scope,targets=expected_targets,queries=len(output),sha256=manifest["sha256"])


if __name__=="__main__":main()
