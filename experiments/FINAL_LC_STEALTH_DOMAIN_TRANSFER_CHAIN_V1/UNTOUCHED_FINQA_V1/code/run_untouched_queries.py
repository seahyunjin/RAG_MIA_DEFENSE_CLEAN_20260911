#!/usr/bin/env python3
"""Generate FinQA MEntA-Q5, DCMI-Std-Q2, and RAG-MIA-Q1 without observing performance."""
from __future__ import annotations
import argparse, csv, json, math, re, sqlite3, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import EXP,QWEN,ROOT,SEED,checkpoint,freeze_json,now,sha_file,sha_text,verify_hashed_json,write_jsonl

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
DCMI_PROMPT="""Rewrite the text below by changing exactly {replace_count} meaning-bearing words or short phrases to clear semantic opposites. Prefer adjectives and adverbs; if there are not enough, use polarity-bearing verbs. You MUST change the text. Keep every other word unchanged. Return only the complete modified text.

Text:
{target_text}"""
DCMI_RETRY="""The previous output did not create a valid perturbation. Rewrite the original text below and MUST change exactly {replace_count} meaning-bearing words or short phrases to semantic opposites. Prefer adjectives and adverbs, then polarity-bearing verbs. Preserve all other content. Return only the complete modified text.

Original text:
{target_text}

Invalid previous output:
{previous_output}"""

def parse_menta(text):
    lines=[x.strip().strip("` ") for x in text.splitlines() if x.strip()]
    if len(lines)!=6:return None
    m=re.fullmatch(r"SUMMARY\s*:\s*(\S(?:.*\S)?)",lines[0],re.I)
    if not m:return None
    qs=[]
    for i,line in enumerate(lines[1:],1):
        q=re.fullmatch(rf"QUERY_{i}\s*:\s*(\S(?:.*\S)?)",line,re.I)
        if not q or not q.group(1).strip().endswith("?"):return None
        qs.append(q.group(1).strip())
    if len({" ".join(q.casefold().split()) for q in qs})!=5:return None
    return {"summary":m.group(1).strip(),"questions":qs}

def database():
    p=EXP/"runtime"/"local_query_generation.sqlite3";con=sqlite3.connect(p,timeout=120)
    con.execute("PRAGMA journal_mode=WAL");con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS generation(attack TEXT,target_id TEXT,membership TEXT,raw_outputs_json TEXT,parsed_json TEXT,valid INTEGER,error TEXT,prompt_sha256 TEXT,output_sha256 TEXT,wall_seconds REAL,completed_utc TEXT,PRIMARY KEY(attack,target_id))""");con.commit();return con

def render(tok,prompt):
    value=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    return value,sha_text(json.dumps(tok(value,add_special_tokens=False).input_ids,separators=(",",":")))

def generate(model,tok,prompts,max_new):
    enc=tok(prompts,add_special_tokens=False,padding=True,truncation=True,max_length=6144,return_tensors="pt").to("cuda");started=time.perf_counter()
    with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=max_new,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tok.eos_token_id)
    return [x.strip() for x in tok.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)],time.perf_counter()-started

def run_attack(con,model,tok,targets,attack):
    complete={x[0] for x in con.execute("SELECT target_id FROM generation WHERE attack=?",(attack,))};tasks=[]
    for row in targets:
        if row["target_id"] in complete:continue
        if attack=="MEntA":prompt=MENTA_PROMPT.format(target_text=row["source_text"]);max_new=1000
        else:
            count=max(1,math.floor(.06*len(row["source_text"].split())))
            prompt=DCMI_PROMPT.format(replace_count=count,target_text=row["source_text"])
            max_new=min(4096,len(tok(row["source_text"],add_special_tokens=False).input_ids)+256)
        rendered,prompt_sha=render(tok,prompt);tasks.append((row,rendered,prompt_sha,prompt,max_new))
    tasks.sort(key=lambda x:(x[4],x[0]["target_id"]));offset=0;started=time.monotonic()
    while offset<len(tasks):
        bs=8 if attack=="MEntA" else (1 if tasks[offset][4]>2750 else 2)
        batch=tasks[offset:offset+bs];max_new=max(x[4] for x in batch);outs,elapsed=generate(model,tok,[x[1] for x in batch],max_new)
        for (row,_,prompt_sha,prompt,_),first in zip(batch,outs):
            attempts=[first];raw=first;parsed=parse_menta(raw) if attack=="MEntA" else ({"perturbed_text":raw} if raw and raw!=row["source_text"] else None)
            if parsed is None:
                if attack=="MEntA":retry=MENTA_RETRY.format(target_text=row["source_text"],previous_output=raw);retry_max=1000
                else:
                    count=max(1,math.floor(.06*len(row["source_text"].split())))
                    retry=DCMI_RETRY.format(replace_count=count,target_text=row["source_text"],previous_output=raw);retry_max=max_new
                rr,prompt_sha=render(tok,retry);second,extra=generate(model,tok,[rr],retry_max);raw=second[0];attempts.append(raw);elapsed+=extra
                parsed=parse_menta(raw) if attack=="MEntA" else ({"perturbed_text":raw} if raw and raw!=row["source_text"] else None)
            con.execute("INSERT INTO generation VALUES (?,?,?,?,?,?,?,?,?,?,?)",(attack,row["target_id"],row["membership"],json.dumps(attempts,ensure_ascii=False),json.dumps(parsed or {},ensure_ascii=False,sort_keys=True),int(parsed is not None),None if parsed else "INVALID_AFTER_FIXED_RETRY",prompt_sha,sha_text(raw),elapsed/len(batch),now()))
        con.commit();offset+=len(batch);rate=offset/max(time.monotonic()-started,1e-9)
        checkpoint("FINQA_LOCAL_QUERY_PROGRESS",scope=len(targets),attack=attack,completed=len(targets)-len(tasks)+offset,total=len(targets),eta_seconds=round((len(tasks)-offset)/max(rate,1e-9)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--scope",choices=("screen","all"),required=True);args=ap.parse_args()
    verify_hashed_json(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json")
    targets=list(csv.DictReader((EXP/"inputs"/"FINQA_TARGETS_1000_1000.csv").open(encoding="utf-8")))
    if args.scope=="screen":targets=[r for r in targets if r["screen"].casefold()=="true"]
    expected_targets=200 if args.scope=="screen" else 2000
    if len(targets)!=expected_targets:raise RuntimeError(f"target drift {len(targets)}/{expected_targets}")
    np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    checkpoint("FINQA_LOCAL_QUERY_GENERATOR_LOADING",scope=args.scope,targets=len(targets),gpu_free_bytes=torch.cuda.mem_get_info()[0])
    tok=AutoTokenizer.from_pretrained(QWEN,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(QWEN,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval();con=database()
    for attack in ("MEntA","DCMI-Std-Q2"):run_attack(con,model,tok,targets,attack)
    del model;torch.cuda.empty_cache();stored={(a,b):(json.loads(c),bool(d),e,f,g) for a,b,c,d,e,f,g in con.execute("SELECT attack,target_id,parsed_json,valid,error,prompt_sha256,output_sha256 FROM generation")}
    output=[];fail=[]
    for row in sorted(targets,key=lambda x:x["target_id"]):
        for attack in ("MEntA","DCMI-Std-Q2"):
            parsed,valid,error,prompt_sha,output_sha=stored[(attack,row["target_id"])]
            if not valid:fail.append({"attack":attack,"target_id":row["target_id"],"error":error});continue
            if attack=="MEntA":
                for i,q in enumerate(parsed["questions"],1):
                    query=f"{parsed['summary']} {q}";output.append({"attack":"MEntA","session_id":f"finqa-menta::{row['target_id']}","query_id":f"finqa-menta::{row['target_id']}::q{i:02d}","query_index":i,"target_id":row["target_id"],"membership":row["membership"],"domain":"finance/FinQA","query":query,"query_sha256":sha_text(query),"prompt_sha256":prompt_sha,"output_sha256":output_sha,"protocol":"new-domain MEntA Q5 instantiation"})
            else:
                for i,(variant,sample) in enumerate((("original",row["source_text"]),("perturbed",parsed["perturbed_text"])),1):
                    query=f'Is this: "{sample}" right? Answer with Yes or No.';output.append({"attack":"DCMI-Std-Q2","session_id":f"finqa-dcmi::{row['target_id']}","query_id":f"finqa-dcmi::{row['target_id']}::{variant}","query_index":i,"variant":variant,"target_id":row["target_id"],"membership":row["membership"],"domain":"finance/FinQA","query":query,"query_sha256":sha_text(query),"prompt_sha256":prompt_sha,"output_sha256":output_sha,"protocol":"standardized DCMI differential Q2; not Original DCMI"})
        query=f'Is this: "{row["source_text"]}" right? Answer with Yes or No.';output.append({"attack":"RAG-MIA","session_id":f"finqa-ragmia::{row['target_id']}","query_id":f"finqa-ragmia::{row['target_id']}::q01","query_index":1,"target_id":row["target_id"],"membership":row["membership"],"domain":"finance/FinQA","query":query,"query_sha256":sha_text(query),"prompt_sha256":"deterministic-paper-prompt-2","output_sha256":sha_text(query),"protocol":"paper Prompt #2"})
    expected=expected_targets*8
    if fail or len(output)!=expected:
        write_jsonl(EXP/"audits"/f"FINQA_{args.scope.upper()}_LOCAL_QUERY_FAILURES.jsonl",fail);raise RuntimeError(f"query generation fail-close failures={len(fail)} queries={len(output)}/{expected}")
    path=EXP/"inputs"/f"FINQA_{args.scope.upper()}_LOCAL_ATTACK_QUERIES.jsonl";write_jsonl(path,output)
    digest=freeze_json(EXP/"configs"/f"FINQA_{args.scope.upper()}_LOCAL_QUERY_MANIFEST.json",{"scope":args.scope,"targets":expected_targets,"queries":len(output),"path":str(path),"sha256":sha_file(path),"performance_observed":False})
    checkpoint("FINQA_LOCAL_QUERY_BUNDLE_FROZEN",scope=args.scope,targets=expected_targets,queries=len(output),manifest_sha256=digest)

if __name__=="__main__":main()
