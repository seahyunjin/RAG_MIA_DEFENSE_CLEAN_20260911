#!/usr/bin/env python3
"""Combine condition-independent local and API attack queries after validity passes."""
from __future__ import annotations
import argparse,json
from common import EXP,checkpoint,freeze_json,read_jsonl,sha_file,write_jsonl

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--scope",choices=("screen","all"),required=True);args=ap.parse_args();upper=args.scope.upper();expected=200 if args.scope=="screen" else 2000
    local_manifest=json.loads((EXP/"configs"/f"FINQA_{upper}_LOCAL_QUERY_MANIFEST.json").read_text(encoding="utf-8"));local_path=EXP/"inputs"/f"FINQA_{upper}_LOCAL_ATTACK_QUERIES.jsonl"
    if sha_file(local_path)!=local_manifest["sha256"]:raise RuntimeError("local query bundle drift")
    ia_result_path=EXP/"IA_STD_Q15_API1"/f"FINQA_{upper}_RESULT.json";ia_result=json.loads(ia_result_path.read_text(encoding="utf-8"))
    if ia_result.get("verdict")!="FINQA_IA_STD_Q15_API1_READY":raise RuntimeError(f"IA API input not READY: {ia_result.get('verdict')}")
    ia_path=EXP/"IA_STD_Q15_API1"/"inputs"/"IA_API1_FULL_QUERIES.jsonl";api_rows=read_jsonl(ia_path);rows=read_jsonl(local_path)
    for row in api_rows:
        rows.append({"attack":"IA-Std-Q15-API1","session_id":row["session_id"],"query_id":row["_id"],"query_index":int(row["turn"]),"target_id":row["target_doc_id"],"canonical_target_id":row["canonical_target_id"],"membership":row["_membership"],"domain":"finance/FinQA","query":row["text"],"protocol":"IA-Std-Q15-API1 frozen structured output"})
    if len(rows)!=expected*23:raise RuntimeError(f"combined query count drift {len(rows)}/{expected*23}")
    attacks={x["attack"] for x in rows};required={"MEntA","RAG-MIA","DCMI-Std-Q2","IA-Std-Q15-API1"}
    if attacks!=required:raise RuntimeError(f"attack set drift {attacks}")
    path=EXP/"inputs"/f"FINQA_{upper}_ATTACK_QUERIES.jsonl";write_jsonl(path,sorted(rows,key=lambda x:(x["attack"],x["session_id"],int(x["query_index"]))))
    digest=freeze_json(EXP/"configs"/f"FINQA_{upper}_QUERY_MANIFEST.json",{"scope":args.scope,"targets":expected,"queries":len(rows),"attacks":sorted(attacks),"path":str(path),"sha256":sha_file(path),"local_manifest_sha256":sha_file(EXP/"configs"/f"FINQA_{upper}_LOCAL_QUERY_MANIFEST.json"),"ia_result_sha256":sha_file(ia_result_path),"condition_independent_once_only":True})
    checkpoint("FINQA_QUERY_BUNDLE_FROZEN",scope=args.scope,targets=expected,queries=len(rows),manifest_sha256=digest)

if __name__=="__main__":main()
