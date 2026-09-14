#!/usr/bin/env python3
"""Reparse all 2,000 frozen IA sessions under parser v2 and apply the frozen recovery gate."""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from collections import Counter

from common import EXP, atomic_json, checkpoint, freeze_json, now, sha_file, sha_text, verify_hashed_json, write_csv, write_jsonl
from ia_parser_v2 import audit_category, parse_session


def main():
    spec=verify_hashed_json(EXP/"configs"/"IA_STD_Q15_PARSER_V2_SPEC.json")
    if sha_file(EXP/"code"/"ia_parser_v2.py")!=spec["parser"]["sha256"] or sha_file(EXP/"tests"/"test_ia_parser_v2.py")!=spec["unit_tests"]["sha256"]:
        raise RuntimeError("parser v2 code/test drift")
    con=sqlite3.connect(EXP/"runtime"/"standardized_query_generation.sqlite3")
    db_rows=list(con.execute("SELECT target_id,membership,raw_output,valid,error,prompt_sha256,output_sha256 FROM generation WHERE attack='IA-Std-Q15' ORDER BY target_id"))
    if len(db_rows)!=2000 or Counter(row[1] for row in db_rows)!={"member":1000,"nonmember":1000}:raise RuntimeError("IA frozen raw cohort drift")
    targets={row["document_id"]:row for row in csv.DictReader((__import__("common").LC/"inputs"/"LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))}
    audit=[];queries=[];provenance=[]
    for target_id,membership,raw_json,v1_valid,v1_error,prompt_sha,output_sha in db_rows:
        attempts=json.loads(raw_json);outcome,selected_attempt=parse_session(attempts);category="V1_VALID"
        if not v1_valid:category=audit_category(outcome)
        audit.append({"target_id":target_id,"membership":membership,"v1_valid":bool(v1_valid),"v1_error":v1_error or "",
            "v2_valid":outcome.valid,"audit_category":category,"v2_reason":outcome.reason,"record_count":outcome.record_count,
            "selected_frozen_attempt":selected_attempt or "","questions":len(outcome.items),"binary_judgments":len(outcome.items),
            "inferred_fields":outcome.inferred_fields,"raw_attempt_count":len(attempts)})
        provenance.append({"target_id":target_id,"membership":membership,"raw_attempts_sha256":[sha_text(value) for value in attempts],
            "selected_frozen_attempt":selected_attempt,"v2_valid":outcome.valid,"reason":outcome.reason,"inferred_fields":outcome.inferred_fields})
        if outcome.valid:
            target=targets[target_id]
            for index,(question,label) in enumerate(outcome.items,1):
                queries.append({"attack":"IA-Std-Q15","session_id":f"ia_std::{target_id}","query_id":f"ia_std::{target_id}::q{index:02d}",
                    "query_index":index,"target_id":target_id,"membership":membership,"domain":target["domain"],"query":question,
                    "query_sha256":sha_text(question),"query_generator_answer_not_scoring_gt":label,"selected_frozen_attempt":selected_attempt,
                    "protocol":"IA-Std-Q15 parser-v2 syntax-only recovery","claim_boundary":"STANDARDIZED_Q15_NOT_PAPER_EXACT_Q30"})
    write_csv(EXP/"audits"/"IA_STD_Q15_PARSER_V2_SESSION_AUDIT.csv",audit);write_jsonl(EXP/"runtime"/"IA_STD_Q15_PARSER_V2_PROVENANCE.jsonl",provenance)
    valid=[row for row in audit if row["v2_valid"]];member=sum(row["membership"]=="member" for row in valid);nonmember=sum(row["membership"]=="nonmember" for row in valid)
    valid_rate=len(valid)/2000;member_rate=member/1000;nonmember_rate=nonmember/1000;gap=abs(member_rate-nonmember_rate)
    inferred=sum(int(row["inferred_fields"]) for row in audit);categories=Counter(row["audit_category"] for row in audit if not row["v1_valid"])
    checks={"valid_sessions_at_least_95pct":valid_rate>=.95,"member_nonmember_validity_gap_at_most_2pp":gap<=.02+1e-12,
        "exact_q15":all(row["questions"]==15 for row in valid),"exact_15_binary_judgments":all(row["binary_judgments"]==15 for row in valid),"inferred_fields_zero":inferred==0}
    passed=all(checks.values());verdict="IA_STD_Q15_PARSER_READY" if passed else "IA_STD_Q15_FORMAT_RECOVERY_FAILED"
    query_path=EXP/"inputs"/"IA_STD_Q15_QUERIES_V2.jsonl";write_jsonl(query_path,queries)
    result={"campaign":EXP.name,"verdict":verdict,"completed_utc":now(),"sessions":2000,"valid_sessions":len(valid),"valid_rate":valid_rate,
        "member_valid":member,"nonmember_valid":nonmember,"member_valid_rate":member_rate,"nonmember_valid_rate":nonmember_rate,"validity_gap":gap,
        "v1_invalid_audit_categories":dict(categories),"inferred_fields":inferred,"valid_queries":len(queries),"checks":checks,
        "query_artifact":{"path":str(query_path),"sha256":sha_file(query_path)},"parser_spec_sha256":sha_file(EXP/"configs"/"IA_STD_Q15_PARSER_V2_SPEC.json"),
        "next_stage":"CORE6_DETECTION" if passed else "CORE5_DETECTION_E2E_GOLD_NO_IA_REGENERATION"}
    atomic_json(EXP/"IA_PARSER_V2_RESULT.json",result)
    freeze_json(EXP/"configs"/"IA_STD_Q15_PARSER_V2_OUTPUT_MANIFEST.json",{"result_sha256":sha_file(EXP/"IA_PARSER_V2_RESULT.json"),
        "query_sha256":sha_file(query_path),"provenance_sha256":sha_file(EXP/"runtime"/"IA_STD_Q15_PARSER_V2_PROVENANCE.jsonl"),"audit_sha256":sha_file(EXP/"audits"/"IA_STD_Q15_PARSER_V2_SESSION_AUDIT.csv")})
    checkpoint(verdict,valid_sessions=len(valid),valid_rate=round(valid_rate,6),member_valid=member,nonmember_valid=nonmember,validity_gap=round(gap,6),categories=dict(categories),next_stage=result["next_stage"])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
