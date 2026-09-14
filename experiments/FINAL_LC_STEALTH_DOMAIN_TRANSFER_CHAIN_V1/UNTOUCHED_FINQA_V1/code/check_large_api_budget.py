#!/usr/bin/env python3
"""Fail closed before large FinQA IA expansion if the shared USD 12 cap cannot cover it."""
import json,sqlite3
from common import EXP,PARENT,atomic_json,checkpoint,now
CAP=12.0

def spend(path):
    con=sqlite3.connect(path);v=float(con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM call").fetchone()[0]);con.close();return v

def main():
    parent=spend(PARENT/"IA_STD_Q15_API1"/"runtime"/"ia_api1.sqlite3")
    local_db=EXP/"IA_STD_Q15_API1"/"runtime"/"finqa_screen_ia_api1.sqlite3";local=spend(local_db)
    result=json.loads((EXP/"IA_STD_Q15_API1"/"FINQA_SCREEN_RESULT.json").read_text(encoding="utf-8"));valid=max(1,int(result.get("sessions",200)))
    per_target=local/valid;remaining_targets=1800;projected=parent+local+per_target*remaining_targets
    value={"checked_utc":now(),"shared_cap_usd":CAP,"parent_spend_usd":parent,"screen_spend_usd":local,"observed_cost_per_screen_target_usd":per_target,"remaining_targets":remaining_targets,"projected_total_usd":projected,"large_authorized":projected<=CAP-.05}
    atomic_json(EXP/"audits"/"FINQA_LARGE_API_COST_PROJECTION.json",value)
    if not value["large_authorized"]:
        checkpoint("FINQA_LARGE_API_COST_CAP_REACHED_BEFORE_REQUESTS",projected_total_usd=round(projected,4),cap_usd=CAP,large_requests_sent=0);raise SystemExit(42)
    checkpoint("FINQA_LARGE_API_COST_FEASIBLE",projected_total_usd=round(projected,4),cap_usd=CAP)

if __name__=="__main__":main()
