#!/usr/bin/env python3
"""Run the frozen IA-Std-Q15-API1 protocol on FinQA while sharing the USD 12 campaign cap."""
from __future__ import annotations
import argparse,csv,json,math,sqlite3,sys
from pathlib import Path
from common import EXP,PARENT,ROOT,atomic_json,checkpoint,freeze_json,now,sha_file,verify_hashed_json

PARENT_CODE=PARENT/"code"
sys.path.insert(0,str(PARENT_CODE))
import run_phase_c_api as api

LOCAL_IA=EXP/"IA_STD_Q15_API1"
PHASE_A_DB=PARENT/"IA_STD_Q15_API1"/"runtime"/"ia_api1.sqlite3"
TOTAL_CAP=12.0

def phase_a_spend():
    con=sqlite3.connect(PHASE_A_DB);value=float(con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM call").fetchone()[0]);con.close();return value

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--scope",choices=("screen","all"),required=True);args=ap.parse_args()
    cross=verify_hashed_json(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json")
    targets=list(csv.DictReader((EXP/"inputs"/"FINQA_TARGETS_1000_1000.csv").open(encoding="utf-8")))
    if args.scope=="screen":targets=[r for r in targets if r["screen"].casefold()=="true"]
    expected=200 if args.scope=="screen" else 2000
    if len(targets)!=expected:raise RuntimeError(f"FinQA IA target drift {len(targets)}/{expected}")
    parent_protocol=PARENT_CODE/"ia_api_protocol.py";local_protocol=EXP/"code"/"ia_api_protocol.py"
    if sha_file(parent_protocol)!=sha_file(local_protocol):raise RuntimeError("IA API protocol copy drift")
    pre_path=EXP/"configs"/f"FINQA_{args.scope.upper()}_IA_API1_PRECOMMIT.json"
    target_ids=[r["target_id"] for r in targets]
    if not pre_path.is_file():
        pre={"campaign":EXP.name,"scope":args.scope,"created_utc":now(),"protocol":"IA-Std-Q15-API1",
             "question_model":api.QUESTION_MODEL,"gt_model":api.GT_MODEL,"q":15,
             "protocol_sha256":sha_file(parent_protocol),"parent_cross_domain_precommit_sha256":sha_file(EXP/"configs"/"CROSS_DOMAIN_PRECOMMIT.json"),
             "target_ids":target_ids,"target_count":expected,"member":sum(r["membership"]=="member" for r in targets),"nonmember":sum(r["membership"]=="nonmember" for r in targets),
             "validity_gate":{"coverage":">=0.95","member_nonmember_gap":"<=0.02","exact_q15_gt15":True,"duplicate_zero":True,"inferred_fields_zero":True},
             "cost":{"shared_campaign_cap_usd":TOTAL_CAP,"includes_parent_phase_a":True,"hard_stop_before_exceeding":True},
             "forbidden":["attack-result tuning","prompt change","schema change","retry change","invalid imputation"],
             "performance_metrics_computed_during_generation":False}
        freeze_json(pre_path,pre)
    else:
        pre=verify_hashed_json(pre_path)
        if pre["target_ids"]!=target_ids:raise RuntimeError("FinQA IA precommitted target IDs drift")
    LOCAL_IA.mkdir(parents=True,exist_ok=True)
    for d in ("runtime","logs","outputs","inputs","audits"): (LOCAL_IA/d).mkdir(parents=True,exist_ok=True)
    target_csv=EXP/"runtime"/f"FINQA_{args.scope.upper()}_IA_API_TARGETS.csv"
    from common import write_csv
    if not target_csv.is_file():write_csv(target_csv,targets)
    api.IA=LOCAL_IA;api.PRECOMMIT=pre_path;api.TARGETS=target_csv;api.DB=LOCAL_IA/"runtime"/f"finqa_{args.scope}_ia_api1.sqlite3";api.LOG=LOCAL_IA/"logs"/f"FINQA_{args.scope}_IA_API1.log";api.STATUS=LOCAL_IA/"STATUS.md";api.HEARTBEAT=LOCAL_IA/"HEARTBEAT.json"
    original_spend=api.estimated_spend
    parent_spend=phase_a_spend()
    api.estimated_spend=lambda con: parent_spend+original_spend(con)
    key=api.KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:raise RuntimeError("OpenAI key unavailable")
    con=api.init_db()
    for target_id,stage,attempt in con.execute("SELECT target_id,stage,attempt FROM call WHERE status='INFLIGHT'").fetchall():
        con.execute("UPDATE call SET status='ORPHANED_INFLIGHT',completed_utc=?,error_type='ORPHANED_INFLIGHT',error_message='ambiguous interrupted request; not regenerated' WHERE target_id=? AND stage=? AND attempt=?",(now(),target_id,stage,attempt))
    con.commit();checkpoint("FINQA_IA_API_STARTED",scope=args.scope,targets=expected,parent_spend_usd=round(parent_spend,6),shared_cap_usd=TOTAL_CAP)
    api.execute_stage(con,targets,"questions",key,TOTAL_CAP,f"FINQA_{args.scope.upper()}")
    api.execute_stage(con,targets,"gt",key,TOTAL_CAP,f"FINQA_{args.scope.upper()}")
    audits,summary=api.session_audit(con,targets)
    checks={"valid_at_least_95pct":summary["valid"]>=math.ceil(.95*expected),"validity_gap_at_most_2pp":summary["validity_gap"]<=.02+1e-12,"exact_q15_gt15":summary["exact_q15_gt15"],"duplicate_zero":summary["duplicate_zero"],"inferred_fields_zero":summary["inferred_fields"]==0}
    atomic_json(LOCAL_IA/"audits"/f"FINQA_{args.scope.upper()}_SESSION_AUDIT.json",audits)
    local_spend=original_spend(con);shared=parent_spend+local_spend
    budget_stops=con.execute("SELECT COUNT(*) FROM call WHERE status='BUDGET_STOP'").fetchone()[0]
    if budget_stops:
        result={"verdict":"FINQA_IA_API_COST_CAP_REACHED","scope":args.scope,**summary,"checks":checks,"parent_spend_usd":parent_spend,"local_spend_usd":local_spend,"shared_spend_usd":shared,"cap_usd":TOTAL_CAP,"budget_stop_rows":budget_stops}
    elif all(checks.values()):
        result=api.export_full(con,targets,{**summary,"checks":checks});result["verdict"]="FINQA_IA_STD_Q15_API1_READY";result.update({"scope":args.scope,"parent_spend_usd":parent_spend,"local_spend_usd":local_spend,"shared_spend_usd":shared,"cap_usd":TOTAL_CAP})
    else:result={"verdict":"FINQA_IA_API_VALIDITY_FAILED","scope":args.scope,**summary,"checks":checks,"parent_spend_usd":parent_spend,"local_spend_usd":local_spend,"shared_spend_usd":shared,"cap_usd":TOTAL_CAP}
    atomic_json(LOCAL_IA/f"FINQA_{args.scope.upper()}_RESULT.json",result);con.execute("PRAGMA wal_checkpoint(TRUNCATE)");con.close();checkpoint(result["verdict"],scope=args.scope,valid=f"{summary['valid']}/{expected}",shared_spend_usd=round(shared,6));print(json.dumps(result,indent=2))

if __name__=="__main__":main()
