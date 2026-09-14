#!/usr/bin/env python3
"""Resume frozen IA API1 with a separately authorized cost-cap amendment."""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import run_phase_c_api as frozen
from common import atomic_json, checkpoint, sha_file, verify_hashed_json

AMENDMENT = frozen.IA / "configs" / "IA_API1_COST_CAP_AMENDMENT.json"
ORIGINAL_VERIFY = frozen.verify_hashed_json


def load_amendment() -> dict:
    amendment = verify_hashed_json(AMENDMENT)
    if amendment.get("change_scope") != "API_COST_CAP_ONLY":
        raise RuntimeError("invalid cost amendment scope")
    if float(amendment.get("authorized_cost_cap_usd", -1)) != 12.0:
        raise RuntimeError("authorized cap is not exactly USD 12.00")
    if amendment.get("original_precommit_sha256") != sha_file(frozen.PRECOMMIT):
        raise RuntimeError("original precommit hash mismatch")
    preflight = frozen.IA / "PREFLIGHT_RESULT.json"
    if amendment.get("passed_preflight_sha256") != sha_file(preflight):
        raise RuntimeError("passed preflight hash mismatch")
    result = json.loads(preflight.read_text(encoding="utf-8"))
    if result.get("verdict") != "IA_API1_PROTOCOL_READY":
        raise RuntimeError("preflight was not READY")
    return amendment


def amended_verify(path: Path) -> dict:
    value = ORIGINAL_VERIFY(path)
    if Path(path) == frozen.PRECOMMIT:
        value = copy.deepcopy(value)
        value["spending_safety"]["campaign_estimated_cost_ceiling_usd"] = 12.0
    return value


def main() -> None:
    amendment = load_amendment()
    frozen.verify_hashed_json = amended_verify
    checkpoint("IA_API_FULL_RESUME_AUTHORIZED", cost_cap_usd=12.0, amendment_sha256=sha_file(AMENDMENT), preflight_reused=True)
    frozen.log(f"cost-only amendment verified; resuming frozen full cohort with USD {amendment['authorized_cost_cap_usd']:.2f} hard cap")
    frozen.main()
    con = sqlite3.connect(frozen.DB)
    budget_stops = con.execute("SELECT COUNT(*) FROM call WHERE status='BUDGET_STOP'").fetchone()[0]
    spend = float(con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM call").fetchone()[0])
    con.close()
    if budget_stops:
        result = json.loads((frozen.IA / "FINAL_RESULT.json").read_text(encoding="utf-8"))
        result.update({"verdict":"IA_API_COST_CAP_REACHED","estimated_campaign_spend_usd":spend,"authorized_cost_cap_usd":12.0,"budget_stop_rows":budget_stops})
        atomic_json(frozen.IA / "FINAL_RESULT.json", result)
        checkpoint("IA_API_COST_CAP_REACHED", actual_cumulative_api_cost_usd=spend, cost_cap_usd=12.0)


if __name__ == "__main__":
    main()
