#!/usr/bin/env python3
"""Run the frozen IA API1 detection using the already-audited computation module."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from common import EXP, ROOT, atomic_json


IA = EXP / "IA_STD_Q15_API1"
POST = IA / "post_ready"
LEGACY_PATH = EXP / "code" / "ia_detection_computation_id_repair.py"


def load_module():
    spec = importlib.util.spec_from_file_location("frozen_ia_detection_computation", LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.EXP = IA
    module.POST = POST
    module.PRECOMMIT = POST / "configs" / "IA_API1_DETECTION_ID_REPAIR_PRECOMMIT.json"
    return module


def relabel(value):
    if isinstance(value, str):
        return value.replace("IA_STEALTH_CONFIRMATION_V1", EXP.name).replace("IA-Std-Q15-ST1", "IA-Std-Q15-API1").replace("IA-ST1", "IA-API1")
    if isinstance(value, list):
        return [relabel(item) for item in value]
    if isinstance(value, dict):
        return {key: relabel(item) for key, item in value.items()}
    return value


def main() -> None:
    module = load_module()
    pre = module.verify_precommit()
    full = json.loads(Path(pre["full_result"]["path"]).read_text(encoding="utf-8"))
    if full.get("verdict") != "IA_STD_Q15_API1_READY":
        raise RuntimeError("IA API1 artifact no longer READY")
    query_rows = module.normalize_queries(module.read_jsonl(Path(pre["queries"]["path"])))
    rows, runtime = module.score_queries(query_rows)
    result = relabel(module.evaluate(rows, runtime, pre))
    primary = result["primary"]
    result["checks"] = {
        "final_lc_tpr_strictly_greater": primary["Final LC_tpr"] > primary["MIRABEL_tpr"],
        "cluster_bootstrap_ci_lower_above_zero": primary["delta_ci95_low"] > 0,
    }
    result["preferred_delta_at_least_3pp"] = primary["delta_tpr"] >= 0.03
    result["verdict"] = "IA_STEALTH_DETECTION_PASS" if all(result["checks"].values()) else "IA_STEALTH_DETECTION_NOT_SUPPORTED"
    result["next"] = "IA_MATCHED_BUDGET_E2E" if result["verdict"] == "IA_STEALTH_DETECTION_PASS" else "STOP"
    atomic_json(POST / "IA_API1_DETECTION_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
