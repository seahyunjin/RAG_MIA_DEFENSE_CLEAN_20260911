#!/usr/bin/env python3
"""Score frozen IA API1 branches and emit an explicitly labelled result."""
from __future__ import annotations

import importlib.util
import json

from common import EXP, ROOT, atomic_json


IA = EXP / "IA_STD_Q15_API1"
POST = IA / "post_ready"
LEGACY = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1" / "post_ready" / "run_e2e_scoring.py"


def relabel(value):
    if isinstance(value, str):
        return value.replace("IA_STEALTH_CONFIRMATION_V1", EXP.name).replace("IA-Std-Q15-ST1", "IA-Std-Q15-API1").replace("IA-ST1", "IA-API1")
    if isinstance(value, list): return [relabel(item) for item in value]
    if isinstance(value, dict): return {key: relabel(item) for key, item in value.items()}
    return value


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_ia_e2e_scoring", LEGACY)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    module.EXP = IA; module.POST = POST
    module.PRECOMMIT = POST / "configs" / "IA_API1_E2E_PRECOMMIT.json"
    module.main()
    source = POST / "IA_ST1_E2E_RESULT.json"
    result = relabel(json.loads(source.read_text(encoding="utf-8")))
    primary = result["primary"]
    point_better = primary["final_lc_matched_2_5"] < primary["mirabel_matched_2_5"]
    point_not_worse = primary["final_lc_matched_2_5"] <= primary["mirabel_matched_2_5"]
    noninferior = primary["bootstrap_delta_vs_mirabel"]["ci95"][1] <= 0.03
    result["checks"] = {
        "clear_privacy_reduction_vs_no_defense_3pp": primary["final_lc_matched_2_5"] <= primary["no_defense"] - 0.03,
        "better_or_noninferior_to_mirabel_with_nonworse_point": point_better or (point_not_worse and noninferior),
    }
    result["verdict"] = "IA_STEALTH_CONFIRMATION_PASS" if all(result["checks"].values()) else "IA_STEALTH_E2E_FAILED"
    result["next"] = "UNTOUCHED_DOMAIN_PRIVACY" if result["verdict"] == "IA_STEALTH_CONFIRMATION_PASS" else "STOP"
    atomic_json(POST / "IA_API1_E2E_RESULT.json", result)
    atomic_json(IA / "POST_READY_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
