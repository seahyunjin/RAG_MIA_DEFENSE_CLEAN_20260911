#!/usr/bin/env python3
"""Generate the frozen IA API1 A0/A_HIDE branches once."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from common import EXP, ROOT


IA = EXP / "IA_STD_Q15_API1"
POST = IA / "post_ready"
LEGACY = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1" / "post_ready" / "run_e2e_generation.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_ia_e2e_generation", LEGACY)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    module.EXP = IA; module.POST = POST
    module.PRECOMMIT = POST / "configs" / "IA_API1_E2E_PRECOMMIT.json"
    module.DB = POST / "runtime" / "ia_api1_generation.sqlite3"
    module.needs_hidden = lambda row, pre: (
        float(row["M"]) > float(pre["thresholds"]["MIRABEL"]["threshold"])
        or float(row["R_LC"]) > float(pre["thresholds"]["Final LC"]["threshold"])
    )
    module.main()


if __name__ == "__main__":
    main()
