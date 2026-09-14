#!/usr/bin/env python3
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp195", HERE / "run_exp195.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

assert mod.ALPHAS == (0.005, 0.01, 0.02, 0.05)
assert mod.PRIMARY == "QLL_GUARD_1PCT"
assert mod.ABSTENTION == "I can't provide additional source-specific detail for this request."
assert mod.quantile_budget([0, 1, 2, 3, 4], .20) == 3.0
assert mod.quantile_budget([0, 1, 2, 3, 4], .01) == 4.0
assert len(mod.INPUTS) >= 10 and all(path.exists() for path in mod.INPUTS.values())
source = (HERE / "run_exp195.py").read_text()
for forbidden in ("attack-specific threshold", "rank-specific threshold", "alternative answer regeneration"):
    assert forbidden not in source.lower()
print("EXP195_UNIT_TESTS_PASS")
