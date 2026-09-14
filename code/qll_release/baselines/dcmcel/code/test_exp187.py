#!/usr/bin/env python3
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_exp187", root / "code/run_exp187.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

assert module.BUDGETS == tuple(range(10, 271, 20))
assert len(module.BUDGETS) == 14
assert module.TOTAL_BUDGET == 128.0 and module.DECISION_BUDGET == 16.0
assert module.PRIMARY == "DC_MCEL_SESSION_T128_D16"
assert module.RESET == "DC_MCEL_RESET_T128_D16"
assert module.normalized_claim_hash("  A  B ") == module.normalized_claim_hash("a b")
print("6 tests passed")
