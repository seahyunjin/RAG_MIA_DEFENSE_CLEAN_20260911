#!/usr/bin/env python3
from pathlib import Path
import ast

source=Path(__file__).with_name("run_exp192.py").read_text()
ast.parse(source)
assert "AutoModelForCausalLM" not in source
assert "generate(" not in source
assert "roc_auc_score" not in source
assert "effective_auc" not in source.lower()
assert "max(auc" not in source.lower()
assert "PER_SOURCE_C1_UNAVAILABLE" in source
assert "EXP192_INPUT_INSUFFICIENT" in source
assert "membership_not_used_in_s_star" in source
assert "target_rank_not_used_in_s_star" in source
assert "attack_family_not_used_in_s_star" in source
print("Exp192 unit tests: PASS")
