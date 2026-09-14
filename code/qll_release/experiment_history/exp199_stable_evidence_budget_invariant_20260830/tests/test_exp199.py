#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter/exp199_stable_evidence_budget_invariant_20260830")
spec = importlib.util.spec_from_file_location("run_exp199", ROOT / "code/run_exp199.py")
module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)


class Tokenizer:
    def __call__(self, value, add_special_tokens=False):
        return type("Encoded", (), {"input_ids": list(range(len(str(value).split())))})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{index}" for index in ids)


def test_contract():
    config = json.loads((ROOT / "configs/PRECOMMIT.json").read_text())
    assert config["canonical_source_view"]["query_independent"] is True
    assert config["canonical_source_view"]["rank_independent"] is True
    assert config["budget_invariant_release"]["caller_budget_used_by_policy"] is False
    assert config["budget_invariant_release"]["server_max_new_tokens"] == 64
    view, selected, used, coverage = module.facility_view(
        ["one two three", "four five", "six seven eight"], np.eye(3, dtype=np.float32), Tokenizer(), budget=4)
    assert used <= 4
    assert selected == sorted(selected)
    assert coverage >= 0
    assert module.token_recall("alpha beta", "alpha gamma") == 0.5
    assert module.exact_contains("Alpha Beta", "x alpha beta y")


if __name__ == "__main__":
    test_contract(); print("EXP199_UNIT_TEST_PASS")
