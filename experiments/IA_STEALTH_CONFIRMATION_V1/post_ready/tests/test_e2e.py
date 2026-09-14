#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import sys


def load(name: str):
    path = Path(__file__).resolve().parents[1] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = load("run_e2e_generation.py")
SCORE = load("run_e2e_scoring.py")


def test_waterfill_preserves_budget():
    assert GEN.waterfill([1000, 1000, 1000, 1000], 2048) == [512, 512, 512, 512]
    caps = GEN.waterfill([10, 1000, 1000], 100)
    assert caps == [10, 45, 45]


def test_answer_parser_is_exact_and_fail_closed():
    assert SCORE.normalize_answer("Yes.") == "Yes"
    assert SCORE.normalize_answer("No") == "No"
    assert SCORE.normalize_answer("I don't know.") == "UNK"
    assert SCORE.normalize_answer("Yes or no") == "UNK"
    assert SCORE.normalize_answer("maybe") == "UNK"


def test_condition_branch_uses_strict_operator():
    pre = {"thresholds": {"MIRABEL": {"threshold": 1.0}, "Final LC": {"threshold": 2.0}}}
    row = {"M": 1.0, "R_LC": 2.0}
    assert SCORE.condition_branch(row, "MIRABEL_MATCHED_2_5", pre) == "A0"
    assert SCORE.condition_branch(row, "FINAL_LC_MATCHED_2_5", pre) == "A0"
    row = {"M": 1.01, "R_LC": 2.01}
    assert SCORE.condition_branch(row, "MIRABEL_MATCHED_2_5", pre) == "A_HIDE"
    assert SCORE.condition_branch(row, "FINAL_LC_MATCHED_2_5", pre) == "A_HIDE"
