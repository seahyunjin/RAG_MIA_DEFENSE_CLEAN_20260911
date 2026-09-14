#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import sys


PATH = Path(__file__).resolve().parents[1] / "run_detection.py"
SPEC = importlib.util.spec_from_file_location("ia_st1_detection", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_threshold_strict_budget_and_ties():
    threshold, alarms = MODULE.threshold_at_budget([5, 4, 4, 3, 2], 0.4)
    assert threshold == 4
    assert alarms == 1


def test_query_normalization_requires_exact_q15():
    rows = [{"_id": f"q{i}", "text": "x", "target_doc_id": "d", "canonical_target_id": "c", "_membership": "member", "session_id": "s", "turn": i} for i in range(1, 16)]
    result = MODULE.normalize_queries(rows)
    assert len(result) == 15
    try:
        MODULE.normalize_queries(rows[:-1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Q14 should fail closed")


def test_primary_budget_is_present():
    assert 0.025 in MODULE.BUDGETS
