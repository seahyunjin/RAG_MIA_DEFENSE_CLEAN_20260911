#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import sys


PATH = Path(__file__).resolve().parents[1] / "code" / "run_phase_a1.py"
SPEC = importlib.util.spec_from_file_location("phase_a1", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_safe_mean_empty_is_missing_not_zero():
    assert MODULE.safe_mean([]) is None


def test_safe_mean_values():
    assert MODULE.safe_mean([True, False, True, False]) == 0.5


def test_scope_constants_are_frozen():
    assert MODULE.VERSIONS == ("V0", "V10", "V25", "V50")
    assert set(MODULE.ATTACKS) == {"MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2"}
