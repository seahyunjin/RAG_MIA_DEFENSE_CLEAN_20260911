#!/usr/bin/env python3
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("exp194",HERE/"run_exp194.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

assert abs(mod.rouge1_f1("a b c","a b")-.8)<1e-12
assert mod.rouge1_f1("","")==0.0
assert mod.REPLICATES==5000
assert mod.INPUTS["exp193_final"].exists()
assert all(map(lambda x: x == x, mod.bootstrap_difference([1,2,3],[0,1,2],replicates=99)))
print("EXP194_UNIT_TESTS_PASS")
