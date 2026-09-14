#!/usr/bin/env python3
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"code/run_exp191.py"

tree=ast.parse(SOURCE.read_text())
text=SOURCE.read_text()
assert "REPLICATES = 20_000" in text
assert "MIN_ABS_RHO = 0.10" in text
assert "new_defense\": False" in text
assert "E-AUC" in text
assert "roc_auc_score" not in text
assert "LogisticRegression" not in text
assert "AutoModelForCausalLM" not in text
assert "generate(" not in text
assert "FINAL_PRINT.md" in text
assert "[Attack Data Used To Choose Future Thresholds?]" in text
assert set(["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10"]).issubset(set(text.split('"')))
assert set(["C1","C2","C3","C4","C5","C6","C7","C8"]).issubset(set(text.split('"')))
print("Exp191 unit tests: PASS")
