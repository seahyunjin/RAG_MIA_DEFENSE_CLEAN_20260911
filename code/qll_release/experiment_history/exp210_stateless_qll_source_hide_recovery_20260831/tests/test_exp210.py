import importlib.util
from pathlib import Path
import pandas as pd

CODE=Path(__file__).resolve().parents[1]/"code/run_exp210.py"
spec=importlib.util.spec_from_file_location("exp210",CODE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_threshold(): assert m.THRESHOLD==0.5300846414247485
def test_zero_parameters(): assert m.STATELESS=="STATELESS_QLL_SOURCE_HIDE"
def test_refusal(): assert m.refusal("I don't know.") and not m.refusal("Yes")
def test_token_f1(): assert m.token_f1("a b","a b")==1 and m.token_f1("a","b")==0
def test_bool_parser(): assert m.bools(pd.Series(["True","False"])).tolist()==[True,False]
def test_no_attack_rule_in_inference():
    source=CODE.read_text();body=source.split("def stateless_rows",1)[1].split("def stage_a",1)[0]
    assert "attack_family" not in body and "target_rank" not in body and "member" not in body
