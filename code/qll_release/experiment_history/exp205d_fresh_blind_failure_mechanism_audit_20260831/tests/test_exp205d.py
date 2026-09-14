import importlib.util
from pathlib import Path
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code/run_exp205d.py"
spec = importlib.util.spec_from_file_location("exp205d", CODE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_threshold():
    assert module.THRESHOLD == 0.5300846414247485


def test_refusal():
    assert module.refusal("I don't know.")
    assert not module.refusal("Yes")


def test_extract_dcmi_quote():
    assert module.extract_dcmi_quote('Does this:\n"abc"\nappear in the context? Answer Yes') == "abc"


def test_bool_series():
    assert module.bool_series(pd.Series(["True", "False"])).tolist() == [True, False]


def test_no_generation_api_in_source():
    source = CODE.read_text(encoding="utf-8")
    assert "model.generate(" not in source
    assert "LocalQwen(" not in source
