import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "code/run_exp204.py"
spec = importlib.util.spec_from_file_location("exp204", SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def test_strict_threshold():
    assert not (exp.THRESHOLD > exp.THRESHOLD)
    assert exp.THRESHOLD + 1e-12 > exp.THRESHOLD


def test_waterfill_no_hide():
    caps = exp.waterfill([1000, 1000, 1000, 1000], None)
    assert caps == [512, 512, 512, 512]
    assert sum(caps) == 2048


def test_waterfill_hide_one():
    caps = exp.waterfill([1000, 1000, 1000, 1000], 1)
    assert caps[1] == 0
    assert sum(caps) == 2048
    assert caps == [683, 0, 683, 682]


def test_policy_names_are_explicit():
    assert "PER_QUERY" in exp.LOCAL
    assert "SESSION_STICKY" in exp.STICKY
