from pathlib import Path
import importlib.util
import json


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("exp156", ROOT / "code/run_exp156.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_is_frozen_global_cap64():
    protocol = json.loads((ROOT / "configs/PRECOMMIT.json").read_text())
    assert protocol["candidate"] == "GLOBAL_CAP64"
    assert protocol["policy"]["rank1_token_cap"] == 64


def test_runtime_has_no_learned_gate_or_calibration():
    runtime = json.loads((ROOT / "configs/PRECOMMIT.json").read_text())["runtime"]
    assert not runtime["detector"] and not runtime["lora"] and not runtime["classifier"]
    assert runtime["normal_calibration_rows"] == runtime["attack_calibration_rows"] == 0


def test_no_document_deletion_or_request_rejection():
    protocol = json.loads((ROOT / "configs/PRECOMMIT.json").read_text())
    assert protocol["policy"]["rank1_token_cap"] > 0
    assert not protocol["policy"]["full_document_deletion"]
    assert not protocol["runtime"]["request_rejection"]
