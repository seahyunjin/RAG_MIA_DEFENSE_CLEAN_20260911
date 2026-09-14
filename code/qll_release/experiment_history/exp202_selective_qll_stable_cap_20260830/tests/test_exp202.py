import importlib.util
from pathlib import Path

import numpy as np


PATH=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter/exp202_selective_qll_stable_cap_20260830/code/run_exp202.py")
spec=importlib.util.spec_from_file_location("exp202",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_waterfill_no_trigger_preserves_budget():
    caps=module.waterfill([1000,1000,1000,1000],None)
    assert sum(caps)==2048
    assert max(caps)-min(caps)<=1


def test_waterfill_selected_source_is_capped():
    caps=module.waterfill([1000,1000,1000,1000],2)
    assert sum(caps)==2048
    assert caps[2]==64


def test_waterfill_short_docs_never_exceeds_length():
    lengths=[20,100,200,400]
    caps=module.waterfill(lengths,1)
    assert np.all(np.asarray(caps)<=np.asarray(lengths))
    assert caps[1]<=64


def test_waterfill_supports_three_source_retrieval():
    caps=module.waterfill([1000,1000,1000],1)
    assert sum(caps)==2048
    assert caps[1]==64
