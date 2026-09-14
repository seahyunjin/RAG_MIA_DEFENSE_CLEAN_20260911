import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location("final",ROOT/"qll_source_hide_final.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location("cal",ROOT/"qll_calibration.py");c=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(c)
def test_softmax_and_dominance():
 p=m.qll_distribution([1,2,3,4]); assert np.isclose(p.sum(),1); assert m.decision([1,2,3,4],.4)["source_index"]==3
def test_strict_boundary():
 d=m.decision([0,0,0,0],.25); assert not d["hide"]
def test_safe_and_risky_actions():
 assert not m.decision([0,0,0,0],.3)["hide"]; assert m.decision([10,0,0,0],.9)["hide"]
def test_allocators():
 assert m.equal_redistribution([1000]*4,None)==[512]*4
 assert m.equal_redistribution([1000]*4,1)==[683,0,683,682]
 assert m.no_redistribution([1000]*4,1)==[512,0,512,512]
def test_calibration_deterministic():
 x=np.arange(100); assert c.calibrate(x)==c.calibrate(x)
def test_no_attack_or_rank_branch():
 text=(ROOT/"qll_source_hide_final.py").read_text(); assert "attack_family" not in text; assert "target_rank" not in text
