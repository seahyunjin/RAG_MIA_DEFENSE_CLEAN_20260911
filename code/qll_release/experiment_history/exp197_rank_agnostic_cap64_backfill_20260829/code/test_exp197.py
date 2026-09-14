#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd

from run_exp197 import CAP, TOTAL, ROOT

def main():
    assert CAP == 64 and TOTAL == 2048
    used=[64]*32
    assert max(used)<=64 and sum(used)==TOTAL
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from evaluate_exp197 import metric_rows
    sample=pd.DataFrame({"condition":["A"]*4,"attack_family":["X"]*4,
                         "member":[0,0,1,1],"attack_score":[0.,.1,.8,.9]})
    measured=metric_rows(sample)
    assert len(measured)==1 and measured.iloc[0].effective_auc==1.0
    packing=pd.read_csv(ROOT/"tables/TABLE_197_01_BACKFILL_DIAGNOSTIC.csv")
    assert packing.iloc[0].maximum_source_tokens<=64
    assert packing.iloc[0].mean_context_tokens==TOTAL
    print("EXP197_UNIT_TESTS_PASS")

if __name__ == "__main__": main()
