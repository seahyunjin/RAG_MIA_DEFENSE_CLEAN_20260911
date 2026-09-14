#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter/exp193_top4_context_rebase_20260829")

def main():
    pre=json.loads((ROOT/"configs/PRECOMMIT.json").read_text())
    cases=pd.read_pickle(ROOT/"private/EXP193_CASES.private.pkl.gz",compression="gzip")
    assert len(cases)==35680 and cases.session_id.nunique()==5600
    assert cases.source_ids.map(len).eq(4).all()
    assert pre["only_experimental_change"].startswith("actual generation context")
    assert not pre.get("threshold_search",False) and "new defense" in pre["forbidden"]
    source=pd.read_csv(ROOT/"audits/TOP4_SOURCE_ORDER.csv.gz")
    assert len(source)==len(cases) and source[[f"source_rank_{i}_id" for i in range(1,5)]].notna().all().all()
    result={"preflight":True,"cohort_cases":len(cases),"cohort_sessions":cases.session_id.nunique(),
            "top4_every_case":True,"new_defense":False,"threshold_search":False}
    if (ROOT/"FINAL_RESULT.json").exists():
        final=json.loads((ROOT/"FINAL_RESULT.json").read_text())
        assert final["final_verdict"] in {"EXP193_INPUT_INSUFFICIENT","EXP193_TOP4_REBASE_VALID_REVERSAL_REPRODUCED",
                                         "EXP193_TOP4_REBASE_VALID_REVERSAL_NOT_REPRODUCED"}
        assert final["historical_frozen_artifacts_unchanged"] is True
        result["final_verdict"]=final["final_verdict"]
    (ROOT/"tests").mkdir(exist_ok=True);(ROOT/"tests/TEST_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
