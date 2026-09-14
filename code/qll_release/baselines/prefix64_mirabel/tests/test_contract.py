import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_precommit():
  value=json.loads((ROOT/'configs/PRECOMMIT.json').read_text())
  assert value['written_before_candidate_generation_and_metrics'] is True
  assert value['candidate']=='PREFIX64_MIRABEL_TO_HIDE'
  assert value['cohort']['member']==value['cohort']['nonmember']==900
  assert value['paid_api_calls']==0 and value['fresh_blind'] is False
