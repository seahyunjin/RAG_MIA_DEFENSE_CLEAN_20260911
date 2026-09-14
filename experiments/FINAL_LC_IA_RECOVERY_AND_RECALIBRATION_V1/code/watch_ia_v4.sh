#!/usr/bin/env bash
set -u
cd /home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1
child=IA_STD_Q15_V4_TWO_STAGE
session=final-lc-ia-v4
python=/home/traffic_3/workspace/miniconda3/envs/torch/bin/python

terminal_result() {
  "$python" - <<'PY'
from pathlib import Path
import json, sys
p=Path('IA_STD_Q15_V4_TWO_STAGE')
checks=[
 ('IA_V4_PREFLIGHT_RESULT.json', {'IA_STD_Q15_V4_FORMAT_PREFLIGHT_FAILED'}),
 ('IA_V4_FULL_VALIDITY_RESULT.json', {'IA_STD_Q15_V4_FAILED'}),
 ('IA_V4_DETECTION_RESULT.json', {'IA_V4_DETECTION_CATASTROPHIC_FAILURE'}),
 ('IA_V4_E2E_RESULT.json', {'IA_V4_MATCHED_BUDGET_E2E_COMPLETE'}),
]
for name, terminal in checks:
 f=p/name
 if f.exists() and json.loads(f.read_text()).get('verdict') in terminal:
  sys.exit(0)
sys.exit(1)
PY
}

while true; do
  if terminal_result; then
    printf '%s terminal result reached\n' "$(date -u +%FT%TZ)" >> logs/IA_V4_WATCHDOG.log
    exit 0
  fi
  if ! tmux has-session -t "$session" 2>/dev/null; then
    printf '%s worker absent; resumable restart\n' "$(date -u +%FT%TZ)" >> logs/IA_V4_WATCHDOG.log
    tmux new-session -d -s "$session" "cd '$PWD' && '$python' code/run_ia_v4.py >> logs/IA_V4.log 2>&1"
  fi
  sleep 30
done
