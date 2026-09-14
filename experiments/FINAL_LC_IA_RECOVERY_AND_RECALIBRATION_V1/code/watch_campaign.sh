#!/usr/bin/env bash
set -u
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"

while [[ ! -f "$EXP/GOLD_RECALIBRATION_RESULT.json" ]]; do
  if ! tmux has-session -t final-lc-recal-gold 2>/dev/null; then
    tmux new-session -d -s final-lc-recal-gold "cd '$EXP' && '$PY' code/run_gold_recalibration.py >> logs/GOLD_RECALIBRATION.log 2>&1"
    printf '[%s] restarted Gold recalibration\n' "$(date -u +%FT%TZ)" >> "$EXP/logs/WATCHDOG.log"
  fi
  sleep 30
done

while true; do
  if [[ -f "$EXP/IA_V3_E2E_RESULT.json" ]]; then break; fi
  if [[ -f "$EXP/IA_V3_VALIDITY_RESULT.json" ]] && grep -q 'IA_STD_Q15_V3_FAILED' "$EXP/IA_V3_VALIDITY_RESULT.json"; then break; fi
  if [[ -f "$EXP/IA_V3_DETECTION_RESULT.json" ]] && grep -q 'CATASTROPHIC_FAILURE' "$EXP/IA_V3_DETECTION_RESULT.json"; then break; fi
  if [[ -f "$EXP/IA_V3_GT_RESULT.json" ]] && grep -q 'IA_V3_GT_FAILED' "$EXP/IA_V3_GT_RESULT.json"; then break; fi
  if ! tmux has-session -t final-lc-ia-v3 2>/dev/null; then
    tmux new-session -d -s final-lc-ia-v3 "cd '$EXP' && '$PY' code/run_ia_v3.py >> logs/IA_V3.log 2>&1"
    printf '[%s] started/restarted IA-v3\n' "$(date -u +%FT%TZ)" >> "$EXP/logs/WATCHDOG.log"
  fi
  sleep 30
done

"$PY" "$EXP/code/finalize.py" >> "$EXP/logs/FINALIZE.log" 2>&1
