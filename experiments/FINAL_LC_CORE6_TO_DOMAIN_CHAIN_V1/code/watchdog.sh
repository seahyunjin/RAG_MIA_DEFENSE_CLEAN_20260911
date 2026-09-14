#!/usr/bin/env bash
set -u
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
SESSION="final-lc-core6-domain-chain"
while true; do
  if [[ -f "$EXP/FINAL_CHAIN_RESULT.json" ]]; then exit 0; fi
  if [[ -f "$EXP/PHASE_A_RESULT.json" ]] && grep -q 'CORE6_DETECTION_FAILED' "$EXP/PHASE_A_RESULT.json"; then exit 0; fi
  if [[ -f "$EXP/PHASE_B_RESULT.json" ]] && grep -q 'CORE6_E2E_PRIVACY_FAILED' "$EXP/PHASE_B_RESULT.json"; then exit 0; fi
  if [[ -f "$EXP/PHASE_C_RESULT.json" ]] && grep -q 'GOLD_QA_UTILITY_FAILED' "$EXP/PHASE_C_RESULT.json"; then exit 0; fi
  if [[ -f "$EXP/PHASE_D_RESULT.json" ]]; then exit 0; fi
  if [[ -f "$EXP/PHASE_D_SCREEN_DETECTION_RESULT.json" ]] && grep -q 'PHASE_D_SCREEN_DETECTION_FAILED' "$EXP/PHASE_D_SCREEN_DETECTION_RESULT.json"; then exit 0; fi
  if [[ -f "$EXP/PHASE_D_ALL_DETECTION_RESULT.json" ]] && grep -q 'PHASE_D_ALL_DETECTION_FAILED' "$EXP/PHASE_D_ALL_DETECTION_RESULT.json"; then exit 0; fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" "bash '$EXP/code/run_chain.sh'"
    echo "[$(date -u +%FT%TZ)] restarted $SESSION" >>"$EXP/logs/WATCHDOG.log"
  fi
  sleep 30
done
