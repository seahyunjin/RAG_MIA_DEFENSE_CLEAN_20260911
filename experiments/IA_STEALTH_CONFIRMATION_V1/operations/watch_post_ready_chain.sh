#!/usr/bin/env bash
set -u

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911"
EXP="$ROOT/experiments/IA_STEALTH_CONFIRMATION_V1"
POST="$EXP/post_ready"
SESSION="ia-stealth-post-ready"
LOG="$EXP/logs/IA_ST1_POST_READY_WATCHDOG.log"
PYTHON_BIN="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"

while true; do
  if [[ -f "$POST/IA_ST1_E2E_RESULT.json" ]]; then
    printf '[%s] E2E terminal result present; watchdog exiting\n' "$(date -u +%FT%TZ)" >> "$LOG"
    exit 0
  fi

  if [[ -f "$POST/IA_ST1_DETECTION_RESULT.json" ]]; then
    verdict="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$POST/IA_ST1_DETECTION_RESULT.json")"
    if [[ "$verdict" == "IA_STEALTH_DETECTION_FAILED" ]]; then
      printf '[%s] detection gate terminal FAIL; watchdog exiting\n' "$(date -u +%FT%TZ)" >> "$LOG"
      exit 0
    fi
  fi

  if [[ -f "$EXP/FINAL_RESULT.json" ]]; then
    full_verdict="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$EXP/FINAL_RESULT.json")"
    if [[ "$full_verdict" != "IA_STD_Q15_ST1_READY" ]]; then
      printf '[%s] full validity terminal verdict=%s; watchdog exiting\n' "$(date -u +%FT%TZ)" "$full_verdict" >> "$LOG"
      exit 0
    fi
  fi

  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    printf '[%s] post-ready chain absent; restarting checkpoint-safe chain\n' "$(date -u +%FT%TZ)" >> "$LOG"
    tmux new-session -d -s "$SESSION" "cd '$ROOT' && '$EXP/operations/continue_after_full.sh'"
  fi
  sleep 30
done
