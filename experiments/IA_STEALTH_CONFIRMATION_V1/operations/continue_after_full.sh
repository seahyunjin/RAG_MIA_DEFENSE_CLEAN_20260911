#!/usr/bin/env bash
set -u

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911"
EXP="$ROOT/experiments/IA_STEALTH_CONFIRMATION_V1"
POST="$EXP/post_ready"
PYTHON_BIN="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
LOG="$EXP/logs/IA_ST1_POST_READY_CHAIN.log"

while [[ ! -f "$EXP/FINAL_RESULT.json" ]]; do
  sleep 15
done

verdict="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$EXP/FINAL_RESULT.json")"
printf '[%s] full artifact verdict=%s\n' "$(date -u +%FT%TZ)" "$verdict" >> "$LOG"

if [[ "$verdict" != "IA_STD_Q15_ST1_READY" ]]; then
  printf '[%s] detection skipped; cross-domain remains next per protocol\n' "$(date -u +%FT%TZ)" >> "$LOG"
  exit 0
fi

"$PYTHON_BIN" "$POST/prepare_detection.py" >> "$LOG" 2>&1 || exit $?
"$PYTHON_BIN" "$POST/run_detection.py" >> "$LOG" 2>&1
rc=$?
printf '[%s] detection runner exited rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi

detection_verdict="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$POST/IA_ST1_DETECTION_RESULT.json")"
if [[ "$detection_verdict" != "IA_STEALTH_DETECTION_PASS" ]]; then
  printf '[%s] E2E skipped by frozen detection gate; cross-domain remains next\n' "$(date -u +%FT%TZ)" >> "$LOG"
  exit 0
fi

"$PYTHON_BIN" "$POST/prepare_e2e.py" >> "$LOG" 2>&1 || exit $?
"$PYTHON_BIN" "$POST/run_e2e_generation.py" >> "$LOG" 2>&1 || exit $?
"$PYTHON_BIN" "$POST/run_e2e_scoring.py" >> "$LOG" 2>&1
rc=$?
printf '[%s] E2E scoring exited rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG"
exit "$rc"
