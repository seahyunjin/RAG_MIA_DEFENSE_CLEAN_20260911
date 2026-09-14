#!/usr/bin/env bash
set -u

CAMPAIGN_ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911"
EXP="$CAMPAIGN_ROOT/experiments/FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1"
CODE="$EXP/code"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
LOG="$EXP/logs/CHAIN.log"
LOCK="$EXP/runtime/chain.lock"
export PYTHONPATH="$CODE"
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK"
flock -n 9 || exit 0
stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
note() { printf '%s %s\n' "$(stamp)" "$*" >> "$LOG"; }
run_stage() {
  stage="$1"; shift
  note "START $stage"
  "$@" >> "$LOG" 2>&1
  rc=$?
  note "END $stage rc=$rc"
  return "$rc"
}

IA="$EXP/IA_STD_Q15_API1"
note "Phase-A chain started pid=$$"

if [[ ! -f "$IA/FINAL_RESULT.json" ]]; then
  run_stage IA_API_GENERATION "$PY" "$CODE/run_phase_c_api.py" || exit 11
fi

if ! grep -q '"verdict": "IA_STD_Q15_API1_READY"' "$IA/FINAL_RESULT.json"; then
  note "IA protocol did not reach full READY; campaign correctly stopped"
  exit 0
fi

if [[ ! -f "$IA/post_ready/configs/IA_API1_DETECTION_PRECOMMIT.json" ]]; then
  run_stage IA_DETECTION_PREPARE "$PY" "$CODE/prepare_phase_c_detection.py" || exit 12
fi
if [[ ! -f "$IA/post_ready/IA_API1_DETECTION_RESULT.json" ]]; then
  run_stage IA_DETECTION "$PY" "$CODE/run_phase_c_detection.py" || exit 13
fi
if ! grep -q '"verdict": "IA_STEALTH_DETECTION_PASS"' "$IA/post_ready/IA_API1_DETECTION_RESULT.json"; then
  note "IA strict detection gate failed; campaign correctly stopped"
  exit 0
fi

if [[ ! -f "$IA/post_ready/configs/IA_API1_E2E_PRECOMMIT.json" ]]; then
  run_stage IA_E2E_PREPARE "$PY" "$CODE/prepare_phase_c_e2e.py" || exit 14
fi
if [[ ! -f "$IA/post_ready/runtime/IA_ST1_GENERATION_MANIFEST.json" ]]; then
  run_stage IA_E2E_GENERATION "$PY" "$CODE/run_phase_c_e2e_generation.py" || exit 15
fi
if [[ ! -f "$IA/post_ready/IA_API1_E2E_RESULT.json" ]]; then
  run_stage IA_E2E_SCORING "$PY" "$CODE/run_phase_c_e2e_scoring.py" || exit 16
fi
if ! grep -q '"verdict": "IA_STEALTH_CONFIRMATION_PASS"' "$IA/post_ready/IA_API1_E2E_RESULT.json"; then
  note "IA E2E gate failed; campaign correctly stopped"
  exit 0
fi

note "PHASE_A_PASS; untouched-domain phase is now eligible"
if [[ -x "$CODE/watch_transfer_chain.sh" ]]; then
  exec "$CODE/watch_transfer_chain.sh"
fi
note "transfer-chain implementation unavailable; fail-closed before untouched-domain execution"
