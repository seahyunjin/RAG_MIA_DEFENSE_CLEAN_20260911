#!/usr/bin/env bash
set -u

ROOT="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911"
EXP="$ROOT/experiments/FINAL_LC_DEPLOYMENT_GAPS_AND_IA_V1"
CODE="$EXP/code"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
LOG="$EXP/logs/CHAIN_WATCHDOG.log"
LOCK="$EXP/runtime/chain_watchdog.lock"
export PYTHONPATH="$CODE"
export CUDA_VISIBLE_DEVICES=0

exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
note() { printf '%s %s\n' "$(stamp)" "$*" >> "$LOG"; }
run_logged() {
  stage="$1"; shift
  note "START $stage"
  "$@" >> "$LOG" 2>&1
  rc=$?
  note "END $stage rc=$rc"
  return "$rc"
}

note "watchdog started pid=$$"
while [[ ! -f "$EXP/PHASE_A2_RESULT.json" ]]; do
  if ! pgrep -f 'python .*run_phase_a2_generation.py$' >/dev/null && ! pgrep -f 'python .*run_phase_a2_scoring_repair_v2.py$' >/dev/null; then
    if [[ -f "$EXP/runtime/PHASE_A2_GENERATION_MANIFEST.json" ]]; then
      note "A2 generation is complete; running precommitted metadata-only scoring repair"
      if [[ ! -f "$EXP/configs/PHASE_A2_SCORING_REPAIR_V2_PRECOMMIT.json" ]]; then
        run_logged A2_SCORING_REPAIR_V2_PREPARE "$PY" "$CODE/prepare_phase_a2_scoring_repair_v2.py" || exit 12
      fi
      run_logged A2_SCORING_REPAIR "$PY" "$CODE/run_phase_a2_scoring_repair_v2.py" || exit 13
    else
      note "A2 worker absent; resuming checkpointed generation"
      run_logged A2_GENERATION "$PY" "$CODE/run_phase_a2_generation.py" || true
    fi
  fi
  sleep 20
done

if [[ ! -f "$EXP/configs/FP_CLOSED_BOOK_FALLBACK_PRECOMMIT.json" ]]; then
  run_logged B_PREPARE "$PY" "$CODE/prepare_phase_b.py" || exit 20
fi
if [[ ! -f "$EXP/runtime/PHASE_B_GENERATION_MANIFEST.json" ]]; then
  run_logged B_GENERATION "$PY" "$CODE/run_phase_b_generation.py" || exit 21
fi
if [[ ! -f "$EXP/PHASE_B_RESULT.json" ]]; then
  if [[ ! -f "$EXP/configs/PHASE_B_SCORING_REPAIR_PRECOMMIT.json" ]]; then
    run_logged B_SCORING_REPAIR_PREPARE "$PY" "$CODE/prepare_phase_b_scoring_repair.py" || exit 22
  fi
  run_logged B_SCORING_REPAIR "$PY" "$CODE/run_phase_b_scoring_repair.py" || exit 23
fi

IA="$EXP/IA_STD_Q15_API1"
if [[ ! -f "$IA/configs/IA_STD_Q15_API1_PRECOMMIT.json" && ! -f "$IA/FINAL_RESULT.json" ]]; then
  run_logged C_PREPARE "$PY" "$CODE/prepare_phase_c.py" || exit 30
fi
if [[ -f "$IA/configs/IA_STD_Q15_API1_PRECOMMIT.json" && ! -f "$IA/FINAL_RESULT.json" ]]; then
  run_logged C_API_GENERATION "$PY" "$CODE/run_phase_c_api.py" || exit 31
fi

if [[ -f "$IA/FINAL_RESULT.json" ]] && grep -q 'IA_STD_Q15_API1_READY' "$IA/FINAL_RESULT.json"; then
  if [[ ! -f "$IA/post_ready/configs/IA_API1_DETECTION_PRECOMMIT.json" ]]; then
    run_logged C_DETECTION_PREPARE "$PY" "$CODE/prepare_phase_c_detection.py" || exit 32
  fi
  if [[ ! -f "$IA/post_ready/IA_API1_DETECTION_RESULT.json" ]]; then
    run_logged C_DETECTION "$PY" "$CODE/run_phase_c_detection.py" || exit 33
  fi
  if grep -q '"no_catastrophic_stealth_miss": true' "$IA/post_ready/IA_API1_DETECTION_RESULT.json"; then
    if [[ ! -f "$IA/post_ready/configs/IA_API1_E2E_PRECOMMIT.json" ]]; then
      run_logged C_E2E_PREPARE "$PY" "$CODE/prepare_phase_c_e2e.py" || exit 34
    fi
    if [[ ! -f "$IA/post_ready/runtime/IA_ST1_GENERATION_MANIFEST.json" ]]; then
      run_logged C_E2E_GENERATION "$PY" "$CODE/run_phase_c_e2e_generation.py" || exit 35
    fi
    if [[ ! -f "$IA/post_ready/IA_API1_E2E_RESULT.json" ]]; then
      run_logged C_E2E_SCORING "$PY" "$CODE/run_phase_c_e2e_scoring.py" || exit 36
    fi
  else
    note "IA detection catastrophic-miss gate failed; E2E correctly skipped"
  fi
fi

run_logged FINALIZE "$PY" "$CODE/finalize_campaign.py" || exit 40
note "campaign chain complete; automatic stop"
