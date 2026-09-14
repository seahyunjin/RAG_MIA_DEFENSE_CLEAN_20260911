#!/usr/bin/env bash
set -u
EXP="/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/FINAL_LC_STEALTH_DOMAIN_TRANSFER_CHAIN_V1/UNTOUCHED_FINQA_V1"
CODE="$EXP/code"
PY="/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
LOG="$EXP/logs/UNTOUCHED_CHAIN.log"
LOCK="$EXP/runtime/untouched-chain.lock"
export PYTHONPATH="$CODE"
export CUDA_VISIBLE_DEVICES=0
exec 9>"$LOCK"; flock -n 9 || exit 0
stamp(){ date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
note(){ printf "%s %s\n" "$(stamp)" "$*" >> "$LOG"; }
run_stage(){ stage="$1"; shift; note "START $stage"; "$@" >> "$LOG" 2>&1; rc=$?; note "END $stage rc=$rc"; return "$rc"; }
note "FinQA untouched-domain chain started pid=$$; no performance has been observed before precommit"
run_stage PREPARE "$PY" "$CODE/prepare_untouched_finqa.py" || exit 31
run_stage SCREEN_LOCAL_QUERIES "$PY" "$CODE/run_untouched_queries.py" --scope screen || exit 32
run_stage SCREEN_IA_API1 "$PY" "$CODE/run_untouched_ia_api.py" --scope screen || exit 33
grep -q '"verdict": "FINQA_IA_STD_Q15_API1_READY"' "$EXP/IA_STD_Q15_API1/FINQA_SCREEN_RESULT.json" || { note "screen IA API validity/cost gate failed"; exit 0; }
run_stage SCREEN_COMBINE "$PY" "$CODE/combine_untouched_queries.py" --scope screen || exit 34
run_stage SCREEN_DETECTION "$PY" "$CODE/run_untouched_detection.py" --scope screen || exit 35
grep -q '"verdict": "PHASE_D_SCREEN_DETECTION_PASS"' "$EXP/PHASE_D_SCREEN_DETECTION_RESULT.json" || { note "FinQA detection screen failed; E2E prohibited"; exit 0; }
run_stage SCREEN_E2E "$PY" "$CODE/run_untouched_e2e.py" --scope screen || exit 36
grep -q '"verdict": "CROSS_DOMAIN_SCREEN_PASS"' "$EXP/PHASE_D_SCREEN_E2E_RESULT.json" || { note "FinQA E2E screen failed; large expansion prohibited"; exit 0; }
run_stage LARGE_API_COST_PROJECTION "$PY" "$CODE/check_large_api_budget.py"
rc=$?
if [[ $rc -eq 42 ]]; then note "FinQA screen passed but large expansion is cost-blocked under shared USD12 cap"; exit 0; fi
[[ $rc -eq 0 ]] || exit 37
run_stage ALL_LOCAL_QUERIES "$PY" "$CODE/run_untouched_queries.py" --scope all || exit 38
run_stage ALL_IA_API1 "$PY" "$CODE/run_untouched_ia_api.py" --scope all || exit 39
grep -q '"verdict": "FINQA_IA_STD_Q15_API1_READY"' "$EXP/IA_STD_Q15_API1/FINQA_ALL_RESULT.json" || { note "large IA API validity/cost gate failed"; exit 0; }
run_stage ALL_COMBINE "$PY" "$CODE/combine_untouched_queries.py" --scope all || exit 40
run_stage ALL_DETECTION "$PY" "$CODE/run_untouched_detection.py" --scope all || exit 41
grep -q '"verdict": "PHASE_D_ALL_DETECTION_PASS"' "$EXP/PHASE_D_ALL_DETECTION_RESULT.json" || { note "FinQA large detection failed; large E2E prohibited"; exit 0; }
run_stage ALL_E2E "$PY" "$CODE/run_untouched_e2e.py" --scope all || exit 42
grep -q '"verdict": "CROSS_DOMAIN_ALL_PASS"' "$EXP/PHASE_D_ALL_E2E_RESULT.json" && note "CROSS_DOMAIN_LARGE_PASS" || note "DOMAIN_GENERALIZATION_FAILED"
