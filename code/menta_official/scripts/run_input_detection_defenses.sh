#!/usr/bin/env bash
# Detector-side defenses: GPT-4 based input detection and Mirabel (embedding
# spike) detection. Both consume the unified attack+benign query dataset
# produced by utils/create_queries_dataset.py.
#
# Prerequisite: run each attack's `scripts/run_*_pipeline.sh --online` first so
# that the per-attack query files exist. Run this script with --online for the
# GPT/API detector and --offline for the local/GPU Mirabel detector.
#
# Skips steps when outputs already exist. Set FORCE=1 in .env to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_detection_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_input_detection_defenses.sh --online
  bash scripts/run_input_detection_defenses.sh --offline

Run --online after attack online phases to build the unified query dataset and
run GPT-based detection. Run --offline inside the GPU allocation to run Mirabel.
All runtime configuration is read from .env.

Environment:
  DATASETS     Space- or comma-separated dataset names to run
  FORCE=1      Re-run steps even when output files already exist

Detection dataset sizes:
  normal runs: 1000 benign + 1000 queries per attack type
  smoke tests: TEST_TOTAL_DOCS benign + all generated smoke-test attack queries
EOF
}

parse_env_only_pipeline_args print_detection_usage "$@"
init_pipeline_env
set_hf_cache_args

ENV_PATH="${ENV_PATH:-.env}"
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
BENIGN_DATA_DIR="${BENIGN_DATA_DIR:-${ORIGINAL_DATA_DIR:-${DATA_DIR}}}"
NUM_QUERIES="${NUM_QUERIES:-5}"
OPENAI_DETECTOR_MODEL="${OPENAI_DETECTOR_MODEL:-gpt-4o-mini}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
PIPELINE_SMOKE_TEST="${PIPELINE_SMOKE_TEST:-0}"
DETECTION_BENIGN_QUERIES="${DETECTION_BENIGN_QUERIES:-1000}"
DETECTION_ATTACK_QUERIES_PER_TYPE="${DETECTION_ATTACK_QUERIES_PER_TYPE:-1000}"
DETECTION_DATASET_SEED="${DETECTION_DATASET_SEED:-42}"
PIPELINE_PHASE="${PIPELINE_PHASE:-all}"
apply_pipeline_phase_arg

DATASET_CONFIG="${DATASETS:-BeIR_nfcorpus BeIR_scidocs BeIR_trec-covid}"
split_pipeline_env_list DATASETS "${DATASET_CONFIG}"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: no datasets selected. Set DATASETS in ${ENV_PATH}." >&2
    exit 2
fi

print_pipeline_phase_summary

queries_dataset_markers=()
gpt_detection_markers=()
mirabel_markers=()
for ds in "${DATASETS[@]}"; do
    queries_dataset_markers+=("${DATA_DIR}/${ds}/queries_dataset.jsonl")
    gpt_detection_markers+=("${RESULTS_DIR}/gpt_detection/${ds}/detection_results.jsonl")
    mirabel_markers+=(
        "${RESULTS_DIR}/mirabel_detection/${ds}/benign_mirabel_results.jsonl"
        "${RESULTS_DIR}/mirabel_detection/${ds}/attack_mirabel_results.jsonl"
    )
done

query_dataset_args=(
    --n_benign "${DETECTION_BENIGN_QUERIES}"
    --n_attack_per_type "${DETECTION_ATTACK_QUERIES_PER_TYPE}"
)
if [[ "${PIPELINE_SMOKE_TEST}" == "1" ]]; then
    query_dataset_args=(
        --n_benign "${TEST_TOTAL_DOCS:-20}"
        --all_attack_queries
    )
fi

if [[ "${FORCE}" == "1" ]]; then
    for marker in "${queries_dataset_markers[@]}"; do
        force_remove_marker "${marker}"
    done
fi

echo "[1/3] Building unified benign + attack query dataset ..."
python utils/create_queries_dataset.py \
    --datasets "${DATASETS[@]}" \
    --data_base "${DATA_DIR}" \
    --results_base "${RESULTS_DIR}" \
    --benign_data_base "${BENIGN_DATA_DIR}" \
    --attack_types IA-MIA MBA MEntA S2-MIA DCMI \
    --seed "${DETECTION_DATASET_SEED}" \
    --ia_queries_file "IA-MIA/{dataset}/queries/queries_30.jsonl" \
    --menta_queries_file "MEntA/{dataset}/queries/queries_${NUM_QUERIES}v.jsonl" \
    "${query_dataset_args[@]}"

if ! pipeline_run_online_enabled; then
    echo "[SKIP] [2/3] GPT-based attack detector (${OPENAI_DETECTOR_MODEL}) (online phase only)"
else
    if [[ "${FORCE}" == "1" ]]; then
        for marker in "${gpt_detection_markers[@]}"; do
            force_remove_marker "${marker}"
        done
    fi
    echo "[2/3] Running GPT-based attack detector (${OPENAI_DETECTOR_MODEL}) ..."
    python defense/gpt_detection.py \
        --datasets "${DATASETS[@]}" \
        --data_base "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/gpt_detection" \
        --model "${OPENAI_DETECTOR_MODEL}" \
        --temperature 0.0 \
        --env_path "${ENV_PATH}"
fi

if ! pipeline_run_offline_enabled; then
    echo "[SKIP] [3/3] Mirabel similarity-spike detector (offline phase only)"
else
    if [[ "${FORCE}" == "1" ]]; then
        for marker in "${mirabel_markers[@]}"; do
            force_remove_marker "${marker}"
        done
    fi
    echo "[3/3] Running Mirabel similarity-spike detector ..."
    python defense/mirabel_detection.py \
        --datasets "${DATASETS[@]}" \
        --data_base "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/mirabel_detection" \
        --embedding_model "${RETRIEVER_MODEL}" \
        "${HF_CACHE_ARGS[@]}" \
        --rho 0.05 \
        --use_gpu
fi

echo "Defense evaluations finished for phase ${PIPELINE_PHASE}. See ${RESULTS_DIR}/{gpt,mirabel}_detection/."
