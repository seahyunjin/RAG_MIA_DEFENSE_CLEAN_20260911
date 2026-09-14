#!/usr/bin/env bash
# DCMI baseline pipeline (Gao et al., "Differential Calibration MIA" 2025).
# Configure datasets and modes in .env:
#   DATASETS="BeIR_nfcorpus BeIR_scidocs"
#   DCMI_MODES="default dp rerank prompt_instruction paraphrase"
#
# Each mode is run as a separate attack pass. Modes are never combined within
# one pass; setting multiple DCMI_MODES runs them sequentially.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_dcmi_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_dcmi_pipeline.sh --online
  bash scripts/run_dcmi_pipeline.sh --offline

Run --online first on a networked node, then --offline inside the GPU allocation.
All other runtime configuration is read from .env.

Environment:
  DATASETS       Space- or comma-separated dataset names to run
  DCMI_MODES     Space- or comma-separated modes to run sequentially
  FORCE=1        Re-run steps even when output files already exist

Modes:
  default, dp, rerank, prompt_instruction, paraphrase
EOF
}

parse_env_only_pipeline_args print_dcmi_usage "$@"
init_pipeline_env
set_hf_cache_args

ENV_PATH="${ENV_PATH:-.env}"
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}/DCMI"
TOP_K="${TOP_K:-3}"
PERTURB_MAG="${PERTURB_MAG:-0.06}"
OPENAI_QUERY_MODEL="${OPENAI_QUERY_MODEL:-gpt-4.1-nano}"
RAG_GENERATOR_MODEL="${RAG_GENERATOR_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
DP_EPSILON="${DP_EPSILON:-0.1}"
DP_LEVEL="${DP_LEVEL:-output}"
RERANK_STRATEGY="${RERANK_STRATEGY:-shuffle}"
QUERY_PARAPHRASE_METHOD="${QUERY_PARAPHRASE_METHOD:-openai}"
QUERY_PARAPHRASE_MODEL="${QUERY_PARAPHRASE_MODEL:-${OPENAI_QUERY_MODEL}}"
QUERY_PARAPHRASE_TRANSFORMERS_MODEL="${QUERY_PARAPHRASE_TRANSFORMERS_MODEL:-${RAG_GENERATOR_MODEL}}"
QUERY_PARAPHRASE_BATCH_SIZE="${QUERY_PARAPHRASE_BATCH_SIZE:-32}"
PIPELINE_PHASE="${PIPELINE_PHASE:-all}"
apply_pipeline_phase_arg
PERTURBED_DATA_DIR="${DCMI_PERTURBED_DATA_DIR:-${DATA_DIR}}"

DATASET_CONFIG="${DATASETS:-BeIR_nfcorpus BeIR_scidocs BeIR_trec-covid}"
split_pipeline_env_list DATASETS "${DATASET_CONFIG}"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: no datasets selected. Set DATASETS in ${ENV_PATH}." >&2
    exit 2
fi

RAW_DCMI_MODES=()
split_pipeline_env_list RAW_DCMI_MODES "${DCMI_MODES:-default}"
if [[ ${#RAW_DCMI_MODES[@]} -eq 0 ]]; then
    echo "Error: no modes selected. Set DCMI_MODES in ${ENV_PATH}." >&2
    exit 2
fi
DCMI_RUN_MODES=()
for raw_mode in "${RAW_DCMI_MODES[@]}"; do
    DCMI_RUN_MODES+=("$(normalize_attack_mode "${raw_mode}")")
done
dedupe_default_first_modes DCMI_RUN_MODES

RETRIEVER_TAG="${RETRIEVER_MODEL//\//--}"
GENERATOR_TAG="${RAG_GENERATOR_MODEL//\//--}"

echo "DCMI modes: ${DCMI_RUN_MODES[*]}"
echo "Datasets: ${DATASETS[*]}"
print_pipeline_phase_summary

dcmi_perturb_markers=()
dcmi_query_markers=()
for ds in "${DATASETS[@]}"; do
    dcmi_perturb_markers+=(
        "${PERTURBED_DATA_DIR}/${ds}/perturbed_corpus_member.jsonl"
        "${PERTURBED_DATA_DIR}/${ds}/perturbed_corpus_nonmember.jsonl"
    )
    dcmi_query_markers+=("${RESULTS_DIR}/${ds}/queries/queries.jsonl")
done

if ! pipeline_run_online_enabled; then
    echo "[SKIP] [1/?] Perturbing target documents (online phase only)"
elif skip_if_all_exist "[1/?] Perturbing target documents" "${dcmi_perturb_markers[@]}"; then
    :
else
    echo "[1/?] Perturbing target documents (magnitude=${PERTURB_MAG}) ..."
    python DCMI/perturb_docs.py \
        --datasets "${DATASETS[@]}" \
        --data_base "${DATA_DIR}" \
        --output_data_base "${PERTURBED_DATA_DIR}" \
        --model "${OPENAI_QUERY_MODEL}" \
        --perturbation_magnitude "${PERTURB_MAG}" \
        --env_path "${ENV_PATH}"
fi

if skip_if_all_exist "[2/?] Building DCMI yes/no queries" "${dcmi_query_markers[@]}"; then
    :
else
    echo "[2/?] Building DCMI yes/no queries ..."
    python DCMI/generate_queries.py \
        --datasets "${DATASETS[@]}" \
        --data_base "${DATA_DIR}" \
        --perturbed_data_base "${PERTURBED_DATA_DIR}" \
        --results_base "${RESULTS_DIR}"
fi

run_dcmi_mode() {
    local run_mode="$1"
    validate_attack_mode_config "${run_mode}"

    echo "=================================================="
    echo "DCMI mode: ${run_mode}"
    echo "=================================================="

    local defense_args=()
    build_defense_args defense_args "${run_mode}"

    for ds in "${DATASETS[@]}"; do
        echo "=================================================="
        echo "Dataset: ${ds}"
        echo "Mode: ${run_mode}"
        echo "=================================================="

        local member="${DATA_DIR}/${ds}/corpus_member.jsonl"
        local nonmember="${DATA_DIR}/${ds}/corpus_nonmember.jsonl"
        local index="${DATA_DIR}/${ds}/faiss_indices/${RETRIEVER_TAG}.faiss"
        local base_queries_file="${RESULTS_DIR}/${ds}/queries/queries.jsonl"
        local queries_file="${base_queries_file}"
        local retrieval_dir="${RESULTS_DIR}/${ds}/retrieval"
        local answers_dir
        answers_dir="$(mode_dirname answers "${run_mode}")"
        local eval_base_dir
        eval_base_dir="$(mode_dirname evaluation "${run_mode}")"

        if [[ "${run_mode}" == "paraphrase" ]]; then
            queries_file="${RESULTS_DIR}/${ds}/queries_paraphrased/queries.jsonl"
            retrieval_dir="${RESULTS_DIR}/${ds}/retrieval_paraphrased"

            if [[ "${QUERY_PARAPHRASE_METHOD}" == "openai" ]] && ! pipeline_run_online_enabled; then
                echo "[SKIP] [2b/5] Paraphrasing DCMI queries (online phase only)"
            elif [[ "${QUERY_PARAPHRASE_METHOD}" != "openai" ]] && ! pipeline_run_offline_enabled; then
                echo "[SKIP] [2b/5] Paraphrasing DCMI queries (offline phase only)"
            elif skip_if_exists "${queries_file}" "[2b/5] Paraphrasing DCMI queries"; then
                :
            else
                echo "[2b/5] Paraphrasing DCMI queries ..."
                local paraphrase_gpu_args=()
                if [[ "${QUERY_PARAPHRASE_METHOD}" == "transformers" ]]; then
                    paraphrase_gpu_args=(--use_gpu)
                fi
                python defense/query_paraphrase.py \
                    --dcmi_queries "${base_queries_file}" \
                    --generation_method "${QUERY_PARAPHRASE_METHOD}" \
                    --openai_model "${QUERY_PARAPHRASE_MODEL}" \
                    --transformers_model "${QUERY_PARAPHRASE_TRANSFORMERS_MODEL}" \
                    --generation_batch_size "${QUERY_PARAPHRASE_BATCH_SIZE}" \
                    "${HF_CACHE_ARGS[@]}" \
                    "${paraphrase_gpu_args[@]}"
            fi
        fi

        if ! pipeline_run_offline_enabled; then
            echo "[STOP] Online phase completed for ${ds} (${run_mode}); skipping offline GPU/local stages."
            continue
        fi

        local retrieval_file="${retrieval_dir}/topk${TOP_K}_results.json"
        local retrieval_force_mode="mode"
        if [[ "${run_mode}" != "default" && "${run_mode}" != "paraphrase" ]]; then
            retrieval_force_mode="shared_prereq"
        fi
        local answer_file="${RESULTS_DIR}/${ds}/${answers_dir}/topk${TOP_K}/${GENERATOR_TAG}/answers.jsonl"
        local eval_dir="${RESULTS_DIR}/${ds}/${eval_base_dir}/topk${TOP_K}/${GENERATOR_TAG}"
        local eval_marker="${eval_dir}/dcmi_evaluation_${GENERATOR_TAG}.json"

        if [[ "${retrieval_force_mode}" == "shared_prereq" && -s "${retrieval_file}" ]]; then
            echo "[SKIP] [3/5] Ensuring top-${TOP_K} contexts"
            echo "       reusing shared retrieval: ${retrieval_file}"
        elif [[ "${retrieval_force_mode}" == "mode" ]] && skip_if_exists "${retrieval_file}" "[3/5] Retrieving top-${TOP_K} contexts"; then
            :
        else
            echo "[3/5] Retrieving top-${TOP_K} contexts ..."
            python DCMI/retrieve.py \
                --queries_file "${queries_file}" \
                --index_path "${index}" \
                --corpus_member_file "${member}" \
                --corpus_nonmember_file "${nonmember}" \
                --output_dir "${retrieval_dir}" \
                --retrieval_model "${RETRIEVER_MODEL}" \
                "${HF_CACHE_ARGS[@]}" \
                --top_k "${TOP_K}" \
                --use_gpu
        fi

        if skip_if_exists "${answer_file}" "[4/5] Generating RAG answers"; then
            :
        else
            echo "[4/5] Generating RAG answers with ${RAG_GENERATOR_MODEL} ..."
            mkdir -p "$(dirname "${answer_file}")"
            python DCMI/generate_output.py \
                --queries_file "${queries_file}" \
                --retrieval_file "${retrieval_file}" \
                --output_file "${answer_file}" \
                --corpus_member_file "${member}" \
                --corpus_nonmember_file "${nonmember}" \
                --generation_method transformers \
                --transformers_model "${RAG_GENERATOR_MODEL}" \
                "${HF_CACHE_ARGS[@]}" \
                --max_context_docs "${TOP_K}" \
                "${defense_args[@]}" \
                --use_gpu
        fi

        if skip_if_exists "${eval_marker}" "[5/5] Evaluating"; then
            :
        else
            echo "[5/5] Evaluating ..."
            mkdir -p "${eval_dir}"
            python DCMI/evaluate.py \
                --answers_file "${answer_file}" \
                --corpus_member "${member}" \
                --corpus_nonmember "${nonmember}" \
                --output_dir "${eval_dir}"
        fi
    done
}

for run_mode in "${DCMI_RUN_MODES[@]}"; do
    run_dcmi_mode "${run_mode}"
done

echo "DCMI pipeline finished for phase ${PIPELINE_PHASE}. Metrics under ${RESULTS_DIR}/<dataset>/evaluation*/..."
