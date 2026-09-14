#!/usr/bin/env bash
# MBA baseline pipeline (Liu et al., "Mask-based MIA" 2025).
# Configure datasets and modes in .env:
#   DATASETS="BeIR_nfcorpus BeIR_scidocs"
#   MBA_MODES="default dp rerank prompt_instruction paraphrase"
#
# Each mode is run as a separate attack pass. Modes are never combined within
# one pass; setting multiple MBA_MODES runs them sequentially.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_mba_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_mba_pipeline.sh --online
  bash scripts/run_mba_pipeline.sh --offline

Run --online first on a networked node, then --offline inside the GPU allocation.
All other runtime configuration is read from .env.

Environment:
  DATASETS      Space- or comma-separated dataset names to run
  MBA_MODES     Space- or comma-separated modes to run sequentially
  FORCE=1       Re-run steps even when output files already exist

Modes:
  default, dp, rerank, prompt_instruction, paraphrase
EOF
}

parse_env_only_pipeline_args print_mba_usage "$@"
init_pipeline_env
set_hf_cache_args

DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}/MBA"
TOP_K="${TOP_K:-3}"
NUM_MASKS="${NUM_MASKS:-5}"
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

DATASET_CONFIG="${DATASETS:-BeIR_nfcorpus BeIR_scidocs BeIR_trec-covid}"
split_pipeline_env_list DATASETS "${DATASET_CONFIG}"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: no datasets selected. Set DATASETS in .env." >&2
    exit 2
fi

RAW_MBA_MODES=()
split_pipeline_env_list RAW_MBA_MODES "${MBA_MODES:-default}"
if [[ ${#RAW_MBA_MODES[@]} -eq 0 ]]; then
    echo "Error: no modes selected. Set MBA_MODES in .env." >&2
    exit 2
fi
MBA_RUN_MODES=()
for raw_mode in "${RAW_MBA_MODES[@]}"; do
    MBA_RUN_MODES+=("$(normalize_attack_mode "${raw_mode}")")
done
dedupe_default_first_modes MBA_RUN_MODES

RETRIEVER_TAG="${RETRIEVER_MODEL//\//--}"
GENERATOR_TAG="${RAG_GENERATOR_MODEL//\//--}"

echo "MBA modes: ${MBA_RUN_MODES[*]}"
echo "Datasets: ${DATASETS[*]}"
print_pipeline_phase_summary

for ds in "${DATASETS[@]}"; do
    echo "=================================================="
    echo "Dataset: ${ds}"
    echo "Shared MBA setup"
    echo "=================================================="

    member="${DATA_DIR}/${ds}/corpus_member.jsonl"
    nonmember="${DATA_DIR}/${ds}/corpus_nonmember.jsonl"
    queries_file="${RESULTS_DIR}/${ds}/queries/queries.jsonl"

    mask_gpu_args=()
    if pipeline_run_offline_enabled; then
        mask_gpu_args=(--use_gpu)
    fi
    if skip_if_exists "${queries_file}" "[1/4] Generating masked queries"; then
        :
    else
        echo "[1/4] Generating masked queries (num_masks=${NUM_MASKS}, proxy=${RAG_GENERATOR_MODEL}) ..."
        mkdir -p "$(dirname "${queries_file}")"
        python MBA/mask_gen.py \
            --corpus_member_file "${member}" \
            --corpus_nonmember_file "${nonmember}" \
            --output_file "${queries_file}" \
            --num_masks "${NUM_MASKS}" \
            --proxy_model "${RAG_GENERATOR_MODEL}" \
            "${HF_CACHE_ARGS[@]}" \
            "${mask_gpu_args[@]}"
    fi
done

run_mba_mode() {
    local run_mode="$1"
    validate_attack_mode_config "${run_mode}"

    echo "=================================================="
    echo "MBA mode: ${run_mode}"
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
                echo "[SKIP] [1b/4] Paraphrasing MBA queries (online phase only)"
            elif [[ "${QUERY_PARAPHRASE_METHOD}" != "openai" ]] && ! pipeline_run_offline_enabled; then
                echo "[SKIP] [1b/4] Paraphrasing MBA queries (offline phase only)"
            elif skip_if_exists "${queries_file}" "[1b/4] Paraphrasing MBA queries"; then
                :
            else
                echo "[1b/4] Paraphrasing MBA queries ..."
                local paraphrase_gpu_args=()
                if [[ "${QUERY_PARAPHRASE_METHOD}" == "transformers" ]]; then
                    paraphrase_gpu_args=(--use_gpu)
                fi
                python defense/query_paraphrase.py \
                    --mba_queries "${base_queries_file}" \
                    --dcmi_queries "" \
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
        local eval_file="${RESULTS_DIR}/${ds}/${eval_base_dir}/topk${TOP_K}/${GENERATOR_TAG}/evaluation.json"

        if [[ "${retrieval_force_mode}" == "shared_prereq" && -s "${retrieval_file}" ]]; then
            echo "[SKIP] [2/4] Ensuring top-${TOP_K} contexts"
            echo "       reusing shared retrieval: ${retrieval_file}"
        elif [[ "${retrieval_force_mode}" == "mode" ]] && skip_if_exists "${retrieval_file}" "[2/4] Retrieving top-${TOP_K} contexts"; then
            :
        else
            echo "[2/4] Retrieving top-${TOP_K} contexts ..."
            python MBA/retrieve.py \
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

        if skip_if_exists "${answer_file}" "[3/4] Generating RAG answers"; then
            :
        else
            echo "[3/4] Generating RAG answers with ${RAG_GENERATOR_MODEL} ..."
            mkdir -p "$(dirname "${answer_file}")"
            python MBA/gen_rag_output.py \
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

        if skip_if_exists "${eval_file}" "[4/4] Evaluating"; then
            :
        else
            echo "[4/4] Evaluating ..."
            mkdir -p "$(dirname "${eval_file}")"
            python MBA/evaluate.py \
                --rag_outputs_file "${answer_file}" \
                --output_file "${eval_file}"
        fi
    done
}

for run_mode in "${MBA_RUN_MODES[@]}"; do
    run_mba_mode "${run_mode}"
done

echo "MBA pipeline finished for phase ${PIPELINE_PHASE}. Metrics under ${RESULTS_DIR}/<dataset>/evaluation*/..."
