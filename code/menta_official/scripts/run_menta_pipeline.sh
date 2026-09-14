#!/usr/bin/env bash
# End-to-end MEntA pipeline.
#
# Configure all runtime choices in .env:
#   DATASETS="BeIR_nfcorpus BeIR_scidocs"
#   MENTA_MODES="default dp rerank prompt_instruction paraphrase similarity generic summary_only"
#
# Each mode is run as a separate attack pass. Modes are never combined within
# one pass; setting multiple MENTA_MODES runs them sequentially.
#
# Default pass:
#   1. generate diverse queries (OpenAI)
#   2. retrieve top-k contexts
#   3. generate RAG outputs
#   4. compute entailment between outputs and candidate documents
#   5. evaluate (AUC / accuracy / TPR@FPR)
#
# Skips a step when its main output file already exists. Set FORCE=1 in .env to
# re-run completed steps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_menta_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_menta_pipeline.sh --online
  bash scripts/run_menta_pipeline.sh --offline

Run --online first on a networked node, then --offline inside the GPU allocation.
All other runtime configuration is read from .env.

Environment:
  DATASETS        Space- or comma-separated dataset names to run
  MENTA_MODES     Space- or comma-separated modes to run sequentially
  FORCE=1         Re-run steps even when output files already exist

Modes:
  default             Document-specific queries + NLI entailment
  dp                  DP defense at RAG generation
  rerank              Retrieval reranking defense at RAG generation
  prompt_instruction  Instruction-based defense at RAG generation
  paraphrase          Query paraphrase defense before retrieval
  similarity          Similarity scoring instead of NLI entailment
  generic             Generic topical queries
  summary_only        Use only target summaries during RAG generation
EOF
}

parse_menta_pipeline_args() {
    parse_pipeline_phase_args print_menta_usage "$@"
}

normalize_menta_mode() {
    local raw="${1:-default}"
    case "${raw}" in
        ""|none|default|paper) printf 'default' ;;
        dp) printf 'dp' ;;
        rerank) printf 'rerank' ;;
        prompt|prompt_instruction) printf 'prompt_instruction' ;;
        paraphrase|paraphrased) printf 'paraphrase' ;;
        similarity|similarity_scoring) printf 'similarity' ;;
        generic|generic_queries) printf 'generic' ;;
        summary|summary_only|summary-only|disable_retrieval) printf 'summary_only' ;;
        *)
            echo "Error: unknown MEntA mode '${raw}'." >&2
            print_menta_usage >&2
            return 1
            ;;
    esac
}

split_env_list() {
    local -n out_array="$1"
    local raw="${2:-}"
    local normalized="${raw//,/ }"
    read -r -a out_array <<< "${normalized}"
}

validate_mode_config() {
    local run_mode="$1"

    if [[ "${run_mode}" == "dp" ]]; then
        case "${DP_LEVEL}" in
            retrieval|output|both) ;;
            *)
                echo "Error: DP_LEVEL must be one of: retrieval, output, both." >&2
                exit 2
                ;;
        esac
    fi

    if [[ "${run_mode}" == "rerank" ]]; then
        case "${RERANK_STRATEGY}" in
            reverse|shuffle|score_noise) ;;
            *)
                echo "Error: RERANK_STRATEGY must be one of: reverse, shuffle, score_noise." >&2
                exit 2
                ;;
        esac
    fi

    if [[ "${run_mode}" == "paraphrase" ]]; then
        case "${QUERY_PARAPHRASE_METHOD}" in
            openai|transformers) ;;
            *)
                echo "Error: QUERY_PARAPHRASE_METHOD must be one of: openai, transformers." >&2
                exit 2
                ;;
        esac
    fi
}

parse_menta_pipeline_args "$@"
init_pipeline_env
set_hf_cache_args

ENV_PATH="${ENV_PATH:-.env}"
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}/MEntA"
NUM_QUERIES="${NUM_QUERIES:-5}"
TOP_K="${TOP_K:-3}"
OPENAI_QUERY_MODEL="${OPENAI_QUERY_MODEL:-gpt-4.1-nano}"
RAG_GENERATOR_MODEL="${RAG_GENERATOR_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
MENTA_MODE_CONFIG="${MENTA_MODES:-default}"
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
split_env_list DATASETS "${DATASET_CONFIG}"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: no datasets selected. Set DATASETS in ${ENV_PATH}." >&2
    exit 2
fi

RAW_MENTA_MODES=()
split_env_list RAW_MENTA_MODES "${MENTA_MODE_CONFIG}"
if [[ ${#RAW_MENTA_MODES[@]} -eq 0 ]]; then
    echo "Error: no modes selected. Set MENTA_MODES in ${ENV_PATH}." >&2
    exit 2
fi

MENTA_RUN_MODES=()
for raw_mode in "${RAW_MENTA_MODES[@]}"; do
    normalized_mode="$(normalize_menta_mode "${raw_mode}")"
    already_selected=0
    for selected_mode in "${MENTA_RUN_MODES[@]}"; do
        if [[ "${selected_mode}" == "${normalized_mode}" ]]; then
            already_selected=1
            break
        fi
    done
    if [[ "${already_selected}" == "0" ]]; then
        MENTA_RUN_MODES+=("${normalized_mode}")
    fi
done

# Several MEntA modes consume baseline artifacts owned by default. Running
# default first avoids a dependent mode creating those artifacts and default
# deleting/recreating them later when FORCE=1.
MENTA_ORDERED_MODES=()
for selected_mode in "${MENTA_RUN_MODES[@]}"; do
    if [[ "${selected_mode}" == "default" ]]; then
        MENTA_ORDERED_MODES+=("default")
        break
    fi
done
for selected_mode in "${MENTA_RUN_MODES[@]}"; do
    if [[ "${selected_mode}" != "default" ]]; then
        MENTA_ORDERED_MODES+=("${selected_mode}")
    fi
done
MENTA_RUN_MODES=("${MENTA_ORDERED_MODES[@]}")

RETRIEVER_TAG="${RETRIEVER_MODEL//\//--}"
GENERATOR_TAG="${RAG_GENERATOR_MODEL//\//--}"

echo "MEntA modes: ${MENTA_RUN_MODES[*]}"
echo "Datasets: ${DATASETS[*]}"
print_pipeline_phase_summary

run_menta_mode() {
    local run_mode="$1"
    validate_mode_config "${run_mode}"

    echo "=================================================="
    echo "MEntA mode: ${run_mode}"
    echo "=================================================="

    local eval_dir="evaluation"

    for ds in "${DATASETS[@]}"; do
        echo "=================================================="
        echo "Dataset: ${ds}"
        echo "Mode: ${run_mode}"
        echo "=================================================="

        local member="${DATA_DIR}/${ds}/corpus_member.jsonl"
        local nonmember="${DATA_DIR}/${ds}/corpus_nonmember.jsonl"
        local summary="${DATA_DIR}/${ds}/summary.jsonl"
        local index="${DATA_DIR}/${ds}/faiss_indices/${RETRIEVER_TAG}.faiss"

        local query_dir="queries"
        local retrieval_dir="retrieval"
        local answers_dir="answers"
        local score_dir="entailment"
        eval_dir="evaluation"
        local query_mode_args=()
        local rag_mode_args=()
        local defense_args=()
        local score_kind="entailment"

        case "${run_mode}" in
            default)
                ;;
            dp)
                answers_dir="answers_dp"
                score_dir="entailment_dp"
                eval_dir="evaluation_dp"
                defense_args=(--defense dp --dp_epsilon "${DP_EPSILON}" --dp_level "${DP_LEVEL}")
                ;;
            rerank)
                answers_dir="answers_rerank"
                score_dir="entailment_rerank"
                eval_dir="evaluation_rerank"
                defense_args=(--defense rerank --rerank_strategy "${RERANK_STRATEGY}")
                ;;
            prompt_instruction)
                answers_dir="answers_prompt_instruction"
                score_dir="entailment_prompt_instruction"
                eval_dir="evaluation_prompt_instruction"
                defense_args=(--defense prompt_instruction)
                ;;
            paraphrase)
                query_dir="queries_paraphrased"
                retrieval_dir="retrieval_paraphrased"
                answers_dir="answers_paraphrased"
                score_dir="entailment_paraphrased"
                eval_dir="evaluation_paraphrased"
                defense_args=(--defense paraphrase)
                ;;
            similarity)
                score_dir="similarity"
                eval_dir="evaluation_similarity"
                score_kind="similarity"
                ;;
            generic)
                query_dir="queries_generic"
                retrieval_dir="retrieval_generic"
                answers_dir="answers_generic"
                score_dir="entailment_generic"
                eval_dir="evaluation_generic"
                query_mode_args=(--query_mode generic)
                rag_mode_args=(--generic_queries)
                ;;
            summary_only)
                answers_dir="answers_disable_retrieval"
                score_dir="entailment_disable_retrieval"
                eval_dir="evaluation_disable_retrieval"
                rag_mode_args=(--summary_path "${summary}" --disable_retrieval)
                ;;
        esac

        local base_query_file="${RESULTS_DIR}/${ds}/queries/queries_${NUM_QUERIES}v.jsonl"
        local query_file="${RESULTS_DIR}/${ds}/${query_dir}/queries_${NUM_QUERIES}v.jsonl"
        local query_generation_file="${query_file}"
        local query_generation_label="[1/5] Generating queries"
        local query_generation_force_mode="mode"
        case "${run_mode}" in
            default|generic)
                query_generation_force_mode="mode"
                ;;
            paraphrase)
                query_generation_file="${base_query_file}"
                query_generation_label="[1/5] Ensuring base queries"
                query_generation_force_mode="shared_prereq"
                ;;
            *)
                query_generation_file="${base_query_file}"
                query_generation_label="[1/5] Ensuring base queries"
                query_generation_force_mode="shared_prereq"
                ;;
        esac

        local retrieval_force_mode="mode"
        if [[ "${retrieval_dir}" == "retrieval" && "${run_mode}" != "default" ]]; then
            retrieval_force_mode="shared_prereq"
        fi

        local answer_force_mode="mode"
        if [[ "${answers_dir}" == "answers" && "${run_mode}" != "default" ]]; then
            answer_force_mode="shared_prereq"
        fi

        local retrieval_file="${RESULTS_DIR}/${ds}/${retrieval_dir}/target_summary/topk${TOP_K}/retrieval_${NUM_QUERIES}v.json"
        local answer_file="${RESULTS_DIR}/${ds}/${answers_dir}/target_summary/topk${TOP_K}/${GENERATOR_TAG}/answers_${NUM_QUERIES}v.json"
        local entailment_file="${RESULTS_DIR}/${ds}/${score_dir}/target_summary/topk${TOP_K}/${GENERATOR_TAG}/entailment_${NUM_QUERIES}v.json"
        local similarity_file="${RESULTS_DIR}/${ds}/${score_dir}/target_summary/topk${TOP_K}/${GENERATOR_TAG}/similarity_${NUM_QUERIES}v.json"
        local eval_file="${RESULTS_DIR}/${ds}/${eval_dir}/target_summary/topk${TOP_K}/${GENERATOR_TAG}/evaluation_${NUM_QUERIES}v"
        local eval_marker="${eval_file}/mia_evaluation_results.json"

        # ---------------------------------------------------------------
        # 1) Generate queries
        # ---------------------------------------------------------------
        if ! pipeline_run_online_enabled; then
            echo "[SKIP] ${query_generation_label} (online phase only)"
        elif [[ "${query_generation_force_mode}" == "shared_prereq" && -s "${query_generation_file}" ]]; then
            echo "[SKIP] ${query_generation_label}"
            echo "       reusing shared queries: ${query_generation_file}"
        elif [[ "${query_generation_force_mode}" == "mode" ]] && skip_if_exists "${query_generation_file}" "${query_generation_label}"; then
            :
        else
            echo "[1/5] Generating queries ..."
            python MEntA/generate_queries.py \
                --dataset "${ds}" \
                --member_path "${member}" \
                --nonmember_path "${nonmember}" \
                --output_base_dir "${RESULTS_DIR}" \
                --generation_method openai \
                --generation_model "${OPENAI_QUERY_MODEL}" \
                --num_queries "${NUM_QUERIES}" \
                "${query_mode_args[@]}" \
                --env_path "${ENV_PATH}"
        fi

        if [[ "${run_mode}" == "paraphrase" ]]; then
            if [[ "${QUERY_PARAPHRASE_METHOD}" == "openai" ]] && ! pipeline_run_online_enabled; then
                echo "[SKIP] [1b/5] Paraphrasing queries (online phase only)"
            elif [[ "${QUERY_PARAPHRASE_METHOD}" != "openai" ]] && ! pipeline_run_offline_enabled; then
                echo "[SKIP] [1b/5] Paraphrasing queries (offline phase only)"
            elif skip_if_exists "${query_file}" "[1b/5] Paraphrasing queries"; then
                :
            else
                echo "[1b/5] Paraphrasing queries ..."
                local paraphrase_gpu_args=()
                if [[ "${QUERY_PARAPHRASE_METHOD}" == "transformers" ]]; then
                    paraphrase_gpu_args=(--use_gpu)
                fi
                python defense/query_paraphrase.py \
                    --menta_queries "${base_query_file}" \
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

        # ---------------------------------------------------------------
        # 2) Retrieve top-k contexts
        # ---------------------------------------------------------------
        if [[ "${retrieval_force_mode}" == "shared_prereq" && -s "${retrieval_file}" ]]; then
            echo "[SKIP] [2/5] Ensuring top-${TOP_K} contexts"
            echo "       reusing shared retrieval: ${retrieval_file}"
        elif [[ "${retrieval_force_mode}" == "mode" ]] && skip_if_exists "${retrieval_file}" "[2/5] Retrieving top-${TOP_K} contexts"; then
            :
        else
            echo "[2/5] Retrieving top-${TOP_K} contexts ..."
            local retrieval_output_base="${RESULTS_DIR}"
            local retrieval_copy_source="${retrieval_file}"
            if [[ "${retrieval_dir}" != "retrieval" ]]; then
                retrieval_output_base="${RESULTS_DIR}/_tmp_${retrieval_dir}"
                retrieval_copy_source="${retrieval_output_base}/${ds}/retrieval/target_summary/topk${TOP_K}/retrieval_${NUM_QUERIES}v.json"
            fi

            python MEntA/retrieve.py \
                --dataset "${ds}" \
                --member_corpus_path "${member}" \
                --nonmember_corpus_path "${nonmember}" \
                --index_path "${index}" \
                --query_path "${query_file}" \
                --summary_path "${summary}" \
                --retrieval_model "${RETRIEVER_MODEL}" \
                --output_base_dir "${retrieval_output_base}" \
                "${HF_CACHE_ARGS[@]}" \
                --top_k "${TOP_K}" \
                --use_target_summary \
                --use_gpu

            if [[ "${retrieval_dir}" != "retrieval" ]]; then
                mkdir -p "$(dirname "${retrieval_file}")"
                cp "${retrieval_copy_source}" "${retrieval_file}"
                rm -rf "${retrieval_output_base}"
            fi
        fi

        # ---------------------------------------------------------------
        # 3) Generate RAG outputs
        # ---------------------------------------------------------------
        if [[ "${answer_force_mode}" == "shared_prereq" && -s "${answer_file}" ]]; then
            echo "[SKIP] [3/5] Ensuring RAG outputs"
            echo "       reusing shared answers: ${answer_file}"
        elif [[ "${answer_force_mode}" == "mode" ]] && skip_if_exists "${answer_file}" "[3/5] Generating RAG outputs"; then
            :
        else
            echo "[3/5] Generating RAG outputs with ${RAG_GENERATOR_MODEL} ..."
            python MEntA/generate_rag_output.py \
                --retrieval_file "${retrieval_file}" \
                --corpus_member_path "${member}" \
                --corpus_nonmember_path "${nonmember}" \
                --generation_method transformers \
                --generator_model "${RAG_GENERATOR_MODEL}" \
                "${HF_CACHE_ARGS[@]}" \
                --max_context_docs "${TOP_K}" \
                --env_path "${ENV_PATH}" \
                "${rag_mode_args[@]}" \
                "${defense_args[@]}" \
                --use_gpu
        fi

        # ---------------------------------------------------------------
        # 4) Score generated claims
        # ---------------------------------------------------------------
        if [[ "${score_kind}" == "similarity" ]]; then
            if skip_if_exists "${similarity_file}" "[4/5] Computing similarity"; then
                :
            else
                echo "[4/5] Computing similarity ..."
                python MEntA/compute_similarity.py \
                    --output_file "${answer_file}" \
                    --corpus_member "${member}" \
                    --corpus_nonmember "${nonmember}" \
                    --embedding_model "${RETRIEVER_MODEL}" \
                    "${HF_CACHE_ARGS[@]}" \
                    --use_gpu
            fi
        else
            if skip_if_exists "${entailment_file}" "[4/5] Computing entailment"; then
                :
            else
                echo "[4/5] Computing entailment ..."
                python MEntA/compute_entailment.py \
                    --output_file "${answer_file}" \
                    --corpus_member "${member}" \
                    --corpus_nonmember "${nonmember}" \
                    "${HF_CACHE_ARGS[@]}" \
                    --use_gpu
            fi
        fi

        # ---------------------------------------------------------------
        # 5) Evaluate
        # ---------------------------------------------------------------
        if skip_if_exists "${eval_marker}" "[5/5] Evaluating"; then
            :
        else
            echo "[5/5] Evaluating ..."
            mkdir -p "$(dirname "${eval_file}")"
            if [[ "${score_kind}" == "similarity" ]]; then
                python MEntA/evaluate_similarity.py \
                    --similarity_file "${similarity_file}" \
                    --corpus_member "${member}" \
                    --corpus_nonmember "${nonmember}" \
                    --output_dir "${eval_file}" \
                    --lambda_idk 1.0 \
                    --threshold_method accuracy
            else
                python MEntA/evaluate.py \
                    --entailment_file "${entailment_file}" \
                    --corpus_member "${member}" \
                    --corpus_nonmember "${nonmember}" \
                    --output_dir "${eval_file}" \
                    --lambda_idk 1.0 \
                    --threshold_method accuracy
            fi
        fi
    done

    echo "MEntA mode ${run_mode} finished. Metrics under ${RESULTS_DIR}/<dataset>/${eval_dir}/..."
}

for run_mode in "${MENTA_RUN_MODES[@]}"; do
    run_menta_mode "${run_mode}"
done

echo "MEntA pipeline finished for phase ${PIPELINE_PHASE}."
