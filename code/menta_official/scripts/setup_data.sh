#!/usr/bin/env bash
# Prepare datasets, benign queries, retrieval indices, and document summaries.
# Run --online on a networked node, then --offline inside the GPU allocation.
#
# Skips steps when outputs already exist. Set FORCE=1 in .env to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/lib_pipeline.sh"

print_setup_data_usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup_data.sh --online
  bash scripts/setup_data.sh --offline

Run --online first on a networked node to download/process BEIR data, download
benign queries, and generate OpenAI summaries. Run --offline inside the GPU
allocation to build FAISS indices from the prepared corpora.

All runtime configuration is read from .env:
  DATASETS         Space- or comma-separated dataset names to prepare
  DATA_DIR         Output data directory
  RETRIEVER_MODEL  Model used for FAISS indexing
  OPENAI_QUERY_MODEL  Model used for summaries
  FORCE=1          Re-run steps even when outputs already exist
EOF
}

parse_env_only_pipeline_args print_setup_data_usage "$@"
init_pipeline_env
set_hf_cache_args

ENV_PATH="${ENV_PATH:-.env}"
DATA_DIR="${DATA_DIR:-data}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
OPENAI_QUERY_MODEL="${OPENAI_QUERY_MODEL:-gpt-4.1-nano}"
PIPELINE_PHASE="${PIPELINE_PHASE:-all}"
apply_pipeline_phase_arg

DATASET_CONFIG="${DATASETS:-BeIR_nfcorpus BeIR_scidocs BeIR_trec-covid}"
split_pipeline_env_list DATASETS "${DATASET_CONFIG}"
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "Error: no datasets selected. Set DATASETS in ${ENV_PATH}." >&2
    exit 2
fi

RETRIEVER_TAG="${RETRIEVER_MODEL//\//--}"

corpus_markers=()
benign_markers=()
index_markers=()
summary_markers=()
for ds in "${DATASETS[@]}"; do
    corpus_markers+=(
        "${DATA_DIR}/${ds}/corpus_member.jsonl"
        "${DATA_DIR}/${ds}/corpus_nonmember.jsonl"
    )
    benign_markers+=("${DATA_DIR}/${ds}/benign_queries.jsonl")
    index_markers+=("${DATA_DIR}/${ds}/faiss_indices/${RETRIEVER_TAG}.faiss")
    summary_markers+=("${DATA_DIR}/${ds}/summary.jsonl")
done

print_pipeline_phase_summary

echo "Datasets: ${DATASETS[*]}"
echo "Data dir: ${DATA_DIR}"

if pipeline_run_online_enabled; then
    if skip_if_all_exist "[1/4] Downloading + deduplicating BEIR datasets" "${corpus_markers[@]}"; then
        :
    else
        echo "[1/4] Downloading + deduplicating BEIR datasets ..."
        python utils/process_dataset.py             --datasets "${DATASETS[@]}"             --data_base "${DATA_DIR}"
    fi

    if skip_if_all_exist "[2/4] Downloading benign query pool" "${benign_markers[@]}"; then
        :
    else
        echo "[2/4] Downloading benign query pool ..."
        python utils/load_benign_queries.py             --datasets "${DATASETS[@]}"             --output_base "${DATA_DIR}"             --cache_dir "${HF_HUB_CACHE_DIR}"
    fi

    echo "[3/4] Generating per-document summaries (used by MEntA retrieval boost) ..."
    for ds in "${DATASETS[@]}"; do
        summary_path="${DATA_DIR}/${ds}/summary.jsonl"
        if skip_if_exists "${summary_path}" "  summaries for ${ds}"; then
            continue
        fi
        echo "  -> ${ds}"
        python utils/summarize.py             --member_path    "${DATA_DIR}/${ds}/corpus_member.jsonl"             --nonmember_path "${DATA_DIR}/${ds}/corpus_nonmember.jsonl"             --output_path    "${summary_path}"             --method openai             --model "${OPENAI_QUERY_MODEL}"             --env_path "${ENV_PATH}"
    done
else
    echo "[SKIP] [1/4] Downloading + deduplicating BEIR datasets (online phase only)"
    echo "[SKIP] [2/4] Downloading benign query pool (online phase only)"
    echo "[SKIP] [3/4] Generating per-document summaries (online phase only)"
fi

if pipeline_run_offline_enabled; then
    if skip_if_all_exist "[4/4] Building FAISS indices" "${index_markers[@]}"; then
        :
    else
        echo "[4/4] Building FAISS indices with ${RETRIEVER_MODEL} ..."
        python utils/create_index.py             --model "${RETRIEVER_MODEL}"             --data_base "${DATA_DIR}"             "${HF_CACHE_ARGS[@]}"             --datasets "${DATASETS[@]}"             --use_gpu
    fi
else
    echo "[SKIP] [4/4] Building FAISS indices (offline phase only)"
fi

echo "Done. Dataset artifacts live under ${DATA_DIR}/"
