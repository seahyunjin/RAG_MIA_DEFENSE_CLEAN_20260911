#!/usr/bin/env bash
# Shared helpers for menta/scripts/* pipelines.
#
# Usage (at top of a pipeline script, after REPO_ROOT is set):
#   source "${REPO_ROOT}/scripts/lib_pipeline.sh"
#   parse_pipeline_args "$@"
#   # optional: default datasets if PIPELINE_DATASETS is empty
#
# Skip behavior:
#   - If an expected output file/dir already exists, the step is skipped.
#   - Pass --force (or -f) to re-run all steps anyway.
#   - Or set FORCE=1 in the environment.

# shellcheck disable=SC2034
FORCE="${FORCE:-0}"
PIPELINE_DATASETS=()
HF_HUB_CACHE_DIR=""

# Load `.env` and export HuggingFace cache paths for pipeline Python steps.
# Call after `cd` to REPO_ROOT (and after `parse_pipeline_args` if you pass "$@").
init_pipeline_env() {
    local env_file="${ENV_PATH:-.env}"
    if [[ -f "${env_file}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi

    # Expand ${PWD}, ~, etc. from .env (e.g. HF_HOME=${PWD}/../hf_cache).
    if [[ -n "${HF_HOME:-}" ]]; then
        HF_HOME="$(eval echo "${HF_HOME}")"
        export HF_HOME
    fi
    if [[ -n "${HF_HUB_CACHE:-}" ]]; then
        HF_HUB_CACHE="$(eval echo "${HF_HUB_CACHE}")"
        export HF_HUB_CACHE
    fi
    if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
        TRANSFORMERS_CACHE="$(eval echo "${TRANSFORMERS_CACHE}")"
        export TRANSFORMERS_CACHE
    fi

    HF_HUB_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:+${HF_HOME}/hub}}"
    HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
    HF_HUB_CACHE_DIR="$(eval echo "${HF_HUB_CACHE_DIR}")"
    export HF_HUB_CACHE_DIR HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"

    if [[ -n "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN
    fi
    if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
        export HF_HUB_OFFLINE
    fi
    if [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
        export TRANSFORMERS_OFFLINE
    fi
}

# Append to python invocations that load HuggingFace models from cache.
HF_CACHE_ARGS=()
set_hf_cache_args() {
    HF_CACHE_ARGS=(--cache_dir "${HF_HUB_CACHE_DIR}")
}

parse_pipeline_args() {
    FORCE="${FORCE:-0}"
    local -a remaining=()
    for arg in "$@"; do
        case "${arg}" in
            --force|-f) FORCE=1 ;;
            -h|--help)
                cat <<'EOF'
Pipeline options:
  --force, -f    Re-run steps even when output files already exist
  <datasets...>  Dataset names (default varies per script)

Environment:
  FORCE=1        Same as --force
EOF
                exit 0
                ;;
            *) remaining+=("${arg}") ;;
        esac
    done
    PIPELINE_DATASETS=("${remaining[@]}")
}

force_remove_marker() {
    local marker="$1"
    if [[ -z "${marker}" || "${FORCE}" != "1" || ! -e "${marker}" ]]; then
        return 0
    fi
    echo "[FORCE] Removing existing output: ${marker}"
    rm -rf -- "${marker}"
}

# Returns 0 if the step should be skipped, 1 if it should run.
should_skip_step() {
    local marker="$1"
    if [[ "${FORCE}" == "1" ]]; then
        return 1
    fi
    if [[ -f "${marker}" && ! -s "${marker}" ]]; then
        return 1
    fi
    if [[ -e "${marker}" ]]; then
        return 0
    fi
    return 1
}

# Returns 0 if ALL markers exist (skip step), 1 otherwise.
should_skip_step_all() {
    if [[ "${FORCE}" == "1" ]]; then
        return 1
    fi
    for marker in "$@"; do
        if [[ ! -e "${marker}" ]]; then
            return 1
        fi
        if [[ -f "${marker}" && ! -s "${marker}" ]]; then
            return 1
        fi
    done
    return 0
}

# Echo skip message and return 0 to skip; return 1 to run the step.
skip_if_exists() {
    local marker="$1"
    local label="$2"
    if [[ "${FORCE}" == "1" ]]; then
        force_remove_marker "${marker}"
        return 1
    fi
    if should_skip_step "${marker}"; then
        echo "[SKIP] ${label}"
        echo "       exists: ${marker}  (set FORCE=1 to re-run)"
        return 0
    fi
    return 1
}

skip_if_all_exist() {
    local label="$1"
    shift
    if [[ "${FORCE}" == "1" ]]; then
        for marker in "$@"; do
            force_remove_marker "${marker}"
        done
        return 1
    fi
    if should_skip_step_all "$@"; then
        echo "[SKIP] ${label}"
        echo "       all outputs exist (set FORCE=1 to re-run):"
        for marker in "$@"; do
            echo "         - ${marker}"
        done
        return 0
    fi
    return 1
}

# IA-MIA query file: IA always builds a 30-question pool; NUM_QUERIES only
# selects the evaluation subset size.
# Optional: IA_QUERIES_FILE can override the resolved path for custom runs.
resolve_ia_queries_file() {
    local dataset="$1"
    local ia_results_dir="${2:-${RESULTS_DIR:-results/IA-MIA}}"

    if [[ -n "${IA_QUERIES_FILE:-}" ]]; then
        local path="${IA_QUERIES_FILE//\{dataset\}/${dataset}}"
        if [[ "${path}" == /* ]]; then
            printf '%s' "${path}"
        elif [[ "${path}" == IA-MIA/* ]]; then
            printf '%s' "${ia_results_dir%/*}/${path}"
        else
            printf '%s' "${ia_results_dir}/${path}"
        fi
        return
    fi
    printf '%s' "${ia_results_dir}/${dataset}/queries/queries_30.jsonl"
}


# Attack runners accept only the phase selector and --help. Datasets, modes,
# models, and defense parameters stay in .env.
PIPELINE_PHASE_ARG=""
parse_pipeline_phase_args() {
    local usage_fn="$1"
    shift

    if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
        "${usage_fn}"
        exit 0
    fi
    if [[ $# -ne 1 ]]; then
        echo "Error: choose exactly one phase: --online or --offline." >&2
        "${usage_fn}" >&2
        exit 2
    fi

    case "$1" in
        --online) PIPELINE_PHASE_ARG="online" ;;
        --offline) PIPELINE_PHASE_ARG="offline" ;;
        *)
            echo "Error: unknown option '$1'. Choose --online or --offline." >&2
            "${usage_fn}" >&2
            exit 2
            ;;
    esac
}

apply_pipeline_phase_arg() {
    if [[ -n "${PIPELINE_PHASE_ARG:-}" ]]; then
        PIPELINE_PHASE="${PIPELINE_PHASE_ARG}"
    fi
    init_pipeline_phase
}

# Backward-compatible alias for existing runner scripts.
parse_env_only_pipeline_args() {
    parse_pipeline_phase_args "$@"
}

split_pipeline_env_list() {
    local -n out_array="$1"
    local raw="${2:-}"
    local normalized="${raw//,/ }"
    read -r -a out_array <<< "${normalized}"
}

# Deduplicate a normalized mode array in place and run default first when it is
# selected. Defense modes share default retrieval artifacts, so this avoids a
# dependent mode creating them before default deletes/recreates them under FORCE=1.
dedupe_default_first_modes() {
    local -n mode_array="$1"
    local deduped=()
    local ordered=()
    local mode=""
    local selected_mode=""
    local already_selected=0

    for mode in "${mode_array[@]}"; do
        already_selected=0
        for selected_mode in "${deduped[@]}"; do
            if [[ "${selected_mode}" == "${mode}" ]]; then
                already_selected=1
                break
            fi
        done
        if [[ "${already_selected}" == "0" ]]; then
            deduped+=("${mode}")
        fi
    done

    for mode in "${deduped[@]}"; do
        if [[ "${mode}" == "default" ]]; then
            ordered+=("default")
            break
        fi
    done
    for mode in "${deduped[@]}"; do
        if [[ "${mode}" != "default" ]]; then
            ordered+=("${mode}")
        fi
    done

    mode_array=("${ordered[@]}")
}


normalize_pipeline_phase() {
    local raw="${1:-all}"
    case "${raw}" in
        ""|all|both|full) printf 'all' ;;
        online|openai) printf 'online' ;;
        offline|gpu|local) printf 'offline' ;;
        *)
            echo "Error: PIPELINE_PHASE must be one of: all, online, offline." >&2
            return 1
            ;;
    esac
}

init_pipeline_phase() {
    PIPELINE_PHASE="$(normalize_pipeline_phase "${PIPELINE_PHASE:-all}")"
    export PIPELINE_PHASE

    case "${PIPELINE_PHASE}" in
        online)
            export HF_HUB_OFFLINE="${PIPELINE_ONLINE_HF_HUB_OFFLINE:-0}"
            export TRANSFORMERS_OFFLINE="${PIPELINE_ONLINE_TRANSFORMERS_OFFLINE:-0}"
            ;;
        offline)
            export HF_HUB_OFFLINE="${PIPELINE_OFFLINE_HF_HUB_OFFLINE:-1}"
            export TRANSFORMERS_OFFLINE="${PIPELINE_OFFLINE_TRANSFORMERS_OFFLINE:-1}"
            ;;
    esac
}

pipeline_run_online_enabled() {
    [[ "${PIPELINE_PHASE:-all}" == "all" || "${PIPELINE_PHASE:-all}" == "online" ]]
}

pipeline_run_offline_enabled() {
    [[ "${PIPELINE_PHASE:-all}" == "all" || "${PIPELINE_PHASE:-all}" == "offline" ]]
}

print_pipeline_phase_summary() {
    echo "Pipeline phase: ${PIPELINE_PHASE:-all}"
    case "${PIPELINE_PHASE:-all}" in
        online) echo "Online phase: OpenAI/API stages and required prerequisites only." ;;
        offline) echo "Offline phase: local/GPU stages only; OpenAI/API stages are skipped." ;;
    esac
}

normalize_attack_mode() {
    local raw="${1:-default}"
    case "${raw}" in
        ""|none|default|paper) printf 'default' ;;
        dp) printf 'dp' ;;
        rerank) printf 'rerank' ;;
        prompt|prompt_instruction) printf 'prompt_instruction' ;;
        paraphrase|paraphrased) printf 'paraphrase' ;;
        *)
            echo "Error: unknown attack mode '${raw}'." >&2
            return 1
            ;;
    esac
}

validate_attack_mode_config() {
    local run_mode="$1"

    if [[ "${run_mode}" == "dp" ]]; then
        case "${DP_LEVEL:-output}" in
            retrieval|output|both) ;;
            *)
                echo "Error: DP_LEVEL must be one of: retrieval, output, both." >&2
                exit 2
                ;;
        esac
    fi

    if [[ "${run_mode}" == "rerank" ]]; then
        case "${RERANK_STRATEGY:-shuffle}" in
            reverse|shuffle|score_noise) ;;
            *)
                echo "Error: RERANK_STRATEGY must be one of: reverse, shuffle, score_noise." >&2
                exit 2
                ;;
        esac
    fi

    if [[ "${run_mode}" == "paraphrase" ]]; then
        case "${QUERY_PARAPHRASE_METHOD:-openai}" in
            openai|transformers) ;;
            *)
                echo "Error: QUERY_PARAPHRASE_METHOD must be one of: openai, transformers." >&2
                exit 2
                ;;
        esac
    fi
}

mode_dirname() {
    local base="$1"
    local run_mode="$2"
    case "${run_mode}" in
        default) printf '%s' "${base}" ;;
        paraphrase) printf '%s_paraphrased' "${base}" ;;
        dp|rerank|prompt_instruction) printf '%s_%s' "${base}" "${run_mode}" ;;
        *)
            echo "Error: cannot build path for unknown mode '${run_mode}'." >&2
            return 1
            ;;
    esac
}

build_defense_args() {
    local -n out_array="$1"
    local run_mode="$2"
    out_array=()

    case "${run_mode}" in
        default)
            ;;
        dp)
            out_array=(--defense dp --dp_epsilon "${DP_EPSILON:-0.1}" --dp_level "${DP_LEVEL:-output}")
            ;;
        rerank)
            out_array=(--defense rerank --rerank_strategy "${RERANK_STRATEGY:-shuffle}")
            ;;
        prompt_instruction)
            out_array=(--defense prompt_instruction)
            ;;
        paraphrase)
            out_array=(--defense paraphrase)
            ;;
        *)
            echo "Error: cannot build defense args for unknown mode '${run_mode}'." >&2
            return 1
            ;;
    esac
}
