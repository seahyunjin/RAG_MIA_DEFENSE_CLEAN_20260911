# MEntA Reproduction Guide

This repository contains MEntA and the baseline attacks used in the paper:
`IA-MIA`, `DCMI`, `MBA`, and `S2-MIA`. The recommended workflow is to run the
Apptainer bash jobs in `scripts/`. Runtime choices live in `.env`; do not pass
experiment parameters to the bash scripts except `--online` or `--offline`.

> **Paper:** *Five Queries Are Enough: Query-Efficient and Surrogate-Free
> Membership Inference Attacks on RAG via Entailment.*

## Quick Start

1. Build the Apptainer image and configure `.env`.
2. Download models and prepare datasets.
3. **Run the smoke test first** to verify the pipeline on a small dataset.
4. Run the full evaluation once the smoke test passes.

See [Hardware Requirements](#hardware-requirements) before launching full
offline runs, especially for baseline methods.

## 1. Set Up Apptainer

Install Apptainer on the host following the official documentation:

<https://apptainer.org/docs/admin/main/installation.html>

**Build privileges.** `apptainer build` typically requires root or fakeroot on
many clusters. On shared systems, ask your administrators how to build images,
or build on a node where you have the required privileges and copy `menta.sif`
to the compute nodes.

**Nested or cloud environments.** If the host is already containerized (for
example, a cloud VM running inside another container), GPU passthrough and bind
mounts may need cluster-specific configuration. Test with the smoke test before
committing to a long full evaluation.

If your cluster needs Apptainer cache/tmp directories on project storage, copy
and edit the example setup script first:

```bash
cp example_apptainer_setup.sh apptainer_setup.sh
# edit apptainer_setup.sh to set APPTAINER_CACHEDIR and APPTAINER_TMPDIR
source apptainer_setup.sh
```

Build the container from the repository root:

```bash
apptainer build menta.sif Apptainer.def
```

Define a helper so each command uses the same `.env` values and HuggingFace
cache bind:

```bash
apptainer_run() {
    local hf_home
    hf_home="$(set -a; source .env; printf '%s' "$HF_HOME")"

    apptainer exec --nv \
        --env-file .env \
        --bind "$PWD:/workspace" \
        --bind "$hf_home:$hf_home" \
        menta.sif "$@"
}
export -f apptainer_run
```

If your cluster drops shell functions after `srun --pty bash`, paste this helper
again inside the allocation.

## 2. Configure `.env`

Create the local config:

```bash
cp .env.example .env
```

Edit `.env`. The most important variables are:

| Variable | Meaning |
| --- | --- |
| `OPENAI_API_KEY` | Required for OpenAI query generation, paraphrasing, summaries, and GPT detection. |
| `OPENAI_INFERENCE_MODE` | `batch` for OpenAI Batch API, `single` for sequential Chat Completions. |
| `HF_HOME` | Host HuggingFace cache root. Models are stored under `${HF_HOME}/hub`. |
| `HF_TOKEN` | Required for gated HuggingFace models. |
| `DATA_DIR` | Dataset location, normally `data`. |
| `RESULTS_DIR` | Output location, normally `results`. |
| `DATASETS` | Space- or comma-separated datasets, e.g. `BeIR_nfcorpus BeIR_scidocs`. |
| `RAG_GENERATOR_MODEL` | HuggingFace generator model for local RAG inference. |
| `RETRIEVER_MODEL` | Dense retriever used for FAISS indexing and retrieval. |
| `NUM_QUERIES` | MEntA query budget. IA-MIA still builds 30 queries and evaluates the selected subset. |
| `TOP_K` | Retrieval depth. |
| `FORCE` | `0` skips existing outputs; `1` removes expected outputs before rerunning. |

Attack modes are also configured in `.env`:

```bash
MENTA_MODES="default dp rerank prompt_instruction paraphrase similarity generic summary_only"
IA_MODES="default dp rerank prompt_instruction paraphrase"
DCMI_MODES="default dp rerank prompt_instruction paraphrase"
MBA_MODES="default dp rerank prompt_instruction paraphrase"
S2MIA_MODES="default dp rerank prompt_instruction paraphrase"
```

Multiple modes are run sequentially as separate attack passes. One attack run
never combines multiple modes.

Defense knobs:

```bash
DP_EPSILON=0.1
DP_LEVEL=output
RERANK_STRATEGY=shuffle
QUERY_PARAPHRASE_METHOD=openai
DETECTION_BENIGN_QUERIES=1000
DETECTION_ATTACK_QUERIES_PER_TYPE=1000
```

**Generator model choice.** The paper's main results use
`CohereLabs/c4ai-command-r7b-12-2024` (the `.env.example` default). DP-defense
experiments in the paper use `microsoft/phi-4` (Phi-4-14B), which requires
substantially more GPU memory. If Phi-4 does not fit on your GPUs, a smaller
generator such as `c4ai-command-r7b-12-2024` can still run, but DP-defense
attack performance may differ from the paper. See
[Hardware Requirements](#hardware-requirements).

## 3. Prepare Models and Data

Download/cache the HuggingFace models configured in `.env`. Run this on a
networked node:

```bash
apptainer_run bash scripts/setup_model.sh
```

Prepare datasets in two phases. Run online setup on a networked node:

```bash
apptainer_run bash scripts/setup_data.sh --online
```

Then build FAISS indices inside the GPU allocation:

```bash
srun --pty bash
# define apptainer_run again if needed
apptainer_run bash scripts/setup_data.sh --offline
```

`setup_model.sh` skips models already present in `${HF_HOME}/hub`.
`setup_data.sh --online` downloads/processes BEIR corpora, downloads benign
queries, and generates OpenAI summaries. `setup_data.sh --offline` builds FAISS
indices with `RETRIEVER_MODEL`. Both phases use `DATASETS`, `DATA_DIR`, and cache
values from `.env`.

## 4. Smoke Test

Run this **before** the full evaluation to verify functionality on a small copied
dataset. It uses the first entry in `DATASETS`, writes to `TEST_RESULTS_DIR`,
and uses `TEST_DATA_DIR` for the small data copy.

```bash
apptainer_run bash scripts/run_test_mode.sh --online
apptainer_run bash scripts/run_test_mode.sh --offline
```

Smoke mode differs from `run_all.sh` in these ways:

- uses only the first dataset in `DATASETS`
- copies `TEST_DOCS_PER_SPLIT` member and nonmember documents
- forces all attack modes for all attacks
- uses `TEST_RAG_GENERATOR_MODEL`, default `google/gemma-2b-it`
- maps `TEST_FORCE` to `FORCE`
- builds a small FAISS index in offline mode if needed
- detection uses `TEST_TOTAL_DOCS` benign queries and all generated smoke attack queries

The smoke-test offline stage typically fits in about **40 GB VRAM**. If the
smoke test passes, proceed to the full evaluation in the next section.

## 5. Run Full Evaluation

Use `run_all.sh` for normal experiments. It runs all attacks configured in
`.env`, then runs input detection defenses.

Run OpenAI/API stages first on a networked node:

```bash
apptainer_run bash scripts/run_all.sh --online
```

Then run local/GPU stages inside the GPU allocation:

```bash
srun --pty bash
# define apptainer_run again if needed
apptainer_run bash scripts/run_all.sh --offline
```

`run_all.sh` ignores `TEST_*` variables. It only follows the normal `.env`
configuration.

**IA-MIA online phase.** IA-MIA builds a 30-query pool per document and
generates shadow ground-truth answers. Its online phase is noticeably slower
than the other attacks.

## 6. Run Individual Jobs

Use these when you only want one attack or only input detection:

```bash
# MEntA
apptainer_run bash scripts/run_menta_pipeline.sh --online
apptainer_run bash scripts/run_menta_pipeline.sh --offline

# Baselines
apptainer_run bash scripts/run_ia_pipeline.sh --online
apptainer_run bash scripts/run_ia_pipeline.sh --offline

apptainer_run bash scripts/run_dcmi_pipeline.sh --online
apptainer_run bash scripts/run_dcmi_pipeline.sh --offline

apptainer_run bash scripts/run_mba_pipeline.sh --online
apptainer_run bash scripts/run_mba_pipeline.sh --offline

apptainer_run bash scripts/run_s2mia_pipeline.sh --online
apptainer_run bash scripts/run_s2mia_pipeline.sh --offline

# Input detection defenses
apptainer_run bash scripts/run_input_detection_defenses.sh --online
apptainer_run bash scripts/run_input_detection_defenses.sh --offline
```

Input detection expects attack query files to already exist. Its online phase
builds `queries_dataset.jsonl` and runs the GPT detector. Its offline phase runs
Mirabel.

## 7. Pipeline Chain

The scripts follow the same high-level chain:

```text
online phase:
  OpenAI/API stages and query prerequisites, then GPT detection
  examples: MEntA/IA OpenAI queries, DCMI perturbation, OpenAI paraphrasing,
  IA shadow answers

offline phase:
  FAISS retrieval, local/GPU RAG generation, scoring, evaluation, Mirabel detection
```

Attack-specific flow:

```text
MEntA:  queries -> retrieval -> RAG answers -> entailment/similarity -> evaluation
IA-MIA: query pool + shadow answers -> retrieval -> RAG answers -> evaluation
DCMI:   perturb docs -> yes/no queries -> retrieval -> RAG answers -> evaluation
MBA:    masked queries -> retrieval -> RAG answers -> evaluation
S2-MIA: half-doc queries -> retrieval -> RAG answers -> evaluation
```

Existing output files are skipped. Set `FORCE=1` in `.env` to rerun expected
outputs.

## 8. Hardware Requirements

VRAM needs depend on the stage and attack method. These are approximate values
observed during artifact evaluation on NVIDIA A800 GPUs:

| Stage / method | Approx. VRAM |
| --- | --- |
| Smoke test (offline) | ~40 GB |
| MEntA full evaluation (offline) | ~75 GB |
| Baselines: IA-MIA, DCMI, MBA, S2-MIA (offline) | >100 GB |
| DP defense with `microsoft/phi-4` | Higher than MEntA default; may exceed 80 GB |

**Practical guidance:**

- A single **40 GB GPU is enough for the smoke test**, but not for the full
  MEntA or baseline offline evaluations.
- **MEntA full offline evaluation** needs roughly **75 GB VRAM** with the
  default `c4ai-command-r7b-12-2024` generator.
- **Baseline offline evaluations** load additional models and typically need
  **more than 100 GB VRAM**. On 80 GB A800 GPUs they may fail immediately with
  out-of-memory errors at the start of the offline phase.
- For DP-defense experiments, the paper uses `microsoft/phi-4`. If that model
  does not fit, use a smaller `RAG_GENERATOR_MODEL` and expect DP-defense
  numbers to differ from the paper.

If you have limited GPU memory, run attacks individually (Section 6) and start
with MEntA offline evaluation, which has the lowest memory footprint among full
experiments.

## 9. Checkpointing and Incremental Saves

Online OpenAI stages now support **checkpointing** and **incremental output writes**:

- After each OpenAI batch (or each single-mode request) completes, results are
  merged into the target output file immediately.
- Checkpoint state is stored next to batch inputs as
  `{base_filename}_checkpoint.json` under the step's output directory.
- If a run is interrupted, rerun the same `--online` command. Completed batches
  are skipped, in-flight batch jobs are resumed when possible, and only missing
  documents or queries are regenerated.

Steps with resume/checkpoint support include:

- MEntA query generation (`MEntA/generate_queries.py`)
- IA-MIA query pool and shadow ground-truth generation
- Dataset summarization (`utils/summarize.py`)
- DCMI corpus perturbation (`DCMI/perturb_docs.py`)
- Query paraphrase defense (`defense/query_paraphrase.py`)
- GPT input detection (`defense/gpt_detection.py`)

To force a clean rerun for a step, delete its output file and the matching
`*_checkpoint.json` file, or set `FORCE=1` in `.env` where the pipeline script
supports it.

## 10. Other Notes

**Apptainer deployment.** Building the image may require elevated privileges
(see Section 1). In shared or nested-container environments, coordinate with
your cluster administrators on image build, GPU passthrough, and bind mounts.

## 11. Interactive Container Shell

For debugging only:

```bash
set -a; source .env; set +a

apptainer shell --nv \
    --env-file .env \
    --bind "$PWD:/workspace" \
    --bind "$HF_HOME:$HF_HOME" \
    menta.sif

cd /workspace
```

Then inspect script help:

```bash
bash scripts/run_all.sh --help
python MEntA/generate_queries.py --help
```

The bash jobs are still the recommended interface because they encode the
online/offline split, skip logic, and mode expansion.

## Recommended Citation

Please cite our work if you use this codebase, data, or experimental results in your research.

```bibtex
@misc{nguyen2026queriesenoughqueryefficientsurrogatefree,
      title={Five Queries Are Enough: Query-Efficient and Surrogate-Free Membership Inference Attacks on RAG via Entailment}, 
      author={Nguyen Linh Bao Nguyen and Wanlun Ma and Viet Vo and Alsharif Abuadbba and Minghong Fang and Jun Zhang and Yang Xiang},
      year={2026},
      eprint={2605.24312},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.24312}, 
}
```
