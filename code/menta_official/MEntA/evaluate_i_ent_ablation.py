import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json
from MEntA.evaluate import (
    detect_idk_queries,
    evaluate_with_threshold,
    calculate_auc_with_roc,
)


DEFAULT_DATASETS = ["BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"]


def deserialize_entailment_matrix(entailment_matrix_serializable: Dict[str, dict]) -> Dict[Tuple[str, str, int], dict]:
    """Convert serialized entailment keys back to tuples."""
    entailment_matrix = {}
    for key_str, value in entailment_matrix_serializable.items():
        parts = key_str.split("||")
        if len(parts) != 3:
            continue
        query_id, doc_id, claim_idx_str = parts
        entailment_matrix[(query_id, doc_id, int(claim_idx_str))] = value
    return entailment_matrix


def extract_target_doc_from_query_id(query_id: str) -> str:
    """Extract target document ID from MEntA query_id."""
    parts = query_id.split("_")
    if len(parts) >= 2:
        if parts[0] == "q" and parts[-1].startswith("v"):
            return "_".join(parts[1:-1])
        if parts[0] == "q":
            return "_".join(parts[1:])
    return query_id


def calculate_mia_scores_min_entailed(
    queries: List[dict],
    query_to_claims: Dict[str, List[str]],
    idk_query_ids: Set[str],
    entailment_matrix: Dict[Tuple[str, str, int], dict],
    min_entailed_claims: int,
    lambda_idk: float = 1.0,
) -> Dict[str, dict]:
    """Calculate MIA scores where ent=1 only if at least min_entailed_claims are entailed."""
    doc_to_queries = {}
    for query_data in queries:
        query_id = query_data["query_id"]
        target_doc = extract_target_doc_from_query_id(query_id)
        if not target_doc:
            continue
        doc_to_queries.setdefault(target_doc, []).append(query_data)

    doc_scores = {}
    for doc_id, doc_queries in doc_to_queries.items():
        total_score = 0.0
        query_details = []

        for query_data in doc_queries:
            query_id = query_data["query_id"]
            is_idk = query_id in idk_query_ids

            if is_idk:
                query_contribution = -lambda_idk
                query_details.append(
                    {
                        "query_id": query_id,
                        "is_idk": True,
                        "ent": 0,
                        "entailed_claim_count": 0,
                        "min_entailed_claims": min_entailed_claims,
                        "query_contribution": query_contribution,
                    }
                )
                total_score += query_contribution
                continue

            claims = query_to_claims.get(query_id, [])
            entailed_claim_count = 0
            claim_scores = []

            for claim_idx, claim in enumerate(claims):
                entailment_info = entailment_matrix.get((query_id, doc_id, claim_idx))
                if entailment_info:
                    ent_prob = entailment_info["entailment_prob"]
                    neu_prob = entailment_info["neutral_prob"]
                    con_prob = entailment_info["contradiction_prob"]
                    is_entailment_highest = ent_prob >= neu_prob and ent_prob >= con_prob
                    if is_entailment_highest:
                        entailed_claim_count += 1
                else:
                    ent_prob = 0.0
                    neu_prob = 0.0
                    con_prob = 0.0
                    is_entailment_highest = False

                claim_scores.append(
                    {
                        "claim": claim,
                        "ent_prob": ent_prob,
                        "neu_prob": neu_prob,
                        "con_prob": con_prob,
                        "is_entailment_highest": is_entailment_highest,
                    }
                )

            ent = 1 if entailed_claim_count >= min_entailed_claims else 0
            query_contribution = ent
            total_score += query_contribution
            query_details.append(
                {
                    "query_id": query_id,
                    "is_idk": False,
                    "ent": ent,
                    "entailed_claim_count": entailed_claim_count,
                    "min_entailed_claims": min_entailed_claims,
                    "query_contribution": query_contribution,
                    "claim_scores": claim_scores,
                }
            )

        n_variations = len(doc_queries)
        doc_scores[doc_id] = {
            "mia_score": total_score / n_variations if n_variations > 0 else 0.0,
            "n_variations": n_variations,
            "total_score": total_score,
            "query_details": query_details,
        }

    return doc_scores


def resolve_default_entailment_file(results_base: Path, dataset: str, top_k: int, model_dir: str, num_queries: int) -> Path:
    """Build the default entailment path for one dataset."""
    return (
        results_base
        / dataset
        / "entailment"
        / "target_summary"
        / f"topk{top_k}"
        / model_dir
        / f"entailment_{num_queries}v.json"
    )


DATASET_DISPLAY = {
    "BeIR_nfcorpus": "NFCorpus",
    "BeIR_scidocs": "SCIDOCS",
    "BeIR_trec-covid": "TREC-COVID",
}


def _short_dataset_label(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def _set_tight_y_axis(ax, values: List[float], tick_step: float = 0.04):
    """Y-limits flush to data ticks — no extra space above or below."""
    y_min, y_max = min(values), max(values)
    tick_lo = np.floor(y_min / tick_step) * tick_step
    tick_hi = np.ceil(y_max / tick_step) * tick_step
    ticks = np.arange(tick_lo, tick_hi + tick_step / 2, tick_step)
    ax.set_ylim(tick_lo, tick_hi)
    ax.set_yticks(ticks)
    ax.margins(y=0)


PLOT_FIGSIZE = (6, 2.8)  # height: 2.5=shorter, 3.5=taller
PLOT_DPI = 300
Y_TICK_STEP = 0.04


def plot_auc_lines(dataset_to_auc: Dict[str, Dict[int, float]], output_path: Path):
    """Plot AUC vs n_entailed for each dataset."""
    label_fs = 14
    tick_fs = 12
    legend_fs = 11

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for dataset, auc_by_n in dataset_to_auc.items():
        n_values = sorted(auc_by_n.keys())
        auc_values = [auc_by_n[n] for n in n_values]
        ax.plot(
            n_values,
            auc_values,
            marker="o",
            markersize=7,
            linewidth=2,
            label=_short_dataset_label(dataset),
        )

    all_aucs = [v for auc_by_n in dataset_to_auc.values() for v in auc_by_n.values()]
    _set_tight_y_axis(ax, all_aucs, tick_step=Y_TICK_STEP)

    ax.set_xlabel("Minimum Entailed Atomic Sentences ($n$)", fontsize=label_fs)
    ax.set_ylabel("AUC", fontsize=label_fs)
    ax.set_xticks(sorted({n for auc_by_n in dataset_to_auc.values() for n in auc_by_n}))
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        fontsize=legend_fs,
        framealpha=0.9,
    )

    fig.subplots_adjust(bottom=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_auc_bars(dataset_to_auc: Dict[str, Dict[int, float]], output_path: Path):
    """Plot grouped bar chart of AUC by dataset and n."""
    datasets = list(dataset_to_auc.keys())
    all_n_values = sorted({n for auc_by_n in dataset_to_auc.values() for n in auc_by_n})
    x = np.arange(len(datasets))
    width = 0.8 / max(len(all_n_values), 1)

    plt.figure(figsize=(12, 6))
    for idx, n_value in enumerate(all_n_values):
        auc_values = [dataset_to_auc[dataset].get(n_value, np.nan) for dataset in datasets]
        offset = (idx - (len(all_n_values) - 1) / 2) * width
        plt.bar(x + offset, auc_values, width=width, label=f"n={n_value}")

    plt.xticks(x, datasets)
    plt.ylabel("AUC")
    plt.title("I_ent Ablation: AUC Comparison by Dataset")
    plt.legend(title="Min entailed")
    plt.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_dataset_ablation(
    entailment_file: Path,
    corpus_member_path: Path,
    corpus_nonmember_path: Path,
    n_values: List[int],
    lambda_idk: float,
) -> Dict[str, dict]:
    """Run the I_ent ablation for a single dataset."""
    entailment_data = read_json(str(entailment_file))
    entailment_matrix = deserialize_entailment_matrix(entailment_data["entailment_matrix"])
    query_to_claims = entailment_data["query_to_claims"]
    idk_detection_details = entailment_data.get("idk_detection_details", {})
    queries = [{"query_id": qid} for qid in query_to_claims.keys()]

    corpus_member = read_jsonl(str(corpus_member_path))
    corpus_nonmember = read_jsonl(str(corpus_nonmember_path))
    member_docs = {doc["_id"] for doc in corpus_member}
    nonmember_docs = {doc["_id"] for doc in corpus_nonmember}

    idk_query_ids, _ = detect_idk_queries(idk_detection_details)

    results = {}
    for n_value in n_values:
        doc_scores = calculate_mia_scores_min_entailed(
            queries,
            query_to_claims,
            idk_query_ids,
            entailment_matrix,
            min_entailed_claims=n_value,
            lambda_idk=lambda_idk,
        )
        all_target_docs = set(doc_scores.keys())
        auc_data = calculate_auc_with_roc(doc_scores, member_docs, all_target_docs)
        eval_result = evaluate_with_threshold(
            doc_scores,
            member_docs,
            all_target_docs,
            auc_data["best_threshold_accuracy"],
        )

        results[str(n_value)] = {
            "auc": auc_data["auc"],
            "best_threshold_accuracy": auc_data["best_threshold_accuracy"],
            "best_accuracy": auc_data["best_accuracy"],
            "best_threshold_f1": auc_data["best_threshold_f1"],
            "best_f1": auc_data["best_f1"],
            "evaluation_at_best_accuracy": eval_result,
            "num_docs": len(all_target_docs),
            "num_queries": len(queries),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation for I_ent definition: ent=1 when at least n atomic sentences are entailed"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--results_base",
        type=str,
        default="results/MEntA",
        help="Base MEntA results directory",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="microsoft--phi-4",
        help="Model directory name under results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k setting to evaluate",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=5,
        help="Number of query variations in the entailment files",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Values of n for the I_ent ablation",
    )
    parser.add_argument(
        "--lambda_idk",
        type=float,
        default=1.0,
        help="Penalty weight for IDK responses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/MEntA/i_ent_ablation/target_summary/topk5/microsoft--phi-4",
        help="Directory to save ablation outputs",
    )

    args = parser.parse_args()

    results_base = Path(args.results_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "metadata": {
            "datasets": args.datasets,
            "model_dir": args.model_dir,
            "top_k": args.top_k,
            "num_queries": args.num_queries,
            "n_values": args.n_values,
            "lambda_idk": args.lambda_idk,
        },
        "datasets": {},
        "missing_files": [],
    }

    dataset_to_auc = {}

    for dataset in args.datasets:
        entailment_file = resolve_default_entailment_file(
            results_base,
            dataset,
            args.top_k,
            args.model_dir,
            args.num_queries,
        )
        corpus_member_path = Path(f"data/{dataset}/corpus_member.jsonl")
        corpus_nonmember_path = Path(f"data/{dataset}/corpus_nonmember.jsonl")

        if not entailment_file.exists():
            all_results["missing_files"].append(str(entailment_file))
            print(f"Missing entailment file, skipping: {entailment_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"Entailment file: {entailment_file}")
        print(f"{'=' * 60}")

        dataset_results = evaluate_dataset_ablation(
            entailment_file,
            corpus_member_path,
            corpus_nonmember_path,
            args.n_values,
            args.lambda_idk,
        )
        all_results["datasets"][dataset] = {
            "entailment_file": str(entailment_file),
            "results_by_n": dataset_results,
        }
        dataset_to_auc[dataset] = {
            int(n_value): metrics["auc"] for n_value, metrics in dataset_results.items()
        }

    summary_path = output_dir / "i_ent_ablation_summary.json"
    write_json(all_results, str(summary_path))

    if dataset_to_auc:
        plot_auc_lines(dataset_to_auc, output_dir / "auc_vs_n_by_dataset.png")
        plot_auc_bars(dataset_to_auc, output_dir / "auc_grouped_bar.png")
        print(f"\nSaved plots to: {output_dir}")

    print(f"Saved ablation summary to: {summary_path}")
    if all_results["missing_files"]:
        print("\nMissing files:")
        for missing in all_results["missing_files"]:
            print(f"  - {missing}")


if __name__ == "__main__":
    main()
