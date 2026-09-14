#!/usr/bin/env python3
"""
Generate Mirabel Detection Evaluation Table
Compares Mirabel's ability to detect malicious queries from different attack methods.
Shows that Mirabel can detect all attacks (MBA, S2-MIA, MEntA, IA-MIA) but has high FPR on benign queries.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_embedding_model
from defense.mirabel import mirabel_detection


# Configuration
DATASETS = ["BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"]
DATASET_DISPLAY = {
    "BeIR_nfcorpus": "NFCorpus",
    "BeIR_scidocs": "SCIDOCS",
    "BeIR_trec-covid": "TREC-COVID"
}

ATTACKS = ["MEntA", "IA-MIA", "S2-MIA", "MBA", "DCMI"]
ATTACK_DISPLAY = {
    "MEntA": "MEntA (Ours)",
    "IA-MIA": "IA-MIA",
    "S2-MIA": "S²MIA",
    "MBA": "MBA",
    "DCMI": "DCMI"
}


def load_all_queries(dataset: str, data_base: str = "data") -> Tuple[List[Dict], List[Dict]]:
    """Load all attack and benign queries from queries_dataset.jsonl."""
    queries_path = str(Path(data_base) / dataset / "queries_dataset.jsonl")

    if not os.path.exists(queries_path):
        print(f"Error: Queries dataset not found: {queries_path}")
        return [], []

    all_queries = read_jsonl(queries_path)
    print(f"Loaded {len(all_queries)} total queries from {queries_path}")

    benign_queries = []
    attack_queries = []

    for query in all_queries:
        # Use is_attack field (set by create_queries_dataset.py)
        is_attack = query.get('is_attack', None)
        attack_type = query.get('attack_type', '').upper()

        if is_attack is True or (is_attack is None and attack_type not in ('', 'BENIGN')):
            attack_queries.append(query)
        else:
            benign_queries.append(query)

    print(f"  - {len(benign_queries)} BENIGN queries")
    print(f"  - {len(attack_queries)} ATTACK queries")

    return benign_queries, attack_queries


def load_corpus(dataset: str, data_base: str = "data") -> Tuple[List[Dict], List[str]]:
    """Load corpus (member + nonmember documents)."""
    member_path = str(Path(data_base) / dataset / "corpus_member.jsonl")
    nonmember_path = str(Path(data_base) / dataset / "corpus_nonmember.jsonl")

    corpus = []

    if os.path.exists(member_path):
        corpus_member = read_jsonl(member_path)
        corpus.extend(corpus_member)
        print(f"  - Member docs: {len(corpus_member)}")

    if os.path.exists(nonmember_path):
        corpus_nonmember = read_jsonl(nonmember_path)
        corpus.extend(corpus_nonmember)
        print(f"  - Nonmember docs: {len(corpus_nonmember)}")

    doc_ids = [doc['_id'] for doc in corpus]

    return corpus, doc_ids


def compute_embeddings(
    texts: List[str],
    model,
    batch_size: int = 32,
    description: str = "Computing embeddings"
) -> np.ndarray:
    """Compute embeddings for texts using the embedding model."""
    print(f"\n{description}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"✓ Computed {len(embeddings)} embeddings")
    return embeddings


def load_cached_results(results_path: Path) -> Tuple[List[Dict], set]:
    """Load cached results and return (results, set of processed query_ids)."""
    if not results_path.exists():
        return [], set()
    results = read_jsonl(str(results_path))
    processed_keys = {(r.get('attack_type', ''), r.get('query_id', '')) for r in results}
    print(f"  Loaded {len(results)} cached results ({len(processed_keys)} unique query keys)")
    return results, processed_keys


def apply_mirabel_detection_incremental(
    queries: List[Dict],
    corpus_embeddings: np.ndarray,
    doc_ids: List[str],
    true_label: str,
    embedding_model,
    results_path: Path,
    rho: float = 0.05,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Apply Mirabel detection incrementally — only processes queries not already cached.
    Appends new results to the cache file and returns the full result list.
    """
    existing_results, processed_keys = load_cached_results(results_path)

    # Filter to only missing queries. Include attack_type so different attacks
    # with the same query/document id are processed separately.
    missing_queries = [
        q for q in queries
        if (q.get('attack_type', ''), q.get('_id', q.get('query_id', ''))) not in processed_keys
    ]

    if not missing_queries:
        print(f"  ✓ All {len(existing_results)} {true_label} queries already processed, skipping")
        return existing_results

    print(f"  {len(existing_results)} cached, {len(missing_queries)} missing — processing new queries...")

    # Compute embeddings only for missing queries
    missing_texts = [q.get('text', q.get('query', '')) for q in missing_queries]
    missing_embeddings = compute_embeddings(
        missing_texts,
        embedding_model,
        batch_size=batch_size,
        description=f"Computing {true_label} embeddings for {len(missing_queries)} missing queries"
    )

    new_results = []
    for i, query in enumerate(tqdm(missing_queries, desc=f"Mirabel {true_label}")):
        mirabel_result = mirabel_detection(
            missing_embeddings[i],
            corpus_embeddings,
            doc_ids,
            rho=rho
        )

        query_id = query.get('_id', query.get('query_id', ''))
        result = {
            'query_id': query_id,
            'query_text': query.get('text', query.get('query', '')),
            'attack_type': query.get('attack_type', ''),
            'true_label': true_label,
            'predicted_label': 'ATTACK' if mirabel_result['detected_as_attack'] else 'BENIGN',
            'correct': (
                (true_label == 'ATTACK' and mirabel_result['detected_as_attack']) or
                (true_label == 'BENIGN' and not mirabel_result['detected_as_attack'])
            ),
            's_max': mirabel_result['mirabel_s_max'],
            'threshold': mirabel_result['mirabel_threshold'],
            'target_doc_id': mirabel_result['mirabel_target_doc_id']
        }
        new_results.append(result)

    # Append new results to cache
    all_results = existing_results + new_results
    write_jsonl(all_results, str(results_path))
    print(f"  ✓ Saved {len(all_results)} total results ({len(new_results)} new) to: {results_path}")

    return all_results


def calculate_metrics(results: List[Dict], attack_label: str) -> Dict:
    """Calculate detection metrics for pairwise evaluation (BENIGN vs specific ATTACK)."""

    benign_results = [r for r in results if r['true_label'] == 'BENIGN']
    attack_results = [r for r in results if r['true_label'] == 'ATTACK']

    tp = sum(1 for r in attack_results if r['predicted_label'] == 'ATTACK')
    fn = sum(1 for r in attack_results if r['predicted_label'] == 'BENIGN')
    tn = sum(1 for r in benign_results if r['predicted_label'] == 'BENIGN')
    fp = sum(1 for r in benign_results if r['predicted_label'] == 'ATTACK')

    total = len(results)
    total_attack = len(attack_results)
    total_benign = len(benign_results)

    recall = tp / total_attack if total_attack > 0 else 0
    fpr = fp / total_benign if total_benign > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        'attack': attack_label,
        'recall': recall,
        'fpr': fpr,
        'precision': precision,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'tpr': recall,
        'tnr': 1 - fpr,
        'true_positives': tp,
        'false_negatives': fn,
        'true_negatives': tn,
        'false_positives': fp,
        'total': total,
        'total_attack': total_attack,
        'total_benign': total_benign
    }


def process_dataset(
    dataset: str,
    output_dir: Path,
    embedding_model_name: str,
    use_gpu: bool,
    cache_dir: str,
    rho: float,
    data_base: str = "data"
):
    """Process a single dataset: apply Mirabel detection and compute metrics."""

    print(f"\n{'='*60}")
    print(f"Processing dataset: {DATASET_DISPLAY.get(dataset, dataset)}")
    print(f"{'='*60}")

    dataset_output_dir = output_dir / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    benign_queries, attack_queries = load_all_queries(dataset, data_base)

    if not benign_queries or not attack_queries:
        print(f"Error: Missing benign or attack queries for {dataset}, skipping...")
        return None

    # Load corpus
    print("\nLoading corpus...")
    corpus, doc_ids = load_corpus(dataset, data_base)
    print(f"✓ Loaded {len(corpus)} documents")

    # Load embedding model
    print(f"\nLoading embedding model: {embedding_model_name}...")
    embedding_model = load_embedding_model(
        embedding_model_name,
        use_gpu=use_gpu,
        cache_dir=cache_dir
    )
    print(f"✓ Embedding model loaded")

    # Compute or load corpus embeddings
    corpus_embeddings_path = dataset_output_dir / "corpus_embeddings.npy"

    if corpus_embeddings_path.exists():
        print("\nLoading cached corpus embeddings...")
        corpus_embeddings = np.load(str(corpus_embeddings_path))
        print(f"✓ Loaded {len(corpus_embeddings)} cached corpus embeddings")
    else:
        corpus_texts = [doc.get('text', doc.get('title', '')) for doc in corpus]
        corpus_embeddings = compute_embeddings(
            corpus_texts,
            embedding_model,
            batch_size=32,
            description="Computing corpus embeddings"
        )
        np.save(str(corpus_embeddings_path), corpus_embeddings)
        print(f"✓ Saved corpus embeddings to: {corpus_embeddings_path}")

    # Incremental Mirabel detection — benign
    print("\nProcessing BENIGN queries...")
    benign_results_path = dataset_output_dir / "benign_mirabel_results.jsonl"
    benign_results = apply_mirabel_detection_incremental(
        benign_queries,
        corpus_embeddings,
        doc_ids,
        'BENIGN',
        embedding_model,
        benign_results_path,
        rho=rho,
    )

    # Incremental Mirabel detection — attack
    print("\nProcessing ATTACK queries...")
    attack_results_path = dataset_output_dir / "attack_mirabel_results.jsonl"
    attack_results = apply_mirabel_detection_incremental(
        attack_queries,
        corpus_embeddings,
        doc_ids,
        'ATTACK',
        embedding_model,
        attack_results_path,
        rho=rho,
    )

    # Overall metrics
    combined_results = benign_results + attack_results
    overall_metrics = calculate_metrics(combined_results, "ALL_ATTACKS")

    print(f"\n{'='*60}")
    print(f"Overall Mirabel Detection Metrics:")
    print(f"{'='*60}")
    print(f"  Recall (Attack Detection): {overall_metrics['recall']:.3f}")
    print(f"  FPR: {overall_metrics['fpr']:.3f}")
    print(f"  TNR: {overall_metrics['tnr']:.3f}")
    print(f"  Precision: {overall_metrics['precision']:.3f}")
    print(f"  F1: {overall_metrics['f1_score']:.3f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.3f}")

    # Per-attack-type breakdown using attack_type field directly
    attack_breakdown = {}
    attack_queries_by_type = {}

    for result in attack_results:
        attack_type = result.get('attack_type', '')
        if not attack_type:
            continue
        attack_queries_by_type.setdefault(attack_type, []).append(result)

    if attack_queries_by_type:
        print(f"\n{'='*60}")
        print("Per-Attack Type Metrics:")
        print(f"{'='*60}")

        for attack_type, attack_type_results in attack_queries_by_type.items():
            combined_for_attack = benign_results + attack_type_results
            metrics = calculate_metrics(combined_for_attack, attack_type)
            attack_breakdown[attack_type] = metrics

            print(f"\n{attack_type} vs BENIGN:")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  FPR: {metrics['fpr']:.3f}")
            print(f"  TNR: {metrics['tnr']:.3f}")
            print(f"  F1: {metrics['f1_score']:.3f}")

    # Save detection statistics (always rewrite with latest metrics)
    stats = {
        'dataset': dataset,
        'embedding_model': embedding_model_name,
        'rho': rho,
        'overall_metrics': overall_metrics,
        'pairwise_evaluation': attack_breakdown
    }

    stats_path = dataset_output_dir / "detection_stats.json"
    write_json(stats, str(stats_path))
    print(f"\n✓ Saved detection statistics to: {stats_path}")

    return stats


def format_value(value):
    """Format a value for LaTeX display."""
    if value is None:
        return "-"
    return f"{value:.3f}"


def generate_latex_table(all_results: dict, output_path: str):
    """Generate LaTeX table showing Mirabel detection performance."""

    # Order: non-MEntA attacks first, MEntA always last
    ATTACK_ORDER = ["IA-MIA", "S2-MIA", "MBA", "DCMI", "MEntA"]

    latex_lines = []

    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Mirabel detector results on attack-generated queries versus benign queries.")
    latex_lines.append(r"We report recall on attack prompts, FPR on benign prompts, F1, and overall accuracy for")
    latex_lines.append(r"MEntA, IA-MIA, MBA, S$^2$MIA, and DCMI across NFCorpus, SCIDOCS, and TREC-COVID.")
    latex_lines.append(r"Higher recall indicates better attack detection, while lower FPR indicates fewer benign queries being incorrectly flagged.}")
    latex_lines.append(r"\label{tab:mirabel_detection}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\setlength{\tabcolsep}{6.5pt}")
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Dataset} & \textbf{Attack} & \textbf{Rec.} & \textbf{FPR} & \textbf{F1} & \textbf{Acc.} \\")
    latex_lines.append(r"\midrule")

    DATASET_LATEX = {
        "BeIR_nfcorpus":   "NFCorpus",
        "BeIR_scidocs":    "SCIDOCS",
        "BeIR_trec-covid": r"\shortstack{TREC-\\COVID}",
    }

    ATTACK_LATEX = {
        "IA-MIA": r"IA~\cite{naseh2025riddle}",
        "S2-MIA": r"S\textsuperscript{2}MIA~\cite{li2024generating}",
        "MBA":    r"MBA~\cite{liu2024mask}",
        "DCMI":   r"DCMI~\cite{dcmi2024}",
        "MEntA":  r"MEntA (Ours)",
    }

    for dataset_idx, dataset in enumerate(DATASETS):
        dataset_display = DATASET_LATEX.get(dataset, dataset)
        stats = all_results.get(dataset)
        breakdown = stats.get('pairwise_evaluation', {}) if stats else {}

        # Only include attacks that exist in the breakdown or all if no data
        attacks_present = [a for a in ATTACK_ORDER if a in breakdown] if breakdown else ATTACK_ORDER
        num_rows = len(attacks_present)

        for row_idx, attack in enumerate(attacks_present):
            metrics = breakdown.get(attack, {})

            recall_val = format_value(metrics.get('recall')) if metrics else "-"
            fpr_val    = format_value(metrics.get('fpr'))    if metrics else "-"
            f1_val     = format_value(metrics.get('f1_score')) if metrics else "-"
            acc_val    = format_value(metrics.get('accuracy')) if metrics else "-"

            attack_latex = ATTACK_LATEX.get(attack, attack)

            if row_idx == 0:
                dataset_col = f"\\multirow{{{num_rows}}}{{*}}{{{dataset_display}}}"
            else:
                dataset_col = ""

            latex_lines.append(
                f"{dataset_col} & {attack_latex} & {recall_val} & {fpr_val} & {f1_val} & {acc_val} \\\\"
            )

        if dataset_idx < len(DATASETS) - 1:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"\n✓ LaTeX table saved to: {output_path}")

 

def generate_json_results(all_results: dict, output_path: str):
    """Generate JSON file with detection results."""

    json_output = {
        "metadata": {
            "description": "Mirabel detection performance on attack vs benign queries",
            "datasets": [DATASET_DISPLAY.get(d, d) for d in DATASETS],
            "attacks": [ATTACK_DISPLAY[a] for a in ATTACKS]
        },
        "results": {},
        "summary": {}
    }

    for dataset in DATASETS:
        dataset_display = DATASET_DISPLAY.get(dataset, dataset)
        stats = all_results.get(dataset)

        if not stats:
            json_output["results"][dataset_display] = None
            continue

        json_output["results"][dataset_display] = {
            "Overall": stats.get('overall_metrics', {})
        }

        pairwise = stats.get('pairwise_evaluation', {})

        recalls, tnrs, fprs = [], [], []

        for attack in ATTACKS:
            attack_display = ATTACK_DISPLAY[attack]

            if attack in pairwise:
                metrics = pairwise[attack]
                json_output["results"][dataset_display][attack_display] = metrics

                if metrics.get('recall') is not None:
                    recalls.append(metrics['recall'])
                if metrics.get('tnr') is not None:
                    tnrs.append(metrics['tnr'])
                if metrics.get('fpr') is not None:
                    fprs.append(metrics['fpr'])
            else:
                json_output["results"][dataset_display][attack_display] = None

        json_output["summary"][dataset_display] = {
            "avg_recall": sum(recalls) / len(recalls) if recalls else None,
            "avg_tnr": sum(tnrs) / len(tnrs) if tnrs else None,
            "avg_fpr": sum(fprs) / len(fprs) if fprs else None,
            "high_detection": sum(1 for r in recalls if r > 0.8),
            "high_false_positive": sum(1 for f in fprs if f > 0.5)
        }

    write_json(json_output, output_path)
    print(f"✓ JSON results saved to: {output_path}")


def main():
    global DATASETS
    parser = argparse.ArgumentParser(description='Generate Mirabel detection evaluation table')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    parser.add_argument('--data_base', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results/mirabel_detection')
    parser.add_argument('--plot_dir', type=str, default='plot')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='../hf_cache/hub')
    parser.add_argument('--rho', type=float, default=0.05)
  
    args = parser.parse_args()

    DATASETS = args.datasets

    if args.cache_dir:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("MIRABEL DETECTION EVALUATION")
    print(f"{'#'*60}")
    print(f"Output directory: {output_dir}")
    print(f"Data base: {args.data_base}")
    print(f"Datasets: {DATASETS}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Rho (significance level): {args.rho}")
    print(f"{'#'*60}")

    all_results = {}

    for dataset in DATASETS:
        stats = process_dataset(
            dataset,
            output_dir,
            args.embedding_model,
            args.use_gpu,
            args.cache_dir,
            args.rho,
            args.data_base
        )
        if stats:
            all_results[dataset] = stats

    # Always rewrite output files with latest results
    latex_path = plot_dir / "mirabel_detection.tex"
    generate_latex_table(all_results, str(latex_path))

    json_path = plot_dir / "mirabel_detection_results.json"
    generate_json_results(all_results, str(json_path))

    print(f"\n{'='*100}")
    print("DETECTION PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    print(f"{'Dataset':<15} {'Type':<15} {'Recall':<10} {'TNR':<10} {'FPR':<10} {'F1':<10} {'Accuracy':<10}")
    print("-" * 100)

    for dataset in DATASETS:
        stats = all_results.get(dataset)
        if stats:
            overall = stats.get('overall_metrics', {})
            print(f"{DATASET_DISPLAY.get(dataset, dataset):<15} {'Overall':<15} {overall.get('recall', 0):<10.3f} {overall.get('tnr', 0):<10.3f} {overall.get('fpr', 0):<10.3f} {overall.get('f1_score', 0):<10.3f} {overall.get('accuracy', 0):<10.3f}")

            breakdown = stats.get('pairwise_evaluation', {})
            for attack in ATTACKS:
                if attack in breakdown:
                    metrics = breakdown[attack]
                    print(f"{'':15} {attack:<15} {metrics.get('recall', 0):<10.3f} {metrics.get('tnr', 0):<10.3f} {metrics.get('fpr', 0):<10.3f} {metrics.get('f1_score', 0):<10.3f} {metrics.get('accuracy', 0):<10.3f}")
            print()

    print(f"{'='*100}")
    print(f"\nOutput files:")
    print(f"  LaTeX table: {latex_path}")
    print(f"  JSON: {json_path}")
    print(f"  Detection results: {output_dir}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()