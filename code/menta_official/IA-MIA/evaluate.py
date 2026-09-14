import os
import sys
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse
import re
import random
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl


def normalize_answer(answer: str) -> str:
    """
    Extract and normalize answer to only Yes, No, or I don't know.

    Returns:
    - "Yes" if answer contains Yes
    - "No" if answer contains No
    - "I don't know" if answer contains variations of "don't know"
    - "I don't know" if no clear Yes/No is found
    """
    answer_lower = answer.lower().strip()

    # Check for "I don't know" variations first
    dont_know_patterns = [
        "i don't know",
        "i do not know",
        "don't know",
        "do not know",
        "i'm not sure",
        "i am not sure",
        "not sure",
        "cannot find",
        "can't find",
        "unable to find"
    ]

    for pattern in dont_know_patterns:
        if pattern in answer_lower:
            return "I don't know"

    # Check for Yes/No
    if answer_lower.startswith("yes") or " yes" in answer_lower or answer_lower == "yes":
        return "Yes"

    if answer_lower.startswith("no") or " no" in answer_lower or answer_lower == "no":
        return "No"

    # If no clear Yes/No found, return "I don't know"
    return "I don't know"


def extract_model_name(rag_answers_path: str) -> str:
    """
    Extract model name from rag_answers_path.
    E.g., .../answers_CohereLabs--c4ai-command-r7b-12-2024.jsonl -> CohereLabs--c4ai-command-r7b-12-2024
    """
    basename = os.path.basename(rag_answers_path)
    match = re.search(r'answers_([^.\/]+)\.jsonl', basename)
    if match:
        return match.group(1)
    return "meta-llama--Llama-3.1-8B-Instruct" 


def extract_doc_id_from_query_id(query_id: str) -> str:
    """
    Extract document ID from query ID.
    E.g., q_MED-4930_v0 -> MED-4930
    """
    # Pattern: q_{doc_id}_v{variation_index}
    match = re.match(r'q_(.+)_v\d+$', query_id)
    if match:
        return match.group(1)
    return query_id


def generate_mia_score_csv(
    rag_answers_path: str,
    gt_answers_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    result_csv_path: str,
    y: int = 1,
    num_questions: int = None
):
    print(f"\n{'#'*60}")
    print(f"Generating MIA scores")
    print(f"RAG answers: {rag_answers_path}")
    print(f"GT answers: {gt_answers_path}")
    print(f"Output CSV: {result_csv_path}")
    if num_questions:
        print(f"Sampling: {num_questions} questions per document")
    print(f"{'#'*60}\n")

    if os.path.exists(result_csv_path):
        print(f"CSV file {result_csv_path} already exists. Loading existing file.")
        return

    # Read files - these are flat lists with one entry per question
    rag_data = read_jsonl(rag_answers_path)
    gt_data = read_jsonl(gt_answers_path)

    print(f"Loaded {len(rag_data)} questions from RAG answers")
    print(f"Loaded {len(gt_data)} questions from GT answers")

    # Load corpus to determine membership
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    member_docs = set(doc['_id'] for doc in corpus_member)
    nonmember_docs = set(doc['_id'] for doc in corpus_nonmember)

    # Create mappings by query ID
    # RAG answers use 'rag_answer' field, GT answers use 'ground_truth_answer' field
    rag_map = {}
    for item in rag_data:
        qid = item.get('_id')
        if qid:
            # Handle different possible answer field names
            answer = item.get('rag_answer') or item.get('answer') or ''
            rag_map[qid] = {
                'answer': answer,
                'target_doc_id': item.get('target_doc_id'),
                'membership': item.get('_membership')
            }

    gt_map = {}
    for item in gt_data:
        qid = item.get('_id')
        if qid:
            # Handle different possible answer field names
            answer = item.get('ground_truth_answer') or item.get('answer') or ''
            gt_map[qid] = {
                'answer': answer,
                'target_doc_id': item.get('target_doc_id'),
                'membership': item.get('_membership')
            }

    # Find common query IDs
    common_qids = set(rag_map.keys()) & set(gt_map.keys())

    print(f"\nQuery overlap:")
    print(f"  Questions only in RAG: {len(rag_map.keys() - gt_map.keys())}")
    print(f"  Questions only in GT: {len(gt_map.keys() - rag_map.keys())}")
    print(f"  Questions in both: {len(common_qids)}")

    if len(common_qids) == 0:
        raise ValueError("No common questions found between RAG and GT answers!")

    # Group questions by document ID
    doc_to_questions = defaultdict(list)
    for qid in common_qids:
        doc_id = rag_map[qid].get('target_doc_id') or extract_doc_id_from_query_id(qid)
        doc_to_questions[doc_id].append(qid)

    print(f"  Documents with questions: {len(doc_to_questions)}")

    # Prepare CSV output
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)

    processed_count = 0
    skipped_count = 0

    # Statistics for answer normalization
    answer_stats = {
        'rag': {'Yes': 0, 'No': 0, "I don't know": 0},
        'gt': {'Yes': 0, 'No': 0, "I don't know": 0}
    }

    with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'type', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for doc_id, question_ids in sorted(doc_to_questions.items()):
            # Determine membership from corpus
            if doc_id in member_docs:
                membership = 'member'
            elif doc_id in nonmember_docs:
                membership = 'nonmember'
            else:
                # Try to get from the data itself
                membership = rag_map[question_ids[0]].get('membership', 'unknown')

            # Determine which questions to sample
            if num_questions is not None and num_questions < len(question_ids):
                selected_qids = sorted(random.sample(question_ids, num_questions))
            else:
                selected_qids = sorted(question_ids)

            score = 0
            num = 0

            for qid in selected_qids:
                rag_answer_raw = rag_map[qid]['answer']
                gt_answer_raw = gt_map[qid]['answer']

                rag_answer = normalize_answer(rag_answer_raw)
                gt_answer = normalize_answer(gt_answer_raw)

                answer_stats['rag'][rag_answer] = answer_stats['rag'].get(rag_answer, 0) + 1
                answer_stats['gt'][gt_answer] = answer_stats['gt'].get(gt_answer, 0) + 1

                # Skip if GT answer is not Yes or No
                if gt_answer not in ["Yes", "No"]:
                    continue

                if rag_answer == gt_answer:
                    score += 1
                elif rag_answer not in ["Yes", "No"]:
                    score -= y

                num += 1

            if num == 0:
                skipped_count += 1
                continue

            score_normalized = round(score / num, 6)

            writer.writerow({
                'id': doc_id,
                'type': membership,
                'score': f"{score_normalized:.6f}"
            })

            processed_count += 1

    print(f"\nAnswer normalization statistics:")
    print(f"  RAG answers:")
    for answer_type, count in answer_stats['rag'].items():
        print(f"    {answer_type}: {count}")
    print(f"  GT answers:")
    for answer_type, count in answer_stats['gt'].items():
        print(f"    {answer_type}: {count}")

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count} documents")
    print(f"  Skipped: {skipped_count} documents")
    print(f"  CSV saved to {result_csv_path}")


def calculate_tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fprs: list) -> dict:
    """
    Calculate TPR at specific FPR values.
    
    Args:
        y_true: Ground truth labels (1 = member, 0 = nonmember)
        y_scores: Predicted scores
        target_fprs: List of target FPR values (e.g., [0.005, 0.01, 0.05])
    
    Returns:
        Dictionary with TPR at each target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    tpr_at_fpr = {}
    for target_fpr in target_fprs:
        # Find the largest FPR that is <= target_fpr
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            # Get the index with the highest TPR among valid FPRs
            best_idx = valid_indices[np.argmax(tpr[valid_indices])]
            actual_fpr = float(fpr[best_idx])
            actual_tpr = float(tpr[best_idx])
        else:
            actual_fpr = 0.0
            actual_tpr = 0.0
        
        tpr_at_fpr[f'tpr_at_fpr_{target_fpr}'] = {
            'target_fpr': target_fpr,
            'actual_fpr': actual_fpr,
            'tpr': actual_tpr
        }
    
    return tpr_at_fpr


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, method: str = 'accuracy') -> dict:
    """
    Find the best threshold based on the specified method.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
        method: 'accuracy', 'f1', or 'youden'
    
    Returns:
        Dictionary with best threshold and associated metrics
    """
    # Get unique thresholds from scores
    unique_scores = np.unique(y_scores)
    # Add boundary thresholds
    thresholds = np.concatenate([[unique_scores.min() - 0.01], unique_scores, [unique_scores.max() + 0.01]])
    
    best_accuracy = 0.0
    best_acc_threshold = 0.5
    best_f1 = 0.0
    best_f1_threshold = 0.5
    best_youden = -1.0
    best_youden_threshold = 0.5
    
    all_metrics = []
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        youden = tpr - fpr
        
        all_metrics.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'f1': f1,
            'youden': youden,
            'precision': precision,
            'recall': recall,
            'tpr': tpr,
            'fpr': fpr
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_threshold = thresh
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = thresh
        
        if youden > best_youden:
            best_youden = youden
            best_youden_threshold = thresh
    
    return {
        'best_threshold_accuracy': float(best_acc_threshold),
        'best_accuracy': float(best_accuracy),
        'best_threshold_f1': float(best_f1_threshold),
        'best_f1': float(best_f1),
        'best_threshold_youden': float(best_youden_threshold),
        'best_youden': float(best_youden),
        'all_metrics': all_metrics
    }


def evaluate_mia_detection(
    csv_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_dir: str,
    threshold_method: str = 'accuracy',
    model_name: str = "unknown-model"
):
    print(f"\n{'#'*60}")
    print(f"Evaluating IA-MIA Membership Detection")
    print(f"Input CSV: {csv_path}")
    print(f"Threshold Method: {threshold_method}")
    print(f"{'#'*60}\n")

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} documents from CSV")

    # Load corpora to get ground truth
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)

    member_docs = set(doc['_id'] for doc in corpus_member)
    nonmember_docs = set(doc['_id'] for doc in corpus_nonmember)

    docs_in_csv = set(df['id'].values)
    gt_members_in_csv = member_docs & docs_in_csv
    gt_nonmembers_in_csv = nonmember_docs & docs_in_csv

    print(f"\nGround truth statistics (full corpus):")
    print(f"  Total member docs: {len(member_docs)}")
    print(f"  Total nonmember docs: {len(nonmember_docs)}")

    print(f"\nEvaluation set (docs in answer files):")
    print(f"  Docs in CSV: {len(docs_in_csv)}")
    print(f"  Members (in CSV): {len(gt_members_in_csv)}")
    print(f"  Nonmembers (in CSV): {len(gt_nonmembers_in_csv)}")

    # Filter to only docs in corpus
    all_corpus_ids = member_docs | nonmember_docs
    docs_not_in_corpus = docs_in_csv - all_corpus_ids
    if docs_not_in_corpus:
        print(f"\n⚠️  Warning: {len(docs_not_in_corpus)} docs in CSV are not in corpus!")
        print(f"  These will be excluded from evaluation.")
        df = df[df['id'].isin(all_corpus_ids)]
        docs_in_csv = set(df['id'].values)
        gt_members_in_csv = member_docs & docs_in_csv
        gt_nonmembers_in_csv = nonmember_docs & docs_in_csv

    # Label: 1 = member, 0 = nonmember
    df['label'] = df['id'].apply(lambda x: 1 if x in member_docs else 0)
    
    y_true = df['label'].values
    y_scores = df['score'].astype(float).values
    doc_ids = df['id'].values

    # Calculate AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # Calculate TPR at specific FPR values
    target_fprs = [0.005, 0.01, 0.05]
    tpr_at_fpr = calculate_tpr_at_fpr(y_true, y_scores, target_fprs)
    
    # Find best thresholds
    threshold_results = find_best_threshold(y_true, y_scores, threshold_method)
    
    # Select threshold based on method
    if threshold_method == 'accuracy':
        threshold = threshold_results['best_threshold_accuracy']
        print(f"\nUsing best accuracy threshold: {threshold:.4f} (Accuracy={threshold_results['best_accuracy']:.4f})")
    elif threshold_method == 'f1':
        threshold = threshold_results['best_threshold_f1']
        print(f"\nUsing best F1 threshold: {threshold:.4f} (F1={threshold_results['best_f1']:.4f})")
    elif threshold_method == 'youden':
        threshold = threshold_results['best_threshold_youden']
        print(f"\nUsing best Youden threshold: {threshold:.4f} (J={threshold_results['best_youden']:.4f})")
    else:
        threshold = 0.5
        print(f"\nUsing default threshold: {threshold}")

    # Predict as member if score > threshold (higher score = more likely member)
    df['predicted'] = (df['score'].astype(float) > threshold).astype(int)
    y_pred = df['predicted'].values

    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    tp_docs = set(doc_ids[tp_mask])
    fp_docs = set(doc_ids[fp_mask])
    fn_docs = set(doc_ids[fn_mask])
    tn_docs = set(doc_ids[tn_mask])

    precision = len(tp_docs) / (len(tp_docs) + len(fp_docs)) if (len(tp_docs) + len(fp_docs)) > 0 else 0
    recall = len(tp_docs) / len(gt_members_in_csv) if len(gt_members_in_csv) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (len(tp_docs) + len(tn_docs)) / len(docs_in_csv) if len(docs_in_csv) > 0 else 0
    
    current_fpr = len(fp_docs) / (len(fp_docs) + len(tn_docs)) if (len(fp_docs) + len(tn_docs)) > 0 else 0
    current_tpr = recall

    print(f"\n{'='*60}")
    print("Membership Detection Metrics:")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.4f} (best {threshold_method.upper()})")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nTPR at Low FPR:")
    for key, value in tpr_at_fpr.items():
        print(f"  TPR@FPR={value['target_fpr']}: {value['tpr']:.4f} (actual FPR={value['actual_fpr']:.4f})")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (correctly predicted members): {len(tp_docs)}")
    print(f"  False Positives (nonmembers predicted as members): {len(fp_docs)}")
    print(f"  False Negatives (members predicted as nonmembers): {len(fn_docs)}")
    print(f"  True Negatives (correctly predicted nonmembers): {len(tn_docs)}")

    # Plot ROC Curve
    fpr_curve, tpr_curve, thresholds_curve = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, linewidth=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title('IA-MIA Membership Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.plot(current_fpr, current_tpr, 'ro', markersize=10,
             label=f'Threshold={threshold:.3f} (FPR={current_fpr:.3f}, TPR={current_tpr:.3f})')
    plt.legend(loc="lower right")

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    roc_plot_path = os.path.join(output_dir, f'roc_curve_{model_name}.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to {roc_plot_path}")
    plt.close()

    results = {
        'membership_detection': {
            'evaluation_scope': {
                'total_member_docs': len(member_docs),
                'total_nonmember_docs': len(nonmember_docs),
                'evaluated_docs': len(docs_in_csv),
                'evaluated_members': len(gt_members_in_csv),
                'evaluated_nonmembers': len(gt_nonmembers_in_csv)
            },
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'auc_roc': auc,
                'tpr': current_tpr,
                'fpr': current_fpr
            },
            'tpr_at_fpr': tpr_at_fpr,
            'threshold_selection': {
                'method': threshold_method,
                'selected_threshold': threshold,
                'best_accuracy_threshold': threshold_results['best_threshold_accuracy'],
                'best_accuracy_score': threshold_results['best_accuracy'],
                'best_f1_threshold': threshold_results['best_threshold_f1'],
                'best_f1_score': threshold_results['best_f1'],
                'best_youden_threshold': threshold_results['best_threshold_youden'],
                'best_youden_score': threshold_results['best_youden']
            },
            'counts': {
                'true_positives': len(tp_docs),
                'false_positives': len(fp_docs),
                'false_negatives': len(fn_docs),
                'true_negatives': len(tn_docs)
            },
            'note': f'Documents with score > {threshold:.4f} are predicted as MEMBER (threshold selected by best {threshold_method})'
        }
    }

    # Save results JSON
    results_json_path = os.path.join(output_dir, f'evaluation_{model_name}.json')
    write_json(results, results_json_path)
    print(f"Results saved to {results_json_path}")

    return results


def generate_comparison_table(results_dict: dict, output_dir: str, model_name: str):
    """Generate comparison table for different question counts."""
    print(f"\n{'='*60}")
    print("Comparison Table: AUC-ROC vs Number of Questions")
    print(f"{'='*60}")

    # Create DataFrame
    data = []
    for num_q, result in sorted(results_dict.items()):
        metrics = result['membership_detection']['metrics']
        counts = result['membership_detection']['counts']
        tpr_at_fpr = result['membership_detection'].get('tpr_at_fpr', {})
        threshold_info = result['membership_detection'].get('threshold_selection', {})
        
        row = {
            'Questions': num_q,
            'AUC-ROC': metrics['auc_roc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1_score'],
            'Accuracy': metrics['accuracy'],
            'Threshold': threshold_info.get('selected_threshold', 0.5),
            'TP': counts['true_positives'],
            'FP': counts['false_positives'],
            'FN': counts['false_negatives'],
            'TN': counts['true_negatives']
        }
        
        # Add TPR@FPR metrics
        for key, value in tpr_at_fpr.items():
            fpr_val = value['target_fpr']
            row[f'TPR@FPR={fpr_val}'] = value['tpr']
        
        data.append(row)

    df = pd.DataFrame(data)

    # Print table
    print("\n" + df.to_string(index=False))

    # Save CSV
    table_path = os.path.join(output_dir, f'comparison_table_{model_name}.csv')
    df.to_csv(table_path, index=False)
    print(f"\nComparison table saved to {table_path}")

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # AUC-ROC plot
    axes[0, 0].plot(df['Questions'], df['AUC-ROC'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Number of Questions')
    axes[0, 0].set_ylabel('AUC-ROC')
    axes[0, 0].set_title('AUC-ROC vs Number of Questions')
    axes[0, 0].grid(True)

    # Precision/Recall/F1 plot
    axes[0, 1].plot(df['Questions'], df['Precision'], marker='o', label='Precision', linewidth=2)
    axes[0, 1].plot(df['Questions'], df['Recall'], marker='s', label='Recall', linewidth=2)
    axes[0, 1].plot(df['Questions'], df['F1'], marker='^', label='F1', linewidth=2)
    axes[0, 1].set_xlabel('Number of Questions')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision/Recall/F1 vs Number of Questions')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy plot
    axes[0, 2].plot(df['Questions'], df['Accuracy'], marker='o', linewidth=2, color='green')
    axes[0, 2].set_xlabel('Number of Questions')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Accuracy vs Number of Questions')
    axes[0, 2].grid(True)

    # TPR@FPR plot
    tpr_columns = [col for col in df.columns if col.startswith('TPR@FPR=')]
    if tpr_columns:
        for col in tpr_columns:
            axes[1, 0].plot(df['Questions'], df[col], marker='o', linewidth=2, label=col)
        axes[1, 0].set_xlabel('Number of Questions')
        axes[1, 0].set_ylabel('TPR')
        axes[1, 0].set_title('TPR at Low FPR vs Number of Questions')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No TPR@FPR data', ha='center', va='center')
        axes[1, 0].set_title('TPR at Low FPR')

    # Threshold plot
    axes[1, 1].plot(df['Questions'], df['Threshold'], marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Number of Questions')
    axes[1, 1].set_ylabel('Threshold')
    axes[1, 1].set_title('Best Threshold vs Number of Questions')
    axes[1, 1].grid(True)

    # Confusion matrix counts
    x = np.arange(len(df))
    width = 0.2
    axes[1, 2].bar(x - 1.5*width, df['TP'], width, label='TP')
    axes[1, 2].bar(x - 0.5*width, df['FP'], width, label='FP')
    axes[1, 2].bar(x + 0.5*width, df['FN'], width, label='FN')
    axes[1, 2].bar(x + 1.5*width, df['TN'], width, label='TN')
    axes[1, 2].set_xlabel('Number of Questions')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Confusion Matrix Counts')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(df['Questions'])
    axes[1, 2].legend()
    axes[1, 2].grid(True, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'comparison_plots_{model_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to {plot_path}")
    plt.close()

    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate IA-MIA membership detection')
    parser.add_argument(
        '--rag_answers',
        type=str,
        default="results/IA-MIA/BeIR_nfcorpus/answers/topk3/answers_google-gemma-2b-it.jsonl",
        help='Path to RAG answers file (e.g., .../answers/topk3/answers_CohereLabs-c4ai-command-r7b-12-2024.jsonl)'
    )
    parser.add_argument( 
        '--gt_answers',
        type=str,
        default="results/IA-MIA/BeIR_nfcorpus/gt/gt.jsonl",
        help='Path to ground truth answers file (e.g., .../gt/gt.jsonl)'
    )
    parser.add_argument(
        '--corpus_member',
        type=str,
        default="data/BeIR_nfcorpus/corpus_member.jsonl",
        help='Path to member corpus file'
    )
    parser.add_argument(
        '--corpus_nonmember',
        type=str,
        default="data/BeIR_nfcorpus/corpus_nonmember.jsonl",
        help='Path to nonmember corpus file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        default='accuracy',
        choices=['accuracy', 'f1', 'youden'],
        help='Method for selecting best threshold (default: accuracy)'
    )
    parser.add_argument(
        '--y',
        type=int,
        default=1,
        help='Penalty for uncertain answers in score calculation'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=None,
        help='Evaluate only this many questions per document. By default, evaluates 5, 10, 20, and 30.'
    )

    args = parser.parse_args()
    if args.num_queries is not None and args.num_queries <= 0:
        parser.error('--num_queries must be positive')

    # Extract model name from RAG answers path
    model_name = extract_model_name(args.rag_answers)

    # Auto-generate output directory if not specified
    # From: .../results/IA-MIA/BeIR_nfcorpus/answers/topk3/answers_model.jsonl
    # To:   .../results/IA-MIA/BeIR_nfcorpus/evaluation/topk3/{model}/
    if args.output_dir is None:
        rag_path = Path(args.rag_answers)
        # Navigate up to get dataset dir and topk
        topk_dir = rag_path.parent.name  # e.g., "topk3"
        dataset_dir = rag_path.parent.parent.parent  # e.g., ".../BeIR_nfcorpus"
        args.output_dir = str(dataset_dir / "evaluation" / topk_dir / model_name)

    print(f"\n{'='*60}")
    print(f"IA-MIA Evaluation")
    print(f"{'='*60}")
    print(f"RAG answers: {args.rag_answers}")
    print(f"GT answers: {args.gt_answers}")
    print(f"Corpus member: {args.corpus_member}")
    print(f"Corpus nonmember: {args.corpus_nonmember}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {model_name}")
    print(f"Threshold method: {args.threshold_method}")
    if args.num_queries is not None:
        print(f"Questions per document: {args.num_queries}")
    print(f"{'='*60}")

    # The IA pipeline always generates and answers a 30-query pool; this option
    # lets the bash runner evaluate the subset size configured by NUM_QUERIES.
    question_counts = [args.num_queries] if args.num_queries is not None else [5, 10, 20, 30]
    all_results = {}

    for num_q in question_counts:
        print(f"\n{'#'*80}")
        print(f"# Evaluating with {num_q} questions per document")
        print(f"{'#'*80}")

        # Generate MIA scores
        csv_path = os.path.join(args.output_dir, f'mia_scores_q{num_q}.csv')
        generate_mia_score_csv(
            args.rag_answers,
            args.gt_answers,
            args.corpus_member,
            args.corpus_nonmember,
            csv_path,
            args.y,
            num_q
        )

        # Evaluate membership detection
        result = evaluate_mia_detection(
            csv_path,
            args.corpus_member,
            args.corpus_nonmember,
            args.output_dir,
            args.threshold_method,
            f"q{num_q}"
        )

        all_results[num_q] = result

    # Generate comparison table
    generate_comparison_table(all_results, args.output_dir, model_name)

    print("\n" + "="*60)
    print("IA-MIA evaluation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()