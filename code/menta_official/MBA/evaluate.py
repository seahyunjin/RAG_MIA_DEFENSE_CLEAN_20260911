#!/usr/bin/env python3
"""
Mask-based MIA Evaluation Script
Evaluates membership inference attack using mask prediction accuracy.
Based on: "Mask-based Membership Inference Attacks for Retrieval-Augmented Generation" (WWW 2025)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import re
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    confusion_matrix,
    average_precision_score
)
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl


def parse_mask_predictions(generated_output: str, num_masks: int) -> Dict[int, str]:
    """
    Parse mask predictions from generated output.
    Expected format: "[Mask_0]: answer_0\n[Mask_1]: answer_1\n..."
    Returns dict mapping mask index to predicted value.
    """
    predictions = {}
    
    # Try to extract predictions using regex
    pattern = r'\[Mask_(\d+)\]:\s*([^\n]+)'
    matches = re.findall(pattern, generated_output, re.IGNORECASE)
    
    for match in matches:
        mask_idx = int(match[0])
        prediction = match[1].strip()
        predictions[mask_idx] = prediction
    
    return predictions


def compute_mask_accuracy(mask_values: List[str], predictions: Dict[int, str]) -> float:
    """
    Compute exact mask prediction accuracy (primary metric in paper).
    Returns the fraction of correctly predicted masks.
    """
    if not mask_values:
        return 0.0
    
    correct = 0
    total = len(mask_values)
    
    for i, true_value in enumerate(mask_values):
        if i in predictions:
            pred_value = predictions[i]
            # Case-insensitive comparison, strip whitespace
            if true_value.lower().strip() == pred_value.lower().strip():
                correct += 1
    
    return correct / total if total > 0 else 0.0


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


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Find the best threshold based on accuracy, F1, and Youden's J statistic.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
    
    Returns:
        Dictionary with best threshold and associated metrics for each method
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
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
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
        'best_youden': float(best_youden)
    }


def evaluate_mask_mia(
    rag_outputs_path: str,
    output_path: str,
    gamma: float = None,  # If None, use best accuracy threshold
    threshold_method: str = 'accuracy'  # 'accuracy', 'f1', or 'youden'
):
    """
    Evaluate Mask-based MIA attack performance following the paper's methodology.
    
    Key metric (Section 5.3 & 5.5):
    - Mask prediction accuracy as membership indicator
    - Binary classification using threshold γ
    - ROC AUC as primary evaluation metric
    
    Args:
        rag_outputs_path: Path to RAG outputs with mask predictions
        output_path: Path to save evaluation results
        gamma: Threshold for membership inference (if None, auto-select based on threshold_method)
        threshold_method: Method for selecting best threshold ('accuracy', 'f1', or 'youden')
    """
    print(f"\n{'#'*60}")
    print(f"MASK-BASED MIA EVALUATION")
    print(f"Based on: Mask-based Membership Inference Attacks (WWW 2025)")
    print(f"RAG outputs: {rag_outputs_path}")
    print(f"Threshold Method: {threshold_method}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    # if os.path.exists(output_path):
    #     print(f"{output_path} already exists. Skipping.")
    #     return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load RAG outputs
    rag_outputs = read_jsonl(rag_outputs_path)
    print(f"Loaded {len(rag_outputs)} RAG outputs")
    
    # Compute mask prediction accuracy for each sample
    print("\nComputing mask prediction accuracies...")
    results = []
    
    for item in tqdm(rag_outputs, desc="Evaluating"):
        doc_id = item['_id']
        mask_values = item['mask_values']
        generated_output = item['generated_output']
        membership_label = item['membership_label']
        
        # Parse mask predictions
        predictions = parse_mask_predictions(generated_output, len(mask_values))
        
        # Exact mask accuracy (primary metric from paper)
        exact_accuracy = compute_mask_accuracy(mask_values, predictions)
        
        result = {
            '_id': doc_id,
            'membership_label': membership_label,
            'mask_prediction_accuracy': exact_accuracy,  # Renamed for clarity
            'mask_values': mask_values,
            'predictions': predictions,
            'generated_output': generated_output,
            'num_masks': len(mask_values),
            'num_predicted': len(predictions),
            'num_correct': int(exact_accuracy * len(mask_values))
        }
        results.append(result)
    
    # Extract labels and scores for metrics computation
    y_true = np.array([r['membership_label'] for r in results])
    accuracy_scores = np.array([r['mask_prediction_accuracy'] for r in results])
    
    print("\n" + "="*60)
    print("MEMBERSHIP INFERENCE ATTACK METRICS")
    print("="*60)
    
    # 1. ROC AUC (Primary metric in paper - Section 5.5)
    auc_score = roc_auc_score(y_true, accuracy_scores)
    print(f"\n🔑 ROC AUC (Primary Metric): {auc_score:.4f}")
    
    # 2. Find best thresholds using different methods
    threshold_results = find_best_threshold(y_true, accuracy_scores)
    
    print(f"\nThreshold Selection Results:")
    print(f"  Best Accuracy Threshold: {threshold_results['best_threshold_accuracy']:.4f} (Acc={threshold_results['best_accuracy']:.4f})")
    print(f"  Best F1 Threshold: {threshold_results['best_threshold_f1']:.4f} (F1={threshold_results['best_f1']:.4f})")
    print(f"  Best Youden Threshold: {threshold_results['best_threshold_youden']:.4f} (J={threshold_results['best_youden']:.4f})")
    
    # Select threshold based on method (or use provided gamma)
    if gamma is not None:
        selected_threshold = gamma
        print(f"\nUsing provided γ={gamma} for binary classification")
    elif threshold_method == 'accuracy':
        selected_threshold = threshold_results['best_threshold_accuracy']
        print(f"\nUsing best accuracy threshold: {selected_threshold:.4f}")
    elif threshold_method == 'f1':
        selected_threshold = threshold_results['best_threshold_f1']
        print(f"\nUsing best F1 threshold: {selected_threshold:.4f}")
    elif threshold_method == 'youden':
        selected_threshold = threshold_results['best_threshold_youden']
        print(f"\nUsing best Youden threshold: {selected_threshold:.4f}")
    else:
        selected_threshold = 0.5
        print(f"\nUsing default threshold: {selected_threshold}")
    
    # 3. Calculate TPR at specific FPR values
    target_fprs = [0.005, 0.01, 0.05]
    tpr_at_fpr = calculate_tpr_at_fpr(y_true, accuracy_scores, target_fprs)
    
    print(f"\nTPR at Low FPR:")
    for key, value in tpr_at_fpr.items(): 
        print(f"  TPR@FPR={value['target_fpr']}: {value['tpr']:.4f} (actual FPR={value['actual_fpr']:.4f})")
    
    # 4. Binary predictions using selected threshold
    # If mask_prediction_accuracy >= threshold, classify as member
    y_pred = (accuracy_scores >= selected_threshold).astype(int)
    
    # 5. Classification metrics (Table 1 in paper)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nClassification Metrics (threshold={selected_threshold:.4f}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (TPR): {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # 6. Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nTrue Positive Rate (TPR): {tpr_val:.4f}")
    print(f"False Positive Rate (FPR): {fpr_val:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # 7. Average precision
    avg_precision = average_precision_score(y_true, accuracy_scores)
    print(f"\nAverage Precision: {avg_precision:.4f}")
    
    # 8. Statistics by membership (Figure 1(d) in paper)
    member_scores = accuracy_scores[y_true == 1]
    nonmember_scores = accuracy_scores[y_true == 0]
    
    print(f"\nMask Prediction Accuracy Statistics:")
    print(f"  Members:     Mean={member_scores.mean():.4f}, Std={member_scores.std():.4f}")
    print(f"  Non-members: Mean={nonmember_scores.mean():.4f}, Std={nonmember_scores.std():.4f}")
    print(f"  Separation:  {member_scores.mean() - nonmember_scores.mean():.4f}")
    
    # 9. Retrieval recall (unique metric for RAG MIA - Section 5.3)
    # Check if target document was retrieved (if hit_rank exists in data)
    retrieval_recall = None
    if 'hit_rank' in rag_outputs[0] or 'is_hit' in rag_outputs[0]:
        retrieval_hits = [item.get('is_hit', False) or item.get('hit_rank', -1) > 0 
                         for item in rag_outputs if item['membership_label'] == 1]
        retrieval_recall = sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0.0
        print(f"\nRetrieval Recall: {retrieval_recall:.4f}")
        print(f"  (Fraction of member docs successfully retrieved in top-K)")
    
    # Prepare output data (following paper's structure)
    output_data = {
        'metadata': {
            'rag_outputs_file': rag_outputs_path,
            'evaluation_method': 'Mask-Based MIA (WWW 2025)',
            'threshold_method': threshold_method,
            'selected_threshold': float(selected_threshold),
            'num_samples': len(results),
            'num_members': int(y_true.sum()),
            'num_nonmembers': int((1 - y_true).sum())
        },
        'metrics': {
            'roc_auc': float(auc_score),  # Primary metric
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall_tpr': float(recall),
            'f1_score': float(f1),
            'fpr': float(fpr_val),
            'average_precision': float(avg_precision),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        },
        'tpr_at_fpr': tpr_at_fpr,
        'threshold_selection': {
            'method': threshold_method,
            'selected_threshold': float(selected_threshold),
            'best_accuracy_threshold': threshold_results['best_threshold_accuracy'],
            'best_accuracy_score': threshold_results['best_accuracy'],
            'best_f1_threshold': threshold_results['best_threshold_f1'],
            'best_f1_score': threshold_results['best_f1'],
            'best_youden_threshold': threshold_results['best_threshold_youden'],
            'best_youden_score': threshold_results['best_youden']
        },
        'score_statistics': {
            'members': {
                'mean_accuracy': float(member_scores.mean()),
                'std_accuracy': float(member_scores.std()),
                'min_accuracy': float(member_scores.min()),
                'max_accuracy': float(member_scores.max())
            },
            'nonmembers': {
                'mean_accuracy': float(nonmember_scores.mean()),
                'std_accuracy': float(nonmember_scores.std()),
                'min_accuracy': float(nonmember_scores.min()),
                'max_accuracy': float(nonmember_scores.max())
            },
            'separation': float(member_scores.mean() - nonmember_scores.mean())
        },
        'detailed_results': results
    }
    
    # Add retrieval recall if available
    if retrieval_recall is not None:
        output_data['metrics']['retrieval_recall'] = float(retrieval_recall)
    
    # Save results
    write_json(output_data, output_path)
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Mask-based MIA attack performance (WWW 2025)'
    )
    parser.add_argument(
        '--rag_outputs_file',
        type=str,
        default="results/MBA/BeIR_nfcorpus/rag/topk3/CohereLabs--c4ai-command-r7b-12-2024/rag_outputs.jsonl",
        help='Path to RAG outputs file with mask predictions'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="results/MBA/BeIR_nfcorpus/results/topk3/CohereLabs--c4ai-command-r7b-12-2024/evaluation_metrics.json",
        help='Path to output evaluation metrics file'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=None,
        help='Membership threshold (if not specified, auto-select based on threshold_method)'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        default='accuracy',
        choices=['accuracy', 'f1', 'youden'],
        help='Method for selecting best threshold (default: accuracy)'
    )
    
    args = parser.parse_args()
    
    evaluate_mask_mia(
        args.rag_outputs_file,
        args.output_file,
        gamma=args.gamma,
        threshold_method=args.threshold_method
    )
    
    print("\n" + "="*60)
    print("Mask-based MIA evaluation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()