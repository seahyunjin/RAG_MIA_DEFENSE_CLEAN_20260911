#!/usr/bin/env python3
"""
S²MIA Evaluation Script
Evaluates membership inference attack using semantic similarity, BLEU score, and perplexity from logprobs.
Computes AUC, accuracy, precision, recall, F1, TPR, FPR, and other metrics.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from tqdm import tqdm
import sys
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_embedding_model


def compute_semantic_similarity(
    text1: str,
    text2: str,
    model: SentenceTransformer
) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.
    Returns similarity score in [0, 1] range (normalized from [-1, 1]).
    """
    emb1 = model.encode(text1, convert_to_numpy=True, normalize_embeddings=True)
    emb2 = model.encode(text2, convert_to_numpy=True, normalize_embeddings=True)
    
    # Cosine similarity (already normalized embeddings)
    similarity = np.dot(emb1, emb2)
    
    # Convert from [-1, 1] to [0, 1] range
    similarity = (similarity + 1) / 2
    
    return float(similarity)


def compute_bleu_score(reference: str, candidate: str) -> float:
    """
    Compute BLEU score between reference and candidate texts.
    Uses smoothing to handle edge cases.
    """
    # Tokenize
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # BLEU score with smoothing
    smoothing = SmoothingFunction()
    score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=smoothing.method1
    )
    
    return float(score)


def compute_perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    Compute perplexity from a list of log probabilities.
    Perplexity = exp(-mean(logprobs))
    """
    if not logprobs or len(logprobs) == 0:
        return None
    
    # Average log probability
    avg_logprob = np.mean(logprobs)
    
    # Perplexity = exp(-avg_logprob)
    perplexity = np.exp(-avg_logprob)
    
    return float(perplexity)


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


def evaluate_s2mia(
    rag_outputs_path: str,
    output_path: str,
    embedding_model_name: str,
    use_bleu: bool = True,
    use_perplexity: bool = True,
    use_gpu: bool = False,
    cache_dir: str = None,
    batch_size: int = 32,
    threshold_method: str = 'accuracy'
):
    """
    Evaluate S²MIA attack performance.
    
    Computes:
    - Semantic similarity scores between generated_output and knowledge_text
    - Optional BLEU scores
    - Perplexity from saved logprobs
    - MIA metrics: AUC, accuracy, precision, recall, F1, TPR, FPR
    
    Args:
        rag_outputs_path: Path to RAG outputs file
        output_path: Path to save evaluation results
        embedding_model_name: Name of embedding model for semantic similarity
        use_bleu: Whether to compute BLEU scores
        use_perplexity: Whether to compute perplexity from logprobs
        use_gpu: Whether to use GPU
        cache_dir: Cache directory for models
        batch_size: Batch size for processing
        threshold_method: Method for selecting best threshold ('accuracy', 'f1', or 'youden')
    """
    print(f"\n{'#'*60}")
    print(f"S²MIA EVALUATION")
    print(f"RAG outputs: {rag_outputs_path}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"Use BLEU: {use_bleu}")
    print(f"Use Perplexity: {use_perplexity}")
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
    
    # Load embedding model for semantic similarity
    print(f"\nLoading embedding model: {embedding_model_name}")
    embedding_model = load_embedding_model(embedding_model_name, use_gpu, cache_dir)
    
    # Compute scores for each sample
    print("\nComputing scores...")
    results = []
    
    for item in tqdm(rag_outputs, desc="Evaluating"):
        doc_id = item['_id']
        knowledge_text = item['knowledge_text']
        generated_output = item['generated_output']
        membership_label = item['membership_label']
        logprobs = item.get('logprobs', None)
        
        # Semantic similarity score (main S²MIA metric)
        sem_sim_score = compute_semantic_similarity(
            knowledge_text,
            generated_output,
            embedding_model
        )
        
        # BLEU score (optional)
        bleu_score = None
        if use_bleu and knowledge_text and generated_output:
            bleu_score = compute_bleu_score(knowledge_text, generated_output)
        
        # Perplexity from logprobs
        ppl_score = None
        if use_perplexity and logprobs:
            ppl_score = compute_perplexity_from_logprobs(logprobs)
        
        result = {
            '_id': doc_id,
            'membership_label': membership_label,
            'semantic_similarity': sem_sim_score,
            'bleu_score': bleu_score,
            'perplexity': ppl_score,
            'knowledge_text': knowledge_text,
            'generated_output': generated_output
        }
        results.append(result)
    
    # Extract labels and scores for metrics computation
    y_true = np.array([r['membership_label'] for r in results])
    sem_sim_scores = np.array([r['semantic_similarity'] for r in results])
    
    # Compute MIA metrics using semantic similarity as attack score
    print("\n" + "="*60)
    print("MEMBERSHIP INFERENCE ATTACK METRICS")
    print("="*60)
    
    # AUC-ROC
    auc_score = roc_auc_score(y_true, sem_sim_scores)
    print(f"\nAUC-ROC: {auc_score:.4f}")
    
    # Find best thresholds using different methods
    threshold_results = find_best_threshold(y_true, sem_sim_scores)
    
    print(f"\nThreshold Selection Results:")
    print(f"  Best Accuracy Threshold: {threshold_results['best_threshold_accuracy']:.4f} (Acc={threshold_results['best_accuracy']:.4f})")
    print(f"  Best F1 Threshold: {threshold_results['best_threshold_f1']:.4f} (F1={threshold_results['best_f1']:.4f})")
    print(f"  Best Youden Threshold: {threshold_results['best_threshold_youden']:.4f} (J={threshold_results['best_youden']:.4f})")
    
    # Select threshold based on method
    if threshold_method == 'accuracy':
        optimal_threshold = threshold_results['best_threshold_accuracy']
        print(f"\nUsing best accuracy threshold: {optimal_threshold:.4f}")
    elif threshold_method == 'f1':
        optimal_threshold = threshold_results['best_threshold_f1']
        print(f"\nUsing best F1 threshold: {optimal_threshold:.4f}")
    elif threshold_method == 'youden':
        optimal_threshold = threshold_results['best_threshold_youden']
        print(f"\nUsing best Youden threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        print(f"\nUsing default threshold: {optimal_threshold}")
    
    # Calculate TPR at specific FPR values
    target_fprs = [0.005, 0.01, 0.05]
    tpr_at_fpr = calculate_tpr_at_fpr(y_true, sem_sim_scores, target_fprs)
    
    print(f"\nTPR at Low FPR:")
    for key, value in tpr_at_fpr.items():
        print(f"  TPR@FPR={value['target_fpr']}: {value['tpr']:.4f} (actual FPR={value['actual_fpr']:.4f})")
    
    # Predictions using selected threshold
    y_pred = (sem_sim_scores >= optimal_threshold).astype(int)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # TPR and FPR
    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nTrue Positive Rate (TPR): {tpr_val:.4f}")
    print(f"False Positive Rate (FPR): {fpr_val:.4f}")
    
    # Confusion matrix details
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Average precision (area under precision-recall curve)
    avg_precision = average_precision_score(y_true, sem_sim_scores)
    print(f"\nAverage Precision: {avg_precision:.4f}")
    
    # Additional statistics by membership
    member_scores = sem_sim_scores[y_true == 1]
    nonmember_scores = sem_sim_scores[y_true == 0]
    
    print(f"\nSemantic Similarity Statistics:")
    print(f"  Members - Mean: {member_scores.mean():.4f}, Std: {member_scores.std():.4f}")
    print(f"  Non-members - Mean: {nonmember_scores.mean():.4f}, Std: {nonmember_scores.std():.4f}")
    
    # BLEU score statistics if available
    bleu_scores = None
    member_bleu = None
    nonmember_bleu = None
    if use_bleu:
        bleu_scores = np.array([r['bleu_score'] for r in results if r['bleu_score'] is not None])
        if len(bleu_scores) > 0:
            member_bleu = np.array([r['bleu_score'] for r in results if r['membership_label'] == 1 and r['bleu_score'] is not None])
            nonmember_bleu = np.array([r['bleu_score'] for r in results if r['membership_label'] == 0 and r['bleu_score'] is not None])
            
            print(f"\nBLEU Score Statistics:")
            print(f"  Members - Mean: {member_bleu.mean():.4f}, Std: {member_bleu.std():.4f}")
            print(f"  Non-members - Mean: {nonmember_bleu.mean():.4f}, Std: {nonmember_bleu.std():.4f}")
    
    # Perplexity statistics if available
    ppl_scores = None
    member_ppl = None
    nonmember_ppl = None
    if use_perplexity:
        ppl_scores = np.array([r['perplexity'] for r in results if r['perplexity'] is not None])
        if len(ppl_scores) > 0:
            member_ppl = np.array([r['perplexity'] for r in results if r['membership_label'] == 1 and r['perplexity'] is not None])
            nonmember_ppl = np.array([r['perplexity'] for r in results if r['membership_label'] == 0 and r['perplexity'] is not None])
            
            print(f"\nPerplexity Statistics:")
            print(f"  Members - Mean: {member_ppl.mean():.4f}, Std: {member_ppl.std():.4f}")
            print(f"  Non-members - Mean: {nonmember_ppl.mean():.4f}, Std: {nonmember_ppl.std():.4f}")
    
    # Prepare output data
    output_data = {
        'metadata': {
            'rag_outputs_file': rag_outputs_path,
            'embedding_model': embedding_model_name,
            'use_bleu': use_bleu,
            'use_perplexity': use_perplexity,
            'threshold_method': threshold_method,
            'num_samples': len(results),
            'num_members': int(y_true.sum()),
            'num_nonmembers': int((1 - y_true).sum())
        },
        'metrics': {
            'auc_roc': float(auc_score),
            'optimal_threshold': float(optimal_threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tpr': float(tpr_val),
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
            'selected_threshold': float(optimal_threshold),
            'best_accuracy_threshold': threshold_results['best_threshold_accuracy'],
            'best_accuracy_score': threshold_results['best_accuracy'],
            'best_f1_threshold': threshold_results['best_threshold_f1'],
            'best_f1_score': threshold_results['best_f1'],
            'best_youden_threshold': threshold_results['best_threshold_youden'],
            'best_youden_score': threshold_results['best_youden']
        },
        'score_statistics': {
            'semantic_similarity': {
                'members_mean': float(member_scores.mean()),
                'members_std': float(member_scores.std()),
                'nonmembers_mean': float(nonmember_scores.mean()),
                'nonmembers_std': float(nonmember_scores.std())
            }
        },
        'detailed_results': results
    }
    
    # Add BLEU statistics if available
    if use_bleu and bleu_scores is not None and len(bleu_scores) > 0:
        output_data['score_statistics']['bleu'] = {
            'members_mean': float(member_bleu.mean()),
            'members_std': float(member_bleu.std()),
            'nonmembers_mean': float(nonmember_bleu.mean()),
            'nonmembers_std': float(nonmember_bleu.std())
        }
    
    # Add perplexity statistics if available
    if use_perplexity and ppl_scores is not None and len(ppl_scores) > 0:
        output_data['score_statistics']['perplexity'] = {
            'members_mean': float(member_ppl.mean()),
            'members_std': float(member_ppl.std()),
            'nonmembers_mean': float(nonmember_ppl.mean()),
            'nonmembers_std': float(nonmember_ppl.std())
        }
    
    # Save results
    write_json(output_data, output_path)
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate S²MIA attack performance')
    parser.add_argument(
        '--rag_outputs_file',
        type=str,
        default='results/S2-MIA/BeIR_nfcorpus/rag/rag_outputs.jsonl',
        help='Path to RAG outputs file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/S2-MIA/BeIR_nfcorpus/evaluation/metrics.json',
        help='Path to output evaluation metrics file'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Embedding model for semantic similarity'
    )
    parser.add_argument(
        '--use_bleu',
        action='store_true',
        default=True,
        help='Compute BLEU scores'
    )
    parser.add_argument(
        '--no_bleu',
        action='store_true',
        help='Do not compute BLEU scores'
    )
    parser.add_argument(
        '--no_perplexity',
        action='store_true',
        help='Do not compute perplexity from logprobs'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        default='accuracy',
        choices=['accuracy', 'f1', 'youden'],
        help='Method for selecting best threshold (default: accuracy)'
    )
    
    args = parser.parse_args()
    
    if args.cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    use_bleu = args.use_bleu and not args.no_bleu
    use_perplexity = not args.no_perplexity
    
    evaluate_s2mia(
        args.rag_outputs_file,
        args.output_file,
        args.embedding_model,
        use_bleu=use_bleu,
        use_perplexity=use_perplexity,
        use_gpu=args.use_gpu,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        threshold_method=args.threshold_method
    )
    
    print("\n" + "="*60)
    print("S²MIA evaluation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()