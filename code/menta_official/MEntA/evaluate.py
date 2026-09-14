import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json


def extract_target_doc_from_query_id(query_id: str) -> str:
    """Extract target document ID from query_id."""
    parts = query_id.split('_')
    if len(parts) >= 2:
        if parts[0] == 'q' and parts[-1].startswith('v'):
            return '_'.join(parts[1:-1])
        elif parts[0] == 'q':
            return '_'.join(parts[1:])
    return query_id


def parse_output_modes_from_path(file_path: str) -> Tuple[str, bool, bool]:
    """Parse defense/query mode flags from entailment or answers path."""
    defense_type = "none"
    generic_queries = False
    disable_retrieval = False

    for part in Path(file_path).parts:
        if part in {"entailment", "answers"}:
            return defense_type, generic_queries, disable_retrieval
        if part.startswith("entailment_"):
            suffix = part[len("entailment_"):]
        elif part.startswith("answers_"):
            suffix = part[len("answers_"):]
        else:
            continue

        if "_generic" in suffix:
            generic_queries = True
            suffix = suffix.replace("_generic", "")
        if "_no_summary" in suffix:
            disable_retrieval = True
            suffix = suffix.replace("_no_summary", "")
        if "_disable_retrieval" in suffix:
            disable_retrieval = True
            suffix = suffix.replace("_disable_retrieval", "")

        if suffix == "paraphrased":
            defense_type = "paraphrase"
        elif suffix:
            defense_type = suffix
        break

    return defense_type, generic_queries, disable_retrieval


def build_mode_label(defense_type: str, generic_queries: bool, disable_retrieval: bool) -> str:
    """Build concise mode label for plots/logging."""
    labels = []
    if defense_type != "none":
        labels.append(f"defense={defense_type}")
    if generic_queries:
        labels.append("queries=generic")
    if disable_retrieval:
        labels.append("disable_retrieval")
    return ", ".join(labels)


def get_evaluation_dirname(defense_type: str, generic_queries: bool, disable_retrieval: bool) -> str:
    """Build evaluation directory name from defense/query mode flags."""
    if defense_type == 'paraphrase':
        dirname = 'evaluation_paraphrased'
    elif defense_type != 'none':
        dirname = f'evaluation_{defense_type}'
    else:
        dirname = 'evaluation'

    if generic_queries:
        dirname = f'{dirname}_generic'
    if disable_retrieval:
        dirname = f'{dirname}_disable_retrieval'

    return dirname


def normalize_evaluation_output_dir(
    entailment_file_path: str,
    output_dir: str,
    defense_type: str,
    generic_queries: bool,
    disable_retrieval: bool
) -> str:
    """Normalize evaluation output directory to match entailment mode suffixes."""
    evaluation_dirname = get_evaluation_dirname(defense_type, generic_queries, disable_retrieval)

    if output_dir:
        parts = list(Path(output_dir).parts)
        replaced = False
        for idx, part in enumerate(parts):
            if part == 'evaluation' or part.startswith('evaluation_'):
                parts[idx] = evaluation_dirname
                replaced = True
                break
        if replaced:
            return str(Path(*parts))

    entailment_path = Path(entailment_file_path)
    parts = list(entailment_path.parent.parts)
    for idx, part in enumerate(parts):
        if part == 'entailment' or part.startswith('entailment_'):
            parts[idx] = evaluation_dirname
            break

    return str(Path(*parts) / entailment_path.stem.replace('entailment', 'evaluation'))


def detect_idk_queries(idk_detection_details: Dict[str, dict]) -> Tuple[Set[str], Dict[str, dict]]:
    """
    Detect IDK queries based on entailment probability being highest.
    
    For each claim:
    1. Check all IDK hypothesis scores
    2. For each hypothesis, check if entailment is highest among (ent, neu, con)
    3. If ANY hypothesis has entailment as highest, claim is IDK
    
    For each query:
    - If ANY claim is IDK, query is IDK
    
    Returns:
        - Set of IDK query IDs
        - Detailed IDK analysis per query
    """
    idk_query_ids = set()
    idk_analysis = {}
    
    print("\nDetecting IDK queries (entailment prob is highest criterion)...")
    
    for query_id, details in idk_detection_details.items():
        claims_analysis = []
        query_has_idk = False
        
        for claim_entry in details['claims']:
            claim = claim_entry['claim']
            idk_scores = claim_entry['idk_entailment_scores']
            
            # For each hypothesis, check if entailment is highest
            hypothesis_results = {}
            claim_is_idk = False
            best_hypothesis = None
            max_ent_score = 0.0
            
            for hypothesis, score_data in idk_scores.items():
                # score_data is just the entailment probability (float)
                ent_prob = score_data
                
                # Consider entailment as "highest" if > 0.5
                is_entailment_highest = ent_prob > 0.5
                
                hypothesis_results[hypothesis] = {
                    'entailment_prob': ent_prob,
                    'is_entailment_highest': is_entailment_highest
                }
                
                if is_entailment_highest:
                    claim_is_idk = True
                    if ent_prob > max_ent_score:
                        max_ent_score = ent_prob
                        best_hypothesis = hypothesis
            
            claims_analysis.append({
                'claim': claim,
                'is_idk': claim_is_idk,
                'best_hypothesis': best_hypothesis,
                'max_entailment_prob': max_ent_score,
                'hypothesis_results': hypothesis_results
            })
            
            if claim_is_idk:
                query_has_idk = True
        
        idk_analysis[query_id] = {
            'is_idk': query_has_idk,
            'claims': claims_analysis,
            'num_idk_claims': sum(1 for c in claims_analysis if c['is_idk']),
            'total_claims': len(claims_analysis)
        }
        
        if query_has_idk:
            idk_query_ids.add(query_id)
    
    print(f"Detected {len(idk_query_ids)} IDK queries")
    
    return idk_query_ids, idk_analysis


def calculate_mia_scores(
    queries: List[dict],
    query_to_claims: Dict[str, List[str]],
    idk_query_ids: Set[str],
    entailment_matrix: Dict[Tuple[str, str, int], dict],
    lambda_idk: float = 1.0
) -> Dict[str, dict]:
    """Calculate MIA score for each document.
    
    FORMULA: MIA score = (1/n) * sum(ent - lambda * idk)
    where:
    - n is the number of query variations for the document
    - ent = 1 if the query output has at least 1 entailed claim (ent_prob is highest), 0 otherwise
    - idk = 1 if the query received an IDK response, 0 otherwise
    - lambda is the penalty weight for IDK responses
    """
    # Group queries by target document
    doc_to_queries = defaultdict(list)
    for query_data in queries:
        query_id = query_data['query_id']
        target_doc = extract_target_doc_from_query_id(query_id)
        if target_doc:
            doc_to_queries[target_doc].append(query_data)
    
    doc_scores = {}
    
    for doc_id, doc_queries in doc_to_queries.items():
        total_score = 0.0
        n_variations = len(doc_queries)
        query_details = []
        
        for query_data in doc_queries:
            query_id = query_data['query_id']
            is_idk = query_id in idk_query_ids
            
            # If IDK, apply penalty
            if is_idk:
                query_contribution = -lambda_idk
                query_details.append({
                    'query_id': query_id,
                    'is_idk': True,
                    'ent': 0,
                    'has_entailed_claim': False,
                    'query_contribution': query_contribution,
                    'claim_scores': []
                })
                total_score += query_contribution
                continue
            
            # Check if this query has at least one entailed claim
            claims = query_to_claims.get(query_id, [])
            has_entailed_claim = False
            claim_scores = []
            
            for claim_idx, claim in enumerate(claims):
                key = (query_id, doc_id, claim_idx)
                entailment_info = entailment_matrix.get(key)
                
                if entailment_info:
                    ent_prob = entailment_info['entailment_prob']
                    neu_prob = entailment_info['neutral_prob']
                    con_prob = entailment_info['contradiction_prob']
                    
                    # Check if entailment is highest (or tied for highest)
                    is_entailment_highest = (ent_prob >= neu_prob and ent_prob >= con_prob)
                    
                    if is_entailment_highest:
                        has_entailed_claim = True
                    
                    claim_scores.append({
                        'claim': claim,
                        'ent_prob': ent_prob,
                        'neu_prob': neu_prob,
                        'con_prob': con_prob,
                        'is_entailment_highest': is_entailment_highest
                    })
                else:
                    claim_scores.append({
                        'claim': claim,
                        'ent_prob': 0.0,
                        'neu_prob': 0.0,
                        'con_prob': 0.0,
                        'is_entailment_highest': False
                    })
            
            # Set ent to 1 if at least one claim is entailed, 0 otherwise
            ent = 1 if has_entailed_claim else 0
            query_contribution = ent  # idk is 0 for non-IDK queries
            
            query_details.append({
                'query_id': query_id,
                'is_idk': False,
                'ent': ent,
                'has_entailed_claim': has_entailed_claim,
                'query_contribution': query_contribution,
                'claim_scores': claim_scores
            })
            total_score += query_contribution
        
        # Calculate final MIA score
        # MIA_score = (1/n) * sum(ent - lambda * idk)
        mia_score = total_score / n_variations if n_variations > 0 else 0.0
        
        doc_scores[doc_id] = {
            'mia_score': mia_score,
            'n_variations': n_variations,
            'total_score': total_score,
            'query_details': query_details
        }
    
    return doc_scores


def evaluate_with_threshold(
    doc_scores: Dict[str, dict],
    member_docs: Set[str],
    all_target_docs: Set[str],
    threshold: float
) -> Dict[str, any]:
    """
    Evaluate predictions using a specific threshold.
    
    For MIA: Higher score → Member (positive class)
             Lower score ≤ threshold → Non-member (negative class)
    """
    # Predict as member if score > threshold
    predicted_members = {doc_id for doc_id, info in doc_scores.items() 
                        if info['mia_score'] > threshold}
    
    tp = predicted_members & member_docs  # Correctly identified members
    fp = predicted_members - member_docs  # Non-members predicted as members
    fn = member_docs - predicted_members  # Members predicted as non-members
    tn = (all_target_docs - predicted_members) - member_docs  # Correctly identified non-members
    
    precision = len(tp) / len(predicted_members) if predicted_members else 0
    recall = len(tp) / len(member_docs) if member_docs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (len(tp) + len(tn)) / len(all_target_docs) if all_target_docs else 0
    
    return {
        'threshold': threshold,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tpr': recall,
            'fpr': len(fp) / (len(fp) + len(tn)) if (len(fp) + len(tn)) > 0 else 0
        },
        'counts': {
            'tp': len(tp),
            'fp': len(fp),
            'fn': len(fn),
            'tn': len(tn),
            'total_docs': len(all_target_docs),
            'predicted_members': len(predicted_members),
            'actual_members': len(member_docs)
        }
    }


def calculate_auc_with_roc(
    doc_scores: Dict[str, dict],
    member_docs: Set[str],
    all_target_docs: Set[str]
) -> Dict[str, any]:
    """Calculate AUC by testing all possible thresholds.
    
    Returns ROC curve data, AUC score, and best threshold based on F1 score.
    """
    # Get all unique MIA scores as potential thresholds
    all_scores = sorted(set(info['mia_score'] for info in doc_scores.values()))
    
    # Add boundary thresholds
    min_score = min(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 1
    thresholds = [min_score - 1] + all_scores + [max_score + 1]
    
    tpr_list = []
    fpr_list = []
    threshold_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    
    # Calculate TPR, FPR, and F1 for each threshold
    for threshold in thresholds:
        result = evaluate_with_threshold(doc_scores, member_docs, all_target_docs, threshold)
        tpr_list.append(result['metrics']['tpr'])
        fpr_list.append(result['metrics']['fpr'])
        f1_list.append(result['metrics']['f1_score'])
        precision_list.append(result['metrics']['precision'])
        recall_list.append(result['metrics']['recall'])
        accuracy_list.append(result['metrics']['accuracy'])
        threshold_list.append(threshold)
    
    # Calculate AUC using trapezoidal rule
    # Sort by FPR (ascending order)
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array([fpr_list[i] for i in sorted_indices])
    tpr_sorted = np.array([tpr_list[i] for i in sorted_indices])
    
    # Calculate AUC using trapezoidal integration
    auc_score = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        auc_score += width * height
    
    # Find best threshold based on F1 score
    best_f1_idx = np.argmax(f1_list)
    best_threshold = threshold_list[best_f1_idx]
    best_f1 = f1_list[best_f1_idx]
    
    # Find best threshold based on Accuracy
    best_acc_idx = np.argmax(accuracy_list)
    best_acc_threshold = threshold_list[best_acc_idx]
    best_accuracy = accuracy_list[best_acc_idx]
    
    # Also calculate Youden's J statistic
    youden_list = [tpr - fpr for tpr, fpr in zip(tpr_list, fpr_list)]
    best_youden_idx = np.argmax(youden_list)
    best_youden_threshold = threshold_list[best_youden_idx]
    best_youden = youden_list[best_youden_idx]
    
    # Calculate TPR at specific FPR values (0.005, 0.01, 0.05)
    target_fprs = [0.005, 0.01, 0.05]
    tpr_at_fpr = {}
    
    for target_fpr in target_fprs:
        # Find the TPR at the largest FPR that is <= target_fpr
        # Use sorted arrays for this
        best_tpr = 0.0
        best_actual_fpr = 0.0
        for i in range(len(fpr_sorted)):
            if fpr_sorted[i] <= target_fpr:
                if tpr_sorted[i] >= best_tpr:
                    best_tpr = tpr_sorted[i]
                    best_actual_fpr = fpr_sorted[i]
        
        tpr_at_fpr[f'tpr_at_fpr_{target_fpr}'] = {
            'target_fpr': target_fpr,
            'actual_fpr': float(best_actual_fpr),
            'tpr': float(best_tpr)
        }
    
    return {
        'auc': auc_score,
        'best_threshold_f1': best_threshold,
        'best_f1': best_f1,
        'best_threshold_accuracy': best_acc_threshold,
        'best_accuracy': best_accuracy,
        'best_threshold_youden': best_youden_threshold,
        'best_youden': best_youden,
        'tpr_at_fpr': tpr_at_fpr,
        'roc_curve': {
            'fpr': fpr_list,
            'tpr': tpr_list,
            'thresholds': threshold_list,
            'f1_scores': f1_list,
            'precision': precision_list,
            'recall': recall_list,
            'accuracy': accuracy_list,
            'youden': youden_list
        },
        'num_thresholds': len(thresholds)
    }


def plot_roc_curve(
    roc_data: Dict,
    save_path: str,
    lambda_idk: float,
    best_threshold: float,
    defense_type: str = "none",
    generic_queries: bool = False,
    disable_retrieval: bool = False
):
    """Plot and save ROC curve with best threshold marked."""
    fpr = roc_data['roc_curve']['fpr']
    tpr = roc_data['roc_curve']['tpr']
    auc_score = roc_data['auc']
    
    plt.figure(figsize=(10, 8))
    
    mode_label = build_mode_label(defense_type, generic_queries, disable_retrieval)
    defense_label = f" [{mode_label}]" if mode_label else ""
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve{defense_label} (AUC = {auc_score:.4f}, λ = {lambda_idk})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark the best threshold (F1-based)
    thresholds = roc_data['roc_curve']['thresholds']
    best_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - best_threshold))
    plt.plot(fpr[best_idx], tpr[best_idx], 'ro', markersize=10, 
             label=f'Best threshold (F1={roc_data["best_f1"]:.3f}, τ={best_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    
    title = 'ROC Curve for Membership Inference Attack'
    if mode_label:
        title += f' ({mode_label})'
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")


def evaluate_mia_from_entailment(
    entailment_file_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_dir: str,
    lambda_idk: float = 1.0,
    threshold_method: str = 'accuracy'
):
    """Evaluate MIA using MIA scores with member/nonmember corpus."""
    
    # Detect defense type from path
    defense_type, generic_queries, disable_retrieval = parse_output_modes_from_path(entailment_file_path)
    
    print(f"\n{'#'*60}")
    print(f"MIA EVALUATION WITH ENTAILMENT-BASED SCORING")
    print(f"Formula: MIA = (1/n) * sum(ent - lambda * idk)")
    print(f"  ent = 1 if query has ≥1 entailed claim, 0 otherwise")
    print(f"  idk = 1 if query is IDK (any claim entails IDK hypothesis), 0 otherwise")
    print(f"IDK Detection: Entailment probability > 0.5")
    print(f"Lambda (IDK penalty): {lambda_idk}")
    print(f"Threshold Selection: Auto (best {threshold_method.upper()})")
    if defense_type != "none":
        print(f"Defense Type: {defense_type}")
    print(f"Generic queries: {generic_queries}")
    print(f"Disable retrieval: {disable_retrieval}")
    print(f"{'#'*60}")
    
    print("\nLoading data...")
    entailment_data = read_json(entailment_file_path)
    
    # Load corpora
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    
    member_docs = set(doc['_id'] for doc in corpus_member)
    nonmember_docs = set(doc['_id'] for doc in corpus_nonmember)
    
    print(f"\nGround Truth:")
    print(f"  Member documents: {len(member_docs)}")
    print(f"  Non-member documents: {len(nonmember_docs)}")
    print(f"  Total documents: {len(member_docs) + len(nonmember_docs)}")
    
    # Deserialize entailment matrix
    entailment_matrix_serializable = entailment_data['entailment_matrix']
    entailment_matrix = {}
    for key_str, value in entailment_matrix_serializable.items():
        parts = key_str.split('||')
        if len(parts) == 3:
            query_id, doc_id, claim_idx_str = parts
            claim_idx = int(claim_idx_str)
            entailment_matrix[(query_id, doc_id, claim_idx)] = value
    
    query_to_claims = entailment_data['query_to_claims']
    idk_detection_details = entailment_data.get('idk_detection_details', {})
    
    # Get defense type from metadata if available
    metadata_defense = entailment_data.get('metadata', {}).get('defense_type', 'none')
    if defense_type == "none" and metadata_defense != "none":
        defense_type = metadata_defense
    generic_queries = entailment_data.get('metadata', {}).get('generic_queries', generic_queries)
    disable_retrieval = entailment_data.get('metadata', {}).get('disable_retrieval', disable_retrieval)
    
    # Build queries list from query_to_claims keys
    queries = [{'query_id': qid} for qid in query_to_claims.keys()]
    
    # Detect IDK queries
    print("\n" + "="*60)
    print("IDK Detection Phase")
    print("="*60)
    idk_query_ids, idk_analysis = detect_idk_queries(idk_detection_details)
    
    print(f"\nCalculating MIA scores (lambda_idk={lambda_idk})...")
    doc_scores = calculate_mia_scores(
        queries,
        query_to_claims,
        idk_query_ids,
        entailment_matrix,
        lambda_idk
    )
    
    all_target_docs = set(doc_scores.keys())
    
    # Print score distribution
    print(f"\n{'='*60}")
    print("Score Distribution Analysis")
    print(f"{'='*60}")
    
    member_scores = [doc_scores[doc_id]['mia_score'] for doc_id in member_docs if doc_id in doc_scores]
    nonmember_scores = [doc_scores[doc_id]['mia_score'] for doc_id in nonmember_docs if doc_id in doc_scores]
    
    print(f"\nMember documents ({len(member_scores)}):")
    print(f"  Mean: {np.mean(member_scores):.4f}")
    print(f"  Median: {np.median(member_scores):.4f}")
    print(f"  Std: {np.std(member_scores):.4f}")
    print(f"  Min: {np.min(member_scores):.4f}")
    print(f"  Max: {np.max(member_scores):.4f}")
    
    print(f"\nNon-member documents ({len(nonmember_scores)}):")
    print(f"  Mean: {np.mean(nonmember_scores):.4f}")
    print(f"  Median: {np.median(nonmember_scores):.4f}")
    print(f"  Std: {np.std(nonmember_scores):.4f}")
    print(f"  Min: {np.min(nonmember_scores):.4f}")
    print(f"  Max: {np.max(nonmember_scores):.4f}")
    
    # Statistics
    total_queries = len(queries)
    total_idk = len(idk_query_ids)
    total_claims = sum(len(claims) for claims in query_to_claims.values())
    total_idk_claims = sum(analysis['num_idk_claims'] for analysis in idk_analysis.values())
    
    print(f"\nQuery Statistics:")
    print(f"  Total queries: {total_queries}")
    print(f"  IDK queries: {total_idk} ({100*total_idk/total_queries:.1f}%)")
    
    print(f"\nClaim Statistics:")
    print(f"  Total claims: {total_claims}")
    print(f"  IDK claims: {total_idk_claims} ({100*total_idk_claims/total_claims:.1f}%)")
    print(f"{'='*60}")
    
    # Calculate AUC and find best threshold
    print(f"\nCalculating AUC and finding best threshold...")
    auc_data = calculate_auc_with_roc(doc_scores, member_docs, all_target_docs)
    
    if threshold_method == 'f1':
        best_threshold = auc_data['best_threshold_f1']
        print(f"\nBest threshold (F1-based): {best_threshold:.4f} (F1={auc_data['best_f1']:.4f})")
    elif threshold_method == 'youden':
        best_threshold = auc_data['best_threshold_youden']
        print(f"\nBest threshold (Youden-based): {best_threshold:.4f} (J={auc_data['best_youden']:.4f})")
    elif threshold_method == 'accuracy':
        best_threshold = auc_data['best_threshold_accuracy']
        print(f"\nBest threshold (Accuracy-based): {best_threshold:.4f} (Acc={auc_data['best_accuracy']:.4f})")
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    print(f"\nAUC Results:")
    print(f"  AUC Score: {auc_data['auc']:.4f}")
    print(f"  Number of thresholds tested: {auc_data['num_thresholds']}")
    
    # Print TPR at specific FPR values
    print(f"\nTPR at specific FPR values:")
    for key, value in auc_data['tpr_at_fpr'].items():
        print(f"  TPR@FPR={value['target_fpr']}: {value['tpr']:.4f} (actual FPR={value['actual_fpr']:.4f})")
    
    # Evaluate with best threshold
    print(f"\nEvaluating with selected threshold={best_threshold:.4f}...")
    eval_result = evaluate_with_threshold(doc_scores, member_docs, all_target_docs, best_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("MIA EVALUATION RESULTS")
    if defense_type != "none":
        print(f"Defense: {defense_type}")
    print(f"Generic queries: {generic_queries}")
    print(f"Disable retrieval: {disable_retrieval}")
    print("="*60)
    print(f"Lambda: {lambda_idk}")
    print(f"Threshold: {eval_result['threshold']:.4f} (best {threshold_method.upper()})")
    print(f"AUC: {auc_data['auc']:.4f}")
    print("\nTPR at Low FPR:")
    for key, value in auc_data['tpr_at_fpr'].items():
        print(f"  TPR@FPR={value['target_fpr']}: {value['tpr']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives (TP):  {eval_result['counts']['tp']}")
    print(f"  False Positives (FP): {eval_result['counts']['fp']}")
    print(f"  True Negatives (TN):  {eval_result['counts']['tn']}")
    print(f"  False Negatives (FN): {eval_result['counts']['fn']}")
    print("\nMetrics:")
    print(f"  Precision: {eval_result['metrics']['precision']:.4f}")
    print(f"  Recall:    {eval_result['metrics']['recall']:.4f}")
    print(f"  F1 Score:  {eval_result['metrics']['f1_score']:.4f}")
    print(f"  Accuracy:  {eval_result['metrics']['accuracy']:.4f}")
    print(f"  TPR:       {eval_result['metrics']['tpr']:.4f}")
    print(f"  FPR:       {eval_result['metrics']['fpr']:.4f}")
    print("="*60)
    
    # Save results
    output_dir = normalize_evaluation_output_dir(
        entailment_file_path,
        output_dir,
        defense_type,
        generic_queries,
        disable_retrieval
    )
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'mia_evaluation_results.json')
    roc_plot_path = os.path.join(output_dir, 'mia_roc_curve.png')
    
    # Plot ROC curve
    print(f"\nGenerating ROC curve plot...")
    plot_roc_curve(
        auc_data,
        roc_plot_path,
        lambda_idk,
        best_threshold,
        defense_type,
        generic_queries,
        disable_retrieval
    )
    
    results_data = {
        'metadata': {
            'entailment_file': str(entailment_file_path),
            'corpus_member': corpus_member_path,
            'corpus_nonmember': corpus_nonmember_path,
            'defense_type': defense_type,
            'generic_queries': generic_queries,
            'query_mode': 'generic' if generic_queries else 'specific',
            'disable_retrieval': disable_retrieval,
            'lambda_idk': lambda_idk,
            'threshold': best_threshold,
            'threshold_selection': f'auto_best_{threshold_method}',
            'evaluation_mode': 'mia_entailment_based',
            'idk_detection_method': 'entailment_prob_highest',
            'formula': '(1/n) * sum(ent - lambda * idk)'
        },
        'evaluation_best_threshold': eval_result,
        'auc_evaluation': {
            'auc': auc_data['auc'],
            'num_thresholds_tested': auc_data['num_thresholds'],
            'best_f1_threshold': auc_data['best_threshold_f1'],
            'best_f1_score': auc_data['best_f1'],
            'best_accuracy_threshold': auc_data['best_threshold_accuracy'],
            'best_accuracy_score': auc_data['best_accuracy'],
            'best_youden_threshold': auc_data['best_threshold_youden'],
            'best_youden_score': auc_data['best_youden'],
            'tpr_at_fpr': auc_data['tpr_at_fpr']
        },
        'ground_truth_summary': {
            'num_members': len(member_docs),
            'num_nonmembers': len(nonmember_docs),
            'num_total': len(all_target_docs)
        },
        'statistics': {
            'member_docs': {
                'count': len(member_scores),
                'mean': float(np.mean(member_scores)),
                'median': float(np.median(member_scores)),
                'std': float(np.std(member_scores)),
                'min': float(np.min(member_scores)),
                'max': float(np.max(member_scores))
            },
            'nonmember_docs': {
                'count': len(nonmember_scores),
                'mean': float(np.mean(nonmember_scores)),
                'median': float(np.median(nonmember_scores)),
                'std': float(np.std(nonmember_scores)),
                'min': float(np.min(nonmember_scores)),
                'max': float(np.max(nonmember_scores))
            },
            'queries': {
                'total': total_queries,
                'idk': total_idk,
                'idk_percentage': 100 * total_idk / total_queries if total_queries > 0 else 0
            },
            'claims': {
                'total': total_claims,
                'idk_claims': total_idk_claims,
                'idk_claims_percentage': 100 * total_idk_claims / total_claims if total_claims > 0 else 0
            }
        },
        'idk_detection_summary': {
            'num_idk_queries': len(idk_query_ids),
            'idk_query_ids': list(idk_query_ids)
        },
        'doc_scores': {doc_id: {
            'mia_score': info['mia_score'],
            'n_variations': info['n_variations'],
            'total_score': info['total_score']
        } for doc_id, info in doc_scores.items()}
    }
    
    write_json(results_data, results_path)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MIA using entailment-based scoring with member/nonmember corpus'
    )
    parser.add_argument(
        '--entailment_file',
        type=str,
        default="results/MEntA/BeIR_scidocs/entailment/target_summary/topk3/meta-llama--Llama-3.1-8B-Instruct/entailment_5v.json",
        help='Path to precomputed entailment matrix JSON file'
    )
    parser.add_argument(
        '--corpus_member', 
        type=str,
        default="data/BeIR_scidocs/corpus_member.jsonl",
        help='Path to member corpus JSONL file'
    )
    parser.add_argument(
        '--corpus_nonmember',
        type=str,
        default="data/BeIR_scidocs/corpus_nonmember.jsonl",
        help='Path to non-member corpus JSONL file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="results/MEntA/BeIR_scidocs/evaluation/target_summary/topk3/meta-llama--Llama-3.1-8B-Instruct/evaluation_5v.json",
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--lambda_idk',
        type=float,
        default=1.0,
        help='Penalty weight for IDK responses (default: 1.0)'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        default='accuracy',
        choices=['f1', 'youden', 'accuracy'],
        help='Method for selecting best threshold (default: accuracy)'
    )
    
    args = parser.parse_args()
    
    evaluate_mia_from_entailment(
        entailment_file_path=args.entailment_file,
        corpus_member_path=args.corpus_member,
        corpus_nonmember_path=args.corpus_nonmember,
        output_dir=args.output_dir,
        lambda_idk=args.lambda_idk,
        threshold_method=args.threshold_method
    )
    
    print("\n" + "="*60)
    print("MIA Evaluation completed successfully!")
    print("="*60)
    

if __name__ == '__main__':
    main()
