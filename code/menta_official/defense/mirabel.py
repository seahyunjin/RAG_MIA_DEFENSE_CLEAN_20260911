import numpy as np
from typing import List, Dict
from tqdm import tqdm


def mirabel_detection(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    doc_ids: List[str],
    rho: float = 0.05
) -> Dict:
    """
    Implement Mirabel detection algorithm (Algorithm 1 from paper).
    
    Args:
        query_embedding: Query embedding vector (normalized)
        corpus_embeddings: All document embeddings (normalized)
        doc_ids: List of document IDs corresponding to corpus_embeddings
        rho: Significance level (default 0.05)
        
    Returns:
        Dictionary with detection results
    """
    similarities = np.dot(corpus_embeddings, query_embedding)
    
    max_idx = np.argmax(similarities)
    s_max = float(similarities[max_idx])
    target_doc_id = doc_ids[max_idx]
    
    similarities_without_max = np.delete(similarities, max_idx)
    mu_q = float(np.mean(similarities_without_max))
    sigma_q = float(np.std(similarities_without_max))
    
    n = len(similarities)
    mu_n = mu_q + sigma_q * np.sqrt(2 * np.log(n))
    
    c = -np.log(-np.log(1 - rho))
    
    tau = mu_n + c * sigma_q / np.sqrt(2 * np.log(n))
    
    is_attack = s_max > tau
    
    return {
        'detected_as_attack': bool(is_attack),
        'mirabel_s_max': s_max,
        'mirabel_threshold': float(tau),
        'mirabel_target_doc_id': target_doc_id,
        'mirabel_mu_q': mu_q,
        'mirabel_sigma_q': sigma_q,
        'mirabel_mu_n': float(mu_n),
        'mirabel_n': n
    }


def apply_mirabel_to_queries(
    queries: List[dict],
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    doc_ids: List[str],
    rho: float = 0.05
) -> List[dict]:
    """
    Apply Mirabel detection to all queries.
    
    Args:
        queries: List of query dictionaries
        query_embeddings: Query embeddings
        corpus_embeddings: Corpus embeddings
        doc_ids: Document IDs
        rho: Significance level
        
    Returns:
        Updated queries with Mirabel detection results
    """
    print(f"\nApplying Mirabel detection to {len(queries)} queries...")
    
    for i, query in enumerate(tqdm(queries, desc="Mirabel detection")):
        mirabel_result = mirabel_detection(
            query_embeddings[i],
            corpus_embeddings,
            doc_ids,
            rho=rho
        )
        
        query.update(mirabel_result)
    
    return queries


def compute_mirabel_statistics(queries: List[dict]) -> Dict:
    """
    Compute statistics for Mirabel detection results.
    Only counts as detected when top-1 Mirabel detection is the target doc.
    
    Args:
        queries: List of queries with Mirabel detection results
        
    Returns:
        Dictionary with statistics
    """
    print(f"\nComputing Mirabel statistics...")
    
    total_queries = len(queries)
    
    detected_with_target_top1 = 0
    detected_raw = 0
    for q in queries:
        if q.get('detected_as_attack', False):
            detected_raw += 1
            target_doc_id = q.get('target_doc_id') or q.get('doc_id')
            mirabel_target = q.get('mirabel_target_doc_id')
            if target_doc_id and mirabel_target and target_doc_id == mirabel_target:
                detected_with_target_top1 += 1
    
    not_detected = total_queries - detected_with_target_top1
    detection_rate = detected_with_target_top1 / total_queries if total_queries > 0 else 0
    
    thresholds = [q['mirabel_threshold'] for q in queries if 'mirabel_threshold' in q]
    s_maxes = [q['mirabel_s_max'] for q in queries if 'mirabel_s_max' in q]
    
    mu_qs = [q['mirabel_mu_q'] for q in queries if 'mirabel_mu_q' in q]
    sigma_qs = [q['mirabel_sigma_q'] for q in queries if 'mirabel_sigma_q' in q]
    
    return {
        'total_queries': total_queries,
        'detected_raw': detected_raw,
        'detected_as_attack': detected_with_target_top1,
        'not_detected_as_attack': not_detected,
        'detection_rate': detection_rate,
        'detection_percentage': detection_rate * 100,
        'threshold_stats': {
            'mean': float(np.mean(thresholds)) if thresholds else 0,
            'median': float(np.median(thresholds)) if thresholds else 0,
            'std': float(np.std(thresholds)) if thresholds else 0,
            'min': float(np.min(thresholds)) if thresholds else 0,
            'max': float(np.max(thresholds)) if thresholds else 0
        },
        's_max_stats': {
            'mean': float(np.mean(s_maxes)) if s_maxes else 0,
            'median': float(np.median(s_maxes)) if s_maxes else 0,
            'std': float(np.std(s_maxes)) if s_maxes else 0,
            'min': float(np.min(s_maxes)) if s_maxes else 0,
            'max': float(np.max(s_maxes)) if s_maxes else 0
        },
        'mu_q_stats': {
            'mean': float(np.mean(mu_qs)) if mu_qs else 0,
            'median': float(np.median(mu_qs)) if mu_qs else 0,
            'std': float(np.std(mu_qs)) if mu_qs else 0,
            'min': float(np.min(mu_qs)) if mu_qs else 0,
            'max': float(np.max(mu_qs)) if mu_qs else 0
        },
        'sigma_q_stats': {
            'mean': float(np.mean(sigma_qs)) if sigma_qs else 0,
            'median': float(np.median(sigma_qs)) if sigma_qs else 0,
            'std': float(np.std(sigma_qs)) if sigma_qs else 0,
            'min': float(np.min(sigma_qs)) if sigma_qs else 0,
            'max': float(np.max(sigma_qs)) if sigma_qs else 0
        }
    }
