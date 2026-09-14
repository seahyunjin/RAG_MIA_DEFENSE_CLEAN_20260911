import json
import os
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_generator_model, load_embedding_model
from defense.mirabel import apply_mirabel_to_queries, compute_mirabel_statistics


def extract_num_variations(query_path: str) -> str:
    """
    Extract number of variations from query file name.
    E.g., 'queries_5v.jsonl' -> '5v'
    """
    filename = Path(query_path).stem
    match = re.search(r'_(\d+v)$', filename)
    if match:
        return match.group(1)
    return "default"


def parse_top_k_range(top_k_str: str) -> List[int]:
    """
    Parse top-k range string into a list of integers.
    
    Supports formats:
    - Single value: "10" -> [10]
    - Comma-separated: "3,5,10,20" -> [3, 5, 10, 20]
    - Range: "3-10" -> [3, 4, 5, 6, 7, 8, 9, 10]
    - Range with step: "3-20-5" -> [3, 8, 13, 18]
    - Mixed: "3,5,10-20-5" -> [3, 5, 10, 15, 20]
    
    Args:
        top_k_str: String specification of top-k values
        
    Returns:
        Sorted list of unique top-k values
    """
    result = set()
    
    parts = top_k_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            range_parts = part.split('-')
            if len(range_parts) == 2:
                start, end = int(range_parts[0]), int(range_parts[1])
                result.update(range(start, end + 1))
            elif len(range_parts) == 3:
                start, end, step = int(range_parts[0]), int(range_parts[1]), int(range_parts[2])
                result.update(range(start, end + 1, step))
            else:
                raise ValueError(f"Invalid range format: {part}")
        else:
            result.add(int(part))
    
    return sorted(list(result))


def load_corpus(corpus_path: str) -> Tuple[List[str], Dict[str, dict]]:
    """
    Load corpus from JSONL file.
    
    Args:
        corpus_path: Path to corpus JSONL file
        
    Returns:
        Tuple of (doc_ids, doc_id_to_doc)
    """
    print(f"Loading corpus from: {corpus_path}")
    corpus = read_jsonl(corpus_path)
    
    doc_ids = []
    doc_id_to_doc = {}
    
    for doc in corpus:
        doc_id = doc['_id']
        doc_ids.append(doc_id)
        doc_id_to_doc[doc_id] = doc
    
    print(f"Loaded {len(doc_ids)} documents from corpus")
    return doc_ids, doc_id_to_doc


def load_member_nonmember_corpus(
    member_corpus_path: str,
    nonmember_corpus_path: str
) -> Tuple[List[str], Dict[str, dict], Set[str], Set[str]]:
    """
    Load both member and non-member corpus files.
    
    Args:
        member_corpus_path: Path to member corpus JSONL file
        nonmember_corpus_path: Path to non-member corpus JSONL file
        
    Returns:
        Tuple of (all_doc_ids, doc_id_to_doc, member_doc_ids, nonmember_doc_ids)
    """
    print(f"Loading member corpus from: {member_corpus_path}")
    member_corpus = read_jsonl(member_corpus_path)
    
    print(f"Loading non-member corpus from: {nonmember_corpus_path}")
    nonmember_corpus = read_jsonl(nonmember_corpus_path)
    
    all_doc_ids = []
    doc_id_to_doc = {}
    member_doc_ids = set()
    nonmember_doc_ids = set()
    
    for doc in member_corpus:
        doc_id = doc['_id']
        all_doc_ids.append(doc_id)
        doc_id_to_doc[doc_id] = doc
        member_doc_ids.add(doc_id)
    
    for doc in nonmember_corpus:
        doc_id = doc['_id']
        if doc_id not in doc_id_to_doc:
            all_doc_ids.append(doc_id)
            doc_id_to_doc[doc_id] = doc
        nonmember_doc_ids.add(doc_id)
    
    print(f"Loaded {len(member_doc_ids)} member documents")
    print(f"Loaded {len(nonmember_doc_ids)} non-member documents")
    print(f"Total unique documents: {len(all_doc_ids)}")
    
    return all_doc_ids, doc_id_to_doc, member_doc_ids, nonmember_doc_ids


def load_summaries(summary_path: str) -> Dict[str, str]:
    """Load document summaries from JSONL file."""
    print(f"Loading summaries from: {summary_path}")
    summaries = {}
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            summaries[doc['_id']] = doc.get('summary', '')
    print(f"Loaded {len(summaries)} summaries")
    return summaries


def load_faiss_index(index_path: str) -> faiss.Index:
    """Load FAISS index."""
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Loaded index with {index.ntotal} documents")
    return index


def load_faiss_index_with_embeddings(
    index_path: str,
    corpus_path: str
) -> Tuple[faiss.Index, List[str], Dict[str, dict], np.ndarray]:
    """Load FAISS index and get corresponding document IDs, documents, and embeddings."""
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)
    
    print(f"Loading corpus: {corpus_path}")
    corpus = read_jsonl(corpus_path)
    doc_ids = [doc['_id'] for doc in corpus]
    doc_id_to_doc = {doc['_id']: doc for doc in corpus}
    
    print(f"Extracting embeddings from FAISS index...")
    corpus_embeddings = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d).copy()
    
    norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print("Normalizing embeddings...")
        corpus_embeddings = corpus_embeddings / norms
    
    print(f"Loaded index with {index.ntotal} documents")
    
    return index, doc_ids, doc_id_to_doc, corpus_embeddings


def evaluate_retrieval_member_only(
    queries: List[dict],
    retrieval_results: Dict[str, List[str]],
    member_doc_ids: Set[str],
    top_k: int,
    query_target_similarities: Dict[str, float] = None
) -> Dict:
    """
    Evaluate retrieval results focusing on MEMBER documents only.
    
    Args:
        queries: List of query dictionaries
        retrieval_results: Dictionary mapping query_id to list of retrieved doc_ids
        member_doc_ids: Set of member document IDs
        top_k: Number of top documents to consider for evaluation
        query_target_similarities: Optional dict of query-target similarities
        
    Returns:
        Dictionary with evaluation metrics
    """
    member_queries = []
    for q in queries:
        target_doc_id = q.get('target_doc_id') or q.get('doc_id')
        membership = q.get('_membership', '')
        if membership == 'member' or target_doc_id in member_doc_ids:
            member_queries.append(q)
    
    hits_member = 0
    total_member = len(member_queries)
    
    mrr_sum = 0.0
    rank_1_hits = 0
    rank_3_hits = 0
    rank_5_hits = 0
    rank_10_hits = 0
    
    for query in member_queries:
        query_id = query['_id']
        target_doc_id = query.get('target_doc_id') or query.get('doc_id')
        
        retrieved = retrieval_results.get(query_id, [])[:top_k]
        
        if query_target_similarities and query_id in query_target_similarities:
            query[f'query_target_similarity_topk{top_k}'] = query_target_similarities[query_id]
        
        if target_doc_id in retrieved:
            hits_member += 1
            rank = retrieved.index(target_doc_id) + 1
            mrr_sum += 1.0 / rank
            
            if rank <= 1:
                rank_1_hits += 1
            if rank <= 3:
                rank_3_hits += 1
            if rank <= 5:
                rank_5_hits += 1
            if rank <= 10:
                rank_10_hits += 1
            
            query[f'target_rank_topk{top_k}'] = rank
            query[f'target_retrieved_topk{top_k}'] = True
        else:
            query[f'target_rank_topk{top_k}'] = -1
            query[f'target_retrieved_topk{top_k}'] = False
    
    hit_rate_member = hits_member / total_member if total_member > 0 else 0
    mrr = mrr_sum / total_member if total_member > 0 else 0
    
    return {
        'member_metrics': {
            'total_member_queries': total_member,
            'hits': hits_member,
            'hit_rate': hit_rate_member,
            'hit_rate_percentage': hit_rate_member * 100,
            'mrr': mrr,
            'recall_at_1': rank_1_hits / total_member if total_member > 0 else 0,
            'recall_at_3': rank_3_hits / total_member if total_member > 0 else 0,
            'recall_at_5': rank_5_hits / total_member if total_member > 0 else 0,
            'recall_at_10': rank_10_hits / total_member if total_member > 0 else 0,
            'recall_at_1_count': rank_1_hits,
            'recall_at_3_count': rank_3_hits,
            'recall_at_5_count': rank_5_hits,
            'recall_at_10_count': rank_10_hits
        }
    }


def compute_query_target_similarities(
    query_embeddings: np.ndarray,
    query_ids: List[str],
    target_doc_ids: List[str],
    corpus_embeddings: np.ndarray,
    doc_ids: List[str]
) -> Dict[str, float]:
    """
    Compute cosine similarity between query embeddings and their target documents.
    Uses pre-computed corpus embeddings from FAISS index instead of re-encoding.
    
    Args:
        query_embeddings: Pre-computed query embeddings
        query_ids: List of query IDs
        target_doc_ids: List of target document IDs corresponding to queries
        corpus_embeddings: Pre-computed corpus embeddings from FAISS index
        doc_ids: List of document IDs corresponding to corpus_embeddings
        
    Returns:
        Dictionary mapping query_id to similarity score
    """
    print(f"\nComputing query-target similarities using pre-computed embeddings...")
    
    # Build doc_id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    target_similarity_dict = {}
    missing_docs = 0
    
    for i, (query_id, target_doc_id) in enumerate(zip(query_ids, target_doc_ids)):
        if target_doc_id and target_doc_id in doc_id_to_idx:
            doc_idx = doc_id_to_idx[target_doc_id]
            doc_embedding = corpus_embeddings[doc_idx]
            # Both embeddings should already be normalized
            similarity = float(np.dot(query_embeddings[i], doc_embedding))
            target_similarity_dict[query_id] = similarity
        else:
            missing_docs += 1
    
    if missing_docs > 0:
        print(f"  Warning: {missing_docs} target documents not found in corpus embeddings")
    
    if target_similarity_dict:
        similarities = list(target_similarity_dict.values())
        print(f"\nQuery-Target Similarity Statistics:")
        print(f"  Computed: {len(target_similarity_dict)} similarities")
        print(f"  Mean:   {np.mean(similarities):.4f}")
        print(f"  Median: {np.median(similarities):.4f}")
        print(f"  Min:    {np.min(similarities):.4f}")
        print(f"  Max:    {np.max(similarities):.4f}")
    
    return target_similarity_dict

def retrieve_documents(
    queries: List[dict],
    index: faiss.Index,
    doc_ids: List[str],
    retrieval_model: SentenceTransformer,
    doc_id_to_doc: Dict[str, dict] = None,
    summaries: Dict[str, str] = None,
    max_top_k: int = 100,
    use_target_summary: bool = False
) -> Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str], List[str]]:
    """
    Retrieve top-k documents for each query.
    
    Args:
        max_top_k: Maximum number of documents to retrieve (will retrieve this many,
                   then slice for different top-k evaluations)
    
    Returns:
        Tuple of (retrieval_results, query_embeddings, query_ids, target_doc_ids, query_texts_with_summaries)
    """
    results = {}
    query_texts_for_encoding = []
    query_ids = []
    target_doc_ids = []
    
    print(f"Constructing {len(queries)} queries...")
    
    for q in tqdm(queries, desc="Constructing queries"):
        query_text = q['text']
        query_id = q['_id']
        target_doc_id = q.get('target_doc_id') or q.get('doc_id')
        
        if use_target_summary:
            if summaries and target_doc_id and target_doc_id in summaries:
                query_text_with_summaries = f"{summaries[target_doc_id]} {query_text}"
            else:
                query_text_with_summaries = query_text
        
        else:
            query_text_with_summaries = query_text
        
        query_texts_for_encoding.append(query_text_with_summaries)
        query_ids.append(query_id)
        target_doc_ids.append(target_doc_id if target_doc_id else '')
    
    print(f"Encoding {len(query_texts_for_encoding)} queries for retrieval...")
    query_embeddings = retrieval_model.encode(
        query_texts_for_encoding,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"Searching index for top-{max_top_k} documents per query...")
    scores, indices = index.search(query_embeddings.astype('float32'), max_top_k)
    
    for i, query_id in enumerate(query_ids):
        retrieved_doc_ids = [doc_ids[idx] for idx in indices[i] if idx < len(doc_ids)]
        results[query_id] = retrieved_doc_ids
    
    return results, query_embeddings, query_ids, target_doc_ids, query_texts_for_encoding


def get_attack_mode_name(
    use_target_summary: bool
) -> str:
    """Get the attack mode folder name based on flags."""
    if use_target_summary:
        return "target_summary"
    else:
        return "baseline"


def perform_retrieval(
    dataset: str,
    member_corpus_path: str,
    nonmember_corpus_path: str,
    index_path: str,
    query_path: str,
    retrieval_model_name: str,
    summary_path: str = None,
    use_gpu: bool = False,
    cache_dir: str = None,
    top_k_values: List[int] = None,
    use_target_summary: bool = False,
    mirabel_rho: float = 0.05,
    output_base_dir: str = "results/MEntA"
):
    """
    Perform retrieval on all queries and evaluate on member documents only.
    Supports multiple top-k values in a single run.
    
    Args:
        top_k_values: List of top-k values to evaluate
    """
    
    if top_k_values is None:
        top_k_values = [3]
    
    attack_mode_name = get_attack_mode_name(use_target_summary)
    
    num_variations = extract_num_variations(query_path)
    
    max_top_k = max(top_k_values)
    
    print(f"\n{'#'*60}")
    print(f"Dataset: {dataset}")
    print(f"Member corpus: {member_corpus_path}")
    print(f"Non-member corpus: {nonmember_corpus_path}")
    print(f"Index: {index_path}")
    print(f"Query file: {query_path}")
    print(f"Query variations: {num_variations}")
    print(f"Top-k values: {top_k_values}")
    print(f"Max top-k (for retrieval): {max_top_k}")
    print(f"Attack mode: {attack_mode_name}")
    print(f"Use target summary: {use_target_summary}")
    print(f"Mirabel rho: {mirabel_rho}")
    print(f"{'#'*60}")
    
    member_corpus_path_obj = Path(member_corpus_path)
    nonmember_corpus_path_obj = Path(nonmember_corpus_path)
    index_path_obj = Path(index_path)
    query_path_obj = Path(query_path)
    
    if not member_corpus_path_obj.exists():
        print(f"Error: Member corpus file not found: {member_corpus_path}")
        return
    
    if not nonmember_corpus_path_obj.exists():
        print(f"Error: Non-member corpus file not found: {nonmember_corpus_path}")
        return
    
    if not index_path_obj.exists():
        print(f"Error: Index file not found: {index_path}")
        return
    
    if not query_path_obj.exists():
        print(f"Error: Query file not found: {query_path}")
        return
    
    all_doc_ids, doc_id_to_doc, member_doc_ids, nonmember_doc_ids = load_member_nonmember_corpus(
        member_corpus_path, nonmember_corpus_path
    )
    
    print(f"Loading queries from: {query_path}")
    queries = read_jsonl(query_path)
    print(f"Loaded {len(queries)} queries (member + nonmember)")
    
    queries_with_doc_id = [q for q in queries if 'target_doc_id' in q or 'doc_id' in q]
    if len(queries_with_doc_id) == 0:
        print("Error: No queries have 'target_doc_id' or 'doc_id' field")
        return
    
    print(f"Found {len(queries_with_doc_id)} queries with target document IDs")
    
    member_queries = []
    nonmember_queries = []
    for q in queries_with_doc_id:
        target_doc_id = q.get('target_doc_id') or q.get('doc_id')
        membership = q.get('_membership', '')
        if membership == 'member' or target_doc_id in member_doc_ids:
            member_queries.append(q)
        else:
            nonmember_queries.append(q)
    
    print(f"Member queries: {len(member_queries)}")
    print(f"Non-member queries: {len(nonmember_queries)}")
    
    summaries = None
    if summary_path and use_target_summary:
        summaries = load_summaries(summary_path)
    
    index, index_doc_ids, _, corpus_embeddings = load_faiss_index_with_embeddings(
        index_path, member_corpus_path
    )
    
    retrieval_model = load_embedding_model(retrieval_model_name, use_gpu, cache_dir)
    
    retrieval_results, query_embeddings, query_ids, target_doc_ids, query_texts_with_summaries = retrieve_documents(
        queries_with_doc_id,
        index,
        index_doc_ids,
        retrieval_model,
        doc_id_to_doc=doc_id_to_doc,
        summaries=summaries,
        max_top_k=max_top_k,
        use_target_summary=use_target_summary
    )
    
    query_id_to_text = {qid: text for qid, text in zip(query_ids, query_texts_with_summaries)}
    for query in queries_with_doc_id:
        if query['_id'] in query_id_to_text:
            query['text_with_augmentation'] = query_id_to_text[query['_id']]
    
    query_target_similarities = compute_query_target_similarities(
        query_embeddings,
        query_ids,
        target_doc_ids,
        corpus_embeddings,  # Use pre-computed embeddings
        index_doc_ids       # Use doc_ids from index
    )
    
    print(f"\nUsing pre-computed embeddings from FAISS index for Mirabel detection...")
    queries_with_doc_id = apply_mirabel_to_queries(
        queries_with_doc_id,
        query_embeddings,
        corpus_embeddings,
        index_doc_ids,
        rho=mirabel_rho
    )
    
    member_queries_for_mirabel = [q for q in queries_with_doc_id 
                                   if q.get('_membership') == 'member' or 
                                   (q.get('target_doc_id') or q.get('doc_id')) in member_doc_ids]
    mirabel_stats = compute_mirabel_statistics(member_queries_for_mirabel)
    
    for top_k in top_k_values:
        print(f"\n{'='*60}")
        print(f"Processing top-k = {top_k}")
        print(f"{'='*60}")
        
        output_dir = Path(output_base_dir) / dataset / "retrieval" / attack_mode_name / f"topk{top_k}"
        output_path = output_dir / f"retrieval_{num_variations}.json"
        
        print(f"Evaluating retrieval on MEMBER documents only (top-k={top_k})...")
        metrics = evaluate_retrieval_member_only(
            queries_with_doc_id,
            retrieval_results,
            member_doc_ids,
            top_k=top_k,
            query_target_similarities=query_target_similarities
        )
        
        metrics['mirabel_detection'] = mirabel_stats
        
        retrieval_results_topk = {
            qid: docs[:top_k] for qid, docs in retrieval_results.items()
        }
        
        output_data = {
            'metadata': {
                'dataset': dataset,
                'member_corpus_file': member_corpus_path,
                'nonmember_corpus_file': nonmember_corpus_path,
                'query_file': query_path,
                'query_variations': num_variations,
                'index_file': index_path,
                'retrieval_model': retrieval_model_name,
                'top_k': top_k,
                'attack_mode': attack_mode_name,
                'use_target_summary': use_target_summary,
                'mirabel_rho': mirabel_rho,
                'num_total_queries': len(queries_with_doc_id),
                'num_member_queries': len(member_queries),
                'num_nonmember_queries': len(nonmember_queries)
            },
            'metrics': metrics,
            'retrieval_results': retrieval_results_topk,
            'queries': queries_with_doc_id
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_data, str(output_path))
        
        print(f"\nMEMBER-ONLY RETRIEVAL METRICS (top-k={top_k}):")
        print(f"  Total member queries: {metrics['member_metrics']['total_member_queries']}")
        print(f"  Hits: {metrics['member_metrics']['hits']}")
        print(f"  Hit rate: {metrics['member_metrics']['hit_rate_percentage']:.2f}%")
        print(f"  MRR: {metrics['member_metrics']['mrr']:.4f}")
        print(f"  Recall@1: {metrics['member_metrics']['recall_at_1']:.4f}")
        print(f"  Recall@3: {metrics['member_metrics']['recall_at_3']:.4f}")
        print(f"  Recall@5: {metrics['member_metrics']['recall_at_5']:.4f}")
        print(f"  Recall@10: {metrics['member_metrics']['recall_at_10']:.4f}")
        print(f"\nResults saved to: {output_path}")
    
    print(f"\n{'#'*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'#'*60}")
    print(f"\nMIRABEL DETECTION STATISTICS (MEMBER ONLY):")
    print(f"  Total queries: {mirabel_stats['total_queries']}")
    print(f"  Detected (target @ top-1): {mirabel_stats['detected_as_attack']}")
    print(f"  Not detected: {mirabel_stats['not_detected_as_attack']}")
    print(f"  Detection rate: {mirabel_stats['detection_percentage']:.2f}%")
    
    print(f"\nHit Rate Summary across top-k values:")
    print(f"{'Top-k':<10} {'Hit Rate %':<15} {'MRR':<10} {'Recall@1':<10}")
    print(f"{'-'*45}")
    
    print(f"\n{'#'*60}")


def main():
    parser = argparse.ArgumentParser(description='Retrieve documents and evaluate on member documents only')
    
    parser.add_argument('--dataset', type=str, default='BeIR_trec-covid',
                       help='Dataset name (used for output folder structure)')
    parser.add_argument('--member_corpus_path', type=str, 
                       default='data/BeIR_trec-covid/corpus_member.jsonl',
                       help='Path to member corpus JSONL file')
    parser.add_argument('--nonmember_corpus_path', type=str, 
                       default='data/BeIR_trec-covid/corpus_nonmember.jsonl',
                       help='Path to non-member corpus JSONL file')
    parser.add_argument('--index_path', type=str,  
                       default='data/BeIR_trec-covid/faiss_indices/thenlper--gte-large.faiss',
                       help='Path to FAISS index')
    parser.add_argument('--query_path', type=str,
                       default='results/MEntA/BeIR_trec-covid/queries/queries_5v.jsonl',
                       help='Path to query JSONL file') 
    parser.add_argument('--summary_path', type=str,
                       default='data/BeIR_trec-covid/summary.jsonl',
                       help='Path to summary JSONL file')
    parser.add_argument('--retrieval_model', type=str,
                       default='thenlper/gte-large',
                       help='Retrieval model') 
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU')
    parser.add_argument('--cache_dir', type=str,
                       default='~/.cache/huggingface/hub',
                       help='Cache directory')
    
    parser.add_argument('--top_k', type=str, default='3',
                       help='Top-k values. Supports: single (3), comma-separated (3,5,10), '
                            'range (3-10), range with step (3-20-5), or mixed (3,5,10-20-5)')
    parser.add_argument('--mirabel_rho', type=float, default=0.05,
                       help='Mirabel significance level')
    
    parser.add_argument('--use_target_summary', action='store_true',
                       help='Append target document summary to query')
    
    parser.add_argument('--output_base_dir', type=str,
                       default='results/MEntA--gte-large',
                       help='Base output directory')
    
    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    if args.cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    top_k_values = parse_top_k_range(args.top_k)
    print(f"Will evaluate top-k values: {top_k_values}")
    
    perform_retrieval(
        dataset=args.dataset,
        member_corpus_path=args.member_corpus_path,
        nonmember_corpus_path=args.nonmember_corpus_path,
        index_path=args.index_path,
        query_path=args.query_path,
        retrieval_model_name=args.retrieval_model,
        summary_path=args.summary_path,
        use_gpu=args.use_gpu,
        cache_dir=args.cache_dir,
        top_k_values=top_k_values,
        use_target_summary=args.use_target_summary,
        mirabel_rho=args.mirabel_rho,
        output_base_dir=args.output_base_dir
    )


if __name__ == '__main__':
    main()
