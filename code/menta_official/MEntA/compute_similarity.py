import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
import torch
import warnings
import re
from collections import defaultdict
import sys

# Add project root to Python path (MUST be before utils import)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_embedding_model

warnings.filterwarnings('ignore')


def split_text(text: str, min_length: int = 10) -> List[str]:
    """
    Universal text splitting function that handles both regular and structured text.
    Splits by sentences first, then by newlines for structured content.
    """
    if not text or not text.strip():
        return []
    
    segments = []
    
    # Split by sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if this "sentence" contains newlines (structured content)
        if '\n' in sentence:
            # Split by newlines and clean up
            lines = sentence.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove common list markers
                line = re.sub(r'^\d+[\.\)]\s+', '', line)
                line = re.sub(r'^[-•*]\s+', '', line)
                line = re.sub(r'^[A-Za-z][\.\)]\s+', '', line)
                
                # Normalize whitespace
                line = re.sub(r'\s+', ' ', line).strip()
                
                # Filter by minimum length
                if len(line) >= min_length:
                    segments.append(line)
        else:
            # Regular sentence
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            if len(sentence) >= min_length:
                segments.append(sentence)
    
    return segments


def compute_similarity_batch(
    texts_a: List[str],
    texts_b: List[str],
    embedding_model,
    batch_size: int = 64,
    desc: str = "Computing similarities"
) -> List[float]:
    """
    Compute cosine similarity between pairs of texts using embedding model.
    
    Args:
        texts_a: List of first texts
        texts_b: List of second texts  
        embedding_model: SentenceTransformer model
        batch_size: Batch size for encoding
        desc: Description for progress bar
        
    Returns:
        List of similarity scores
    """
    assert len(texts_a) == len(texts_b), "Text lists must have same length"
    
    if not texts_a:
        return []
    
    print(f"\n{desc}: Processing {len(texts_a):,} text pairs")
    
    # Encode all texts
    print("Encoding first set of texts...")
    embeddings_a = embedding_model.encode(
        texts_a,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print("Encoding second set of texts...")
    embeddings_b = embedding_model.encode(
        texts_b,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute cosine similarities (dot product since normalized)
    print("Computing similarities...")
    similarities = np.sum(embeddings_a * embeddings_b, axis=1)
    
    return similarities.tolist()


def precompute_document_text_units(
    corpus_map: Dict[str, dict],
    doc_ids: Set[str]
) -> Dict[str, List[str]]:
    """Pre-compute text units for all documents once."""
    doc_text_units = {}
    
    print(f"Pre-processing {len(doc_ids)} documents...")
    for doc_id in tqdm(doc_ids, desc="Processing documents"):
        if doc_id not in corpus_map:
            continue
        
        doc = corpus_map[doc_id]
        doc_fields = {k: v for k, v in doc.items() if k != '_id'}
        doc_text = '; '.join(f"{k}: {v}" for k, v in doc_fields.items() if v).strip()
        
        text_units = split_text(doc_text, min_length=10)
        
        if text_units:
            doc_text_units[doc_id] = text_units
    
    return doc_text_units


def extract_all_claims(
    queries: List[dict]
) -> Dict[str, List[str]]:
    """
    Extract claims for ALL queries using heuristic splitting.
    """
    query_to_claims = {}
    
    print(f"\nExtracting claims using heuristic splitting...")
    
    for query in tqdm(queries, desc="Splitting text"):
        query_id = query['query_id']
        answer = query['generated_answer']
        claims = split_text(answer, min_length=10)
        query_to_claims[query_id] = claims if claims else []
    
    total_claims = sum(len(claims) for claims in query_to_claims.values())
    print(f"Extracted {total_claims} total claims from {len(query_to_claims)} queries")
    
    return query_to_claims


def extract_target_doc_from_query_id(query_id: str) -> str:
    """Extract target document ID from query_id.
    
    Handles formats like:
    - 'q_MED-2450_v0' -> 'MED-2450'
    - 'MED-2450_v0' -> 'MED-2450'
    - 'q_MED-2450' -> 'MED-2450'
    """
    parts = query_id.split('_')
    
    # Handle 'q_XXX_vN' format
    if len(parts) >= 2:
        if parts[0] == 'q' and parts[-1].startswith('v'):
            return '_'.join(parts[1:-1])
        elif parts[0] == 'q':
            return '_'.join(parts[1:])
        elif parts[-1].startswith('v'):
            return '_'.join(parts[:-1])
    
    return query_id


def build_similarity_matrix_with_idk_detection(
    queries: List[dict],
    query_to_claims: Dict[str, List[str]],
    doc_text_units: Dict[str, List[str]],
    corpus_map: Dict[str, dict],
    embedding_model,
    batch_size: int = 64
) -> Tuple[Dict[Tuple[str, str, int], dict], Dict[str, dict]]:
    """
    Build similarity matrix for target documents AND compute IDK similarity scores.
    
    Returns:
        - similarity_matrix: {(query_id, target_doc_id, claim_idx): similarity_info}
        - idk_detection_details: IDK similarity scores per query
    """
    print("\nBuilding similarity matrix with IDK similarity scoring...")
    
    # Multiple IDK hypotheses
    idk_hypotheses = [
        "I don't have enough information to answer this question",
        "The provided text does not contain the answer to this question",
        "I cannot determine the answer from the given information",
        "This information is not mentioned in the provided text",
        "I am unable to answer based on the available information",
        "There is no information provided about this",
        "The text does not specify this information"
    ]
    
    # Collect all (query_id, target_doc_id, claim_idx) tuples
    pair_metadata = []
    
    for query_data in queries:
        query_id = query_data['query_id']
        target_doc = extract_target_doc_from_query_id(query_id)
        
        # Check if target_doc exists in corpus
        if not target_doc or target_doc not in corpus_map:
            continue
        
        # If document not in doc_text_units (maybe no text units extracted), skip
        if target_doc not in doc_text_units:
            continue
        
        claims = query_to_claims.get(query_id, [])
        
        if not claims:
            continue
        
        for claim_idx, claim in enumerate(claims):
            pair_metadata.append({
                'query_id': query_id,
                'doc_id': target_doc,
                'claim': claim,
                'claim_idx': claim_idx
            })
    
    print(f"Total (query, target_doc, claim) combinations: {len(pair_metadata)}")
    
    if not pair_metadata:
        print("WARNING: No valid pairs found for similarity checking!")
        return {}, {}
    
    # Flatten claim-text_unit pairs for document similarity
    all_claims_doc = []
    all_text_units_doc = []
    pair_indices_doc = []
    text_unit_indices_doc = []
    
    for i, meta in enumerate(pair_metadata):
        doc_id = meta['doc_id']
        claim = meta['claim']
        text_units = doc_text_units[doc_id]
        
        # Document similarity checks
        for tu_idx, text_unit in enumerate(text_units):
            all_claims_doc.append(claim)
            all_text_units_doc.append(text_unit)
            pair_indices_doc.append(i)
            text_unit_indices_doc.append(tu_idx)
    
    print(f"Total similarity checks for document: {len(all_claims_doc)}")
    
    # Process document similarity checks
    similarity_matrix = {}
    
    if all_claims_doc:
        print(f"\nComputing document similarities (batch_size={batch_size})...")
        all_similarities_doc = compute_similarity_batch(
            all_claims_doc,
            all_text_units_doc,
            embedding_model,
            batch_size=batch_size,
            desc="Document similarity"
        )
        
        # Aggregate results - find best matching text unit for each claim
        print("Aggregating document similarity results...")
        
        # Group results by pair index
        pair_results = defaultdict(list)
        for idx, (pair_idx, tu_idx, sim) in enumerate(zip(pair_indices_doc, text_unit_indices_doc, all_similarities_doc)):
            pair_results[pair_idx].append((tu_idx, sim))
        
        for pair_idx, results in pair_results.items():
            meta = pair_metadata[pair_idx]
            key = (meta['query_id'], meta['doc_id'], meta['claim_idx'])
            
            # Find best matching text unit
            best_tu_idx, best_sim = max(results, key=lambda x: x[1])
            text_units = doc_text_units[meta['doc_id']]
            best_text_unit = text_units[best_tu_idx] if best_tu_idx < len(text_units) else text_units[0]
            
            # Compute average similarity across all text units
            all_sims = [r[1] for r in results]
            
            similarity_matrix[key] = {
                'claim': meta['claim'],
                'best_text_unit': best_text_unit,
                'best_similarity': best_sim,
                'avg_similarity': float(np.mean(all_sims)),
                'max_similarity': float(np.max(all_sims)),
                'min_similarity': float(np.min(all_sims)),
                'num_text_units': len(results)
            }
    
    # Process IDK similarity scoring
    idk_detection_details = {}
    
    print(f"\nComputing IDK similarity scores with {len(idk_hypotheses)} hypotheses...")
    
    # Pre-encode all IDK hypotheses
    print("Encoding IDK hypotheses...")
    idk_embeddings = embedding_model.encode(
        idk_hypotheses,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Encode all claims once
    all_claims = [meta['claim'] for meta in pair_metadata]
    print(f"Encoding {len(all_claims)} claims for IDK detection...")
    claim_embeddings = embedding_model.encode(
        all_claims,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute similarities between all claims and all IDK hypotheses
    print("Computing claim-IDK similarities...")
    # Shape: (num_claims, num_idk_hypotheses)
    idk_similarities = np.dot(claim_embeddings, idk_embeddings.T)
    
    # Store results
    for pair_idx, meta in enumerate(pair_metadata):
        query_id = meta['query_id']
        claim = meta['claim']
        
        if query_id not in idk_detection_details:
            idk_detection_details[query_id] = {
                'claims': []
            }
        
        # Get similarities for this claim
        claim_idk_sims = idk_similarities[pair_idx]
        
        claim_entry = {
            'claim': claim,
            'idk_similarity_scores': {}
        }
        
        for hyp_idx, idk_hypothesis in enumerate(idk_hypotheses):
            claim_entry['idk_similarity_scores'][idk_hypothesis] = float(claim_idk_sims[hyp_idx])
         
        # Also store aggregate score s
        claim_entry['idk_max_similarity'] = float(np.max(claim_idk_sims))
        claim_entry['idk_avg_similarity'] = float(np.mean(claim_idk_sims))
        
        idk_detection_details[query_id]['claims'].append(claim_entry)
    
    print(f"\nSimilarity matrix built with {len(similarity_matrix)} entries")
    print(f"IDK similarity scores computed for {len(idk_detection_details)} queries")
    
    return similarity_matrix, idk_detection_details


def compute_and_save_similarity(
    output_file_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    cache_dir: str = None,
    use_gpu: bool = False,
    batch_size: int = 64
):
    """
    Compute similarity matrix for target documents with IDK similarity scoring.
    Uses both member and non-member corpus for document lookup.
    
    Args:
        output_file_path: Path to answers JSON file
        corpus_member_path: Path to corpus_member.jsonl
        corpus_nonmember_path: Path to corpus_nonmember.jsonl
        embedding_model_name: Name of the embedding model
        cache_dir: Cache directory for models
        use_gpu: Whether to use GPU
        batch_size: Batch size for encoding
    """
    print(f"\n{'#'*60}")
    print(f"Computing Similarity Matrix with IDK Similarity Scoring")
    print(f"Embedding Model: {embedding_model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"{'#'*60}")
    
    print("\nLoading data...")
    output_data = read_json(output_file_path)
    
    # Load both member and non-member corpus
    print(f"Loading member corpus from: {corpus_member_path}")
    corpus_member = read_jsonl(corpus_member_path)
    print(f"Loading non-member corpus from: {corpus_nonmember_path}")
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    
    # Build combined corpus map with membership labels
    corpus_map = {}
    member_doc_ids = set()
    nonmember_doc_ids = set()
    
    for doc in corpus_member:
        doc_id = doc.get('_id') or doc.get('id')
        if doc_id:
            corpus_map[doc_id] = doc
            member_doc_ids.add(doc_id)
    
    for doc in corpus_nonmember:
        doc_id = doc.get('_id') or doc.get('id')
        if doc_id:
            corpus_map[doc_id] = doc
            nonmember_doc_ids.add(doc_id)
    
    # Handle different answer file formats
    if 'results' in output_data:
        queries = output_data['results']
    elif isinstance(output_data, list):
        queries = output_data
    else:
        # Try to find a list of queries in the data
        for key in output_data:
            if isinstance(output_data[key], list):
                queries = output_data[key]
                break
        else:
            raise ValueError("Could not find query results in the output file")
    
    print(f"Loaded {len(queries)} query outputs")
    print(f"Loaded {len(member_doc_ids)} member documents")
    print(f"Loaded {len(nonmember_doc_ids)} non-member documents")
    print(f"Total documents in corpus: {len(corpus_map)}")
    
    # Find all unique target documents
    all_target_docs = set()
    member_targets = set()
    nonmember_targets = set()
    
    for query_data in queries:
        query_id = query_data.get('query_id') or query_data.get('id')
        if not query_id:
            continue
        target_doc = extract_target_doc_from_query_id(query_id)
        if target_doc in corpus_map:
            all_target_docs.add(target_doc)
            if target_doc in member_doc_ids:
                member_targets.add(target_doc)
            elif target_doc in nonmember_doc_ids:
                nonmember_targets.add(target_doc)
    
    print(f"\nTotal unique target documents to process: {len(all_target_docs)}")
    print(f"  - Member targets: {len(member_targets)}")
    print(f"  - Non-member targets: {len(nonmember_targets)}")
    
    print("\nLoading embedding model...")
    if cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    embedding_model = load_embedding_model(embedding_model_name, use_gpu, cache_dir)
    
    # STEP 1: Pre-compute document text units
    print(f"\nStep 1: Pre-computing document text units...")
    doc_text_units = precompute_document_text_units(
        corpus_map,
        all_target_docs
    )
    print(f"Processed {len(doc_text_units)} target documents")
    
    # Normalize query data format
    normalized_queries = []
    for q in queries:
        normalized_q = {
            'query_id': q.get('query_id') or q.get('id'),
            'generated_answer': q.get('generated_answer') or q.get('answer') or q.get('response', '')
        }
        if normalized_q['query_id'] and normalized_q['generated_answer']:
            normalized_queries.append(normalized_q)
    
    print(f"Normalized {len(normalized_queries)} queries with answers")
    
    # STEP 2: Extract all claims
    print(f"\nStep 2: Extracting claims...")
    query_to_claims = extract_all_claims(normalized_queries)
    total_claims = sum(len(claims) for claims in query_to_claims.values())
    print(f"Extracted {total_claims} total claims from {len(query_to_claims)} queries")
    
    # STEP 3: Build similarity matrix AND compute IDK scores
    print(f"\nStep 3: Building similarity matrix with IDK similarity scoring...")
    similarity_matrix, idk_detection_details = build_similarity_matrix_with_idk_detection(
        normalized_queries,
        query_to_claims,
        doc_text_units,
        corpus_map,
        embedding_model,
        batch_size
    )
    
    # Convert tuple keys to string for JSON serialization
    similarity_matrix_serializable = {
        f"{query_id}||{doc_id}||{claim_idx}": value
        for (query_id, doc_id, claim_idx), value in similarity_matrix.items()
    }
    
    # Build output path: replace 'answers' with 'similarity' in path
    output_file_path = Path(output_file_path)
    
    # Replace 'answers' with 'similarity' in filename
    filename = output_file_path.name.replace('answers', 'similarity')
    
    # Replace 'answers' and 'answers_{defense}' with corresponding 'similarity' patterns in directory path
    parts = list(output_file_path.parent.parts)
    new_parts = []
    for p in parts:
        if p == "answers":
            new_parts.append("similarity")
        elif p.startswith("answers_"):
            # Extract defense suffix and create similarity_{defense}
            defense_suffix = p[len("answers_"):]
            new_parts.append(f"similarity_{defense_suffix}")
        else:
            new_parts.append(p)
    
    base_dir = Path(*new_parts)
    
    # Create output directory if needed
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = base_dir / filename
    
    # Detect defense type from path
    defense_type = "none"
    for part in parts:
        if part.startswith("answers_"):
            defense_type = part[len("answers_"):]
            break
    
    # Add membership info to results
    query_membership = {}
    for query_data in normalized_queries:
        query_id = query_data['query_id']
        target_doc = extract_target_doc_from_query_id(query_id)
        if target_doc in member_doc_ids:
            query_membership[query_id] = 'member'
        elif target_doc in nonmember_doc_ids:
            query_membership[query_id] = 'nonmember'
        else:
            query_membership[query_id] = 'unknown'
    
    results_data = {
        'metadata': {
            'output_file': str(output_file_path),
            'corpus_member': corpus_member_path,
            'corpus_nonmember': corpus_nonmember_path,
            'defense_type': defense_type,
            'embedding_model': embedding_model_name,
            'batch_size': batch_size,
            'num_queries': len(query_to_claims),
            'total_claims': total_claims,
            'num_target_documents': len(doc_text_units),
            'num_member_targets': len(member_targets),
            'num_nonmember_targets': len(nonmember_targets),
            'matrix_size': len(similarity_matrix),
            'note': 'IDK similarity scores computed but NO conclusions made - evaluation happens in evaluate.py'
        },
        'query_to_claims': query_to_claims,
        'query_membership': query_membership,
        'idk_detection_details': idk_detection_details,
        'similarity_matrix': similarity_matrix_serializable
    }
    
    write_json(results_data, str(results_path))
    print(f"\nSimilarity matrix saved to: {results_path}")
    print(f"Defense type detected: {defense_type}")
    
    return results_path


def main():
    parser = argparse.ArgumentParser(
        description='Compute similarity matrix for membership inference attack analysis'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="results/MEntA/BeIR_scidocs/answers/target_summary/topk3/meta-llama--Llama-3.1-8B-Instruct/answers_5v.json",
        help='Path to answers JSON file'
    )
    parser.add_argument(
        '--corpus_member',
        type=str,
        default='data/BeIR_scidocs/corpus_member.jsonl',
        help='Path to member corpus JSONL file'
    )
    parser.add_argument(
        '--corpus_nonmember',
        type=str,
        default='data/BeIR_scidocs/corpus_nonmember.jsonl',
        help='Path to non-member corpus JSONL file'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Embedding model name'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for inference'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for embedding computation'
    )
    
    args = parser.parse_args()
    
    compute_and_save_similarity(
        output_file_path=args.output_file,
        corpus_member_path=args.corpus_member,
        corpus_nonmember_path=args.corpus_nonmember,
        embedding_model_name=args.embedding_model,
        cache_dir=args.cache_dir,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("Similarity computation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()