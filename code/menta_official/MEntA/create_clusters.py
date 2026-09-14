import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from tqdm import tqdm
import sys
# Add project root to Python path (MUST be before utils import)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json

def read_jsonl(file_path: str) -> List[dict]:
    """Read JSONL file and return list of records."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_json(data: dict, file_path: str):
    """Write dictionary to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_embeddings_from_faiss(index_path: str) -> np.ndarray:
    """Load embeddings from FAISS index efficiently."""
    index = faiss.read_index(index_path)
    n = index.ntotal
    d = index.d
    
    # Batch reconstruction for better memory efficiency
    batch_size = 10000
    embeddings = np.zeros((n, d), dtype=np.float32)
    
    for start_idx in tqdm(range(0, n, batch_size), desc="Loading embeddings"):
        end_idx = min(start_idx + batch_size, n)
        batch_vectors = np.zeros((end_idx - start_idx, d), dtype=np.float32)
        for i, idx in enumerate(range(start_idx, end_idx)):
            batch_vectors[i] = index.reconstruct(idx)
        embeddings[start_idx:end_idx] = batch_vectors
    
    return embeddings


def create_knn_clusters(
    embeddings: np.ndarray,
    doc_ids: List[str],
    k: int = 5
) -> Dict[str, List[str]]:
    """
    Create clusters where each document has its own cluster of k nearest neighbors.
    Each document becomes a cluster center with k-1 nearest neighbors PLUS itself.
    
    Args:
        embeddings: Document embeddings (n_docs, d)
        doc_ids: Document IDs
        k: Total cluster size (including the center document itself)
    
    Returns:
        clusters: Dict mapping doc_id to list of k doc_ids (including itself)
    """
    n_docs = len(embeddings)
    
    print(f"\nCreating k-NN clusters:")
    print(f"  Total documents: {n_docs}")
    print(f"  Cluster size (k): {k} (including center document)")
    print(f"  Total clusters to create: {n_docs} (one per document)")
    
    # Build FAISS index for efficient nearest neighbor search
    print("  Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
    index.add(embeddings.astype('float32'))
    
    # Search for k nearest neighbors (including the document itself)
    print(f"  Finding {k} nearest neighbors for each document...")
    distances, indices = index.search(embeddings.astype('float32'), k)
    
    # Create clusters
    clusters = {}
    for i, doc_id in enumerate(tqdm(doc_ids, desc="Creating clusters")):
        # Get k nearest neighbor indices (including the document itself)
        neighbor_indices = indices[i][:k]
        
        # Map indices to doc_ids
        neighbor_doc_ids = [doc_ids[idx] for idx in neighbor_indices]
        
        # Store cluster with doc_id as key
        clusters[doc_id] = neighbor_doc_ids
    
    print(f"\n  Created {len(clusters)} clusters")
    
    return clusters

def process_dataset(
    dataset_path: str,
    model_name: str,
    folder_type: str,
    k: int = 5,
):
    """
    Create k-NN clusters for t_0 corpus of a dataset.
    
    Args:
        dataset_path: Path to dataset folder
        model_name: Sentence transformer model name
        folder_type: 'snapshot' or 'persistent'
        k: Total cluster size (including the center document)
        batch_size: Batch size for encoding
        use_gpu: Whether to use GPU
        cache_dir: Cache directory for models
    """
    print(f"\n{'#'*60}")
    print(f"Processing {folder_type} in {dataset_path}")
    print(f"{'#'*60}")
    
    # Paths
    corpus_path = os.path.join(dataset_path, folder_type, 'corpus', 'corpus_t_0.jsonl')
    output_dir = os.path.join(dataset_path, 'clusters', model_name.replace('/', '--'))
    
    if not os.path.exists(corpus_path):
        print(f"Warning: {corpus_path} does not exist, skipping...")
        return
    
    # Read corpus
    print(f"Reading {corpus_path}...")
    corpus = read_jsonl(corpus_path)
    doc_ids = [doc['_id'] for doc in corpus]
    print(f"Total documents: {len(corpus)}")
    
    # Try to load embeddings from FAISS index
    faiss_index_path = os.path.join(dataset_path, folder_type, 'faiss_indices', model_name.replace('/', '--'), 'index_t_0.faiss')
    
    if os.path.exists(faiss_index_path):
        print(f"Loading embeddings from FAISS index: {faiss_index_path}")
        embeddings = load_embeddings_from_faiss(faiss_index_path)
        
        # Create k-NN clusters
        clusters = create_knn_clusters(embeddings, doc_ids, k)
        
        # Save clustering results
        output_data = {
            'clusters': clusters,
            'metadata': {
                'n_documents': len(corpus),
                'n_clusters': len(clusters),
                'k': k,
                'model': model_name,
                'clustering_type': 'knn',
                'description': f'Each document has its own cluster containing {k} nearest neighbors (INCLUDING itself).'
            }
        }
        
        output_path = os.path.join(output_dir, f'clusters_t_0_knn_k{k}.jsonl')
        write_json(output_data, output_path)
        print(f"\nClusters saved to: {output_path}")

    else:
        print("FAISS index not found")

def main():
    parser = argparse.ArgumentParser(
        description='Create k-NN clusters - each document gets a cluster of k nearest neighbors'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='thenlper/gte-large',
        help='Sentence transformer model name or path'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['BeIR_nfcorpus', 'BeIR_scidocs', 'BeIR_trec-covid'],
        help='Dataset names to process'
    )
    parser.add_argument(
        '--folder_type',
        type=str,
        choices=['snapshot', 'persistent', 'both'],
        default='snapshot',
        help='Which folder type to process'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of nearest neighbors'
    )
    
    args = parser.parse_args()
    
    base_path = 'data'
    folder_types = ['snapshot', 'persistent'] if args.folder_type == 'both' else [args.folder_type]
    
    for dataset in args.datasets:
        dataset_path = os.path.join(base_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue
        
        for folder_type in folder_types:
            process_dataset(
                dataset_path=dataset_path,
                model_name=args.model,
                folder_type=folder_type,
                k=args.k
            )
    
    print("\n" + "="*60)
    print("All clustering completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()