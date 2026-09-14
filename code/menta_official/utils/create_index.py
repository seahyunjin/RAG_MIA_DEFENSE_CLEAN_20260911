import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json
from utils.load_model import load_generator_model, load_embedding_model


def create_faiss_index(
    member_path: str,
    model_name: str,
    output_path: str,
    batch_size: int = 32,
    use_gpu: bool = False,
    cache_dir: str = None,
    text_key: str = "text"
):
    """
    Create FAISS index from member corpus file only.
    
    Args:
        member_path: Path to member corpus.jsonl file
        model_name: Sentence transformer model name or path
        output_path: Path to save FAISS index
        batch_size: Batch size for encoding
        use_gpu: Whether to use GPU for encoding
        cache_dir: Cache directory for models
        text_key: Key to extract text from records
    """
    print(f"\n{'='*60}")
    print(f"Creating index for:")
    print(f"  Members: {member_path}")
    print(f"{'='*60}")
    
    # Read member corpus
    print("Reading member corpus...")
    members = read_jsonl(member_path)
    print(f"Total member documents: {len(members)}")
    
    # Extract texts and create labels
    doc_ids = []
    texts = []
    
    # Process members
    for doc in members:
        doc_ids.append(doc.get('_id', str(len(doc_ids))))
        text = doc.get(text_key, "")
        # If text is empty, try combining other fields
        if not text:
            text_parts = []
            for key, value in doc.items():
                if key != '_id' and value and isinstance(value, str):
                    text_parts.append(value)
            text = ' '.join(text_parts).strip()
        texts.append(text)
    
    print(f"Total documents: {len(texts)}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = load_embedding_model(model_name, use_gpu, cache_dir)
    
    if use_gpu and not model.device.type == 'cuda':
        model = model.to('cuda')
    
    # Encode documents
    print("Encoding documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity if vectors are normalized)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add vectors to index
    index.add(embeddings.astype('float32'))
    
    print(f"Index created with {index.ntotal} vectors")
    
    # Save index
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    faiss.write_index(index, output_path)
    print(f"Index saved to: {output_path}")
    
    # Save metadata (doc_ids only, all are members)
    metadata_path = output_path.replace('.faiss', '_metadata.json')
    metadata = {
        'doc_ids': doc_ids,
        'num_members': len(members)
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    return index, doc_ids


def process_dataset(
    dataset_path: str,
    model_name: str,
    batch_size: int = 32,
    use_gpu: bool = False,
    cache_dir: str = "~/.cache/huggingface/hub",
    text_key: str = "text"
):
    """
    Process a dataset folder and create FAISS index for members only.
    
    Args:
        dataset_path: Path to dataset folder (e.g., data/BeIR_nfcorpus)
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        use_gpu: Whether to use GPU
        cache_dir: Cache directory for models
        text_key: Key to extract text from records
    """
    member_path = os.path.join(dataset_path, 'corpus_member.jsonl')
    
    if not os.path.exists(member_path):
        print(f"Warning: {member_path} does not exist, skipping...")
        return
    
    # Create output path
    model_name_safe = model_name.replace('/', '--')
    index_path = os.path.join(dataset_path, 'faiss_indices', f'{model_name_safe}.faiss')
    
    # Check if index already exists
    if os.path.exists(index_path):
        print(f"Index already exists: {index_path}, skipping...")
        return
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    create_faiss_index(
        member_path=member_path,
        model_name=model_name,
        output_path=index_path,
        batch_size=batch_size,
        use_gpu=use_gpu,
        cache_dir=cache_dir,
        text_key=text_key
    )


def main():
    parser = argparse.ArgumentParser(description='Create FAISS indices for member datasets')
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
        '--data_base',
        type=str,
        default='data',
        help='Base directory containing dataset folders'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for encoding'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--text_key',
        type=str,
        default='text',
        help='Key to extract text from records'
    )
    
    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    # Set offline mode if cache_dir is specified
    if args.cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    base_path = args.data_base
    
    for dataset in args.datasets:
        dataset_path = os.path.join(base_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue
        
        process_dataset(
            dataset_path=dataset_path,
            model_name=args.model,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            cache_dir=args.cache_dir,
            text_key=args.text_key
        )
    
    print("\n" + "="*60)
    print("All indices created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()