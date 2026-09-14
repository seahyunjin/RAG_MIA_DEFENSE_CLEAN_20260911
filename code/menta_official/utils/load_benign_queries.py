#!/usr/bin/env python3
"""
Download Benign Queries from BeIR Datasets
Downloads the original queries from BeIR datasets (nfcorpus, scidocs, trec-covid)
and saves them as benign_queries.jsonl for comparison with MIA attack queries.
"""

import os
import sys
import json
import requests
import zipfile
import csv
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from process_json import write_jsonl


# Dataset mapping: our naming -> BeIR naming
DATASET_MAPPING = {
    'BeIR_nfcorpus': 'nfcorpus',
    'BeIR_scidocs': 'scidocs', 
    'BeIR_trec-covid': 'trec-covid'
}
 
# BeIR download URLs
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def download_file(url: str, output_path: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=f"Downloading {output_path.name}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def load_queries_from_tsv(queries_file: Path) -> List[Dict]:
    """Load queries from TSV file."""
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            query_record = {
                '_id': row.get('_id', row.get('query-id', '')),
                'text': row.get('text', row.get('query', '')),
                'is_benign': True
            }
            queries.append(query_record)
    
    return queries


def download_generated_queries_from_hf(dataset_name: str, cache_dir: str = None) -> List[Dict]:
    """
    Download generated queries from Hugging Face BeIR datasets.
    
    Args:
        dataset_name: BeIR dataset name (e.g., 'nfcorpus', 'scidocs', 'trec-covid')
        cache_dir: Cache directory for downloads
    
    Returns:
        List of query dictionaries
    """
    print(f"\nDownloading generated queries for {dataset_name} from Hugging Face...")
    
    try:
        from datasets import load_dataset
        
        # Set cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
            os.environ['HF_DATASETS_CACHE'] = cache_dir
        
        # Load the generated queries dataset
        hf_dataset = f"BeIR/{dataset_name}-generated-queries"
        print(f"  Loading from: {hf_dataset}")
        
        dataset = load_dataset(
            hf_dataset,
            split="train",
            cache_dir=cache_dir
        )
        
        print(f"  Loaded {len(dataset)} generated queries")
        
        queries = []
        for item in tqdm(dataset, desc="  Processing queries"):
            # Only save _id and text (query)
            query_record = {
                '_id': item.get('_id', ''),
                'text': item.get('query', '')
            }
            queries.append(query_record)
        
        return queries
        
    except ImportError:
        print("  ❌ Error: datasets library not installed. Please install: pip install datasets")
        raise
    except Exception as e:
        print(f"  ❌ Error loading from Hugging Face: {e}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download benign queries from BeIR datasets'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['BeIR_nfcorpus', 'BeIR_scidocs', 'BeIR_trec-covid'],
        help='Datasets to download queries for'
    )
    parser.add_argument(
        '--output_base',
        type=str,
        default='data',
        help='Base output directory'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for downloads'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print("DOWNLOADING BENIGN QUERIES FROM BeIR DATASETS")
    print(f"Datasets: {args.datasets}")
    print(f"Output base: {args.output_base}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"{'#'*60}")
    
    total_queries = 0
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        # Get BeIR dataset name
        if dataset_name not in DATASET_MAPPING:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")
            continue
        
        beir_name = DATASET_MAPPING[dataset_name]
        
        # Output path
        output_dir = Path(args.output_base) / dataset_name
        output_file = output_dir / 'benign_queries.jsonl'
        
        # Check if file exists
        if output_file.exists() and not args.overwrite:
            print(f"⚠️  File already exists: {output_file}")
            print("  Use --overwrite to replace. Skipping...")
            existing = sum(1 for _ in open(output_file))
            print(f"  Existing queries: {existing}")
            total_queries += existing
            continue
        
        # Download queries from Hugging Face generated queries
        try:
            queries = download_generated_queries_from_hf(beir_name, args.cache_dir)
        except Exception as e:
            print(f"❌ Error downloading {beir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if not queries:
            print(f"⚠️  No queries found for {dataset_name}")
            continue
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write queries
        write_jsonl(queries, str(output_file))
        
        print(f"\n✓ Saved {len(queries)} queries to: {output_file}")
        
        total_queries += len(queries)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"Total queries downloaded: {total_queries}")
    print(f"Saved to: {args.output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()