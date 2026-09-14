import os
import argparse
from datasets import load_dataset
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from env_config import ensure_hf_network_access, load_repo_dotenv

DATA_PATH = "data/"

DATASET_MAPPING = {
    "BeIR_nfcorpus": ("BeIR/nfcorpus", "corpus"),
    "BeIR_trec-covid": ("BeIR/trec-covid", "corpus"),
    "BeIR_scidocs": ("BeIR/scidocs", "corpus"),
}


def download_dataset(repo_id: str, config: str, save_dir: str = DATA_PATH):
    """
    Download a dataset from Hugging Face and save it locally.

    Args:
        repo_id (str): The dataset repository ID on Hugging Face (e.g., 'BeIR/nfcorpus').
        save_dir (str): The directory where the dataset should be saved.
    """

    save_path = os.path.join(save_dir, repo_id.replace("/", "_"))

    if os.path.exists(save_path):
        print(
            f"Directory '{save_path}' already exists. Skipping download."
        )
        return

    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading {repo_id} ...")
    ensure_hf_network_access()
    dataset = load_dataset(repo_id, config)

    for split, split_dataset in dataset.items():
        split_path = os.path.join(save_path, f"{split}.jsonl")
        print(f"Saving {split} split to {split_path}")
        split_dataset.to_json(split_path)

    print(f"Download and save complete for {repo_id}.")

def deduplicate_jsonl(input_path: str, output_path: str, key: str = None):
    """
    Deduplicate a JSONL file based on full line content or a specific key.
    
    Args:
        input_path (str): Path to the input .jsonl file.
        output_path (str): Path to save the deduplicated .jsonl file.
        key (str, optional): Specific key to deduplicate on. If None, deduplicate entire records.
    """
    if os.path.exists(output_path):
        print(
            f"Output path '{output_path}' already exists. Skipping deduplicating."
        )
        return
    
    seen = set()
    deduped_records = []
    
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            
            # Choose what to use for checking duplicates
            identifier = json.dumps(record, sort_keys=True) if key is None else record.get(key)
            
            if identifier not in seen:
                seen.add(identifier)
                deduped_records.append(record)
    
    # Save deduplicated records
    with open(output_path, "w", encoding="utf-8") as outfile:
        for record in deduped_records:
            json_record = json.dumps(record, ensure_ascii=False)
            outfile.write(json_record + "\n")

    print(f"Deduplication complete. {len(deduped_records)} unique records saved to {output_path}.")

def split_samples(
    input_path: str,
    member_output_path: str,
    nonmember_output_path: str,
    num_samples: int = 1000,
    similarity_threshold: float = 0.95,
    seed: int = 42,
    text_key: str = "text",
    batch_size: int = 5000,
):
    """
    Split samples into members and non-members following the methodology:
    1. Randomly select non-members from the dataset
    2. Find near-duplicates of non-members in the remaining dataset using TF-IDF
    3. Remove near-duplicates to ensure non-members don't overlap with members
    4. Randomly select members from the cleaned remaining dataset
    """
    if os.path.exists(member_output_path) and os.path.exists(nonmember_output_path):
        print(f"Output paths already exist. Skipping split_samples.")
        return

    import sys
    from scipy import sparse
    sys.stdout.reconfigure(line_buffering=True)

    print("✅ Loading records from:", input_path, flush=True)
    with open(input_path, "r", encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile]

    print(f"✅ Loaded {len(records)} records", flush=True)

    if len(records) < num_samples * 2:
        raise ValueError(f"Not enough records. Need at least {num_samples * 2}, got {len(records)}")

    rng = np.random.default_rng(seed)
    
    # Shuffle all indices
    all_indices = np.arange(len(records))
    rng.shuffle(all_indices)
    
    # Step 1: Select non-member indices (first num_samples after shuffle)
    nonmember_indices_list = list(all_indices[:num_samples])
    remaining_indices = list(all_indices[num_samples:])
    
    print(f"✅ Selected {len(nonmember_indices_list)} non-members", flush=True)
    print(f"✅ Remaining pool size: {len(remaining_indices)}", flush=True)

    # Extract texts
    nonmember_texts = [records[i].get(text_key, "") for i in nonmember_indices_list]
    remaining_texts = [records[i].get(text_key, "") for i in remaining_indices]
    
    # Step 2: Build TF-IDF vectors
    print("✅ Fitting TF-IDF model...", flush=True)
    tfidf = TfidfVectorizer(max_features=10000)  # Limit vocabulary size
    
    # Fit on all texts to ensure consistent vocabulary
    all_texts = nonmember_texts + remaining_texts
    tfidf.fit(all_texts)
    
    # Transform non-members (small, can convert to dense)
    nonmember_matrix = tfidf.transform(nonmember_texts).toarray().astype(np.float32)
    faiss.normalize_L2(nonmember_matrix)
    
    print(f"✅ Non-member matrix shape: {nonmember_matrix.shape}", flush=True)
    print(f"✅ Remaining records to process: {len(remaining_texts)}", flush=True)

    dim = nonmember_matrix.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    # Auto-detect GPU availability
    index = cpu_index
    num_gpus = faiss.get_num_gpus()
    print(f"🔍 FAISS detected {num_gpus} GPU(s)", flush=True)
    
    if num_gpus > 0:
        try:
            print("🔁 GPU detected, initializing FAISS GPU...", flush=True)
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("✅ FAISS is using GPU", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to use GPU: {e}. Falling back to CPU.", flush=True)
            index = cpu_index
    else:
        print("💻 Using CPU FAISS index", flush=True)

    # Step 3: Add ONLY non-member vectors to the index
    index.add(nonmember_matrix)
    
    # Step 4: Search remaining records against non-members IN BATCHES
    print("🔍 Searching remaining records against non-members...", flush=True)
    k = min(10, num_samples)
    
    to_remove = set()
    num_batches = (len(remaining_texts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_texts))
        
        batch_texts = remaining_texts[start_idx:end_idx]
        batch_matrix = tfidf.transform(batch_texts).toarray().astype(np.float32)
        faiss.normalize_L2(batch_matrix)
        
        similarities, _ = index.search(batch_matrix, k=k)
        
        for i, sim_row in enumerate(similarities):
            max_sim = np.max(sim_row)
            if max_sim > similarity_threshold:
                to_remove.add(start_idx + i)

    print(f"🧹 Found {len(to_remove)} near-duplicates to remove", flush=True)
    
    # Step 6: Clean the remaining pool
    cleaned_remaining_indices = [idx for i, idx in enumerate(remaining_indices) if i not in to_remove]
    
    print(f"✅ Cleaned pool size: {len(cleaned_remaining_indices)}", flush=True)

    if len(cleaned_remaining_indices) < num_samples:
        raise ValueError(f"Not enough clean records. Need {num_samples}, got {len(cleaned_remaining_indices)}")

    # Step 7: Randomly select members from cleaned pool
    rng.shuffle(cleaned_remaining_indices)
    member_indices = cleaned_remaining_indices[:num_samples]
    
    # Prepare final records
    member_records = [records[i] for i in member_indices]
    nonmember_records = [records[i] for i in nonmember_indices_list]

    # Save members
    with open(member_output_path, "w", encoding="utf-8") as out:
        for record in member_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(member_records)} member records → {member_output_path}", flush=True)

    # Save non-members
    with open(nonmember_output_path, "w", encoding="utf-8") as out:
        for record in nonmember_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(nonmember_records)} non-member records → {nonmember_output_path}", flush=True)

def process_dataset(repo_id: str, config: str, data_path: str = DATA_PATH):
    """
    Process a dataset by downloading and deduplicating it.

    Args:
        repo_id (str): The dataset repository ID on Hugging Face (e.g., 'BeIR/nfcorpus').
        config (str): The configuration for the dataset.
        data_path (str): Base directory where processed dataset folders are stored.
    """
    save_path = os.path.join(data_path, repo_id.replace("/", "_"))
    origin_path = os.path.join(save_path, f"{config}.jsonl")
    deduped_path = os.path.join(save_path, f"{config}_deduped.jsonl")
    member_path = os.path.join(save_path, f"{config}_member.jsonl")
    nonmember_path = os.path.join(save_path, f"{config}_nonmember.jsonl")

    download_dataset(repo_id, config, save_dir=data_path)
    deduplicate_jsonl(input_path=origin_path, output_path=deduped_path, key="title")

    split_samples(
        input_path=deduped_path,
        member_output_path=member_path,
        nonmember_output_path=nonmember_path,
        num_samples=1000,
        similarity_threshold=0.95,
        seed=42,
        text_key="text",
    )

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess BEIR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"],
        help="Dataset folder names to process",
    )
    parser.add_argument(
        "--data_base",
        default=DATA_PATH,
        help="Base directory for processed dataset folders",
    )
    args = parser.parse_args()

    load_repo_dotenv()
    ensure_hf_network_access()

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_MAPPING:
            print(f"Warning: unknown dataset {dataset_name}, skipping...")
            continue
        repo_id, config = DATASET_MAPPING[dataset_name]
        process_dataset(repo_id, config, data_path=args.data_base)


if __name__ == "__main__":
    main()