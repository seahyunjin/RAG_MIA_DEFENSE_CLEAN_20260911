#!/usr/bin/env python3
"""
Mask-based MIA Document Retrieval Script
Retrieves documents for mask-based queries using FAISS index.
"""


import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys



# Add project root to Python path (MUST be before utils import)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_generator_model, load_embedding_model



def load_faiss_index_simple(index_path: str, corpus_member_path: str, corpus_nonmember_path: str) -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index and corresponding document IDs from corpus jsonl files."""
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)


    # Load member corpus
    corpus_member_path_obj = Path(corpus_member_path)
    if not corpus_member_path_obj.exists():
        raise FileNotFoundError(f"Member corpus file not found: {corpus_member_path_obj}")
    
    print(f"Loading member corpus: {corpus_member_path_obj}")
    corpus_member = read_jsonl(str(corpus_member_path_obj))
    
    # Load nonmember corpus
    corpus_nonmember_path_obj = Path(corpus_nonmember_path)
    if not corpus_nonmember_path_obj.exists():
        raise FileNotFoundError(f"Nonmember corpus file not found: {corpus_nonmember_path_obj}")
    
    print(f"Loading nonmember corpus: {corpus_nonmember_path_obj}")
    corpus_nonmember = read_jsonl(str(corpus_nonmember_path_obj))
    
    # Combine both corpora
    corpus = corpus_member + corpus_nonmember
    doc_ids = [doc["_id"] for doc in corpus]


    if len(doc_ids) != index.ntotal:
        print(
            f"Warning: corpus size ({len(doc_ids)}) "
            f"!= index.ntotal ({index.ntotal})."
        )


    print(f"Loaded index with {index.ntotal} vectors / {len(doc_ids)} doc IDs")
    print(f"  - Member docs: {len(corpus_member)}")
    print(f"  - Nonmember docs: {len(corpus_nonmember)}")
    return index, doc_ids



def retrieve_documents_for_mask_queries(
    queries_path: str,
    index_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_path: str,
    retrieval_model_name: str,
    use_gpu: bool = False,
    cache_dir: str = None,
    top_k: int = 5,
):
    """
    Retrieve documents for Mask-based MIA queries using FAISS index.


    queries_path: JSONL with fields:
        - _id (doc id)
        - masked_text (document with masks)
        - query_prompt (prompt template)
        - mask_values (ground truth for masks)
        - mask_positions
        - membership_label (1=member, 0=non-member)
        - original_text
    
    For MBA, we encode the masked_text to retrieve relevant documents.
    """
    print(f"\n{'#'*60}")
    print(f"Mask-based MIA Document Retrieval")
    print(f"Queries: {queries_path}")
    print(f"Index: {index_path}")
    print(f"Member corpus: {corpus_member_path}")
    print(f"Nonmember corpus: {corpus_nonmember_path}")
    print(f"Retrieval model: {retrieval_model_name}")
    print(f"Top-k: {top_k}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")


    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping.")
        return


    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")


    # Load queries
    queries_data = read_jsonl(queries_path)
    print(f"Loaded {len(queries_data)} queries")


    # Load FAISS index and doc IDs
    index, doc_ids = load_faiss_index_simple(index_path, corpus_member_path, corpus_nonmember_path)


    # Load retrieval model
    retrieval_model = load_embedding_model(retrieval_model_name, use_gpu, cache_dir)


    # Prepare all query texts and metadata
    print("\nPreparing queries for retrieval...")
    all_texts = []
    query_metadata = []


    for q in queries_data:
        doc_id = q["_id"]
        masked_text = q["masked_text"]
        query_prompt = q["query_prompt"]
        mask_values = q["mask_values"]
        mask_positions = q.get("mask_positions", [])
        membership_label = q["membership_label"]
        original_text = q.get("original_text", "")
        masking_strategy = q.get("masking_strategy", "")


        all_texts.append(query_prompt)
        query_metadata.append({
            "doc_id": doc_id,
            "masked_text": masked_text,
            "query_prompt": query_prompt,
            "mask_values": mask_values,
            "mask_positions": mask_positions,
            "membership_label": membership_label,
            "original_text": original_text,
            "masking_strategy": masking_strategy,
        })


    print(f"Total queries to process: {len(all_texts)}")
    member_count = sum(1 for q in queries_data if q["membership_label"] == 1)
    nonmember_count = len(queries_data) - member_count
    print(f"  - Member queries: {member_count}")
    print(f"  - Non-member queries: {nonmember_count}")


    # Encode all queries in batches
    print("\nEncoding queries...")
    query_embeddings = retrieval_model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


    # Search index for all queries
    print(f"\nSearching index for top-{top_k} documents per query...")
    scores, indices = index.search(query_embeddings.astype("float32"), top_k)


    # Organize retrieval results keyed by doc_id and calculate hit rates
    print("\nOrganizing retrieval results and calculating hit rates...")
    retrieval_results = {}
    
    # Hit rate tracking
    total_hits = 0
    member_hits = 0
    nonmember_hits = 0
    member_queries = 0
    nonmember_queries = 0


    for i, (score_list, index_list, meta) in enumerate(
        zip(scores, indices, query_metadata)
    ):
        retrieved_doc_ids = [
            doc_ids[idx] for idx in index_list if 0 <= idx < len(doc_ids)
        ]
        retrieved_scores = [float(s) for s in score_list]
        
        # Check if target document is in retrieved results (hit)
        target_doc_id = meta["doc_id"]
        is_hit = target_doc_id in retrieved_doc_ids
        hit_rank = retrieved_doc_ids.index(target_doc_id) + 1 if is_hit else -1
        
        # Update hit counters
        if is_hit:
            total_hits += 1
            if meta["membership_label"] == 1:
                member_hits += 1
            else:
                nonmember_hits += 1
        
        # Update query counters
        if meta["membership_label"] == 1:
            member_queries += 1
        else:
            nonmember_queries += 1


        retrieval_results[meta["doc_id"]] = {
            "doc_id": meta["doc_id"],
            "masked_text": meta["masked_text"],
            "query_prompt": meta["query_prompt"],
            "mask_values": meta["mask_values"],
            "mask_positions": meta["mask_positions"],
            "membership_label": meta["membership_label"],
            "original_text": meta["original_text"],
            "masking_strategy": meta["masking_strategy"],
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_scores": retrieved_scores,
            "is_hit": is_hit,
            "hit_rank": hit_rank,
        }
    
    # Calculate hit rates
    overall_hit_rate = (total_hits / len(all_texts) * 100) if len(all_texts) > 0 else 0.0
    member_hit_rate = (member_hits / member_queries * 100) if member_queries > 0 else 0.0
    nonmember_hit_rate = (nonmember_hits / nonmember_queries * 100) if nonmember_queries > 0 else 0.0


    # Prepare output data
    output_data = {
        "metadata": {
            "queries_file": queries_path,
            "index_file": index_path,
            "corpus_member_file": corpus_member_path,
            "corpus_nonmember_file": corpus_nonmember_path,
            "retrieval_model": retrieval_model_name,
            "top_k": top_k,
            "num_queries": len(all_texts),
            "num_member_queries": member_count,
            "num_nonmember_queries": nonmember_count,
            # Hit rate statistics
            "hit_rate_overall": round(overall_hit_rate, 2),
            "hit_rate_member": round(member_hit_rate, 2),
            "hit_rate_nonmember": round(nonmember_hit_rate, 2),
            "total_hits": total_hits,
            "member_hits": member_hits,
            "nonmember_hits": nonmember_hits,
        },
        "retrieval_results": retrieval_results,
    }


    # Save results
    write_json(output_data, output_path)


    print(f"\n{'='*60}")
    print(f"Retrieval completed!")
    print(f"Processed {len(retrieval_results)} queries")
    print(f"\nHit Rate Statistics:")
    print(f"  Overall Hit Rate: {overall_hit_rate:.2f}% ({total_hits}/{len(all_texts)})")
    print(f"  Member Hit Rate: {member_hit_rate:.2f}% ({member_hits}/{member_queries})")
    print(f"  Nonmember Hit Rate: {nonmember_hit_rate:.2f}% ({nonmember_hits}/{nonmember_queries})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")



def main():
    parser = argparse.ArgumentParser(
        description="Retrieve documents for Mask-based MIA queries"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default="results/MBA/BeIR_nfcorpus/queries_paraphrased/queries.jsonl",
        help="Path to queries JSONL file",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/BeIR_nfcorpus/faiss_indices/sentence-transformers--all-mpnet-base-v2.faiss",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--corpus_member_file",
        type=str,
        default="data/BeIR_nfcorpus/corpus_member.jsonl",
        help="Path to member corpus JSONL file",
    )
    parser.add_argument(
        "--corpus_nonmember_file",
        type=str,
        default="data/BeIR_nfcorpus/corpus_nonmember.jsonl",
        help="Path to nonmember corpus JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/MBA/BeIR_nfcorpus/retrieval_paraphrased",
        help="Base output directory for retrieval results",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Retrieval model name",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/hub",
        help="Cache directory for models",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of documents to retrieve per query",
    )


    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    if args.cache_dir:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


    # Construct output path with topk{k} subfolder
    output_path = Path(args.output_dir) / f"topk{args.top_k}_results.json"


    retrieve_documents_for_mask_queries(
        queries_path=args.queries_file,
        index_path=args.index_path,
        corpus_member_path=args.corpus_member_file,
        corpus_nonmember_path=args.corpus_nonmember_file,
        output_path=str(output_path),
        retrieval_model_name=args.retrieval_model,
        use_gpu=args.use_gpu,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
    )


    print("\n" + "=" * 60)
    print("Mask-based MIA retrieval completed successfully!")
    print("=" * 60)



if __name__ == "__main__":
    main()
