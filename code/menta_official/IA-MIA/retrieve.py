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


def load_summaries(summary_path: str) -> Dict[str, str]:
    """Load summaries JSONL into a dict: doc_id -> summary."""
    print(f"Loading summaries from: {summary_path}")
    data = read_jsonl(summary_path)
    summary_map = {}

    for item in data:
        # assume keys: "_id" and "summary"
        doc_id = item.get("_id")
        summary = item.get("summary", "")
        if doc_id is not None:
            summary_map[doc_id] = summary

    print(f"Loaded {len(summary_map)} summaries")
    return summary_map


def retrieve_documents_for_queries_flat(
    queries_path: str,
    index_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    summary_path: str,
    output_path: str,
    retrieval_model_name: str,
    use_gpu: bool = False,
    cache_dir: str = None,
    top_k: int = 5,
    use_summary: bool = True,
):
    """
    Retrieve documents for IA-MIA-style queries using FAISS index.

    queries_path: JSONL with fields:
        - _id
        - text
        - target_doc_id
        - _membership
        - variation_index
    summary_path: JSONL with fields:
        - _id (doc id)
        - summary (string)

    When use_summary is True, the encoded text is:
        "I have a question about {summary}. Question: {query}."
    """
    print(f"\n{'#'*60}")
    print(f"IA-MIA Document Retrieval (flat queries)")
    print(f"Queries: {queries_path}")
    print(f"Index: {index_path}")
    print(f"Member corpus: {corpus_member_path}")
    print(f"Nonmember corpus: {corpus_nonmember_path}")
    print(f"Summary file: {summary_path}")
    print(f"Retrieval model: {retrieval_model_name}")
    print(f"Top-k: {top_k}")
    print(f"Use summary: {use_summary}")
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
    if not queries_data:
        raise ValueError(
            f"No queries found in {queries_path}. Run the IA-MIA online phase "
            "successfully first, and remove any stale empty query file before retrying."
        )

    # Load FAISS index and doc IDs
    index, doc_ids = load_faiss_index_simple(index_path, corpus_member_path, corpus_nonmember_path)

    # Load summaries
    summary_map = load_summaries(summary_path) if use_summary else {}

    # Load retrieval model
    retrieval_model = load_embedding_model(retrieval_model_name, use_gpu, cache_dir)

    # Prepare all query texts and metadata
    print("\nPreparing queries for retrieval...")
    all_texts = []
    query_metadata = []

    for q in queries_data:
        qid = q["_id"]
        qtext = q["text"]
        target_doc_id = q.get("target_doc_id")
        membership = q.get("_membership")
        variation_index = q.get("variation_index")

        # Build query text with optional summary
        if use_summary and target_doc_id in summary_map:
            summary = summary_map[target_doc_id]
            query_with_summary = f"I have a question about {summary}. Question: {qtext}."
        else:
            query_with_summary = qtext

        all_texts.append(query_with_summary)
        query_metadata.append(
            {
                "query_id": qid,
                "text": qtext,
                "query_with_summary": query_with_summary if use_summary else None,
                "target_doc_id": target_doc_id,
                "membership": membership,
                "variation_index": variation_index,
            }
        )

    print(f"Total queries to process: {len(all_texts)}")
    if use_summary:
        print('Template: "I have a question about {summary}. Question: {query}."')

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

    # Organize retrieval results keyed by query_id
    print("\nOrganizing retrieval results...")
    retrieval_results = {}

    for i, (score_list, index_list, meta) in enumerate(
        zip(scores, indices, query_metadata)
    ):
        retrieved_doc_ids = [
            doc_ids[idx] for idx in index_list if 0 <= idx < len(doc_ids)
        ]
        retrieved_scores = [float(s) for s in score_list]

        retrieval_results[meta["query_id"]] = {
            "query_id": meta["query_id"],
            "text": meta["text"],
            "query_with_summary": meta["query_with_summary"],
            "target_doc_id": meta["target_doc_id"],
            "membership": meta["membership"],
            "variation_index": meta["variation_index"],
            "retrieved_doc_ids": retrieved_doc_ids,
            "scores": retrieved_scores,
        }

    # Prepare output data
    output_data = {
        "metadata": {
            "queries_file": queries_path,
            "index_file": index_path,
            "corpus_member_file": corpus_member_path,
            "corpus_nonmember_file": corpus_nonmember_path,
            "summary_file": summary_path,
            "retrieval_model": retrieval_model_name,
            "top_k": top_k,
            "use_summary": use_summary,
            "num_queries": len(all_texts),
        },
        "retrieval_results": retrieval_results,
    }

    # Save results
    write_json(output_data, output_path)

    print(f"\n{'='*60}")
    print(f"Retrieval completed!")
    print(f"Processed {len(retrieval_results)} queries")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve documents for IA-MIA flat queries with optional summary context"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default="results/IA-MIA/BeIR_nfcorpus/queries_paraphrased/queries_30.jsonl",
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
        "--summary_file",
        type=str,
        default="data/BeIR_nfcorpus/summary.jsonl",
        help="Path to summary JSONL file with doc summaries",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/IA-MIA/BeIR_nfcorpus/retrieval_paraphrased",
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
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Do NOT concatenate summary with query for retrieval",
    )

    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    if args.cache_dir:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Construct output path with topk{k} subfolder
    output_path = Path(args.output_dir) / f"topk{args.top_k}_results.json"

    retrieve_documents_for_queries_flat(
        queries_path=args.queries_file,
        index_path=args.index_path,
        corpus_member_path=args.corpus_member_file,
        corpus_nonmember_path=args.corpus_nonmember_file,
        summary_path=args.summary_file,
        output_path=str(output_path),
        retrieval_model_name=args.retrieval_model,
        use_gpu=args.use_gpu,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        use_summary=not args.no_summary,
    )

    print("\n" + "=" * 60)
    print("IA-MIA flat retrieval completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
