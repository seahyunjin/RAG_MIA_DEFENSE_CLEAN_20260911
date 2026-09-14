import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import warnings
import re
from collections import defaultdict
import sys

# Add project root to Python path (MUST be before utils import)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_generator_model, load_embedding_model

warnings.filterwarnings('ignore')


def load_splitter_model(cache_dir: str = None, use_gpu: bool = False):
    """Load T5 model for splitting text into atomic claims."""
    from utils.env_config import normalize_cache_dir

    cache_dir = normalize_cache_dir(cache_dir)
    model_name = "Babelscape/t5-base-summarization-claim-extractor"
    print(f"Loading splitter model: {model_name}")
    
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    if cache_dir:
        model_name_safe = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, f"models--{model_name_safe}")
        
        if os.path.exists(model_cache_path):
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                               if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    if use_gpu:
                        model = model.to(device)
                    model.eval()
                    return tokenizer, model, device
        
        raise FileNotFoundError(f"Model not found in cache: {cache_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if use_gpu:
            model = model.to(device)
        model.eval()
        return tokenizer, model, device


def load_nli_model(cache_dir: str = None, use_gpu: bool = False):
    """Load DeBERTa model for Natural Language Inference (entailment)."""
    from utils.env_config import normalize_cache_dir

    cache_dir = normalize_cache_dir(cache_dir)
    model_name = "tasksource/deberta-base-long-nli"
    print(f"Loading NLI model: {model_name}")
    
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    if cache_dir:
        model_name_safe = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, f"models--{model_name_safe}")
        
        if os.path.exists(model_cache_path):
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                               if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    if use_gpu:
                        model = model.to(device)
                    model.eval()
                    return tokenizer, model, device
        
        raise FileNotFoundError(f"Model not found in cache: {cache_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_gpu:
            model = model.to(device)
        model.eval()
        return tokenizer, model, device


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


def split_into_atomic_claims_batch(
    texts: List[str],
    splitter_tokenizer,
    splitter_model,
    device,
    batch_size: int = 32,
    max_length: int = 512,
    min_claim_length: int = 20,
    min_words: int = 5,
    show_progress: bool = True
) -> List[List[str]]:
    """Split multiple texts into atomic claims using T5 model (split_and_rephrase)."""
    
    if not texts:
        return []
    
    # Prepare all inputs with prefix
    input_texts = [f"split: {text}" for text in texts]
    num_batches = (len(input_texts) + batch_size - 1) // batch_size
    
    if show_progress:
        print(f"\nProcessing {len(texts)} texts in {num_batches} batches...")
    
    # Initialize results array
    all_claims = [[] for _ in range(len(texts))]
    
    # Create progress bar for batches
    batch_iterator = range(0, len(input_texts), batch_size)
    if show_progress:
        batch_iterator = tqdm(batch_iterator, 
                             desc="Model splitting", 
                             total=num_batches,
                             unit="batch")
    
    for batch_start in batch_iterator:
        batch_end = min(batch_start + batch_size, len(input_texts))
        batch_input_texts = input_texts[batch_start:batch_end]
        
        # Tokenize entire batch at once
        inputs = splitter_tokenizer(
            batch_input_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate for entire batch at once
        with torch.no_grad():
            outputs = splitter_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode all outputs
        decoded_batch = splitter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Post-process each decoded output
        for idx, decoded in enumerate(decoded_batch):
            original_idx = batch_start + idx
            claims = [c.strip() for c in decoded.split('<sep>') if c.strip()]
            
            if len(claims) == 1:
                claims = [c.strip() for c in decoded.split('.') if c.strip()]
            
            # Merge fragments
            merged_claims = []
            current_claim = ""
            
            for claim in claims:
                claim = claim.strip()
                if not claim:
                    continue
                
                is_fragment = (
                    len(claim) <= 3 or
                    claim.isdigit() or
                    (len(claim.split()) == 1 and claim[0].isupper() and len(claim) == 1) or
                    claim.endswith(':') or
                    (claim[-1] not in '.!?' and len(claim.split()) < 5)
                )
                
                starts_with_number = bool(re.match(r'^\d+\.?\s', claim))
                
                if is_fragment or (current_claim and starts_with_number):
                    if current_claim:
                        current_claim += " " + claim
                    else:
                        current_claim = claim
                else:
                    if current_claim:
                        current_claim = re.sub(r'\s+', ' ', current_claim).strip()
                        if len(current_claim) >= min_claim_length and len(current_claim.split()) >= min_words:
                            merged_claims.append(current_claim)
                    current_claim = claim
            
            if current_claim:
                current_claim = re.sub(r'\s+', ' ', current_claim).strip()
                if len(current_claim) >= min_claim_length and len(current_claim.split()) >= min_words:
                    merged_claims.append(current_claim)
            
            all_claims[original_idx] = merged_claims
    
    return all_claims


def check_entailment_batch(
    premises: List[str],
    hypotheses: List[str],
    nli_tokenizer,
    nli_model,
    device,
    batch_size: int = 16,
    max_length: int = 2048,
    desc: str = "NLI inference"
) -> List[Tuple[float, float, float]]:
    """
    Check entailment for multiple premise-hypothesis pairs in batches.
    Returns list of (entailment_prob, neutral_prob, contradiction_prob) tuples.
    """
    assert len(premises) == len(hypotheses), "Premises and hypotheses must have same length"
    
    all_results = []
    num_pairs = len(premises)
    num_batches = (num_pairs + batch_size - 1) // batch_size
    
    print(f"\n{desc}: Processing {num_pairs:,} premise-hypothesis pairs in {num_batches} batches")
    
    # Create progress bar for batches
    batch_iterator = range(0, len(premises), batch_size)
    if batch_size > 1:
        batch_iterator = tqdm(
            batch_iterator,
            desc=desc,
            total=num_batches,
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    for i in batch_iterator:
        batch_premises = premises[i:i + batch_size]
        batch_hypotheses = hypotheses[i:i + batch_size]
        current_batch_size = len(batch_premises)
        
        inputs = nli_tokenizer(
            batch_premises,
            batch_hypotheses,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        num_classes = probs.shape[1]
        
        if num_classes == 2:
            # Binary classification: entailment vs not_entailment
            entailment_probs = probs[:, 0].cpu().numpy()
            not_entailment_probs = probs[:, 1].cpu().numpy()
            
            for ent_prob, not_ent_prob in zip(entailment_probs, not_entailment_probs):
                all_results.append((float(ent_prob), 0.0, float(not_ent_prob)))
        
        elif num_classes == 3:
            # Three-way classification: entailment, neutral, contradiction
            entailment_probs = probs[:, 0].cpu().numpy()
            neutral_probs = probs[:, 1].cpu().numpy()
            contradiction_probs = probs[:, 2].cpu().numpy()
            
            for ent_prob, neu_prob, cont_prob in zip(entailment_probs, neutral_probs, contradiction_probs):
                all_results.append((float(ent_prob), float(neu_prob), float(cont_prob)))
        
        else:
            raise ValueError(f"Unexpected number of classes: {num_classes}. Expected 2 or 3.")
        
        if batch_size > 1:
            batch_iterator.set_postfix({
                'pairs': len(all_results),
                'batch_sz': current_batch_size
            })
    
    return all_results


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


def extract_all_claims_batch(
    queries: List[dict],
    splitter_tokenizer,
    splitter_model,
    splitter_device,
    splitting_method: str = 'heuristic',
    batch_size: int = 32
) -> Dict[str, List[str]]:
    """
    Extract claims for ALL queries.
    
    Args:
        splitting_method: 'heuristic' or 'split_and_rephrase'
    """
    query_ids = [q['query_id'] for q in queries]
    answers = [q['generated_answer'] for q in queries]
    
    print(f"\nExtracting claims using splitting_method={splitting_method}...")
    
    if splitting_method == 'heuristic':
        # Use simple text splitting
        all_claims = []
        for answer in tqdm(answers, desc="Splitting text"):
            claims = split_text(answer, min_length=10)
            all_claims.append(claims)
    
    elif splitting_method == 'split_and_rephrase':
        # Use T5 model-based splitting
        print(f"Using model-based claim extraction...")
        all_claims = split_into_atomic_claims_batch(
            answers,
            splitter_tokenizer,
            splitter_model,
            splitter_device,
            batch_size=batch_size
        )
    
    else:
        raise ValueError(f"Unknown splitting_method: {splitting_method}. Use 'heuristic' or 'split_and_rephrase'")
    
    # Map back to query IDs
    query_to_claims = {}
    for query_id, claims in zip(query_ids, all_claims):
        query_to_claims[query_id] = claims if claims else []
    
    total_claims = sum(len(claims) for claims in all_claims)
    print(f"Extracted {total_claims} total claims")
    
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


def parse_answers_modes_from_parts(parts: List[str]) -> Tuple[str, bool, bool]:
    """Parse defense/query mode flags from answers path parts."""
    defense_type = "none"
    generic_queries = False
    disable_retrieval = False

    for part in parts:
        if part == "answers":
            return defense_type, generic_queries, disable_retrieval
        if part.startswith("answers_"):
            suffix = part[len("answers_"):]
            if "_generic" in suffix:
                generic_queries = True
                suffix = suffix.replace("_generic", "")
            if "_disable_retrieval" in suffix:
                disable_retrieval = True
                suffix = suffix.replace("_disable_retrieval", "")

            if suffix == "paraphrased":
                defense_type = "paraphrase"
            elif suffix:
                defense_type = suffix
            break

    return defense_type, generic_queries, disable_retrieval


def get_entailment_dirname(defense_type: str, generic_queries: bool, disable_retrieval: bool) -> str:
    """Build entailment directory name from defense/query mode flags."""
    if defense_type == "paraphrase":
        dirname = "entailment_paraphrased"
    elif defense_type != "none":
        dirname = f"entailment_{defense_type}"
    else:
        dirname = "entailment"

    if generic_queries:
        dirname = f"{dirname}_generic"
    if disable_retrieval:
        dirname = f"{dirname}_disable_retrieval"

    return dirname


def build_global_entailment_matrix_with_idk_detection(
    queries: List[dict],
    query_to_claims: Dict[str, List[str]],
    doc_text_units: Dict[str, List[str]],
    corpus_map: Dict[str, dict],
    nli_tokenizer,
    nli_model,
    nli_device,
    nli_batch_size: int = 512
) -> Tuple[Dict[Tuple[str, str, int], dict], Dict[str, dict]]:
    """
    Build entailment matrix for target documents AND compute IDK entailment scores.
    Does NOT make conclusions - just saves probabilities for later evaluation.
    
    Returns:
        - entailment_matrix: {(query_id, target_doc_id, claim_idx): entailment_info}
        - idk_detection_details: IDK entailment scores per query (no conclusions)
    """
    print("\nBuilding global entailment matrix with IDK entailment scoring...")
    
    # Multiple IDK hypotheses - use different phrasings to catch various IDK patterns
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
        print("WARNING: No valid pairs found for entailment checking!")
        return {}, {}
    
    # Flatten premise-hypothesis pairs for document entailment
    all_premises_doc = []
    all_hypotheses_doc = []
    pair_indices_doc = []
    
    for i, meta in enumerate(pair_metadata):
        doc_id = meta['doc_id']
        claim = meta['claim']
        text_units = doc_text_units[doc_id]
        
        # Document entailment checks
        for text_unit in text_units:
            all_premises_doc.append(text_unit)
            all_hypotheses_doc.append(claim)
            pair_indices_doc.append(i)
    
    print(f"Total NLI checks for document entailment: {len(all_premises_doc)}")
    
    # Process document entailment checks
    entailment_matrix = {}
    
    if all_premises_doc:
        print(f"\nRunning document entailment NLI (batch_size={nli_batch_size})...")
        all_results_doc = check_entailment_batch(
            all_premises_doc,
            all_hypotheses_doc,
            nli_tokenizer,
            nli_model,
            nli_device,
            batch_size=nli_batch_size,
            desc="Document entailment"
        )
        
        # Aggregate results
        print("Aggregating document entailment results...")
        current_pair_idx = -1
        current_results = []
        current_text_units = []
        
        for idx, (pair_idx, result) in enumerate(zip(pair_indices_doc, all_results_doc)):
            if pair_idx != current_pair_idx:
                # Save previous pair's results
                if current_pair_idx >= 0:
                    meta = pair_metadata[current_pair_idx]
                    key = (meta['query_id'], meta['doc_id'], meta['claim_idx'])
                    
                    best_result = max(current_results, key=lambda r: r[0])
                    best_idx = current_results.index(best_result)
                    best_text_unit = current_text_units[best_idx] if best_idx < len(current_text_units) else current_text_units[0]
                    
                    entailment_matrix[key] = {
                        'claim': meta['claim'],
                        'best_text_unit': best_text_unit,
                        'entailment_prob': best_result[0],
                        'neutral_prob': best_result[1],
                        'contradiction_prob': best_result[2]
                    }
                
                # Start new pair
                current_pair_idx = pair_idx
                current_results = [result]
                meta = pair_metadata[pair_idx]
                current_text_units = doc_text_units[meta['doc_id']]
            else:
                current_results.append(result)
        
        # Last pair
        if current_pair_idx >= 0:
            meta = pair_metadata[current_pair_idx]
            key = (meta['query_id'], meta['doc_id'], meta['claim_idx'])
            
            best_result = max(current_results, key=lambda r: r[0])
            best_idx = current_results.index(best_result)
            best_text_unit = current_text_units[best_idx] if best_idx < len(current_text_units) else current_text_units[0]
            
            entailment_matrix[key] = {
                'claim': meta['claim'],
                'best_text_unit': best_text_unit,
                'entailment_prob': best_result[0],
                'neutral_prob': best_result[1],
                'contradiction_prob': best_result[2]
            }
    
    # Process IDK entailment scoring with MULTIPLE hypotheses
    # Just save probabilities - NO conclusions made here
    idk_detection_details = {}
    
    print(f"\nComputing IDK entailment scores with {len(idk_hypotheses)} hypotheses...")
    
    # For each claim, test against ALL IDK hypotheses
    for hypothesis_idx, idk_hypothesis in enumerate(idk_hypotheses):
        print(f"\nTesting hypothesis {hypothesis_idx + 1}/{len(idk_hypotheses)}: '{idk_hypothesis}'")
        
        all_premises_idk = []
        all_hypotheses_idk = []
        pair_indices_idk = []
        
        for i, meta in enumerate(pair_metadata):
            claim = meta['claim']
            all_premises_idk.append(claim)
            all_hypotheses_idk.append(idk_hypothesis)
            pair_indices_idk.append(i)
        
        print(f"Total NLI checks: {len(all_premises_idk)}")
        
        all_results_idk = check_entailment_batch(
            all_premises_idk,
            all_hypotheses_idk,
            nli_tokenizer,
            nli_model,
            nli_device,
            batch_size=nli_batch_size,
            desc=f"IDK scoring (hyp {hypothesis_idx + 1})"
        )
        
        # Save IDK entailment scores per query - NO thresholding or conclusions
        for pair_idx, result in zip(pair_indices_idk, all_results_idk):
            meta = pair_metadata[pair_idx]
            query_id = meta['query_id']
            claim = meta['claim']
            entailment_prob = result[0]
            
            if query_id not in idk_detection_details:
                idk_detection_details[query_id] = {
                    'claims': []
                }
            
            # Find or create claim entry
            claim_entry = None
            for c in idk_detection_details[query_id]['claims']:
                if c['claim'] == claim:
                    claim_entry = c
                    break
            
            if claim_entry is None:
                claim_entry = {
                    'claim': claim,
                    'idk_entailment_scores': {}  # Just store raw scores
                }
                idk_detection_details[query_id]['claims'].append(claim_entry)
            
            # Store this hypothesis score (raw probability)
            claim_entry['idk_entailment_scores'][idk_hypothesis] = entailment_prob
    
    print(f"\nEntailment matrix built with {len(entailment_matrix)} entries")
    print(f"IDK entailment scores computed for {len(idk_detection_details)} queries")
    
    return entailment_matrix, idk_detection_details


def compute_and_save_entailment(
    output_file_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    cache_dir: str = None,
    use_gpu: bool = False,
    nli_batch_size: int = 512,
    splitting_method: str = 'heuristic'
):
    """
    Compute entailment matrix for target documents with IDK entailment scoring.
    Uses both member and non-member corpus for document lookup.
    Does NOT make IDK conclusions - just saves probabilities for evaluate.py.
    
    Args:
        output_file_path: Path to answers JSON file
        corpus_member_path: Path to corpus_member.jsonl
        corpus_nonmember_path: Path to corpus_nonmember.jsonl
        splitting_method: 'heuristic', 'split_and_rephrase', or 'both'
    """
    print(f"\n{'#'*60}")
    print(f"Computing Entailment Matrix with IDK Entailment Scoring")
    print(f"Splitting Method: {splitting_method}")
    print(f"NLI Batch Size: {nli_batch_size}")
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
    
    print("\nLoading models...")
    if cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Only load splitter if using split_and_rephrase or both
    if splitting_method in ['split_and_rephrase', 'both']:
        splitter_tokenizer, splitter_model, splitter_device = load_splitter_model(cache_dir, use_gpu)
    else:
        splitter_tokenizer, splitter_model, splitter_device = None, None, None
    
    nli_tokenizer, nli_model, nli_device = load_nli_model(cache_dir, use_gpu)
    
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
    query_to_claims = extract_all_claims_batch(
        normalized_queries,
        splitter_tokenizer,
        splitter_model,
        splitter_device,
        splitting_method=splitting_method,
        batch_size=32
    )
    total_claims = sum(len(claims) for claims in query_to_claims.values())
    print(f"Extracted {total_claims} total claims from {len(query_to_claims)} queries")
    
    # STEP 3: Build entailment matrix AND compute IDK scores (no conclusions)
    print(f"\nStep 3: Building entailment matrix with IDK entailment scoring...")
    entailment_matrix, idk_detection_details = build_global_entailment_matrix_with_idk_detection(
        normalized_queries,
        query_to_claims,
        doc_text_units,
        corpus_map,
        nli_tokenizer,
        nli_model,
        nli_device,
        nli_batch_size
    )
    
    # Convert tuple keys to string for JSON serialization
    entailment_matrix_serializable = {
        f"{query_id}||{doc_id}||{claim_idx}": value
        for (query_id, doc_id, claim_idx), value in entailment_matrix.items()
    }
    
    # Build output path: replace 'answers' with 'entailment' in path
    # Handles both 'answers' and 'answers_{defense}' patterns
    output_file_path = Path(output_file_path)
    
    # Replace 'answers' with 'entailment' in filename
    filename = output_file_path.name.replace('answers', 'entailment')
    
    # Replace answers directory with the normalized entailment naming scheme
    parts = list(output_file_path.parent.parts)
    defense_type, generic_queries, disable_retrieval = parse_answers_modes_from_parts(parts)

    new_parts = []
    entailment_dirname = get_entailment_dirname(defense_type, generic_queries, disable_retrieval)
    for p in parts:
        if p == "answers" or p.startswith("answers_"):
            new_parts.append(entailment_dirname)
        else:
            new_parts.append(p)
    
    base_dir = Path(*new_parts)
    
    # Create output directory if needed
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = base_dir / filename
    
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
            'generic_queries': generic_queries,
            'query_mode': 'generic' if generic_queries else 'specific',
            'disable_retrieval': disable_retrieval,
            'splitting_method': splitting_method,
            'nli_batch_size': nli_batch_size,
            'num_queries': len(query_to_claims),
            'total_claims': total_claims,
            'num_target_documents': len(doc_text_units),
            'num_member_targets': len(member_targets),
            'num_nonmember_targets': len(nonmember_targets),
            'matrix_size': len(entailment_matrix),
            'note': 'IDK entailment scores computed but NO conclusions made - evaluation happens in evaluate.py'
        },
        'query_to_claims': query_to_claims,
        'query_membership': query_membership,
        'idk_detection_details': idk_detection_details,
        'entailment_matrix': entailment_matrix_serializable
    }
    
    write_json(results_data, str(results_path))
    print(f"\nEntailment matrix saved to: {results_path}")
    print(f"Defense type detected: {defense_type}")
    print(f"Generic queries: {generic_queries}")
    print(f"Disable retrieval: {disable_retrieval}")
    
    return results_path

def main():
    parser = argparse.ArgumentParser(
        description='Compute entailment matrix for membership inference attack analysis'
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
        '--cache_dir',
        type=str,
        default='../hf_cache/hub',
        help='Cache directory for models' 
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for inference'
    )
    parser.add_argument(
        '--nli_batch_size',
        type=int,
        default=32, 
        help='Batch size for NLI model inference'
    )
    parser.add_argument(
        '--splitting_method',
        type=str,
        default='heuristic',
        choices=['heuristic', 'split_and_rephrase', 'both'],
        help='Method for splitting text: heuristic (fast regex), split_and_rephrase (T5), or both (combine both)'
    )
    
    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    compute_and_save_entailment(
        output_file_path=args.output_file,
        corpus_member_path=args.corpus_member,
        corpus_nonmember_path=args.corpus_nonmember,
        cache_dir=args.cache_dir,
        use_gpu=args.use_gpu,
        nli_batch_size=args.nli_batch_size,
        splitting_method=args.splitting_method
    )
    
    print("\n" + "="*60)
    print("Entailment computation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
