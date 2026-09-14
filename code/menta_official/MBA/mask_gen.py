#!/usr/bin/env python3
"""
Mask-Based MIA Query Generation Script
Generates masked queries for membership inference attacks on RAG systems.
Follows the approach of masking specific tokens/phrases and using the masked text as queries.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import spacy
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_jsonl


# Template for mask-based queries
MASK_QUERY_TEMPLATE = """Predict the answers of the masks in the given texts:

{masked_document}

You should strictly follow the contexts below to answer, and provide your answers in the specified format: "[Mask_i]: answer_i\\n" for each mask:

{retrieved_documents}"""


class MaskGenerator:
    """
    Generates masks for membership inference attacks on RAG systems.
    Implements multiple masking strategies based on MBA framework.
    """
    
    def __init__(
        self,
        masking_strategy: str = 'important',
        num_masks: int = 5,
        mask_token: str = '[MASK]',
        proxy_model_name: str = None,
        use_gpu: bool = False,
        cache_dir: str = None
    ):
        """
        Initialize mask generator.
        
        Args:
            masking_strategy: Strategy for selecting tokens to mask
                - 'random': Random token selection
                - 'important': Select important tokens (nouns, proper nouns, numbers)
                - 'difficult': Use proxy model to find hardest-to-predict tokens
            num_masks: Number of tokens/phrases to mask
            mask_token: Token to use for masking
            proxy_model_name: Name of proxy model for 'difficult' strategy
            use_gpu: Whether to use GPU for proxy model
            cache_dir: Cache directory for models
        """
        self.masking_strategy = masking_strategy
        self.num_masks = num_masks
        self.mask_token = mask_token
        
        # Load spacy for NER and POS tagging
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spacy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Load proxy model if using 'difficult' strategy
        self.proxy_tokenizer = None
        self.proxy_model = None
        
        if masking_strategy == 'difficult' and proxy_model_name:
            print(f"Loading proxy model: {proxy_model_name}")
            if cache_dir:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.proxy_tokenizer = AutoTokenizer.from_pretrained(
                proxy_model_name,
                cache_dir=cache_dir
            )
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            self.proxy_model = AutoModelForCausalLM.from_pretrained(
                proxy_model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if use_gpu else torch.float32
            ).to(device)
            self.proxy_model.eval()
            print(f"Proxy model loaded on {device}")
    
    def extract_important_tokens(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract important tokens: proper nouns, nouns, and numbers.
        Returns list of (token, start_idx, end_idx).
        """
        doc = self.nlp(text)
        important_tokens = []
        
        for token in doc:
            # Select proper nouns, nouns, and numbers
            if token.pos_ in ['PROPN', 'NOUN', 'NUM']:
                important_tokens.append((token.text, token.idx, token.idx + len(token.text)))
        
        # Also extract named entities
        for ent in doc.ents:
            important_tokens.append((ent.text, ent.start_char, ent.end_char))
        
        # Remove duplicates and sort by position
        important_tokens = list(set(important_tokens))
        important_tokens.sort(key=lambda x: x[1])
        
        return important_tokens
    
    def calculate_token_perplexity(self, text: str) -> List[Tuple[str, int, int, float]]:
        """
        Calculate perplexity for each token using proxy model.
        Returns list of (token, start_idx, end_idx, perplexity).
        """
        if not self.proxy_model or not self.proxy_tokenizer:
            raise ValueError("Proxy model not loaded for difficult masking strategy")
        
        device = self.proxy_model.device
        
        # Tokenize
        inputs = self.proxy_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        
        # Get token probabilities
        with torch.no_grad():
            outputs = self.proxy_model(input_ids, labels=input_ids)
            logits = outputs.logits
        
        # Calculate per-token perplexity
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Get log prob of actual next token
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Convert to perplexity (higher = harder to predict)
        token_perplexities = torch.exp(-token_log_probs).squeeze(0).cpu().numpy()
        
        # Map back to original text tokens
        tokens = self.proxy_tokenizer.convert_ids_to_tokens(input_ids[0])
        
        token_info = []
        for i, (token, ppl) in enumerate(zip(tokens[1:], token_perplexities)):
            # Approximate character positions (simplified)
            token_info.append((token, i, i+1, float(ppl)))
        
        return token_info
    
    def generate_masks(self, text: str) -> List[Dict]:
        """
        Generate masked versions of text based on masking strategy.
        Returns list of dicts with 'masked_text', 'mask_positions', and 'mask_values'.
        """
        if self.masking_strategy == 'random':
            return self._mask_random(text)
        elif self.masking_strategy == 'important':
            return self._mask_important(text)
        elif self.masking_strategy == 'difficult':
            return self._mask_difficult(text)
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
    
    def _mask_random(self, text: str) -> List[Dict]:
        """Random token masking."""
        tokens = word_tokenize(text)
        
        if len(tokens) <= self.num_masks:
            return []
        
        # Randomly select tokens to mask
        mask_indices = random.sample(range(len(tokens)), self.num_masks)
        mask_indices.sort()
        
        mask_values = [tokens[i] for i in mask_indices]
        masked_tokens = tokens.copy()
        
        # Replace with numbered masks [Mask_0], [Mask_1], etc.
        for idx, i in enumerate(mask_indices):
            masked_tokens[i] = f'[Mask_{idx}]'
        
        masked_text = ' '.join(masked_tokens)
        
        return [{
            'masked_text': masked_text,
            'mask_positions': mask_indices,
            'mask_values': mask_values,
            'masking_strategy': 'random'
        }]
    
    def _mask_important(self, text: str) -> List[Dict]:
        """Mask important tokens (nouns, proper nouns, numbers, entities)."""
        important_tokens = self.extract_important_tokens(text)
        
        if len(important_tokens) == 0:
            # Fallback to random if no important tokens found
            return self._mask_random(text)
        
        # Select up to num_masks important tokens
        num_to_mask = min(self.num_masks, len(important_tokens))
        selected_tokens = random.sample(important_tokens, num_to_mask)
        selected_tokens.sort(key=lambda x: x[1])  # Sort by position
        
        # Create masked text with numbered masks
        masked_text = text
        mask_values = []
        mask_positions = []
        
        # Mask from end to start to preserve positions
        for idx, (token, start_idx, end_idx) in enumerate(reversed(selected_tokens)):
            mask_num = len(selected_tokens) - 1 - idx
            mask_values.insert(0, token)
            mask_positions.insert(0, start_idx)
            masked_text = masked_text[:start_idx] + f'[Mask_{mask_num}]' + masked_text[end_idx:]
        
        return [{
            'masked_text': masked_text,
            'mask_positions': mask_positions,
            'mask_values': mask_values,
            'masking_strategy': 'important'
        }]
    
    def _mask_difficult(self, text: str) -> List[Dict]:
        """Mask tokens that are hardest to predict using proxy model."""
        token_info = self.calculate_token_perplexity(text)
        
        if len(token_info) == 0:
            return self._mask_random(text)
        
        # Sort by perplexity (descending) and select top-k hardest tokens
        token_info.sort(key=lambda x: x[3], reverse=True)
        num_to_mask = min(self.num_masks, len(token_info))
        selected_tokens = token_info[:num_to_mask]
        
        # Sort by position for masking
        selected_tokens.sort(key=lambda x: x[1])
        
        mask_values = [t[0] for t in selected_tokens]
        mask_positions = [t[1] for t in selected_tokens]
        
        # Create masked text with numbered masks (simplified - using word tokenization)
        tokens = word_tokenize(text)
        masked_tokens = tokens.copy()
        
        for idx, pos in enumerate(mask_positions):
            if pos < len(masked_tokens):
                masked_tokens[pos] = f'[Mask_{idx}]'
        
        masked_text = ' '.join(masked_tokens)
        
        return [{
            'masked_text': masked_text,
            'mask_positions': mask_positions,
            'mask_values': mask_values,
            'masking_strategy': 'difficult'
        }]


def create_mask_query_prompt(masked_document: str) -> str:
    """
    Create query prompt following the template format.
    The retrieved documents placeholder will be filled by RAG system.
    """
    # Create prompt with placeholders
    # Retrieved documents will be inserted by the RAG system during generation
    query_prompt = f"""Predict the answers of the masks in the given texts:

{masked_document}

You should strictly follow the contexts below to answer, and provide your answers in the specified format: "[Mask_i]: answer_i\\n" for each mask:"""
    
    return query_prompt


def generate_mask_queries(
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_path: str,
    masking_strategy: str = 'important',
    num_masks: int = 5,
    proxy_model: str = None,
    use_gpu: bool = False,
    cache_dir: str = None
):
    """
    Generate masked queries for membership inference attacks.
    """
    print(f"\n{'#'*60}")
    print(f"MASK-BASED MIA QUERY GENERATION")
    print(f"Masking strategy: {masking_strategy}")
    print(f"Number of masks: {num_masks}")
    print(f"Member corpus: {corpus_member_path}")
    print(f"Non-member corpus: {corpus_nonmember_path}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping.")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize mask generator
    mask_gen = MaskGenerator(
        masking_strategy=masking_strategy,
        num_masks=num_masks,
        proxy_model_name=proxy_model,
        use_gpu=use_gpu,
        cache_dir=cache_dir
    )
    
    # Load corpora
    print("Loading corpora...")
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    
    print(f"Loaded {len(corpus_member)} member documents")
    print(f"Loaded {len(corpus_nonmember)} non-member documents")
    
    # Generate masked queries
    all_queries = []
    
    print("\nGenerating masked queries for members...")
    for doc in tqdm(corpus_member, desc="Member docs"):
        doc_id = doc['_id']
        title = doc.get('title', '')
        text = doc.get('text', '')
        
        # Combine title and text
        full_text = f"{title} {text}".strip() if title else text
        
        if not full_text:
            continue
        
        # Generate masks
        masked_versions = mask_gen.generate_masks(full_text)
        
        for masked_ver in masked_versions:
            # Create query prompt using template
            query_prompt = create_mask_query_prompt(masked_ver['masked_text'])
            
            query_record = {
                '_id': doc_id,
                'masked_text': masked_ver['masked_text'],
                'query_prompt': query_prompt,
                'mask_positions': masked_ver['mask_positions'],
                'mask_values': masked_ver['mask_values'],
                'masking_strategy': masked_ver['masking_strategy'],
                'original_text': full_text,
                'membership_label': 1,
                'original_title': title
            }
            all_queries.append(query_record)
    
    print("\nGenerating masked queries for non-members...")
    for doc in tqdm(corpus_nonmember, desc="Non-member docs"):
        doc_id = doc['_id']
        title = doc.get('title', '')
        text = doc.get('text', '')
        
        # Combine title and text
        full_text = f"{title} {text}".strip() if title else text
        
        if not full_text:
            continue
        
        # Generate masks
        masked_versions = mask_gen.generate_masks(full_text)
        
        for masked_ver in masked_versions:
            # Create query prompt using template
            query_prompt = create_mask_query_prompt(masked_ver['masked_text'])
            
            query_record = {
                '_id': doc_id,
                'masked_text': masked_ver['masked_text'],
                'query_prompt': query_prompt,
                'mask_positions': masked_ver['mask_positions'],
                'mask_values': masked_ver['mask_values'],
                'masking_strategy': masked_ver['masking_strategy'],
                'original_text': full_text,
                'membership_label': 0,
                'original_title': title
            }
            all_queries.append(query_record)
    
    # Write output
    write_jsonl(all_queries, output_path)
    
    member_count = sum(1 for q in all_queries if q['membership_label'] == 1)
    nonmember_count = len(all_queries) - member_count
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_queries)} masked queries")
    print(f"  Members: {member_count}")
    print(f"  Non-members: {nonmember_count}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate masked queries for mask-based MIA'
    )
    parser.add_argument(
        '--corpus_member_file',
        type=str,
        default='data/BeIR_trec-covid/corpus_member.jsonl',
        help='Path to member corpus file'
    )
    parser.add_argument(
        '--corpus_nonmember_file',
        type=str,
        default='data/BeIR_trec-covid/corpus_nonmember.jsonl',
        help='Path to non-member corpus file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/MBA/BeIR_trec-covid/queries/queries.jsonl',
        help='Path to output queries file'
    )
    parser.add_argument(
        '--masking_strategy',
        type=str,
        choices=['random', 'important', 'difficult'],
        default='important',
        help='Strategy for selecting tokens to mask'
    )
    parser.add_argument(
        '--num_masks',
        type=int,
        default=10,
        help='Number of tokens/phrases to mask'
    )
    parser.add_argument(
        '--proxy_model',
        type=str,
        default='gpt2',
        help='Proxy model for difficult masking strategy'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for proxy model'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    
    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    if args.cache_dir:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    generate_mask_queries(
        args.corpus_member_file,
        args.corpus_nonmember_file,
        args.output_file,
        args.masking_strategy,
        args.num_masks,
        args.proxy_model,
        args.use_gpu,
        args.cache_dir)
    
    print("\n" + "="*60)
    print("Mask generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
