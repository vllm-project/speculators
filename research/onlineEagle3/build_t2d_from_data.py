#!/usr/bin/env python3
# Build t2d (keep-mask) and d2t (offsets) from JSON data using verifier tokenizer

import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", type=str, required=True, help="HF path to verifier model")
    ap.add_argument("--data", type=str, required=True, help="JSON dataset file")
    ap.add_argument("--samples", type=int, default=50000, help="Number of samples to process")
    ap.add_argument("--drafter-vocab", type=int, default=32000, help="Draft vocabulary size")
    ap.add_argument("--out", type=str, required=True, help="Output path for t2d.npy")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Load tokenizer and get vocab size
    tokenizer = AutoTokenizer.from_pretrained(args.verifier, use_fast=False)
    config = AutoConfig.from_pretrained(args.verifier)
    vocab_size = config.vocab_size  # This should give you 128256
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.data)["train"]
    total_samples = min(args.samples, len(dataset))
    
    # Sample random indices - convert to regular Python ints
    np.random.seed(args.seed)
    indices = np.random.choice(len(dataset), total_samples, replace=False).astype(int)
    
    # Count token frequencies
    counts = np.zeros(vocab_size, dtype=np.int64)
    
    for idx in indices:
        # Extract text from record
        record = dataset[int(idx)]  # Ensure it's a regular int
        if isinstance(record, dict):
            if "conversations" in record:
                # ShareGPT format
                text = "\n".join([f"[{msg.get('from', 'user')}] {msg.get('value', '')}" 
                                for msg in record["conversations"]])
            else:
                # Try common text fields
                for field in ["text", "content", "instruction", "input", "output"]:
                    if field in record and isinstance(record[field], str):
                        text = record[field]
                        break
                else:
                    text = str(record)
        else:
            text = str(record)
        
        # Tokenize and count
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        if tokens:
            for token_id in tokens:
                if 0 <= token_id < vocab_size:
                    counts[token_id] += 1
    
    # Get top tokens by frequency
    top_tokens = np.argsort(-counts)[:args.drafter_vocab]
    
    # Create t2d mask: boolean mask indicating which tokens are in draft vocab
    t2d = np.zeros(vocab_size, dtype=bool)
    t2d[top_tokens] = True
    
    # Create d2t offsets: offset between draft_id and teacher_id
    d2t = top_tokens - np.arange(args.drafter_vocab)
    
    # Save files
    output_dir = os.path.dirname(args.out)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(args.out, t2d)
    
    d2t_path = args.out.replace("t2d", "d2t")
    np.save(d2t_path, d2t)
    
    print(f"Saved t2d.npy: {args.out}")
    print(f"Saved d2t.npy: {d2t_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model config vocab size: {config.vocab_size}")
    print(f"Vocabulary: {vocab_size} -> {args.drafter_vocab}")

if __name__ == "__main__":
    main()
