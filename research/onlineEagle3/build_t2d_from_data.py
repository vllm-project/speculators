#!/usr/bin/env python3
# Build (1) t2d keep-mask [T] and (2) vLLM-ready d2t offsets [N]
# from raw data using the verifier tokenizer.

import argparse
import os
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig


def _gather_text(rec) -> str:
    # ShareGPT-style first
    if isinstance(rec, dict) and "conversations" in rec:
        parts = []
        for t in rec["conversations"]:
            role = t.get("from", "user")
            content = t.get("value", "")
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)
    # Common plain fields
    for k in ("text", "content", "instruction", "input", "output"):
        if isinstance(rec, dict) and k in rec and isinstance(rec[k], str):
            return rec[k]
    # Fallback
    return str(rec)


def _forced_special_ids(tok, teacher_vocab: int) -> List[int]:
    keep = set()

    # Collect from common attributes
    def add(x):
        if x is None:
            return
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, int) and 0 <= t < teacher_vocab:
                    keep.add(int(t))
        elif isinstance(x, int) and 0 <= x < teacher_vocab:
            keep.add(int(x))

    add(getattr(tok, "bos_token_id", None))
    add(getattr(tok, "eos_token_id", None))
    add(getattr(tok, "pad_token_id", None))
    add(getattr(tok, "unk_token_id", None))

    # Some Llama 3.x have multiple EOS-like IDs (list)
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        add(eos)

    # Try tokenizer-provided complete special list if present
    all_special = getattr(tok, "all_special_ids", None)
    if isinstance(all_special, (list, tuple)):
        add([i for i in all_special if isinstance(i, int)])

    return sorted(keep)


def _expand_counts_if_needed(counts: np.ndarray, max_id: int) -> np.ndarray:
    if max_id < counts.shape[0]:
        return counts
    new_size = max(max_id + 1, counts.shape[0] * 2)
    return np.pad(counts, (0, new_size - counts.shape[0]), constant_values=0)

from transformers import AutoTokenizer, AutoConfig
# optional, for the most robust check
try:
    from safetensors import safe_open
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

def _teacher_vocab_size(verifier_path: str) -> int:
    tok = AutoTokenizer.from_pretrained(verifier_path, use_fast=False)
    tok_v = int(getattr(tok, "vocab_size", 0) or 0)
    cfg_v = int(AutoConfig.from_pretrained(verifier_path).vocab_size)
    T = max(tok_v, cfg_v)

    # Optional: if safetensors are present, read lm_head rows exactly
    if _HAVE_ST:
        import json, os
        idx_json = os.path.join(verifier_path, "model.safetensors.index.json")
        if os.path.exists(idx_json):
            with open(idx_json) as f:
                idx = json.load(f)
            wpath = os.path.join(verifier_path, idx["weight_map"]["lm_head.weight"])
            with safe_open(wpath, framework="pt", device="cpu") as sf:
                T = max(T, int(sf.get_slice("lm_head.weight").get_shape()[0]))
    return T



def _build_keep_from_data(
    verifier_path: str,
    data_path: str,
    samples: int,
    draft_vocab: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    tok = AutoTokenizer.from_pretrained(verifier_path, use_fast=False)
    teacher_vocab = _teacher_vocab_size(verifier_path)
    if teacher_vocab <= 0:
        raise ValueError("Could not determine teacher vocab size from tokenizer or config.")

    ds = load_dataset("json", data_files=data_path)["train"]
    total = len(ds)
    if total == 0:
        raise ValueError(f"Dataset {data_path} is empty.")
    samples = min(int(samples), total)

    # Uniform random sampling to avoid ordering bias
    rng = np.random.default_rng(seed)
    idxs = rng.choice(total, size=samples, replace=False).tolist()

    counts = np.zeros(teacher_vocab, dtype=np.int64)
    for i in idxs:
        text = _gather_text(ds[int(i)])   # casting here is also ok

        ids = tok(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        mx = max(ids)
        if mx >= counts.shape[0]:
            counts = _expand_counts_if_needed(counts, mx)
            teacher_vocab = counts.shape[0]
        np.add.at(counts, ids, 1)

    forced = _forced_special_ids(tok, teacher_vocab)
    keep: List[int] = list(forced)

    # Mask forced, then take top-K by frequency for the rest
    pool_mask = np.ones(counts.shape[0], dtype=bool)
    if forced:
        pool_mask[np.array(forced, dtype=np.int64)] = False

    pool_ids = np.nonzero(pool_mask)[0]
    pool_counts = counts[pool_ids]
    order = np.argsort(-pool_counts, kind="mergesort")
    for tid in pool_ids[order]:
        keep.append(int(tid))
        if len(keep) >= draft_vocab:
            break

    if len(keep) < draft_vocab:
        raise RuntimeError(f"Not enough tokens to fill draft_vocab={draft_vocab}. Only got {len(keep)}.")

    # Sort ascending for stable downstream behavior
    keep_sorted = np.array(sorted(keep[:draft_vocab]), dtype=np.int64)
    # Sanity: unique and in bounds
    if np.unique(keep_sorted).size != keep_sorted.size:
        raise AssertionError("keep set contains duplicates.")
    if keep_sorted.min() < 0 or keep_sorted.max() >= teacher_vocab:
        raise AssertionError("keep set contains out-of-bounds ids.")

    return keep_sorted, teacher_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", type=str, required=True,
                    help="HF path to teacher/verifier (for tokenizer & vocab).")
    ap.add_argument("--data", type=str, required=True,
                    help="JSON dataset file used to count token frequencies.")
    ap.add_argument("--samples", type=int, default=50000,
                    help="How many JSON rows to sample for counts.")
    ap.add_argument("--drafter-vocab", type=int, default=32000,
                    help="Draft vocab size N.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output path for t2d.npy (bool keep-mask of length T).")
    ap.add_argument("--out-d2t", type=str, default=None,
                    help="Optional output path for d2t.npy (offsets of length N).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed for dataset rows.")
    args = ap.parse_args()

    N = int(args.drafter_vocab)
    if N <= 0:
        raise ValueError("--drafter-vocab must be > 0")

    keep_sorted, T = _build_keep_from_data(
        verifier_path=args.verifier,
        data_path=args.data,
        samples=int(args.samples),
        draft_vocab=N,
        seed=int(args.seed),
    )

    # --- t2d: bool mask of length T ---
    t2d_mask = np.zeros(T, dtype=bool)
    t2d_mask[keep_sorted] = True
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, t2d_mask)

    # --- d2t: vLLM-style OFFSETS of length N (teacher_id_for_row_j - j) ---
    base = np.arange(N, dtype=np.int64)
    d2t_offsets = keep_sorted.astype(np.int64) - base
    targets = base + d2t_offsets

    # Sanity checks
    if targets.min() < 0 or targets.max() >= T:
        raise RuntimeError("Computed targets out of teacher bounds; something is wrong.")
    if not np.all(np.diff(targets) >= 1):  # strictly increasing if keep_sorted is sorted
        raise RuntimeError("Targets are not strictly increasing; expected sorted keep set.")

    # Determine d2t output path
    out_d2t = args.out_d2t
    if out_d2t is None:
        root, name = os.path.split(args.out)
        if name.endswith(".npy"):
            name = name.replace("t2d", "d2t")
        else:
            name = "d2t.npy"
        out_d2t = os.path.join(root or ".", name)

    np.save(out_d2t, d2t_offsets)

    # --- summary ---
    print(f"[OK] Wrote t2d (mask) -> {args.out}   shape=({T},) bool  kept={N}")
    print(f"[OK] Wrote d2t (offsets) -> {out_d2t} shape=({N},) int64")
    print(f"     teacher_vocab(T)={T}  draft_vocab(N)={N}")
    print(f"     first 5 targets: {targets[:5].tolist()}  last 5: {targets[-5:].tolist()}")


if __name__ == "__main__":
    main()
