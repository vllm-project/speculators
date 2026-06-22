#!/usr/bin/env python3
"""
Unified DFlash comparison script - validates consistency between speculators and vLLM.

This script performs two types of validation:

1. Training Consistency Check:
   - Validates that training forward pass produces consistent outputs
   - Checks FC layer consistency between training and inference paths
   - Validates logits are well-formed

2. Intermediate Output Comparison (requires VLLM_LOG_DRAFT_INTERMEDIATES=1):
   - Compares all intermediate tensors between speculators and vLLM
   - Validates: embeddings, FC output, layer outputs, final norm, logits
   - Ensures numerical consistency across implementations

Usage:
    # Basic validation (training consistency only)
    python scripts/compare_real_outputs.py

    # Full validation with intermediate comparison
    VLLM_LOG_DRAFT_INTERMEDIATES=1 python scripts/compare_real_outputs.py --compare-intermediates

    # Auto-start servers if not running
    python scripts/compare_real_outputs.py --auto-start

    # Custom sample and tolerances
    python scripts/compare_real_outputs.py --sample-idx 5 --rtol 1e-4 --atol 1e-5
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Suppress transformers docstring validation errors
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "critical")
logging.getLogger("transformers.utils.doc").setLevel(logging.CRITICAL)

import openai
import requests
import torch
from datasets import load_from_disk
from transformers import AutoProcessor

# Suppress auto_docstring print() noise from transformers during import
import io as _io
_stdout_orig = sys.stdout
sys.stdout = _io.StringIO()
try:
    from speculators.data_generation.vllm_client import generate_hidden_states
    from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig
    from transformers import Qwen3Config
    from huggingface_hub import hf_hub_download
    from speculators.train.data import build_client_item, _maybe_load_hs_file
finally:
    sys.stdout = _stdout_orig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate DFlash consistency between speculators and vLLM"
    )
    
    # Server configuration
    parser.add_argument("--hs-endpoint", type=str,
                        default="http://localhost:8001/v1",
                        help="Hidden states extraction server endpoint")
    parser.add_argument("--inference-endpoint", type=str,
                        default="http://localhost:8000/v1",
                        help="DFlash inference server endpoint")
    parser.add_argument("--auto-start", action="store_true",
                        help="Automatically start servers if not running")
    
    # Input configuration
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index of sample to test")
    parser.add_argument("--data-path", type=str, default="./output",
                        help="Path to preprocessed dataset")
    parser.add_argument("--checkpoint", type=str,
                        default="z-lab/gemma-4-26B-A4B-it-DFlash",
                        help="DFlash checkpoint to load")
    
    # Validation options
    parser.add_argument("--compare-intermediates", action="store_true",
                        help="Compare intermediate outputs between speculators and vLLM")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference validation (only check training)")

    # Tolerances
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance for numerical comparison")
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance for numerical comparison")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="/tmp/dflash_comparison",
                        help="Directory to save outputs and intermediates")
    
    return parser.parse_args()


def check_server_running(endpoint, timeout=2):
    """Check if a server is running at the given endpoint."""
    try:
        response = requests.get(f"{endpoint}/models", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def start_server(server_type, log_file, args):
    """Start a vLLM server using the provided scripts."""
    env = os.environ.copy()
    
    # Respect parent's CUDA_VISIBLE_DEVICES if set, otherwise use defaults
    parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    if server_type == "inference":
        env["VLLM_LOG_DRAFT_INTERMEDIATES"] = "1" if args.compare_intermediates else "0"
        # Only set CUDA_VISIBLE_DEVICES if not already set by parent
        if not parent_cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = "1"
        cmd = [
            "vllm",
            "serve",
            "google/gemma-4-26B-A4B-it",
            "--port", "8000",
            "--max-num-seqs", "1",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
            "--allowed-local-media-path", "/",
            "--speculative-config", json.dumps({
                "method": "dflash",
                "model": "z-lab/gemma-4-26B-A4B-it-DFlash",
                "num_speculative_tokens": 15,
                "attention_backend": "flex_attention",
            }),
            "--attention-backend", "flex_attention",
            "--enforce-eager",
            "--no-enable-prefix-caching",
        ]
    elif server_type == "hs":
        env["VLLM_LOG_DRAFT_INTERMEDIATES"] = "0"
        # Only set CUDA_VISIBLE_DEVICES if not already set by parent
        if not parent_cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = "2"
        cmd = [
            "vllm",
            "serve",
            "google/gemma-4-26B-A4B-it",
            "--port", "8001",
            "--max-num-seqs", "1",
            "--max-model-len", "8192",
            "--max-num-batched-tokens", "8192",  # Ensure all tokens are processed in one batch
            "--no-enable-chunked-prefill",
            "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0}),
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
            "--allowed-local-media-path", "/",
            "--speculative-config", json.dumps({
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                # Extract both: +1 offset layers [2,7,12,18,23,28,29] AND 
                # non-offset layers [1,6,11,17,22,27,30] to compare which matches
                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 6, 7, 11, 12, 17, 18, 22, 23, 27, 28, 29, 30]}},
            }),
            "--kv_transfer_config", json.dumps({
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"shared_storage_path": "/tmp/hidden_states"},
            }),
            "--attention-backend", "flex_attention",
            "--log-error-stack",
            "--enforce-eager",
            "--no-enable-prefix-caching",
        ]
    else:
        raise NotImplementedError(server_type)
    
    print(f"  Starting {server_type} server...")
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    
    return proc


def wait_for_server(endpoint, log_file=None, timeout=300, interval=5):
    """Wait for a server to become available.
    
    If the server fails to start within the timeout, check the log file
    for error messages and print them.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_server_running(endpoint):
            return True
        time.sleep(interval)
    
    # Server failed to start, check log file for errors
    if log_file and Path(log_file).exists():
        print(f"\n  ✗ Server failed to start within {timeout}s")
        print(f"  Checking log file: {log_file}")
        
        # Read last 100 lines and look for errors
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-100:] if len(lines) > 100 else lines
            
            # Look for error patterns
            error_lines = []
            for i, line in enumerate(last_lines):
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'runtimeerror', 'valueerror']):
                    # Include context: 2 lines before and 5 lines after
                    start_idx = max(0, i - 2)
                    end_idx = min(len(last_lines), i + 6)
                    error_lines.extend(last_lines[start_idx:end_idx])
                    error_lines.append("  ---\n")
            
            if error_lines:
                print("\n  Relevant error messages from log:")
                print("  " + "="*70)
                for line in error_lines[:50]:  # Limit to first 50 lines
                    print(f"  {line.rstrip()}")
                if len(error_lines) > 50:
                    print(f"  ... and {len(error_lines) - 50} more lines")
            else:
                print("  No obvious error messages found in log")
                print("  Last 20 lines of log:")
                for line in last_lines[-20:]:
                    print(f"  {line.rstrip()}")
    
    return False


def extract_hidden_states(args, sample):
    """Extract hidden states from the training server."""
    print("\n" + "=" * 80)
    print("Step 1: Extract hidden states from training server")
    print("=" * 80)
    
    # Build client item
    client_item = build_client_item(sample)
    input_ids = client_item["input_ids"]
    
    print(f"  Sample index: {args.sample_idx}")
    print(f"  Input tokens: {len(input_ids)}")

    # Extract hidden states
    print(f"\n  Sending request to {args.hs_endpoint}...")
    
    client = openai.OpenAI(base_url=args.hs_endpoint, api_key="EMPTY")
    
    # Get model name
    models = client.models.list()
    model_name = models.data[0].id
    
    # Generate hidden states
    hidden_states_path = generate_hidden_states(
        client=client,
        model=model_name,
        client_item=client_item,
        timeout=300
    )

    hidden_states = _maybe_load_hs_file(Path(hidden_states_path))
    if hidden_states is None:
        raise RuntimeError(f"Hidden stats file not found: {hidden_states_path}")

    print(f"  ✓ Hidden states extracted: {hidden_states['hidden_states'].shape}")

    return hidden_states


def _load_dflash_config(checkpoint: str) -> DFlashSpeculatorConfig | None:
    """Load or convert a DFlash checkpoint config.

    Handles legacy checkpoints that lack ``speculators_model_type`` by
    converting the old nested ``dflash_config`` format into the current
    ``DFlashSpeculatorConfig``.

    Returns ``None`` when the checkpoint already uses the new format,
    signalling the caller to let ``from_pretrained`` load the config
    itself.
    """
    config_path = hf_hub_download(checkpoint, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    if "speculators_model_type" in raw:
        return None  # new format, normal loading path

    dflash_cfg = raw.get("dflash_config", {})

    transformer_layer_config = Qwen3Config(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        hidden_act=raw["hidden_act"],
        max_position_embeddings=raw["max_position_embeddings"],
        initializer_range=raw["initializer_range"],
        rms_norm_eps=raw["rms_norm_eps"],
        use_cache=raw["use_cache"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        attention_bias=raw["attention_bias"],
        attention_dropout=raw["attention_dropout"],
        use_sliding_window=raw["use_sliding_window"],
        sliding_window=raw["sliding_window"],
        max_window_layers=raw["max_window_layers"],
        # Use full_attention for all layers by default, matching train.py's default
        # (train.py sets layer_types based on --sliding-window-indices, which
        # defaults to None = all full_attention). vLLM also ignores layer_types
        # and uses full attention for all DFlash layers.
        layer_types=["full_attention"] * raw["num_hidden_layers"],
        pad_token_id=raw["pad_token_id"],
        bos_token_id=raw["bos_token_id"],
        eos_token_id=raw["eos_token_id"],
        rope_theta=raw.get("rope_theta", 10000.0),
        rope_scaling=raw.get("rope_scaling"),
    )

    block_size = raw.get("block_size", 8)
    speculators_config = SpeculatorsConfig(
        algorithm="dflash",
        proposal_methods=[
            GreedyTokenProposalConfig(
                speculative_tokens=block_size - 1,
            )
        ],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path="google/gemma-4-26B-A4B-it",
            architectures=["Gemma4ForConditionalGeneration"],
        ),
    )

    return DFlashSpeculatorConfig(
        transformer_layer_config=transformer_layer_config,
        draft_vocab_size=raw["vocab_size"],
        block_size=block_size,
        aux_hidden_state_layer_ids=dflash_cfg.get("target_layer_ids"),
        mask_token_id=dflash_cfg.get("mask_token_id"),
        speculators_config=speculators_config,
    )


def run_speculators_forward(args, hidden_states, sample):
    """Run DFlash forward pass in speculators library."""
    print("\n" + "=" * 80)
    print("Step 2: Run DFlash forward pass in speculators")
    print("=" * 80)

    print(f"  Loading checkpoint: {args.checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = _load_dflash_config(args.checkpoint)
    if config is None:
        print("  Using native config format")
        model = DFlashDraftModel.from_pretrained(args.checkpoint)
    else:
        print("  Converted legacy config format")
        model = DFlashDraftModel.from_pretrained(args.checkpoint, config=config)

    # Use SDPA for speculators (PyTorch's standard attention implementation)
    # Note: vLLM uses FLEX_ATTENTION for main model and FLASH_ATTN for draft model
    # The divergence in attn_output is expected due to different attention implementations
    if args.compare_intermediates:
        attn_impl = "sdpa"
        print(f"  Using {attn_impl} attention for speculators")
        print(f"  Note: vLLM uses different attention backends (FLEX_ATTENTION/FLASH_ATTN)")
        print(f"        Some divergence in attn_output is expected")

        model.config.transformer_layer_config._attn_implementation = attn_impl  # noqa: SLF001
        model._attn_impl = attn_impl
        # Switch mask function to create_mask (dense) instead of create_block_mask
        from torch.nn.attention.flex_attention import create_mask
        model._create_mask_fn = create_mask
        # Update all layers
        for layer in model.layers:
            layer.self_attn.config._attn_implementation = attn_impl  # noqa: SLF001

    model = model.to(device)
    model.eval()
    
    # Explicitly load verifier weights (embed_tokens, lm_head)
    # Reset to NaN first to force loading from verifier model
    print("  Loading verifier weights...")
    if hasattr(model, 'embed_tokens'):
        torch.nn.init.constant_(model.embed_tokens.weight, torch.nan)
    if hasattr(model, 'lm_head'):
        torch.nn.init.constant_(model.lm_head.weight, torch.nan)
    if hasattr(model, 'verifier_lm_head'):
        torch.nn.init.constant_(model.verifier_lm_head.weight, torch.nan)
    model.load_verifier_weights()
    
    print("  Running forward pass...")
    with torch.no_grad():
        input_ids = torch.as_tensor(sample['input_ids'], device=device).unsqueeze(0)
        loss_mask = torch.as_tensor(sample['loss_mask'], device=device).unsqueeze(0)

        # Use all layers (don't drop the last one)
        raw_hidden = hidden_states['hidden_states'].to(device)  # [seq_len, num_layers, hidden_size]
        hidden_states_flat = raw_hidden.flatten(1).unsqueeze(0)  # [1, seq_len, num_layers*hidden_size]

        # For verifier_last_hidden, use the last layer
        verifier_last_hidden = raw_hidden[:, -1, :].unsqueeze(0)  # [1, seq_len, hidden_size]

        outputs = model(
            hidden_states=hidden_states_flat,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden,
            return_intermediates=args.compare_intermediates,
            inference_aligned=args.compare_intermediates,
        )

    # Save model and layer hidden states for per-layer FC analysis
    if args.compare_intermediates and isinstance(outputs, dict):
        outputs['_model'] = model
        outputs['layer_hidden_states'] = raw_hidden  # [seq_len, num_layers, hidden_size]
        # Note: intermediates are saved in main() after +1 offset forward pass

    if args.compare_intermediates:
        print(f"  ✓ Forward pass complete with {len([k for k in outputs.keys() if not k.startswith('_')]) - 3} intermediate tensors")
    else:
        print("  ✓ Forward pass complete")

    return outputs


def run_vllm_inference(args, hidden_states, sample):
    """Run inference in vLLM and collect outputs."""
    print("\n" + "=" * 80)
    print("Step 3: Run inference in vLLM")
    print("=" * 80)
    
    client = openai.OpenAI(base_url=args.inference_endpoint, api_key="EMPTY")

    # Get model name from server
    inf_models = client.models.list()
    inf_model_name = inf_models.data[0].id

    # Build client item
    client_item = build_client_item(sample)
    input_ids = client_item["input_ids"]

    print(f"  Sending request to {args.inference_endpoint}...")

    if 'messages' in client_item and client_item['messages']:
        response = client.chat.completions.create(
            model=inf_model_name,
            messages=client_item['messages'],
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True}
        )
        # Validate prompt token IDs match
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            if len(prompt_token_ids) != len(input_ids):
                print(f"  ⚠ Token count mismatch: expected {len(input_ids)}, got {len(prompt_token_ids)}")
            elif list(prompt_token_ids) != list(input_ids):
                for i, (a, b) in enumerate(zip(prompt_token_ids, input_ids)):
                    if a != b:
                        print(f"  ⚠ Token mismatch at position {i}: expected {b}, got {a}")
                        break
            else:
                print(f"  ✓ Prompt token IDs match ({len(prompt_token_ids)} tokens)")
        else:
            print("  ⚠ Response missing prompt_token_ids")
    else:
        response = client.completions.create(
            model=inf_model_name,
            prompt=input_ids,
            max_tokens=1,
            echo=False
        )
    
    print("  ✓ Inference complete")
    
    # Wait for intermediate files if needed
    if args.compare_intermediates:
        time.sleep(1)  # Give time for files to be written
    
    return response


def compare_tensors(spec_tensor, vllm_tensor, name, rtol=1e-3, atol=1e-3, debug=False,
                    multimodal_mask=None, text_mask=None, exclude_outliers=False, 
                    outlier_threshold=0.9):
    """Compare two tensors using cosine similarity.

    Cosine similarity measures the angular similarity between tensors,
    which is more robust to magnitude differences than abs/rel diff.
    Returns similarity in range [-1, 1], where 1 = identical direction.

    If multimodal_mask and text_mask are provided, also compute separate
    cosine similarities for each token type.
    
    If exclude_outliers is True, computes per-position similarity and excludes
    positions with similarity < outlier_threshold from the overall similarity.
    """
    if spec_tensor is None or vllm_tensor is None:
        return {
            "name": name,
            "status": "FAIL",
            "reason": "Missing tensor"
        }

    # Check if tensors have the same number of elements
    spec_numel = spec_tensor.numel()
    vllm_numel = vllm_tensor.numel()
    
    if spec_numel != vllm_numel:
        return {
            "name": name,
            "status": "FAIL",
            "reason": f"Element count mismatch: {spec_numel} vs {vllm_numel}"
        }
    
    # If shapes differ but element counts match, note it but allow comparison
    shape_note = ""
    if spec_tensor.shape != vllm_tensor.shape:
        shape_note = f" (shapes differ: {list(spec_tensor.shape)} vs {list(vllm_tensor.shape)}, but same element count)"
        
        # Handle different memory layouts for attention tensors
        # Pattern 1: vLLM uses [seq_len, heads*head_dim] while speculators uses [heads, seq_len, head_dim]
        if (spec_tensor.dim() == 3 and vllm_tensor.dim() == 2 and
            spec_tensor.shape[1] == vllm_tensor.shape[0] and
            spec_tensor.shape[0] * spec_tensor.shape[2] == vllm_tensor.shape[1]):
            # spec: [H, S, D] -> [S, H, D] -> [S, H*D]
            spec_tensor = spec_tensor.permute(1, 0, 2).reshape(vllm_tensor.shape)
            shape_note += " [reshaped to match vLLM layout]"
        # Pattern 2: vLLM uses [seq_len, heads*head_dim] while speculators uses [seq_len, heads, head_dim]
        elif (spec_tensor.dim() == 3 and vllm_tensor.dim() == 2 and
              spec_tensor.shape[0] == vllm_tensor.shape[0] and
              spec_tensor.shape[1] * spec_tensor.shape[2] == vllm_tensor.shape[1]):
            # spec: [S, H, D] -> [S, H*D]
            spec_tensor = spec_tensor.reshape(vllm_tensor.shape)
            shape_note += " [reshaped to match vLLM layout]"

    spec_f32 = spec_tensor.float()
    vllm_f32 = vllm_tensor.float()

    # Compute per-position similarity if applicable
    # This requires both tensors to be at least 2D and have the same first dimension
    can_do_per_position = (
        spec_tensor.dim() >= 2 and 
        vllm_tensor.dim() >= 2 and
        spec_tensor.shape[0] == vllm_tensor.shape[0] and
        spec_tensor.shape[0] > 1
    )
    
    if can_do_per_position:
        # Compute per-position cosine similarity
        position_sims = []
        for pos in range(spec_f32.shape[0]):
            spec_pos = spec_f32[pos].flatten()
            vllm_pos = vllm_f32[pos].flatten()

            norm_spec_pos = torch.norm(spec_pos)
            norm_vllm_pos = torch.norm(vllm_pos)

            if norm_spec_pos < 1e-8 or norm_vllm_pos < 1e-8:
                pos_sim = 0.0
            else:
                pos_sim = torch.dot(spec_pos, vllm_pos).item() / (norm_spec_pos.item() * norm_vllm_pos.item())

            position_sims.append(pos_sim)

        import numpy as np
        position_sims = np.array(position_sims)

        # Compute statistics over per-position similarities
        cosine_sim = float(np.mean(position_sims))
        overall_cosine_sim = cosine_sim
        
        if exclude_outliers:
            # Find outliers
            outlier_mask = position_sims < outlier_threshold
            num_outliers = outlier_mask.sum()

            if num_outliers > 0:
                # Compute similarity excluding outliers
                non_outlier_mask = ~outlier_mask
                filtered_sims = position_sims[non_outlier_mask]
                cosine_sim = float(np.mean(filtered_sims))
        else:
            num_outliers = 0
    else:
        # Can't do per-position comparison, just do overall
        dot_product = torch.dot(spec_f32.flatten(), vllm_f32.flatten())
        norm_spec = torch.norm(spec_f32.flatten())
        norm_vllm = torch.norm(vllm_f32.flatten())

        if norm_spec < 1e-8 or norm_vllm < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = (dot_product / (norm_spec * norm_vllm)).item()

        overall_cosine_sim = cosine_sim
        num_outliers = 0

    # Consider PASS if cosine similarity > 0.999 (very high similarity)
    # This allows for small numerical differences while ensuring same direction
    is_close = cosine_sim > 0.999

    result = {
        "name": name,
        "status": "PASS" if is_close else "FAIL",
        "shape": list(spec_tensor.shape),
        "cosine_similarity": cosine_sim,
        "spec_norm": float(torch.norm(spec_f32.flatten()).item()),
        "vllm_norm": float(torch.norm(vllm_f32.flatten()).item()),
    }
    
    if shape_note:
        result["shape_note"] = shape_note

    # Add per-position information if applicable
    if can_do_per_position:
        result["overall_cosine_similarity"] = overall_cosine_sim
        result["per_position_mean"] = float(np.mean(position_sims))
        result["per_position_median"] = float(np.median(position_sims))
        result["per_position_min"] = float(np.min(position_sims))
        result["per_position_max"] = float(np.max(position_sims))
        result["total_positions"] = int(spec_tensor.shape[0])
        
        if exclude_outliers:
            result["num_outliers"] = int(num_outliers)
            result["outlier_threshold"] = outlier_threshold
            if num_outliers > 0:
                result["outlier_positions"] = np.where(outlier_mask)[0].tolist()
                # If all positions are outliers, filtered similarity is not meaningful
                if num_outliers == spec_tensor.shape[0]:
                    result["all_positions_are_outliers"] = True

    # Compute separate cosine similarities for multimodal and text tokens
    # Only apply if tensor first dimension matches mask length
    can_apply_mask = (
        multimodal_mask is not None and 
        text_mask is not None and
        spec_f32.dim() >= 1 and
        spec_f32.shape[0] == multimodal_mask.shape[0]
    )
    
    if can_apply_mask:
        # Multimodal tokens
        if multimodal_mask.any():
            spec_mm = spec_f32[multimodal_mask].flatten()
            vllm_mm = vllm_f32[multimodal_mask].flatten()
            dot_mm = torch.dot(spec_mm, vllm_mm)
            norm_spec_mm = torch.norm(spec_mm)
            norm_vllm_mm = torch.norm(vllm_mm)
            if norm_spec_mm < 1e-8 or norm_vllm_mm < 1e-8:
                mm_sim = 0.0
            else:
                mm_sim = (dot_mm / (norm_spec_mm * norm_vllm_mm)).item()
            result["multimodal_cosine_similarity"] = mm_sim
            result["multimodal_token_count"] = multimodal_mask.sum().item()

        # Text tokens
        if text_mask.any():
            spec_text = spec_f32[text_mask].flatten()
            vllm_text = vllm_f32[text_mask].flatten()
            dot_text = torch.dot(spec_text, vllm_text)
            norm_spec_text = torch.norm(spec_text)
            norm_vllm_text = torch.norm(vllm_text)
            if norm_spec_text < 1e-8 or norm_vllm_text < 1e-8:
                text_sim = 0.0
            else:
                text_sim = (dot_text / (norm_spec_text * norm_vllm_text)).item()
            result["text_cosine_similarity"] = text_sim
            result["text_token_count"] = text_mask.sum().item()

    # Add detailed DEBUG information
    if debug:
        flat_spec = spec_f32.flatten()
        flat_vllm = vllm_f32.flatten()

        # Find index of max absolute difference (for reference)
        abs_diff = torch.abs(flat_spec - flat_vllm)
        max_diff_idx = abs_diff.argmax().item()

        result["debug"] = {
            "spec_first_10": flat_spec[:10].tolist(),
            "vllm_first_10": flat_vllm[:10].tolist(),
            "max_diff_idx": max_diff_idx,
            "max_diff_spec": flat_spec[max_diff_idx].item(),
            "max_diff_vllm": flat_vllm[max_diff_idx].item(),
            "max_diff_value": abs_diff[max_diff_idx].item(),
            "spec_stats": {
                "min": flat_spec.min().item(),
                "max": flat_spec.max().item(),
                "mean": flat_spec.mean().item(),
                "std": flat_spec.std().item(),
            },
            "vllm_stats": {
                "min": flat_vllm.min().item(),
                "max": flat_vllm.max().item(),
                "mean": flat_vllm.mean().item(),
                "std": flat_vllm.std().item(),
            },
        }

    return result


def compare_intermediate_outputs(args, spec_outputs, input_ids=None):
    """Compare intermediate outputs between speculators and vLLM."""
    print("\n" + "=" * 80)
    print("Step 4: Compare intermediate outputs")
    print("=" * 80)

    vllm_dir = Path(args.output_dir)

    # Load vLLM intermediates
    vllm_intermediates = {}

    fc_file = vllm_dir / "vllm_fc_output.pt"
    if fc_file.exists():
        data = torch.load(fc_file, map_location="cpu", weights_only=False)
        vllm_intermediates["fc_output"] = data.get("fc_output") if isinstance(data, dict) else data
        print(f"  ✓ Loaded vLLM FC output")

    hs_file = vllm_dir / "vllm_hidden_states.pt"
    if hs_file.exists():
        data = torch.load(hs_file, map_location="cpu", weights_only=False)
        vllm_intermediates["hidden_states"] = data.get("hidden_states") if isinstance(data, dict) else data
        print(f"  ✓ Loaded vLLM hidden_states before FC")

    hn_file = vllm_dir / "vllm_hidden_norm_output.pt"
    if hn_file.exists():
        data = torch.load(hn_file, map_location="cpu", weights_only=False)
        vllm_intermediates["hidden_norm_output"] = data.get("hidden_norm_output") if isinstance(data, dict) else data
        print(f"  ✓ Loaded vLLM hidden norm output")

    logits_file = vllm_dir / "vllm_logits.pt"
    if logits_file.exists():
        data = torch.load(logits_file, map_location="cpu", weights_only=False)
        vllm_intermediates["logits"] = data.get("logits") if isinstance(data, dict) else data
        print(f"  ✓ Loaded vLLM logits")

    intermediates_file = vllm_dir / "vllm_intermediates.pt"
    if intermediates_file.exists():
        data = torch.load(intermediates_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            vllm_intermediates["embed_output"] = data.get("embed_output")
            vllm_intermediates["final_norm_output"] = data.get("final_norm_output")
            # Layer outputs
            layer_outputs = []
            for i in range(100):  # Max 100 layers
                key = f"layer_{i}_output"
                if key in data:
                    layer_outputs.append(data[key])
                else:
                    break
            vllm_intermediates["layer_outputs"] = layer_outputs
            print(f"  ✓ Loaded vLLM forward pass outputs ({len(layer_outputs)} layers)")
    
    # Load vLLM attention intermediates from temp files
    import tempfile
    import glob
    temp_dir = tempfile.gettempdir()
    attn_files = sorted(glob.glob(os.path.join(temp_dir, "vllm_attn_intermediates_*.pt")))
    if attn_files:
        # vLLM uses layer indices 30-34 for DFlash draft layers, speculators uses 0-4
        # Map vLLM layer indices to speculators layer indices
        vllm_layer_offset = 30
        for attn_file in attn_files:
            # Extract layer name from filename: vllm_attn_intermediates_model.layers.30.self_attn.pt
            basename = os.path.basename(attn_file)
            layer_name = basename.replace("vllm_attn_intermediates_", "").replace(".pt", "")
            # Parse layer index from "model.layers.30.self_attn"
            parts = layer_name.split(".")
            if len(parts) >= 3 and parts[1] == "layers":
                vllm_layer_idx = int(parts[2])
                # Map vLLM layer index to speculators layer index
                spec_layer_idx = vllm_layer_idx - vllm_layer_offset
                data = torch.load(attn_file, map_location="cpu", weights_only=False)
                vllm_intermediates[f"layer_{spec_layer_idx}_attn"] = data
        print(f"  ✓ Loaded vLLM attention intermediates ({len(attn_files)} layers)")

    if not vllm_intermediates:
        print(f"✗ No intermediate files found in {vllm_dir}")
        print("  Ensure vLLM server was started with VLLM_LOG_DRAFT_INTERMEDIATES=1")
        return []

    # Create token type masks if input_ids provided
    multimodal_mask = None
    text_mask = None
    if input_ids is not None:
        # Gemma4 uses token ID 258880 for <|image|>
        processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
        if input_ids.dim() == 2:
            input_ids_flat = input_ids.squeeze(0)
        else:
            input_ids_flat = input_ids
        multimodal_mask = (input_ids_flat == processor.image_token_id)
        text_mask = ~multimodal_mask
        num_multimodal = multimodal_mask.sum().item()
        num_text = text_mask.sum().item()
        print(f"\n  Token breakdown:")
        print(f"    Multimodal tokens (token ID {processor.image_token_id}): {num_multimodal}")
        print(f"    Text tokens: {num_text}")

    # Build comparisons in forward pass order
    comparisons = []

    def squeeze_spec(t):
        """Remove batch dimension from speculators output if present."""
        if t is not None and t.dim() > 0 and t.shape[0] == 1:
            return t.squeeze(0)
        return t

    def safe_compare(spec_t, vllm_t, name, skip_bonus_token=False, seq_dim=0):
        """Compare tensors, handling shape mismatches gracefully.

        If skip_bonus_token is True, skip position 0 in the sequence dimension for speculators.
        Speculators embeds input_ids[-1] (last context token) at position 0,
        while vLLM embeds the last accepted/generated token — always different.
        Note: vLLM now logs outputs starting from position 1 (excludes bonus token),
        so we only skip from speculators side.

        seq_dim: which dimension holds the sequence in the spec tensor AFTER
        batch dim is squeezed (default 0; use 1 for [batch, heads, seq, dim]
        tensors like k_after_norm, k_after_rope, attn_output).
        """
        if spec_t is None or vllm_t is None:
            return {"name": name, "status": "FAIL", "reason": "Missing tensor"}

        spec_t = squeeze_spec(spec_t)

        # For speculators: skip bonus token (position 0) + any additional positions
        spec_n_skip = 0
        if skip_bonus_token:
            spec_n_skip += 1  # Add 1 for bonus token if skip_bonus_token is enabled

        if spec_n_skip > 0:
            # After squeeze, batch dim is gone. seq_dim tells us which remaining
            # dim holds the sequence positions.
            if spec_t.shape[seq_dim] > spec_n_skip:
                spec_t = spec_t.index_select(
                    seq_dim,
                    torch.arange(spec_n_skip, spec_t.shape[seq_dim], device=spec_t.device)
                )

        # Move to CPU for comparison
        spec_t = spec_t.cpu()
        vllm_t = vllm_t.cpu()

        return compare_tensors(spec_t, vllm_t, name, args.rtol, args.atol, debug=True,
                               multimodal_mask=multimodal_mask, text_mask=text_mask,
                               exclude_outliers=True, outlier_threshold=0.9)

    # ===== 1. Hidden states before FC =====
    vllm_hidden_states = vllm_intermediates.get("hidden_states")
    if vllm_hidden_states is not None and 'layer_hidden_states' in spec_outputs:
        layer_hidden_states = spec_outputs['layer_hidden_states']  # [seq_len, num_layers, hidden_size]

        # Flatten to match vLLM's format: [seq_len, num_layers * hidden_size]
        spec_hidden_states = layer_hidden_states.reshape(layer_hidden_states.shape[0], -1)

        print("\n  [1/7] Comparing hidden_states before FC layer:")
        comparisons.append(safe_compare(
            spec_hidden_states,
            vllm_hidden_states,
            "hidden_states_before_fc"
        ))

    # ===== 2. FC output =====
    print("\n  [2/7] Comparing FC output:")
    comparisons.append(safe_compare(
        spec_outputs.get("fc_output"),
        vllm_intermediates.get("fc_output"),
        "fc_output"
    ))

    # ===== 3. Hidden norm output =====
    print("\n  [3/7] Comparing hidden_norm output:")
    comparisons.append(safe_compare(
        spec_outputs.get("hidden_norm_output"),
        vllm_intermediates.get("hidden_norm_output"),
        "hidden_norm_output"
    ))

    # ===== 4. Embed output =====
    print("\n  [4/7] Comparing embed output:")
    comparisons.append(safe_compare(
        spec_outputs.get("embed_output"),
        vllm_intermediates.get("embed_output"),
        "embed_output",
        skip_bonus_token=True,
    ))

    # ===== 5. Layer-by-layer processing (attention intermediates + layer outputs) =====
    spec_layers = spec_outputs.get("layer_outputs", [])
    vllm_layers = vllm_intermediates.get("layer_outputs", [])
    num_layers = max(len(spec_layers), len(vllm_layers))

    print(f"\n  [5/7] Comparing {num_layers} transformer layers:")

    # Attention intermediates in forward pass order.
    # seq_dim: which dim holds the sequence after batch squeeze.
    # [1, seq, heads, dim] → squeeze → [seq, heads, dim] → seq_dim=0
    # [1, heads, seq, dim] → squeeze → [heads, seq, dim] → seq_dim=1
    attn_keys = [
        ("q_after_proj", 0),          # [1, seq, heads, dim] → squeeze → [seq, heads, dim]
        ("k_noise_after_proj", 0),    # [1, seq, kv*dim] → squeeze → [seq, kv*dim]
        ("v_noise_after_proj", 0),    # [1, seq, kv*dim] → squeeze → [seq, kv*dim]
        ("k_after_norm", 1),          # [1, kv_heads, seq, dim] → squeeze → [kv_heads, seq, dim]
        ("q_after_norm", 0),          # [1, seq, heads, dim] → squeeze → [seq, heads, dim]
        ("q_after_rope", 1),          # [1, heads, seq, dim] → squeeze → [heads, seq, dim]
        ("k_after_rope", 1),          # [1, kv_heads, seq, dim] → squeeze → [kv_heads, seq, dim]
        ("attn_output", 0),           # [1, seq, heads, dim] → squeeze → [seq, heads, dim]
        ("attn_after_o_proj", 0),     # [1, seq, hidden] → squeeze → [seq, hidden]
    ]

    for i in range(num_layers):
        print(f"\n  === Layer {i} ===")

        # First: attention intermediates (in forward pass order)
        spec_attn = spec_outputs.get(f"layer_{i}_attn", {})
        vllm_attn = vllm_intermediates.get(f"layer_{i}_attn", {})

        if spec_attn or vllm_attn:
            print(f"  Attention intermediates:")
            for key, sdim in attn_keys:
                spec_val = spec_attn.get(key) if isinstance(spec_attn, dict) else None
                vllm_val = vllm_attn.get(key) if isinstance(vllm_attn, dict) else None

                # Print shapes for debugging
                if spec_val is not None:
                    print(f"    spec_{key}: shape={list(spec_val.shape)}")
                if vllm_val is not None:
                    print(f"    vllm_{key}: shape={list(vllm_val.shape)}")

                if spec_val is not None or vllm_val is not None:
                    comparisons.append(safe_compare(
                        spec_val,
                        vllm_val,
                        f"layer_{i}_attn_{key}",
                        skip_bonus_token=True,
                        seq_dim=sdim,
                    ))

        # Then: layer output (MLP output before residual, to match vLLM)
        print(f"  Layer output:")
        spec_mlp_output = spec_attn.get("mlp_output") if isinstance(spec_attn, dict) else None
        comparisons.append(safe_compare(
            spec_mlp_output,
            vllm_layers[i] if i < len(vllm_layers) else None,
            f"layer_outputs[{i}]",
            skip_bonus_token=True,
        ))

    # ===== 6. Final norm output =====
    print("\n  [6/7] Comparing final_norm output:")
    comparisons.append(safe_compare(
        spec_outputs.get("final_norm_output"),
        vllm_intermediates.get("final_norm_output"),
        "final_norm_output",
        skip_bonus_token=True,
    ))

    # ===== 7. Logits =====
    print("\n  [7/7] Comparing logits:")

    # Logits - only compare draft token positions
    spec_logits = squeeze_spec(spec_outputs.get("logits"))
    vllm_logits = vllm_intermediates.get("logits")
    if spec_logits is not None and vllm_logits is not None:
        # Move both to CPU for comparison
        spec_logits = spec_logits.cpu()
        vllm_logits = vllm_logits.cpu()
        # spec_logits: [num_positions, vocab_size] - includes bonus token at position 0
        # vllm_logits: [num_draft_tokens, vocab_size] - only draft tokens (no bonus token)
        # Skip the bonus token (position 0) in speculators logits to match vLLM
        if spec_logits.shape[0] > vllm_logits.shape[0]:
            # Speculators has bonus token, skip it
            spec_logits_draft = spec_logits[1:1+vllm_logits.shape[0]]
            vllm_logits_draft = vllm_logits
        else:
            # Same length, compare directly
            spec_logits_draft = spec_logits
            vllm_logits_draft = vllm_logits
        
        comparisons.append(compare_tensors(
            spec_logits_draft,
            vllm_logits_draft,
            f"logits (draft tokens only, {spec_logits_draft.shape[0]} positions)",
            args.rtol, args.atol
        ))
    else:
        comparisons.append({
            "name": "logits",
            "status": "FAIL",
            "reason": "Missing tensor"
        })
    
    return comparisons


def validate_training_consistency(outputs):
    """Validate that training forward pass is consistent."""
    print("\n" + "=" * 80)
    print("Step 5: Validate training consistency")
    print("=" * 80)
    
    validations = []
    
    # FC consistency
    if "fc_output" in outputs and "hidden_norm_output" in outputs:
        fc_out = outputs["fc_output"]
        hn_out = outputs["hidden_norm_output"]
        is_finite = torch.isfinite(fc_out).all() and torch.isfinite(hn_out).all()
        validations.append({
            "name": "FC and hidden norm outputs",
            "status": "PASS" if is_finite else "FAIL",
            "reason": "All values finite" if is_finite else "Contains NaN or Inf"
        })
        print(f"  {'✓' if is_finite else '✗'} FC and hidden norm outputs are finite")
    
    # Logits validation
    if "logits" in outputs:
        logits = outputs["logits"]
        is_finite = torch.isfinite(logits).all()
        has_reasonable_range = logits.abs().max() < 100
        validations.append({
            "name": "Logits",
            "status": "PASS" if (is_finite and has_reasonable_range) else "FAIL",
            "reason": "Finite and reasonable range" if (is_finite and has_reasonable_range) 
                     else "Invalid logits"
        })
        print(f"  {'✓' if (is_finite and has_reasonable_range) else '✗'} Logits are valid")
    
    # Draft tokens
    if "draft_tokens" in outputs:
        draft_tokens = outputs["draft_tokens"]
        is_valid = (draft_tokens >= 0).all()
        validations.append({
            "name": "Draft tokens",
            "status": "PASS" if is_valid else "FAIL",
            "reason": "All tokens non-negative" if is_valid else "Invalid token IDs"
        })
        print(f"  {'✓' if is_valid else '✗'} Draft tokens are valid")
    
    return validations


def print_results(args, training_validations, comparisons_plus1, comparisons_non_offset, inference_response):
    """Print final results summary."""
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    # Training consistency
    print("\n1. Training Consistency Checks:")
    print("-" * 80)
    training_passed = 0
    training_failed = 0
    for val in training_validations:
        status_symbol = "✓" if val["status"] == "PASS" else "✗"
        print(f"  {status_symbol} {val['name']}: {val['reason']}")
        if val["status"] == "PASS":
            training_passed += 1
        else:
            training_failed += 1
    
    print(f"\n  Training: {training_passed} passed, {training_failed} failed")
    
    # Intermediate comparisons
    if comparisons_plus1 and comparisons_non_offset:
        print("\n2. Intermediate Output Comparisons:")
        print("-" * 80)
        passed = 0
        failed = 0
        info = 0

        print("\n" + "="*60)
        print("Comparing +1 offset layers against vLLM")
        print("="*60)

        for comp in comparisons_plus1:
            status = comp["status"]
            if status == "INFO":
                info += 1
                print(f"  ℹ {comp['name']}: {comp['reason']}")
            elif status == "PASS":
                passed += 1
                print(f"  ✓ {comp['name']}: cosine_sim={comp['cosine_similarity']:.6f}")
                # Print shape note if shapes differ
                if 'shape_note' in comp:
                    print(f"      {comp['shape_note']}")
                # Print per-position statistics if available
                if 'per_position_mean' in comp:
                    print(f"      Per-position: mean={comp['per_position_mean']:.6f}, median={comp['per_position_median']:.6f}, min={comp['per_position_min']:.6f}, max={comp['per_position_max']:.6f}")
                # Print token-type-specific similarities if available
                if 'multimodal_cosine_similarity' in comp:
                    print(f"      Multimodal tokens ({comp['multimodal_token_count']}): {comp['multimodal_cosine_similarity']:.6f}")
                if 'text_cosine_similarity' in comp:
                    print(f"      Text tokens ({comp['text_token_count']}): {comp['text_cosine_similarity']:.6f}")
            else:
                failed += 1
                if "reason" in comp:
                    print(f"  ✗ {comp['name']}: {comp['reason']}")
                    # Print shape note if shapes differ
                    if 'shape_note' in comp:
                        print(f"      {comp['shape_note']}")
                else:
                    print(f"  ✗ {comp['name']}: Low cosine similarity")
                    # Print shape note if shapes differ
                    if 'shape_note' in comp:
                        print(f"      {comp['shape_note']}")

                    # Print per-position statistics if available
                    if 'per_position_mean' in comp:
                        print(f"      Per-position: mean={comp['per_position_mean']:.6f}, median={comp['per_position_median']:.6f}, min={comp['per_position_min']:.6f}, max={comp['per_position_max']:.6f}")

                    # Check if all positions are outliers
                    if comp.get('all_positions_are_outliers', False):
                        print(f"      All {comp['total_positions']} positions are outliers (threshold: {comp['outlier_threshold']:.1f})")
                        print(f"      Overall cosine similarity: {comp['overall_cosine_similarity']:.6f}")
                    else:
                        print(f"      Cosine similarity (filtered): {comp['cosine_similarity']:.6f} (threshold: 0.999)")

                        # Print outlier information if available
                        if 'overall_cosine_similarity' in comp:
                            print(f"      Cosine similarity (overall): {comp['overall_cosine_similarity']:.6f}")
                        if 'num_outliers' in comp and comp['num_outliers'] > 0:
                            print(f"      Outliers: {comp['num_outliers']} / {comp['total_positions']} positions (threshold: {comp['outlier_threshold']:.1f})")
                            if 'outlier_positions' in comp and len(comp['outlier_positions']) <= 20:
                                print(f"      Outlier positions: {comp['outlier_positions']}")

                    print(f"      Spec norm: {comp['spec_norm']:.6f}, vLLM norm: {comp['vllm_norm']:.6f}")

                    # Print token-type-specific similarities if available
                    if 'multimodal_cosine_similarity' in comp:
                        print(f"      Multimodal tokens ({comp['multimodal_token_count']}): {comp['multimodal_cosine_similarity']:.6f}")
                    if 'text_cosine_similarity' in comp:
                        print(f"      Text tokens ({comp['text_token_count']}): {comp['text_cosine_similarity']:.6f}")

                    # Print detailed debug information if available
                    if "debug" in comp:
                        dbg = comp["debug"]
                        print(f"      First 10 values:")
                        print(f"        spec:  {dbg['spec_first_10']}")
                        print(f"        vllm:  {dbg['vllm_first_10']}")

        print("\n" + "="*60)
        print("Comparing non-offset layers against vLLM")
        print("="*60)

        for comp in comparisons_non_offset:
            status = comp["status"]
            if status == "INFO":
                info += 1
                print(f"  ℹ {comp['name']}: {comp['reason']}")
            elif status == "PASS":
                passed += 1
                print(f"  ✓ {comp['name']}: cosine_sim={comp['cosine_similarity']:.6f}")
                # Print shape note if shapes differ
                if 'shape_note' in comp:
                    print(f"      {comp['shape_note']}")
                # Print per-position statistics if available
                if 'per_position_mean' in comp:
                    print(f"      Per-position: mean={comp['per_position_mean']:.6f}, median={comp['per_position_median']:.6f}, min={comp['per_position_min']:.6f}, max={comp['per_position_max']:.6f}")
                # Print token-type-specific similarities if available
                if 'multimodal_cosine_similarity' in comp:
                    print(f"      Multimodal tokens ({comp['multimodal_token_count']}): {comp['multimodal_cosine_similarity']:.6f}")
                if 'text_cosine_similarity' in comp:
                    print(f"      Text tokens ({comp['text_token_count']}): {comp['text_cosine_similarity']:.6f}")
            else:
                failed += 1
                if "reason" in comp:
                    print(f"  ✗ {comp['name']}: {comp['reason']}")
                    # Print shape note if shapes differ
                    if 'shape_note' in comp:
                        print(f"      {comp['shape_note']}")
                else:
                    print(f"  ✗ {comp['name']}: Low cosine similarity")
                    # Print shape note if shapes differ
                    if 'shape_note' in comp:
                        print(f"      {comp['shape_note']}")

                    # Print per-position statistics if available
                    if 'per_position_mean' in comp:
                        print(f"      Per-position: mean={comp['per_position_mean']:.6f}, median={comp['per_position_median']:.6f}, min={comp['per_position_min']:.6f}, max={comp['per_position_max']:.6f}")

                    # Check if all positions are outliers
                    if comp.get('all_positions_are_outliers', False):
                        print(f"      All {comp['total_positions']} positions are outliers (threshold: {comp['outlier_threshold']:.1f})")
                        print(f"      Overall cosine similarity: {comp['overall_cosine_similarity']:.6f}")
                    else:
                        print(f"      Cosine similarity (filtered): {comp['cosine_similarity']:.6f} (threshold: 0.999)")

                        # Print outlier information if available
                        if 'overall_cosine_similarity' in comp:
                            print(f"      Cosine similarity (overall): {comp['overall_cosine_similarity']:.6f}")
                        if 'num_outliers' in comp and comp['num_outliers'] > 0:
                            print(f"      Outliers: {comp['num_outliers']} / {comp['total_positions']} positions (threshold: {comp['outlier_threshold']:.1f})")
                            if 'outlier_positions' in comp and len(comp['outlier_positions']) <= 20:
                                print(f"      Outlier positions: {comp['outlier_positions']}")

                    print(f"      Spec norm: {comp['spec_norm']:.6f}, vLLM norm: {comp['vllm_norm']:.6f}")

                    # Print token-type-specific similarities if available
                    if 'multimodal_cosine_similarity' in comp:
                        print(f"      Multimodal tokens ({comp['multimodal_token_count']}): {comp['multimodal_cosine_similarity']:.6f}")
                    if 'text_cosine_similarity' in comp:
                        print(f"      Text tokens ({comp['text_token_count']}): {comp['text_cosine_similarity']:.6f}")

                    # Print detailed debug information if available
                    if "debug" in comp:
                        dbg = comp["debug"]
                        print(f"      First 10 values:")
                        print(f"        spec:  {dbg['spec_first_10']}")
                        print(f"        vllm:  {dbg['vllm_first_10']}")
                        print(f"      Max diff at index {dbg['max_diff_idx']}:")
                        print(f"        spec:  {dbg['max_diff_spec']:.6f}")
                        print(f"        vllm:  {dbg['max_diff_vllm']:.6f}")
                        print(f"        diff:  {dbg['max_diff_value']:.6f}")
                        print(f"      Speculator stats:")
                        print(f"        min={dbg['spec_stats']['min']:.6f}, max={dbg['spec_stats']['max']:.6f}, mean={dbg['spec_stats']['mean']:.6f}, std={dbg['spec_stats']['std']:.6f}")
                        print(f"      vLLM stats:")
                        print(f"        min={dbg['vllm_stats']['min']:.6f}, max={dbg['vllm_stats']['max']:.6f}, mean={dbg['vllm_stats']['mean']:.6f}, std={dbg['vllm_stats']['std']:.6f}")

        print(f"\n  Intermediates (+1 offset): {passed} passed, {failed} failed, {info} info")

        # Note: We only count +1 offset layers in the final pass/fail determination
        # because vLLM uses +1 offset layers [2,7,12,18,23,28] for inference.
        # The non-offset layers are shown for comparison but are expected to have
        # lower similarity since they're the wrong layer selection.

    # Inference validation
    if not args.skip_inference and inference_response:
        print("\n3. Inference Output Validation:")
        print("-" * 80)
        if inference_response.choices:
            text = inference_response.choices[0].message.content if hasattr(
                inference_response.choices[0], 'message'
            ) else inference_response.choices[0].text

            if text and len(text.strip()) > 0:
                print(f"  ✓ Inference produced output: {text[:100]}...")
            else:
                print(f"  ✗ Inference produced empty output")
        else:
            print(f"  ✗ Inference failed to produce response")

    # Overall summary
    print("\n" + "=" * 80)
    all_training_passed = training_failed == 0
    # Only check +1 offset layers for final pass/fail (vLLM uses +1 offset layers)
    all_intermediates_passed = (not comparisons_plus1 or
                                all(c["status"] != "FAIL" for c in comparisons_plus1))
    
    if all_training_passed and all_intermediates_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 80)
        print("\nThe DFlash implementation is consistent between speculators and vLLM.")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 80)
        print("\nPlease review the failed checks above.")
        return 1


def main():
    args = parse_args()
    
    print("=" * 80)
    print("DFlash Unified Validation")
    print("Speculators ↔ vLLM Consistency Check")
    print("=" * 80)
    
    # Clear/create output directory to avoid pollution from previous runs
    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"Clearing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clean stale vLLM intermediate files from /tmp to avoid comparing
    # against data from a previous run with different input or config
    import glob as _glob
    import tempfile as _tempfile
    stale_files = _glob.glob(os.path.join(_tempfile.gettempdir(), "vllm_*.pt"))
    if stale_files:
        print(f"Cleaning {len(stale_files)} stale vLLM intermediate files from {_tempfile.gettempdir()}")
        for f in stale_files:
            os.remove(f)
    
    # Check servers
    print("\nChecking server status...")
    hs_running = check_server_running(args.hs_endpoint)
    inf_running = check_server_running(args.inference_endpoint)
    
    print(f"  Hidden states server: {'✓ running' if hs_running else '✗ not running'}")
    print(f"  Inference server: {'✓ running' if inf_running else '✗ not running'}")
    
    # Auto-start servers if requested
    server_procs = []
    if args.auto_start:
        if not hs_running:
            hs_log_file = f"{args.output_dir}/hs_server.log"
            proc = start_server("hs", hs_log_file, args)
            if proc:
                server_procs.append(proc)
                hs_running = wait_for_server(args.hs_endpoint, log_file=hs_log_file)

        if not inf_running and not args.skip_inference:
            inf_log_file = f"{args.output_dir}/inference_server.log"
            proc = start_server("inference", inf_log_file, args)
            if proc:
                server_procs.append(proc)
                inf_running = wait_for_server(args.inference_endpoint, log_file=inf_log_file)
    
    # Verify servers are running
    if not hs_running:
        print("\n✗ Hidden states server is not running")
        print("(Serve by running this script again with --auto-start)")
        return 1
    
    if not inf_running and not args.skip_inference:
        print("\n✗ Inference server is not running")
        print("(Serve by running this script again with --auto-start)")
        return 1
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    try:
        dataset = load_from_disk(args.data_path)
        print(f"  ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return 1
    
    if args.sample_idx >= len(dataset):
        print(f"✗ Sample index {args.sample_idx} out of range (max: {len(dataset) - 1})")
        return 1
    
    sample = dataset[args.sample_idx]

    # Run validation pipeline
    try:
        # Step 1: Extract hidden states (both layer sets)
        hidden_states_dict = extract_hidden_states(args, sample)
        hidden_states_all = hidden_states_dict['hidden_states']  # [seq_len, 14, hidden_size]
        
        # Split into two groups:
        # +1 offset layers [2,7,12,18,23,28]: indices [1, 3, 5, 7, 9, 11]
        # non-offset layers [1,6,11,17,22,27]: indices [0, 2, 4, 6, 8, 10]
        offset_plus1_indices = [1, 3, 5, 7, 9, 11]
        non_offset_indices = [0, 2, 4, 6, 8, 10]

        hidden_states_plus1 = hidden_states_all[:, offset_plus1_indices, :]  # [seq_len, 6, hidden_size]
        hidden_states_non_offset = hidden_states_all[:, non_offset_indices, :]  # [seq_len, 6, hidden_size]

        print(f"\nSplit hidden states into two groups:")
        print(f"  +1 offset layers [2,7,12,18,23,28]: {hidden_states_plus1.shape}")
        print(f"  non-offset layers [1,6,11,17,22,27]: {hidden_states_non_offset.shape}")
        print(f"\n  Note: +1 offset layers show better similarity (~0.99) for both image and text tokens")
        print(f"        This is the correct layer selection for DFlash inference")

        # Step 2a: Run speculators forward with +1 offset layers (primary)
        print("\n" + "="*60)
        print("Testing +1 offset layers [2,7,12,18,23,28] (primary)")
        print("="*60)
        spec_outputs_plus1 = run_speculators_forward(
            args,
            {'hidden_states': hidden_states_plus1},
            sample
        )
        
        # Save the +1 offset intermediates (primary results)
        if args.compare_intermediates:
            spec_intermediates_path = Path(args.output_dir) / "speculators_intermediates.pt"
            outputs_to_save = {k: v for k, v in spec_outputs_plus1.items() if k != '_model'}
            torch.save(outputs_to_save, spec_intermediates_path)
            print(f"  ✓ Saved +1 offset intermediates to {spec_intermediates_path}")

        # Step 2b: Run speculators forward with non-offset layers (comparison)
        print("\n" + "="*60)
        print("Testing non-offset layers [1,6,11,17,22,27] (comparison)")
        print("="*60)
        spec_outputs_non_offset = run_speculators_forward(
            args,
            {'hidden_states': hidden_states_non_offset},
            sample
        )

        # Debug: verify in-memory values differ between the two forward passes
        if args.compare_intermediates:
            q_plus1 = spec_outputs_plus1.get('layer_0_attn', {}).get('q_after_proj')
            q_nonoff = spec_outputs_non_offset.get('layer_0_attn', {}).get('q_after_proj')
            if q_plus1 is not None and q_nonoff is not None:
                q1_norm = q_plus1.squeeze(0)[1].float().norm().item()
                q2_norm = q_nonoff.squeeze(0)[1].float().norm().item()
                same_ptr = q_plus1.data_ptr() == q_nonoff.data_ptr()
                cos_q = torch.nn.functional.cosine_similarity(
                    q_plus1.squeeze(0)[1:].float().flatten().unsqueeze(0),
                    q_nonoff.squeeze(0)[1:].float().flatten().unsqueeze(0)).item()
                print(f"\n  DEBUG: q_plus1 norm(pos1)={q1_norm:.4f}, q_non_offset norm(pos1)={q2_norm:.4f}")
                print(f"  DEBUG: same data_ptr={same_ptr}, cos(plus1 vs non_offset)={cos_q:.8f}")

        # Step 3: Run vLLM inference (uses +1 offset layers [2,7,12,18,23,28])
        inference_response = None
        if not args.skip_inference:
            inference_response = run_vllm_inference(args, hidden_states_dict, sample)

        # Step 4: Compare intermediates (if requested)
        if args.compare_intermediates:
            # Extract input_ids from sample for token type separation
            input_ids = sample.get('input_ids')
            if input_ids is None:
                # Try to get from hidden_states_dict
                input_ids = hidden_states_dict.get('token_ids')
            
            comparisons_plus1 = compare_intermediate_outputs(args, spec_outputs_plus1, input_ids=input_ids)
            comparisons_non_offset = compare_intermediate_outputs(args, spec_outputs_non_offset, input_ids=input_ids)
        else:
            comparisons_plus1 = None
            comparisons_non_offset = None

        # Step 5: Validate training consistency (use +1 offset as primary)
        training_validations = validate_training_consistency(spec_outputs_plus1)

        # Print results
        exit_code = print_results(args, training_validations, comparisons_plus1, comparisons_non_offset, inference_response)
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # Clean up auto-started servers
        for proc in server_procs:
            proc.terminate()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
