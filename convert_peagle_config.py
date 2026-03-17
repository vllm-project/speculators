#!/usr/bin/env python3
"""Convert P-EAGLE config.json to vLLM-compatible format."""

import json
import sys
from pathlib import Path


def convert_peagle_config(input_path: Path, output_path: Path | None = None):
    """
    Convert P-EAGLE training config to vLLM inference config.

    Args:
        input_path: Path to P-EAGLE checkpoint directory or config.json
        output_path: Optional output path (defaults to input_path/config.json)
    """
    # Handle directory or file input
    if input_path.is_dir():
        config_path = input_path / "config.json"
    else:
        config_path = input_path
        input_path = input_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load P-EAGLE config
    with open(config_path) as f:
        peagle_config = json.load(f)

    # Extract transformer layer config (base model architecture)
    transformer_config = peagle_config["transformer_layer_config"]

    # Create vLLM-compatible config
    vllm_config = {
        # Use verifier architecture (LlamaForCausalLM for Llama models)
        "architectures": ["LlamaForCausalLM"],

        # Flatten transformer config to top level
        **transformer_config,

        # Add P-EAGLE specific fields
        "draft_vocab_size": peagle_config["draft_vocab_size"],
        "ptd_token_id": peagle_config["ptd_token_id"],

        # Keep dtype if specified
        "dtype": peagle_config.get("dtype", "bfloat16"),

        # Preserve transformers version
        "transformers_version": peagle_config.get("transformers_version", "4.57.0"),
    }

    # Add optional fields if they exist in transformer config
    optional_fields = ["bos_token_id", "eos_token_id", "tie_word_embeddings", "bias"]
    for field in optional_fields:
        if field in transformer_config:
            vllm_config[field] = transformer_config[field]

    # Determine output path
    if output_path is None:
        output_path = input_path / "config.json"
        # Backup original
        backup_path = input_path / "config.json.peagle_backup"
        if not backup_path.exists():
            print(f"Backing up original config to {backup_path}")
            with open(config_path) as f:
                original = f.read()
            with open(backup_path, "w") as f:
                f.write(original)

    # Write vLLM config
    with open(output_path, "w") as f:
        json.dump(vllm_config, f, indent=2)

    print(f"Converted P-EAGLE config to vLLM format: {output_path}")
    print(f"\nKey changes:")
    print(f"  - Architecture: {peagle_config['architectures']} -> {vllm_config['architectures']}")
    print(f"  - Flattened transformer_layer_config to top level")
    print(f"  - Preserved P-EAGLE fields: draft_vocab_size={vllm_config['draft_vocab_size']}, "
          f"ptd_token_id={vllm_config['ptd_token_id']}")

    return vllm_config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_peagle_config.py <checkpoint_dir> [output_path]")
        print("\nExample:")
        print("  python convert_peagle_config.py output/peagle_llama3_8b_sharegpt_5k/checkpoints/6")
        print("  python convert_peagle_config.py output/peagle_llama3_8b_sharegpt_5k/checkpoints/6/config.json config_vllm.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    convert_peagle_config(input_path, output_path)
