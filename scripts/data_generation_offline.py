#!/usr/bin/env python3
"""
Offline EAGLE Training Data Generation Pipeline

This script generates training data for EAGLE models by:
1. Automatically preprocessing data if needed (or loading from cache)
2. Using vLLM to extract hidden states from target model
3. Saving each data point as a separate .pt file

Preprocessing is cached automatically by HuggingFace datasets.
Token frequencies are saved in the current directory by default.

Usage:
    python scripts/data_generation_offline.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path sharegpt \
        --output-dir ./training_data \
        --hf-cache-dir /path/to/cache \
        --max-samples 5000

    # Multimodal (VL) example:
    # (Assumes you already converted the dataset to JSONL, e.g. ./data/pokemon_vl.jsonl)
    python scripts/data_generation_offline.py \
        --target-model-path Qwen/Qwen3-VL-8B-Instruct \
        --train-data-path ./data/pokemon_vl.jsonl \
        --output-dir ./training_data_vl \
        --data-mode vl \
        --image-root ./pokemon_images
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from datasets import config as datasets_config
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

# Set vLLM to use 'spawn' instead of 'fork'
# to prevent "Cannot re-initialize CUDA in forked subprocess" errors
from vllm import envs
from vllm.multimodal.utils import fetch_image as vllm_fetch_image

envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

from speculators.data_generation.config_generator import (  # noqa: E402
    DataGenerationConfig,
)
from speculators.data_generation.logging_utils import PipelineLogger  # noqa: E402
from speculators.data_generation.preprocessing import (  # noqa: E402
    load_and_preprocess_dataset,
)
from speculators.data_generation.vllm_hidden_states_generator import (  # noqa: E402
    VllmHiddenStatesGenerator,
)

# Constants
MAX_IO_WORKERS = 4  # Number of parallel file save operations

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = PipelineLogger(__name__)




def parse_args():
    # Parse CLI arguments for offline data generation.
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=torch.accelerator.device_count(),
        help="Tensor parallel size for target model (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Target GPU memory utilization (default: 0.8)",
    )

    # Data arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (same as used in preprocessing)",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["text", "vl"],
        default="text",
        help="Data mode: text (default) or vl for image-text inputs.",
    )
    parser.add_argument(
        "--image-field",
        type=str,
        default="image",
        help=(
            "Field name for image entries in multimodal data (VL mode only). "
            "Default: image."
        ),
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for preprocessing and model (default: 2048)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default="./token_freq.pt",
        help="Path to save token frequency distribution (default: ./token_freq.pt)",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help=(
            "Optional root directory for resolving image paths in multimodal data "
            "(VL mode only)."
        ),
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for HuggingFace datasets cache. "
            "If not specified, uses HF_DATASETS_CACHE env var or default location. "
            "(default: None)"
        ),
    )
    parser.add_argument(
        "--assistant-pattern",
        type=str,
        default=None,
        help=(
            "Custom regex pattern for matching assistant responses. "
            "If not provided, auto-detected from chat template."
        ),
    )
    parser.add_argument(
        "--turn-dropout",
        action="store_true",
        help=(
            "Enable turn dropout: randomly keeps first N consecutive turns "
            "per conversation for data augmentation."
        ),
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save .pt files"
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of layer IDs from which to capture hidden states "
            "(default: auto-select)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for hidden states generation (default: 8)",
    )

    # Processing arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (must match preprocessing seed, default: 0)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for output files (default: 0)",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset preprocessing (default: 8)",
    )
    args = parser.parse_args()

    if args.data_mode != "vl":
        uses_vl_args = args.image_root is not None or args.image_field != "image"
        if uses_vl_args:
            parser.error(
                "Image arguments (--image-field/--image-root) "
                "are only valid with --data-mode vl."
            )

    return args


def find_last_checkpoint(output_dir: str) -> int:
    """Find the last successfully saved file index by scanning existing files."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    max_index = -1
    for file_path in output_path.iterdir():
        if file_path.name.startswith("data_") and file_path.name.endswith(".pt"):
            index_str = file_path.stem[5:]  # Remove "data_" prefix
            try:
                index = int(index_str)
                max_index = max(max_index, index)
            except ValueError:
                continue

    return max_index + 1


def save_sample_to_disk(data_dict, output_path):
    """Save a single sample to disk for async execution."""
    torch.save(data_dict, output_path)
    return output_path


def save_config(args, generator, num_samples, output_dir):
    """Save metadata config file for reproducibility."""
    log.subsection("Saving configuration metadata")

    cache_dir = (
        args.hf_cache_dir if args.hf_cache_dir else datasets_config.HF_DATASETS_CACHE
    )

    config = DataGenerationConfig.from_generator(
        generator=generator,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        cache_dir=str(cache_dir),
        num_samples=num_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    config_path = Path(output_dir) / "data_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    log.info(f"Saved config v{config.version} to {config_path}")


def _resolve_image_ref(
    image_ref: Any,
    image_root: str | None,
    image_fetcher: Any,
) -> Image.Image:
    # Resolve a single image reference into a PIL image.
    if isinstance(image_ref, dict):
        image_ref = (
            image_ref.get("url") or image_ref.get("path") or image_ref.get("image")
        )
    if not isinstance(image_ref, str):
        raise ValueError(f"Unsupported image reference: {type(image_ref)}")

    image_path = image_ref
    if image_path.startswith(("http://", "https://", "data:")):
        return image_fetcher(image_path)

    if image_path.startswith("file://"):
        image_path = image_path[len("file://") :]

    if os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")

    candidate_path = None
    if image_root and not os.path.isabs(image_ref):
        candidate_path = os.path.join(image_root, image_ref)
        if os.path.exists(candidate_path):
            return Image.open(candidate_path).convert("RGB")

    if candidate_path is None:
        candidate_path = image_path

    abs_path = os.path.abspath(candidate_path)
    return image_fetcher(f"file://{abs_path}")


def _build_multimodal_inputs(
    batch_mm_data: list[dict[str, list[Any]]],
    image_root: str | None,
    image_fetcher: Any,
    image_field: str,
    placeholder_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    # Convert multimodal data refs into vLLM MultiModalDataDict inputs.
    # NOTE: This path only supports a single image per sample for now.
    # TODO(VL): Extend to multi-image prompts once alignment is stable.
    multimodal_inputs: list[dict[str, Any]] = []
    for idx, mm_data in enumerate(batch_mm_data):
        if not mm_data:
            multimodal_inputs.append({})
            continue
        images = []
        for image_ref in mm_data.get(image_field, []):
            images.append(_resolve_image_ref(image_ref, image_root, image_fetcher))
        if placeholder_counts is not None and images:
            target_count = placeholder_counts[idx]
            if target_count == 0:
                raise ValueError(
                    "Multimodal data provided but prompt has no image placeholders."
                )
            if target_count != len(images):
                raise ValueError(
                    "Single-image mode requires exactly one image placeholder."
                )
        if len(images) != 1:
            raise ValueError("Single-image mode requires exactly one image ref.")
        mm_input: dict[str, Any] = {}
        if images:
            # vLLM expects a list for image modality; keep length=1.
            mm_input["image"] = images
        multimodal_inputs.append(mm_input)
    return multimodal_inputs


def _align_loss_mask_to_vllm(
    vllm_input_ids: list[int],
    pre_input_ids: list[int],
    pre_loss_mask: torch.Tensor,
    image_pad_id: int | None,
) -> torch.Tensor:
    # Align preprocessing loss_mask to vLLM tokenization by expanding image pads.
    if image_pad_id is None:
        return pre_loss_mask[: len(vllm_input_ids)]
    aligned: list[int] = []
    i = 0
    j = 0
    pre_mask_list = pre_loss_mask.tolist()
    while i < len(vllm_input_ids) and j < len(pre_input_ids) and j < len(pre_mask_list):
        v_id = vllm_input_ids[i]
        p_id = pre_input_ids[j]
        if v_id == image_pad_id and p_id == image_pad_id:
            aligned.append(pre_mask_list[j])
            i += 1
            j += 1
            continue
        if v_id == image_pad_id and p_id != image_pad_id:
            aligned.append(0)
            i += 1
            continue
        aligned.append(pre_mask_list[j])
        i += 1
        j += 1
    # If vLLM has extra tokens beyond preprocessing, pad with 0 mask.
    while i < len(vllm_input_ids):
        aligned.append(0)
        i += 1
    # If preprocessing had extra tokens, ignore the tail (already truncated).
    return torch.tensor(aligned, dtype=torch.long)


def generate_and_save_hidden_states(args, dataset):  # noqa: C901, PLR0915
    """Generate hidden states and save each sample as a .pt file"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start_file_idx = find_last_checkpoint(args.output_dir)
    sample_lengths = {}

    if start_file_idx > 0:
        log.subsection(f"Resuming: {start_file_idx} files already exist")

    num_samples = len(dataset)
    start_sample_idx = start_file_idx - args.start_idx

    if start_sample_idx >= num_samples:
        log.info("All samples already processed!")
        return 0

    log.subsection("Initializing vLLM hidden states generator")
    generator = VllmHiddenStatesGenerator(
        model_path=args.target_model_path,
        layer_ids=args.layer_ids,
        max_model_len=args.seq_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    log.info(f"Processing {num_samples - start_sample_idx}/{num_samples} samples")
    file_idx = start_file_idx

    num_batches = (
        num_samples - start_sample_idx + args.batch_size - 1
    ) // args.batch_size

    # Use ThreadPoolExecutor for async file I/O
    max_io_workers = MAX_IO_WORKERS

    pbar = tqdm(
        range(start_sample_idx, num_samples, args.batch_size),
        desc="Generating hidden states",
        total=num_batches,
    )

    image_fetcher = None
    image_pad_id = None

    with ThreadPoolExecutor(max_workers=max_io_workers) as thread_executor:
        futures = []

        for i in pbar:
            batch_end = min(i + args.batch_size, num_samples)
            batch = dataset[i:batch_end]
            batch_input_ids = batch["input_ids"]
            batch_loss_mask = batch["loss_mask"]
            if args.data_mode == "vl":
                batch_mm_data = batch.get("multi_modal_data")
                batch_prompts = batch.get("prompt")

                if batch_mm_data is None or batch_prompts is None:
                    raise ValueError(
                        "VL mode requires multi_modal_data and prompt fields "
                        "from preprocessing. Ensure the dataset uses image-text "
                        "format and preprocessing supports multimodal outputs."
                    )

                if image_fetcher is None:
                    image_fetcher = vllm_fetch_image
                mm_inputs = _build_multimodal_inputs(
                    batch_mm_data,
                    args.image_root,
                    image_fetcher,
                    args.image_field,
                )
                request_ids = [f"sample_{i + j}" for j in range(len(batch_prompts))]
                results = generator.generate(
                    batch_input_ids,
                    prompt_texts=batch_prompts,
                    multimodal_inputs=mm_inputs,
                    request_ids=request_ids,
                )
            else:
                results = generator.generate(batch_input_ids)

            # Submit save operations to thread pool (async I/O)
            for j, result in enumerate(results):
                # Truncate loss_mask to match input_ids length (generator may truncate)
                input_ids_list = (
                    result["input_ids"].tolist()
                    if isinstance(result["input_ids"], torch.Tensor)
                    else result["input_ids"]
                )
                input_len = len(input_ids_list)
                sample_lengths[file_idx] = input_len
                if args.data_mode == "vl":
                    if image_pad_id is None:
                        image_pad_id = generator.tokenizer.convert_tokens_to_ids(
                            "<|image_pad|>"
                        )
                    pre_ids = (
                        batch_input_ids[j].tolist()
                        if isinstance(batch_input_ids[j], torch.Tensor)
                        else batch_input_ids[j]
                    )
                    loss_mask = _align_loss_mask_to_vllm(
                        input_ids_list,
                        pre_ids,
                        batch_loss_mask[j],
                        image_pad_id,
                    )[:input_len]
                else:
                    loss_mask = batch_loss_mask[j][:input_len]

                result_cleaned = {
                    "input_ids": torch.as_tensor(input_ids_list, dtype=torch.long),
                    "hidden_states": [h.contiguous() for h in result["hidden_states"]],
                    "loss_mask": loss_mask,
                }
                output_path = Path(args.output_dir) / f"data_{file_idx}.pt"
                future = thread_executor.submit(
                    save_sample_to_disk, result_cleaned, output_path
                )
                futures.append(future)
                file_idx += 1

        log.info("Waiting for remaining file saves to complete...")
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Saving files"
        ):
            future.result()

    samples_saved = file_idx - start_file_idx

    sample_lengths_output_path = Path(args.output_dir) / "sample_lengths.json"
    with open(sample_lengths_output_path, "w") as f:
        json.dump(sample_lengths, f, indent=2)

    log.info(f"Saved {samples_saved} new data points to {args.output_dir}")

    save_config(args, generator, num_samples, args.output_dir)

    return samples_saved


def main():
    # Entry point for offline data generation.
    args = parse_args()

    log.section("EAGLE Offline Data Generation")
    log.config(
        {
            "Target Model": args.target_model_path,
            "Dataset": args.train_data_path,
            "Output Dir": args.output_dir,
            "Tensor Parallel": args.tensor_parallel_size,
            "Batch Size": args.batch_size,
        }
    )

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=args.token_freq_path,
        cache_dir=args.hf_cache_dir,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
    )
    num_saved = generate_and_save_hidden_states(args, dataset)

    log.section("Data generation complete!")
    log.info(f"Saved {num_saved} files to {args.output_dir}")


if __name__ == "__main__":
    main()
