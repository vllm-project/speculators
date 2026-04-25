#!/bin/bash
# Online Eagle3 Training Script for Qwen3-VL-4B on hao05/llava-cot-5k-reannotated
#
# Runs the full online training pipeline:
#   1. Download the multimodal Parquet dataset snapshot from Hugging Face
#   2. Materialize local image files and an absolute-path JSONL
#   3. Prepare arrow data with multimodal preprocessing
#   4. Launch a hidden-state extraction vLLM server
#   5. Train Eagle3 with on-the-fly hidden-state generation
#
# Usage:
#   bash examples/train/eagle3_qwen3_vl_4b_llava_cot_5k_online.sh
#
# Notes:
# - `prepare_data.py` currently accepts local json/jsonl files or built-in dataset
#   aliases. This example snapshots the public HF Parquet dataset locally first.
# - The uploaded dataset stores image bytes in Parquet and preserves original
#   relative paths in `image_path`. This script materializes image files and a
#   JSONL with absolute image paths so vLLM can load images reliably during
#   online training.
# - For more data and a longer training run that can improve accuracy, replace
#   `hao05/llava-cot-5k-reannotated` with `hao05/llava-cot-48k-reannotated`
#   and adjust `MAX_SAMPLES` / `EPOCHS` as needed.
#
# ### Example E2E run for Qwen3-VL-4B on 5k samples from LLaVA-CoT ###
#
# Note: This 5k setup is primarily a pipeline sanity check. It is enough to
# verify that multimodal online training, hidden-state generation, and
# checkpointing all work end-to-end, but it is not intended to represent final
# model quality.
#
# Timing from an observed run on 4x NVIDIA GeForce RTX 4090 24GB GPUs
# (vLLM on GPUs 0,1 and training on GPUs 2,3):
# Data preprocessing: 8 seconds
# vLLM server startup: 54 seconds
# Training (5 epochs): 1337 seconds (22 mins 17 secs)
# Total (prepare_data start to checkpoint save): 1427 seconds (23 mins 47 secs)
#
# Final validation metrics from that run:
# val/loss_epoch: 8.4479
# val/full_acc_0_epoch: 58.55%
# val/full_acc_1_epoch: 32.46%
# val/full_acc_2_epoch: 18.22%

set -euo pipefail

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_REPO="${DATASET_REPO:-hao05/llava-cot-5k-reannotated}"
DATASET_DIR="${DATASET_DIR:-./data/llava-cot-5k-reannotated}"
DATASET_JSONL="$DATASET_DIR/train.absolute_paths.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-./output_qwen3_vl_4b_llava_cot_online}"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states_online"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5120}"
VLLM_TP="${VLLM_TP:-2}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# GPU assignments
VLLM_GPUS="${VLLM_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
# =======================================

# Optional mirror for environments without direct access to huggingface.co
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR"
read -r -a VLLM_EXTRA_ARR <<< "$VLLM_EXTRA_ARGS"

echo "=== Step 1: Downloading dataset snapshot ==="
hf download "$DATASET_REPO" \
    --repo-type dataset \
    --local-dir "$DATASET_DIR" \
    --include "README.md" \
    --include "data/*.parquet"

echo "=== Step 2: Materializing Parquet dataset to absolute-path JSONL ==="
python - "$DATASET_DIR" "$MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

from datasets import Image, load_dataset

dataset_dir = Path(sys.argv[1]).resolve()
max_samples_arg = sys.argv[2]
max_samples = None
if max_samples_arg and max_samples_arg.lower() not in {"0", "all", "none"}:
    max_samples = int(max_samples_arg)
dst = dataset_dir / "train.absolute_paths.jsonl"
parquet_files = sorted((dataset_dir / "data").glob("train-*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No Parquet shards found under {dataset_dir / 'data'}")


def absolutize_image_ref(image_ref: object) -> object:
    if not isinstance(image_ref, str):
        return image_ref
    if image_ref.startswith(("http://", "https://", "/")):
        return image_ref
    return str((dataset_dir / image_ref).resolve())


def safe_relative_path(image_path: object, row_idx: int) -> Path:
    path_text = str(image_path) if isinstance(image_path, str) else f"images/{row_idx:08d}.jpg"
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        path = Path("images") / path.name
    return path


def materialize_image(sample: dict, row_idx: int) -> str:
    image = sample.get("image")
    image_path = sample.get("image_path")
    image_bytes = None
    if isinstance(image, dict):
        image_path = image_path or image.get("path")
        image_bytes = image.get("bytes")
    rel_path = safe_relative_path(image_path, row_idx)
    image_file = dataset_dir / rel_path
    if image_bytes is not None:
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.write_bytes(image_bytes)
    elif not image_file.exists():
        raise FileNotFoundError(f"Missing image bytes and file for row {row_idx}: {image_path}")
    return str(image_file.resolve())


ds = load_dataset(
    "parquet",
    data_files={"train": [str(path) for path in parquet_files]},
    split="train",
).cast_column("image", Image(decode=False))

count = 0
with dst.open("w", encoding="utf-8") as fout:
    for row_idx, sample in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break

        sample["image"] = materialize_image(sample, row_idx)
        sample.pop("image_path", None)

        for turn in sample.get("conversations", []):
            content = turn.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"image", "image_url"}:
                    if "image" in item:
                        item["image"] = absolutize_image_ref(item["image"])
                    elif isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url")
                        item["image_url"]["url"] = absolutize_image_ref(url)

        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        count += 1

print(f"Wrote {count} rows to {dst}")
PY

echo "=== Step 3: Preparing multimodal data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET_JSONL" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --multimodal

echo "=== Step 4: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image":1}' \
    "${VLLM_EXTRA_ARR[@]}" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 5: Online training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --num-layers 1 \
    --ttt-steps 3 \
    --ttt-step-loss-decay 1.0 \
    --on-missing generate \
    --on-generate cache \
    --run-name eagle3_qwen3_vl_4b_llava_cot_5k_online

echo "Done. Checkpoints saved to $CHECKPOINT_DIR"
