#!/bin/bash
# Online Eagle3 training for Qwen3-VL-4B on the 5k LLaVA-CoT dataset.
#
# This is a small end-to-end validation run: download and materialize the
# multimodal dataset, prepare Arrow data, start vLLM, and train Eagle3.
#
# Latest server validation (2026-07-17, RTX 5090 x4, 5k VL run):
# MAX_SAMPLES=5000, EPOCHS=10, SEQ_LENGTH=7680, VLLM_MAX_MODEL_LEN=8704.
# Runtime: 31m05s (2026-07-17T01:20:16+08:00 to 01:51:21+08:00).
# Result: rc=0, hidden_states=5000/5000, vLLM POST 500=0,
# hidden-state cache failures=0, prompt token mismatches=0.
# Final epoch: val/loss_epoch=6.038,
# val/full_acc_0_epoch=0.738, val/cond_acc_0_epoch=0.738,
# val/full_acc_1_epoch=0.512, val/cond_acc_1_epoch=0.693,
# val/full_acc_2_epoch=0.361, val/cond_acc_2_epoch=0.705.
# Best loss: val/loss_epoch=5.686 at epoch 6/10.

set -euo pipefail

# Override these values through the environment when needed.
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_DIR="${DATASET_DIR:-./data/llava-cot-5k-reannotated}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_qwen3_vl_4b_llava_cot_5k_online}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-7680}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"

VLLM_GPUS="${VLLM_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
if [[ -z "${VLLM_EXTRA_ARGS+x}" ]]; then
    VLLM_EXTRA_ARGS="--enforce-eager"
fi

DATASET_JSONL="$DATASET_DIR/train.absolute_paths.jsonl"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states_online"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
VLLM_LOG_FILE="$OUTPUT_DIR/vllm_server.log"
VLLM_MAX_MODEL_LEN="$((SEQ_LENGTH + 1024))"

IFS=',' read -r -a TRAIN_GPU_ARR <<< "$TRAIN_GPUS"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARR[@]}"
read -r -a VLLM_EXTRA_ARR <<< "$VLLM_EXTRA_ARGS"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR"
VLLM_ALLOWED_LOCAL_MEDIA_PATH="$(cd "$DATASET_DIR" && pwd)"

echo "=== Step 1: Downloading dataset snapshot ==="
hf download hao05/llava-cot-5k-reannotated \
    --repo-type dataset \
    --revision "$DATASET_REVISION" \
    --local-dir "$DATASET_DIR" \
    --include "README.md" \
    --include "data/*.parquet"

echo "=== Step 2: Materializing Parquet dataset ==="
python scripts/materialize_multimodal_parquet.py \
    --dataset-dir "$DATASET_DIR" \
    --max-samples "$MAX_SAMPLES"

if [[ ! -s "$DATASET_JSONL" ]]; then
    echo "Materializer did not produce $DATASET_JSONL" >&2
    exit 1
fi

echo "=== Step 3: Preparing multimodal data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET_JSONL" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --multimodal

if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "Port $VLLM_PORT already has a healthy server." >&2
    exit 1
fi

echo "=== Step 4: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image":1}' \
    --allowed-local-media-path "$VLLM_ALLOWED_LOCAL_MEDIA_PATH" \
    "${VLLM_EXTRA_ARR[@]}" \
    >"$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
VLLM_START_TIME="$(date +%s)"
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming healthy. See $VLLM_LOG_FILE" >&2
        wait "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    if (( $(date +%s) - VLLM_START_TIME >= VLLM_STARTUP_TIMEOUT )); then
        echo "Timed out waiting for vLLM. See $VLLM_LOG_FILE" >&2
        exit 1
    fi
    sleep 2
done

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
    --on-missing generate \
    --on-generate cache \
    --run-name eagle3_qwen3_vl_4b_llava_cot_5k_online

echo "Done. Checkpoints saved to $CHECKPOINT_DIR"
