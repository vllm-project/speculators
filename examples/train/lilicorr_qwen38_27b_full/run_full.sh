#!/bin/bash
# Full run: Qwen3.8-27B LiLiCorr drafter on the complete regenerated
# perfectblend dataset (1.88M records -> ~1.75M rows after 8192 truncation
# and zero-supervision drops, ~4.4B tokens, 2 epochs). lr 1.0e-3 per user.
# 2 extraction GPUs (0,1) + 6 train GPUs (2-7). Config: full.yaml.
# Server flags identical to the validated smoke + 5k runs.
set -euo pipefail

export PYTHONPATH=/workspace/speculators/.claude/worktrees/lilicorr-minimal/src
export PATH=/workspace/speculators/.venv/bin:$PATH
cd /workspace/speculators/.claude/worktrees/lilicorr-minimal

MODEL="/tmp/models/Qwen3.8-27B"
CONFIG="/tmp/lilicorr_check/qwen38-perfectblend/full.yaml"
RUN_DIR="/tmp/lilicorr_check/qwen38-perfectblend/runfull"
VLLM_PORT=8000
TARGET_LAYER_IDS="4 12 20 28 36 44 52 60"

VLLM_GPUS="0,1"
TRAIN_GPUS="2,3,4,5,6,7"
NUM_TRAIN_GPUS=6

mkdir -p "$RUN_DIR"

echo "=== Step 1: Launching vLLM extraction server (GPUs $VLLM_GPUS) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" \
    --max-model-len 8448 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.95 --no-enable-prefix-caching \
    > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died during startup" >&2
        tail -40 "$RUN_DIR/vllm.log"
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 2: Training (GPUs $TRAIN_GPUS, 6 ranks, full dataset, 2 epochs) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train --config "$CONFIG" \
    > "$RUN_DIR/train.log" 2>&1

echo "Done. Checkpoints under $RUN_DIR/checkpoints/."
