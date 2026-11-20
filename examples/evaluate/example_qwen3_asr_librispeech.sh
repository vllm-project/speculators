#!/bin/bash
# Example: evaluate the published Qwen3-ASR-1.7B DFlash head.
#
# Prerequisites: install current vLLM from source and the evaluation dependencies.
#   pip install -r scripts/evaluate/requirements.txt
#
# Usage (one GPU, HTTP concurrency 8 by default):
#   CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_asr_librispeech.sh
#
# AIPerf loads LibriSpeech test-clean audio directly from Hugging Face.
# Results: evaluation/acceptance.csv, native AIPerf reports,
# server.log, and launch/evaluation provenance in OUT_DIR. This is not a WER test.

set -euo pipefail

MODEL="Qwen/Qwen3-ASR-1.7B-hf"
SPEC_MODEL="mattymchen/Qwen3-ASR-1.7B-speculator.dflash"
SPEC_TOKENS=7
MAX_REQUESTS="${MAX_REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_PORT="${VLLM_PORT:-8112}"
SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
OUT_DIR="${OUT_DIR:-./output/qwen3_asr_dflash_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$EXAMPLE_DIR/../../scripts" && pwd)"

if ! command -v aiperf &> /dev/null; then
    echo "ERROR: aiperf not found. Install dependencies first:" >&2
    echo "  pip install -r $SCRIPT_DIR/evaluate/requirements.txt" >&2
    exit 1
fi

if curl -sf --max-time 2 "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: a server is already running at ${SERVER_URL}; set VLLM_PORT." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

python "$SCRIPT_DIR/launch_vllm.py" eval "$MODEL" \
    --spec-model "$SPEC_MODEL" --spec-method dflash --spec-tokens "$SPEC_TOKENS" \
    --provenance-dir "$OUT_DIR" \
    -- \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-model-len 4096 --max-num-seqs "$CONCURRENCY" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt '{"audio":1}' \
    --no-enable-prefix-caching --mm-processor-cache-gb 0 \
    --per-request-spec-decode-metrics summary \
    --generation-config vllm > "$OUT_DIR/server.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM; see $OUT_DIR/server.log"
DEADLINE=$((SECONDS + 1800))
until curl -sf --max-time 2 "${SERVER_URL}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || (( SECONDS >= DEADLINE )); then
        echo "ERROR: vLLM failed to start; see $OUT_DIR/server.log" >&2
        exit 1
    fi
    sleep 2
done

python "$SCRIPT_DIR/evaluate/evaluate.py" acceptance \
    --url "$SERVER_URL" \
    --model "$MODEL" \
    --dataset librispeech \
    --request-count "$MAX_REQUESTS" --concurrency "$CONCURRENCY" \
    --extra-inputs '{"temperature":0,"max_completion_tokens":1024}' \
    --output-artifact-dir "$OUT_DIR/evaluation"

echo "Done. Results in $OUT_DIR"
