#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash vLLM server on 8x GPUs (TP=8).
#
# Required env vars:
#   GPU_IDS   — comma-separated GPU IDs (e.g. "0,1,2,3,4,5,6,7")
#
# Optional env vars:
#   PORT        — host port to bind (default: 8000)
#   SERVER_LOG  — log file path (default: server.log in current dir)

set -euo pipefail

: "${GPU_IDS:?GPU_IDS must be set (e.g. export GPU_IDS=0,1,2,3,4,5,6,7)}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-server.log}"

echo "[INFO] Starting DeepSeek-V4-Flash on GPUs: ${GPU_IDS} (port ${PORT})"

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --attention_config.use_fp4_indexer_cache=True \
    --moe-backend deep_gemm_mega_moe \
    --speculative_config '{"method":"mtp","num_speculative_tokens":2}' \
    2>&1 | tee "${SERVER_LOG}"
