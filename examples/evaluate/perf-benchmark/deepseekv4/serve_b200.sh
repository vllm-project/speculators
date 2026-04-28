#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash vLLM server on 4x B200 GPUs.
#
# Required env vars:
#   GPU_IDS   — comma-separated GPU IDs (e.g. "0,1,2,3")
#
# Optional env vars:
#   PORT        — host port to bind (default: 8000)
#   HF_HUB_CACHE — HuggingFace cache dir (default: from environment)
#   SERVER_LOG  — log file path (default: server.log in current dir)

set -euo pipefail

: "${GPU_IDS:?GPU_IDS must be set (e.g. export GPU_IDS=0,1,2,3)}"
: "${HF_HUB_CACHE:?HF_HUB_CACHE must be set}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-server.log}"

CONTAINER_NAME="vllm-deepseek-${PORT}"
echo "[INFO] Starting DeepSeek-V4-Flash on B200 GPUs: ${GPU_IDS} (port ${PORT}, container: ${CONTAINER_NAME})"

# Write the container name to a lockfile so kill_server.sh can find it after a crash
echo "${CONTAINER_NAME}" > "/tmp/vllm_container_${PORT}.lock"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "\"device=${GPU_IDS}\"" \
    --privileged --ipc=host \
    -p "${PORT}:8000" \
    -v "${HF_HUB_CACHE}:/root/.cache/huggingface" \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    vllm/vllm-openai:deepseekv4-cu130 \
    deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --data-parallel-size 4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --attention_config.use_fp4_indexer_cache=True \
    --speculative_config '{"method":"mtp","num_speculative_tokens":10}' \
    2>&1 | tee "${SERVER_LOG}"

rm -f "/tmp/vllm_container_${PORT}.lock"
