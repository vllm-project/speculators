#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash via Docker.
#
# Required env vars:
#   GPU_IDS   — comma-separated GPU IDs
#   HARDWARE  — h100 | b200 | h200
#
# Optional env vars:
#   PORT        — host port to bind (default: 8000)
#   HF_HUB_CACHE — HuggingFace model cache dir (default: from environment)
#   SERVER_LOG  — log file path (default: server.log)

set -euo pipefail

: "${GPU_IDS:?GPU_IDS must be set}"
: "${HARDWARE:?HARDWARE must be set (h100 or b200)}"
: "${HF_HUB_CACHE:?HF_HUB_CACHE must be set}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-server.log}"

CONTAINER_NAME="vllm-deepseek-${PORT}"

case "${HARDWARE}" in
    h100)
        IMAGE="vllm/vllm-openai:deepseekv4-cu129"
        GPUS_FLAG="all"
        GPU_ENV="-e CUDA_VISIBLE_DEVICES=${GPU_IDS}"
        CONTAINER_CMD="docker"
        PODMAN_EXTRA=""
        CACHE_ENV=""
        VOLUME_SUFFIX=""
        SPEC_TOKENS=2
        HW_ARGS=(
            --data-parallel-size 4
        )
        ;;
    b200)
        IMAGE="vllm/vllm-openai:deepseekv4-cu130"
        GPUS_FLAG="\"device=${GPU_IDS}\""
        GPU_ENV="-e CUDA_VISIBLE_DEVICES=${GPU_IDS}"
        CONTAINER_CMD="docker"
        PODMAN_EXTRA=""
        CACHE_ENV=""
        VOLUME_SUFFIX=""
        SPEC_TOKENS=2
        HW_ARGS=(
            --data-parallel-size 4
            --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
            --attention_config.use_fp4_indexer_cache=True
        )
        ;;
    h200)
        IMAGE="vllm/vllm-openai:latest"
        GPUS_FLAG="all"
        GPU_ENV="-e CUDA_VISIBLE_DEVICES=${GPU_IDS}"
        CONTAINER_CMD="docker"
        PODMAN_EXTRA=""
        CACHE_ENV=""
        VOLUME_SUFFIX=""
        SPEC_TOKENS=1
        HW_ARGS=()
        ;;
    *)
        echo "[ERROR] Unknown hardware: ${HARDWARE}. Valid: h100, b200, h200" >&2
        exit 1
        ;;
esac

echo "[INFO] Starting DeepSeek-V4-Flash (Docker) on ${HARDWARE} GPUs: ${GPU_IDS} (port ${PORT}, container: ${CONTAINER_NAME})"

# Write the container name so stop_server can find it after a crash
echo "${CONTAINER_NAME}" > "/tmp/vllm_container_${PORT}.lock"

# shellcheck disable=SC2086  # GPU_ENV, CACHE_ENV, PODMAN_EXTRA are intentionally unquoted (separate args)
${CONTAINER_CMD} run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPUS_FLAG}" \
    --privileged --ipc=host \
    ${PODMAN_EXTRA} \
    -p "${PORT}:8000" \
    -v "${HF_HUB_CACHE}:/root/.cache/huggingface${VOLUME_SUFFIX}" \
    ${CACHE_ENV} \
    ${GPU_ENV} \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    "${IMAGE}" \
    deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    "${HW_ARGS[@]}" \
    --speculative_config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}" \
    2>&1 | tee "${SERVER_LOG}"

rm -f "/tmp/vllm_container_${PORT}.lock"
