#!/usr/bin/env bash
# Start the DeepSeek-V4-Flash vLLM server with speculative decoding (MTP).
# Usage: bash start_server.sh

set -euo pipefail

chg run --gpus 8 -- docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v /raid/engine/hub:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --speculative_config '{"method":"mtp","num_speculative_tokens":10}' 2>&1 | tee server-logs.log
