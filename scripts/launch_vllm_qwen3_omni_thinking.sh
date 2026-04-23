#!/usr/bin/env bash
# Launch vLLM for DFlash hidden-states extraction on Qwen3-Omni-Thinking.
#
# Wraps speculators' scripts/launch_vllm.py which injects the required
# --speculative_config (method=extract_hidden_states) and --kv_transfer_config
# (ExampleHiddenStatesConnector -> hs_*.safetensors). Without these, the client
# in scripts/data_generation_offline2.py hits InvalidResponseError because
# kv_transfer_params is missing from the response.
#
# Usage:
#   bash scripts/launch_vllm_qwen3_omni_thinking.sh
#   PORT=8001 TP=4 bash scripts/launch_vllm_qwen3_omni_thinking.sh
#   DRY_RUN=1 bash scripts/launch_vllm_qwen3_omni_thinking.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Thinking}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Must match DataGenArgs.layer_ids minus the trailing num_hidden_layers
# (launch_vllm.py re-appends the last layer via --include-last-layer).
# llava example uses [2, 23, 45] + 48 for Qwen3-Omni-Thinking.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 23 45}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/tmp/qwen3_omni_hidden_states}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${HIDDEN_STATES_PATH}"

LAUNCHER=(
  python "${REPO_ROOT}/scripts/launch_vllm.py"
  "${MODEL}"
  --hidden-states-path "${HIDDEN_STATES_PATH}"
  --target-layer-ids ${TARGET_LAYER_IDS}
  --include-last-layer
)

VLLM_PASSTHRU=(
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --trust-remote-code
  --dtype bfloat16
  --enforce-eager
  --limit-mm-per-prompt '{"video": 1, "image": 0, "audio": 0}'
)

if [[ "${DRY_RUN}" == "1" ]]; then
  LAUNCHER+=(--dry-run)
fi

echo "[serve] MODEL=${MODEL}"
echo "[serve] PORT=${PORT} TP=${TP} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[serve] TARGET_LAYER_IDS=[${TARGET_LAYER_IDS}] + last"
echo "[serve] HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH}"

exec "${LAUNCHER[@]}" -- "${VLLM_PASSTHRU[@]}"
