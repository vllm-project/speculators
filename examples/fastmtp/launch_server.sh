#!/usr/bin/env bash
# Launch a vLLM server configured to extract the last transformer layer's hidden states.
# For FastMTP we only need the final hidden layer (the input to the MTP head).
#
# Run this in a dedicated terminal. Keep it running while you call run_datagen.sh.
#
# Usage:
#   MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
#   CUDA_VISIBLE_DEVICES=0,4,5,6 \
#   bash examples/fastmtp/launch_server.sh
#
# Optional overrides:
#   TP_SIZE=4                    (tensor parallel size, default 4)
#   MAX_MODEL_LEN=8192           (max sequence length, default 8192)
#   HIDDEN_STATES_PATH=./hidden_states   (where vLLM writes temp files)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-$REPO_ROOT/hidden_states}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,5,6}"

# Auto-detect num_hidden_layers; vLLM uses this value as the last layer index
LAST_LAYER=$(python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('$MODEL')
cfg = getattr(cfg, 'text_config', cfg)
print(cfg.num_hidden_layers)
")

echo "=== vLLM Hidden States Server ==="
echo "  Model:              $MODEL"
echo "  Tensor parallel:    $TP_SIZE  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Extracting layer:   $LAST_LAYER  (last hidden layer for FastMTP)"
echo "  Hidden states path: $HIDDEN_STATES_PATH"
echo ""

mkdir -p "$HIDDEN_STATES_PATH"

# launch_vllm.py calls os.execvp — exec preserves the PID and SIGTERM propagates cleanly
exec python "$REPO_ROOT/scripts/launch_vllm.py" \
    "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_PATH" \
    --layers "$LAST_LAYER" \
    -- \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tokenizer-mode auto \
    --no-enable-chunked-prefill
