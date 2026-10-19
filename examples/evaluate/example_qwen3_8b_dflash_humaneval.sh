#!/bin/bash
# Run a short acceptance evaluation on HumanEval.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SPEC_MODEL="${SPEC_MODEL:-RedHatAI/Qwen3-8B-speculator.dflash}"
VLLM_PORT="${VLLM_PORT:-8108}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dflash-$(date +%Y%m%d_%H%M%S)}"
PROVENANCE_DIR="${OUTPUT_DIR}-server"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

"$PYTHON" scripts/launch_vllm.py eval "$MODEL" \
    --spec-model "$SPEC_MODEL" --provenance-dir "$PROVENANCE_DIR" \
    -- --port "$VLLM_PORT" --per-request-spec-decode-metrics summary &
VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for ((attempt=0; attempt<300; attempt++)); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready" >&2
        exit 1
    fi
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    echo "Timed out waiting for vLLM" >&2
    exit 1
fi

"$PYTHON" scripts/evaluate/evaluate.py acceptance \
    --dataset openai/openai_humaneval --split test --prompt-column prompt \
    --max-samples 80 --max-tokens 1024 --target "http://localhost:${VLLM_PORT}" \
    --model "$MODEL" --max-requests 80 --output-dir "$OUTPUT_DIR"
cp "$PROVENANCE_DIR"/* "$OUTPUT_DIR/"
