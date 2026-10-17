#!/bin/bash
# Example: evaluate the published Qwen3-ASR-1.7B DFlash head.
#
# Prerequisites: install Speculators, current vLLM from source, and GuideLLM audio.
#   pip install "guidellm[audio]>=0.8.1"
#
# Usage (one GPU, HTTP concurrency 8 by default):
#   CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_asr_librispeech.sh
#
# GuideLLM loads the selected split's first MAX_REQUESTS clips from Hugging Face.
# Results: acceptance.csv, artifacts/run_*.json,
# server.log, and launch/evaluation provenance in OUT_DIR. This is not a WER test.

set -euo pipefail

MODEL="Qwen/Qwen3-ASR-1.7B-hf"
SPEC_MODEL="mattymchen/Qwen3-ASR-1.7B-speculator.dflash"
SPEC_TOKENS=7
MAX_REQUESTS="${MAX_REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DATA_CONFIG="${DATA_CONFIG:-clean}"      # clean or other
DATA_SPLIT="${DATA_SPLIT:-test}"         # test or validation
VLLM_PORT="${VLLM_PORT:-8112}"
SERVER_URL="http://127.0.0.1:${VLLM_PORT}"
OUT_DIR="${OUT_DIR:-./output/qwen3_asr_dflash_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$EXAMPLE_DIR/../../scripts" && pwd)"

python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

if Version(version("guidellm")) < Version("0.8.1"):
    raise SystemExit("Install guidellm[audio]>=0.8.1 for audio evaluation.")
PY

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

# OpenAI input_audio needs WAV/MP3; map only audio, excluding reference transcripts.
python "$SCRIPT_DIR/evaluate/evaluate.py" throughput \
    --target "${SERVER_URL}/v1" \
    --dataset openslr/librispeech_asr \
    --dataset-load-kwargs "{\"name\":\"${DATA_CONFIG}\",\"data_files\":{\"${DATA_SPLIT}\":\"${DATA_CONFIG}/${DATA_SPLIT}/*.parquet\"},\"split\":\"${DATA_SPLIT}[:${MAX_REQUESTS}]\"}" \
    --subsets "librispeech_${DATA_CONFIG}_${DATA_SPLIT}" \
    --data-column-mapper "kind=generative_column_mapper,column_mappings.audio_column=audio" \
    --data-preprocessor "kind=encode_media,audio_kwargs.audio_format=wav" \
    --max-concurrency "$CONCURRENCY" --max-requests "$MAX_REQUESTS" \
    --gen-kwargs '{"temperature":0,"max_completion_tokens":1024}' \
    --output-dir "$OUT_DIR"

echo "Done. Results in $OUT_DIR"
