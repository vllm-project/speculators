#!/usr/bin/env bash
set -euo pipefail

# — overrideable env vars with defaults —
: "${FIXED_ACCEPTANCE_RATE:=-1}"
: "${CUDA_VISIBLE_DEVICES:?Please set CUDA_VISIBLE_DEVICES}"
: "${NUM_SPEC_TOKENS:=3}"

# — parse GPUs & compute TP size + port —
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
TP_SIZE=${#GPUS[@]}
FIRST_GPU=${GPUS[0]}
PORT=$((8000 + FIRST_GPU))

# — model & spec settings —
MODEL_PATH="/home/linghao/models/Qwen3-0.6B/"
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=$(basename "${MODEL_PATH%/}")
# SPEC_MODEL_PATH="/nm/drive0/linghao/${MODEL_NAME}-speculator"
# SPEC_MODEL_PATH="/nm/drive0/linghao/Qwen3-8B-4L-speculator-ckpt/state_-1"
SPEC_MODEL_PATH="/nm/drive0/linghao/Qwen3-0.6B-speculator-ckpt/state_-1"
# SPEC_MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
if [ "$FIXED_ACCEPTANCE_RATE" = "-1" ]; then
  SPEC_SETTING="RS"
else
  SPEC_SETTING="FS_alpha-${FIXED_ACCEPTANCE_RATE}"
fi

PROMPT_TOKENS=512
OUTPUT_TOKENS=128
MAX_CONCURRENCY=128
# sum the prompt and output tokens and ad an extra 128
MAX_TOKENS=$((PROMPT_TOKENS + OUTPUT_TOKENS + 128))

# — output path for benchmark —
# OUTPUT_PATH=~/speculators/throughput/output_"${MODEL_NAME}"_"${SPEC_SETTING}"_draft-"${NUM_SPEC_TOKENS}".json
OUTPUT_PATH=~/speculators/throughput/output_"${MODEL_NAME}"_"${SPEC_SETTING}"_draft-"${NUM_SPEC_TOKENS}"_prompt-"${PROMPT_TOKENS}"_output-"${OUTPUT_TOKENS}"_concurrency-"${MAX_CONCURRENCY}".json
# — activate conda env —
CONDA_ENV="speculators_v3"
source /home/linghao/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# — launch server into a temp log, capture PID, rename log —
TMP_LOG="server_${PORT}.log"
: > "$TMP_LOG"
FIXED_ACCEPTANCE_RATE="$FIXED_ACCEPTANCE_RATE" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VLLM_USE_V1=1 \
vllm serve "$MODEL_PATH" \
  --seed 42 \
  -tp "$TP_SIZE" \
  --max_model_len "$MAX_TOKENS" \
  --gpu_memory_utilization 0.9 \
  --enable_chunked_prefill \
  --speculative-config "{\"model\":\"${SPEC_MODEL_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"method\":\"eagle\",\"draft_tensor_parallel_size\":1}" \
  --port "$PORT" \
  > "$TMP_LOG" 2>&1 &
SERVER_PID=$!
LOG_FILE="server_${SERVER_PID}.log"
mv "$TMP_LOG" "$LOG_FILE"

# Trap to ensure server is killed and log cleaned up on Ctrl+C or termination
cleanup() {
  echo "Caught SIGINT or SIGTERM, killing server (PID=${SERVER_PID})"
  kill -2 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$LOG_FILE"
  exit 1
}
trap cleanup SIGINT SIGTERM

echo "[$(date +"%T")] vLLM server (PID=${SERVER_PID}) starting on port ${PORT}, log = ${LOG_FILE}"

# — wait for readiness —
echo -n "Waiting for startup"
while ! grep -q "INFO:     Application startup complete\." "$LOG_FILE"; do
  sleep 1
  echo -n "."
done
echo " ready!"

# — run benchmark —
echo "[$(date +"%T")] Running benchmark → ${OUTPUT_PATH}"
GUIDELLM__MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
GUIDELLM__PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:${PORT}/v1" \
  --model "$MODEL_PATH" \
  --output-path "$OUTPUT_PATH" \
  --data "prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  --rate-type sweep \
  --rate 10 \
  --max-seconds 90 \
  --backend-args "{
    \"extra_body\": {
      \"chat_completions\": {
        \"temperature\": 0.6
      },
      \"max_output_tokens\":${MAX_TOKENS}
    }
  }"

# — shut down server —
echo "[$(date +"%T")] Stopping server (PID=${SERVER_PID})"
kill -2 "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true

# — cleanup log —
rm -f "$LOG_FILE"

echo "[$(date +"%T")] All done."
