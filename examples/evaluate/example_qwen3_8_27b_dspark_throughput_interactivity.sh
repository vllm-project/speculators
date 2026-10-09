#!/bin/bash
# Example: throughput vs. interactivity of Qwen3.8-27B alone, with its DSpark
# speculator, and with its own MTP head, on the HumanEval and math_reasoning
# subsets of RedHatAI/speculator_benchmarks.
#
# For each of the three server configurations this script launches vLLM, runs an
# InferenceX-style closed-loop sweep (N = 1, 2, 4, ... requests kept in flight)
# on each dataset, stops the server, and finally draws one chart per dataset
# with all curves. The x axis is output tokens per second per user
# (1000 / mean inter-token latency), the y axis output tokens per second for
# the GPU. The further right the speculator's curve sits at a given
# throughput, the more each user gains from it.
#
# Prerequisites:
#   pip install "guidellm>=0.8.0" pillow     (plus a vLLM that serves the model)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh
#   Environment: OUT_DIR (results directory), VLLM_PORT (default 8110; set a different
#   port and GPU to run a second copy of this script on the same host).
#
# Output (in $OUT_DIR, default ./qwen3_8_27b_dspark_<timestamp>):
#   <config>_<subset>/            raw GuideLLM JSON per point, acceptance
#                                 sidecars, bench_command.txt (provenance)
#   <config>_<subset>.csv         one row per point and repeat
#   <subset>.png                  the chart, all configurations
#   serve_<config>.log            vLLM server logs and the exact serve command
#
# Expect about 40 minutes per configuration (three configurations) with the defaults below. Raise
# REPEATS to 3 before quoting a number; the validate step reports the spread.

set -euo pipefail

# ============ Configuration ============
TARGET_MODEL="Qwen/Qwen3.8-27B"
SPECULATOR="RedHatAI/Qwen3.8-27B-speculator.dspark"
SPEC_TOKENS=7                  # DSpark draft tokens per step (per the speculator's model card)
MTP_SPEC_TOKENS=2              # MTP draft tokens per step; the head has one layer, so vLLM's
                               # default would be 1, and the vLLM recipe for this model suggests 3
DATASET="RedHatAI/speculator_benchmarks"
SUBSETS="HumanEval math_reasoning"
STREAMS_SHORT="1,2,4,8,16"     # 30 s warmup + 90 s window each
STREAMS_LONG="32,64,128"       # 60 s warmup + 120 s window each
MAX_TOKENS=1024
REPEATS=1
MAX_MODEL_LEN=16384
# Pin the scheduler limits: vLLM picks them from GPU memory otherwise (max-num-seqs 256 to
# 1024, batched tokens 2048 to 16384), and the batch cap must be at least the largest N.
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=16384
# Prefix caching is off so the curves measure the drafter and nothing else: the
# repeated prompts would otherwise be prefilled from cache, and GuideLLM cannot see
# cache hits over HTTP. Serve on one GPU, the way a 27B model is deployed.
VLLM_EXTRA_ARGS=(--no-enable-prefix-caching)
VLLM_PORT="${VLLM_PORT:-8110}"
SERVER_START_TIMEOUT=1800      # seconds to wait for the server's /health before giving up
SERVER_URL="http://localhost:${VLLM_PORT}"
OUT_DIR="${OUT_DIR:-./qwen3_8_27b_dspark_$(date +%Y%m%d_%H%M%S)}"
# Uses CUDA_VISIBLE_DEVICES from the environment (set it before running).
# =======================================

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$EXAMPLE_DIR/../../scripts/evaluate" && pwd)/throughput_interactivity.py"

if ! command -v guidellm &> /dev/null; then
    echo "ERROR: guidellm not found. Install it first: pip install 'guidellm>=0.8.0' pillow"
    exit 1
fi
mkdir -p "$OUT_DIR"

VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (pid $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap cleanup EXIT

start_server() {  # <config> [extra vllm args...]
    local config=$1; shift
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "ERROR: something already answers on ${SERVER_URL}; stop it or set VLLM_PORT"
        exit 1
    fi
    local cmd=(vllm serve "$TARGET_MODEL" --port "$VLLM_PORT" --max-model-len "$MAX_MODEL_LEN"
               --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
               "${VLLM_EXTRA_ARGS[@]}" "$@")
    echo "=== Launching vLLM ($config): ${cmd[*]}"
    printf '%s\n' "${cmd[*]}" > "$OUT_DIR/serve_${config}_command.txt"
    "${cmd[@]}" > "$OUT_DIR/serve_${config}.log" 2>&1 &
    VLLM_PID=$!
    local waited=0
    until curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM exited; see $OUT_DIR/serve_${config}.log"
            exit 1
        fi
        if (( waited >= SERVER_START_TIMEOUT )); then
            echo "ERROR: vLLM not healthy after ${SERVER_START_TIMEOUT}s; see $OUT_DIR/serve_${config}.log"
            exit 1
        fi
        sleep 5; waited=$((waited + 5))
    done
    echo "vLLM server ready."
}

INCOMPLETE=()  # <config>_<subset> sweeps with failed points (collect exits non-zero)
sweep() {  # <config> <subset>
    local config=$1 subset=$2 ok=1
    local common=(--target "$SERVER_URL" --model "$TARGET_MODEL"
                  --dataset "$DATASET" --subset "$subset" --max-tokens "$MAX_TOKENS"
                  --repeats "$REPEATS" --out-dir "$OUT_DIR/${config}_${subset}"
                  --label "$config" --csv "$OUT_DIR/${config}_${subset}.csv" --keep-going)
    python "$BENCH" collect "${common[@]}" --streams "$STREAMS_SHORT" \
        --max-seconds 90 --warmup-seconds 30 || ok=0
    python "$BENCH" collect "${common[@]}" --streams "$STREAMS_LONG" \
        --max-seconds 120 --warmup-seconds 60 || ok=0
    if (( ! ok )); then
        echo "WARNING: some points of ${config}/${subset} failed; its CSV is incomplete"
        INCOMPLETE+=("${config}_${subset}")
    fi
}

# Configuration 1: the target model alone.
start_server baseline
for subset in $SUBSETS; do sweep baseline "$subset"; done
cleanup

# Configuration 2: the target model with the DSpark speculator.
start_server dspark --speculative-config \
    "{\"model\":\"${SPECULATOR}\",\"num_speculative_tokens\":${SPEC_TOKENS},\"method\":\"dspark\"}"
for subset in $SUBSETS; do sweep dspark "$subset"; done
cleanup

# Configuration 3: the target model with its own MTP head (shipped in the checkpoint).
start_server mtp --speculative-config \
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPEC_TOKENS}}"
for subset in $SUBSETS; do sweep mtp "$subset"; done
cleanup

# One chart per dataset, all configurations.
for subset in $SUBSETS; do
    python "$BENCH" plot \
        --series "$OUT_DIR/baseline_${subset}.csv:${TARGET_MODEL}, no speculator:#7a8c3f" \
        --series "$OUT_DIR/dspark_${subset}.csv:+ ${SPECULATOR} (${SPEC_TOKENS} draft tokens):#b52513" \
        --series "$OUT_DIR/mtp_${subset}.csv:+ MTP head (${MTP_SPEC_TOKENS} draft tokens):#2c6e9b" \
        --title "Output Throughput vs. Interactivity: ${subset}" --label-format "N={streams:.0f}" \
        --subtitle "${TARGET_MODEL} · ${DATASET} ${subset} · max_tokens ${MAX_TOKENS} · closed loop, N = ${STREAMS_SHORT},${STREAMS_LONG} · ${REPEATS} run(s) per point" \
        --out "$OUT_DIR/${subset}.png"
done

echo ""
echo "Done. Results in $OUT_DIR: <subset>.png, <config>_<subset>.csv, and raw runs."
if (( ${#INCOMPLETE[@]} )); then
    echo "Incomplete sweeps (some points failed, see their logs): ${INCOMPLETE[*]}"
    exit 1
fi
