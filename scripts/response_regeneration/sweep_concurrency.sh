#!/bin/bash
#
# Sweep concurrency values to find optimal throughput for response regeneration.
#
# Starts a vLLM server once, then runs script.py at each concurrency level for
# 60 seconds, counts completed samples, and reports the best throughput.
#
# Usage:
#   ./sweep_concurrency.sh --model google/gemma-4-31B-it --tp-size 2 --dp-size 4 --dataset tulu3
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
PORT=8000
MODEL=""
DP_SIZE=""
TP_SIZE=""
MAX_MODEL_LEN=""
REASONING_PARSER=""
GPUS=""
DURATION=60
CONCURRENCIES=(64 128 256 512)

# Collect args for script.py (dataset, shuffle, etc.)
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            PYTHON_ARGS+=("--model" "$2")
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --concurrencies)
            IFS=',' read -ra CONCURRENCIES <<< "$2"
            shift 2
            ;;
        --concurrency)
            echo "Ignoring --concurrency (this script sweeps concurrency values automatically)"
            shift 2
            ;;
        --outfile)
            echo "Ignoring --outfile (this script manages its own output files)"
            shift 2
            ;;
        --resume)
            shift
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: --model is required."
    echo "Usage: $0 --model MODEL [--tp-size N] [--dp-size N] [--dataset DATASET] [--duration SECS] [--concurrencies 64,128,256,512]"
    exit 1
fi

# Build vllm serve command
VLLM_CMD=(vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key "")
[ -n "$DP_SIZE" ] && VLLM_CMD+=(--data-parallel-size "$DP_SIZE")
[ -n "$TP_SIZE" ] && VLLM_CMD+=(--tensor-parallel-size "$TP_SIZE")
[ -n "$MAX_MODEL_LEN" ] && VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$REASONING_PARSER" ] && VLLM_CMD+=(--reasoning-parser "$REASONING_PARSER")

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        sleep 3
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -9 "$VLLM_PID"
        fi
    fi
}
trap cleanup EXIT

echo "========================================="
echo "Concurrency Sweep"
echo "========================================="
echo "  Model: $MODEL"
echo "  Concurrencies: ${CONCURRENCIES[*]}"
echo "  Duration per run: ${DURATION}s"
[ -n "$DP_SIZE" ] && echo "  Data parallel size: $DP_SIZE"
[ -n "$TP_SIZE" ] && echo "  Tensor parallel size: $TP_SIZE"
echo ""

# Start server
echo "Starting vLLM server..."
if [ -n "$GPUS" ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" "${VLLM_CMD[@]}" > "$SCRIPT_DIR/vllm_sweep.log" 2>&1 &
else
    "${VLLM_CMD[@]}" > "$SCRIPT_DIR/vllm_sweep.log" 2>&1 &
fi
VLLM_PID=$!

# Wait for server
ENDPOINT="http://127.0.0.1:$PORT/v1/models"
MAX_RETRIES=300
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server died. Last 20 lines:"
        tail -20 "$SCRIPT_DIR/vllm_sweep.log"
        exit 1
    fi
    if curl -s --connect-timeout 5 --max-time 10 "$ENDPOINT" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "Server failed to start."
        tail -20 "$SCRIPT_DIR/vllm_sweep.log"
        exit 1
    fi
    [ $((RETRY % 5)) -eq 0 ] && echo "  Waiting... ($RETRY retries)"
    sleep 2
done
echo ""

# Run sweep
declare -a RESULTS_CONC
declare -a RESULTS_COUNT
declare -a RESULTS_RATE

BEST_CONC=0
BEST_RATE=0

for CONC in "${CONCURRENCIES[@]}"; do
    OUTFILE="sweep_concurrency_${CONC}.jsonl"
    rm -f "$OUTFILE"

    echo "-----------------------------------------"
    echo "Testing concurrency=$CONC for ${DURATION}s..."

    timeout "$DURATION" python "$SCRIPT_DIR/script.py" \
        "${PYTHON_ARGS[@]}" \
        --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
        --concurrency "$CONC" \
        --outfile "$OUTFILE" \
        --shuffle \
        2>&1 || true

    COUNT=0
    [ -f "$OUTFILE" ] && COUNT=$(wc -l < "$OUTFILE" | tr -d ' ')
    RATE=$(echo "scale=2; $COUNT / $DURATION" | bc)

    echo "  Completed: $COUNT samples ($RATE samples/sec)"
    echo ""

    RESULTS_CONC+=("$CONC")
    RESULTS_COUNT+=("$COUNT")
    RESULTS_RATE+=("$RATE")

    if (( $(echo "$RATE > $BEST_RATE" | bc -l) )); then
        BEST_RATE=$RATE
        BEST_CONC=$CONC
    fi
done

# Print summary
echo "========================================="
echo "Results"
echo "========================================="
printf "%-15s %-15s %-15s\n" "Concurrency" "Samples" "Samples/sec"
printf "%-15s %-15s %-15s\n" "-----------" "-------" "-----------"
for i in "${!RESULTS_CONC[@]}"; do
    MARKER=""
    [ "${RESULTS_CONC[$i]}" = "$BEST_CONC" ] && MARKER=" <-- best"
    printf "%-15s %-15s %-15s%s\n" "${RESULTS_CONC[$i]}" "${RESULTS_COUNT[$i]}" "${RESULTS_RATE[$i]}" "$MARKER"
done
echo ""
echo "Best concurrency: $BEST_CONC ($BEST_RATE samples/sec)"
echo "========================================="
