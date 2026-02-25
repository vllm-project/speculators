#!/bin/bash
#
# Run the complete response regeneration pipeline:
# start a vLLM server (with optional data/tensor parallelism), regenerate
# responses for the dataset, and stop the server.
#
# Usage examples:
#   ./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie --limit 100
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dp-size 4 --tp-size 2 --dataset magpie
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --gpus 0,1,2,4 --tp-size 4 --dataset magpie
#   ./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset magpie --keep-server
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
PORT=8000
MODEL=""
DP_SIZE=""
TP_SIZE=""
GPUS=""
KEEP_SERVER=false

# Parse arguments
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
        --model)
            MODEL="$2"
            PYTHON_ARGS+=("--model" "$2")
            shift 2
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --ports)
            echo "Error: $1 has been removed. Use --dp-size and --tp-size instead."
            echo "  Migration: --ports '8000,8001' becomes --dp-size 2"
            exit 1
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: --model is required."
    echo "Usage: $0 --model MODEL [--gpus GPUS] [--dp-size N] [--tp-size N] [--dataset DATASET] [...]"
    exit 1
fi

# Build vllm serve command
VLLM_CMD=(vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --api-key "")
[ -n "$DP_SIZE" ] && VLLM_CMD+=(--data-parallel-size "$DP_SIZE")
[ -n "$TP_SIZE" ] && VLLM_CMD+=(--tensor-parallel-size "$TP_SIZE")

# Cleanup function
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        sleep 3
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing vLLM server..."
            kill -9 "$VLLM_PID"
        fi
    fi
}

echo "========================================="
echo "Response Regeneration Pipeline"
echo "========================================="
echo ""

# Step 1: Start server
echo "Step 1: Starting vLLM server on port $PORT"
echo "  Model: $MODEL"
[ -n "$GPUS" ] && echo "  GPUs: $GPUS"
[ -n "$DP_SIZE" ] && echo "  Data parallel size: $DP_SIZE"
[ -n "$TP_SIZE" ] && echo "  Tensor parallel size: $TP_SIZE"
echo "  Command: ${VLLM_CMD[*]}"
echo ""

if [ -n "$GPUS" ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" "${VLLM_CMD[@]}" > "$SCRIPT_DIR/vllm_server.log" 2>&1 &
else
    "${VLLM_CMD[@]}" > "$SCRIPT_DIR/vllm_server.log" 2>&1 &
fi
VLLM_PID=$!

# Set up cleanup trap (unless --keep-server)
if [ "$KEEP_SERVER" = false ]; then
    trap cleanup EXIT
fi

# Step 2: Health check
echo "Step 2: Waiting for server to be ready..."
echo "  (Large models may take several minutes to load)"
ENDPOINT="http://127.0.0.1:$PORT/v1/models"
MAX_RETRIES=300  # Up to 12s per retry (2s sleep + 10s curl timeout); large models may need time for compilation
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  vLLM server process died. Last 20 lines of log:"
        tail -20 "$SCRIPT_DIR/vllm_server.log"
        exit 1
    fi
    if curl -s --connect-timeout 5 --max-time 10 "$ENDPOINT" > /dev/null 2>&1; then
        echo "  Server ready (after $RETRY retries)"
        break
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "  Server failed to start after $MAX_RETRIES retries"
        echo "  Last 20 lines of log:"
        tail -20 "$SCRIPT_DIR/vllm_server.log"
        exit 1
    fi
    [ $((RETRY % 5)) -eq 0 ] && echo "  Still waiting... ($RETRY retries)"
    sleep 2
done
echo ""

# Step 3: Run response regeneration
echo "Step 3: Running response regeneration..."
PYTHON_ARGS+=("--endpoint" "http://127.0.0.1:$PORT/v1/chat/completions")
echo "Arguments: ${PYTHON_ARGS[*]}"
echo ""
python "$SCRIPT_DIR/script.py" "${PYTHON_ARGS[@]}"
PYTHON_EXIT_CODE=$?
echo ""

# Step 4: Cleanup
if [ "$KEEP_SERVER" = true ]; then
    echo "Keeping vLLM server running (PID $VLLM_PID)"
    echo "Stop with: kill $VLLM_PID"
fi

echo "========================================="
echo "Pipeline complete!"
echo "========================================="

exit $PYTHON_EXIT_CODE
