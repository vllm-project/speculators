#!/bin/bash
#
# Start vLLM servers with GPU assignment
#
# Usage:
#   ./start_vllm_servers.sh --model "meta-llama/Llama-3.3-70B-Instruct" --ports "8000,8001" --gpus "0,1:2,3"
#   ./start_vllm_servers.sh --ports "8000" --gpus "0,1,2,3"
#   ./start_vllm_servers.sh  # Single server on port 8000, all GPUs, default model
#

# Default configuration
MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct"
HOST="127.0.0.1"
PORTS="8000"
GPUS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ports)
            PORTS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--ports PORT_LIST] [--gpus GPU_LIST] [--model MODEL_NAME]"
            echo "Example: $0 --ports \"8000,8001\" --gpus \"0,1:2,3\""
            exit 1
            ;;
    esac
done

# Parse ports into array
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"

# Parse GPU groups into array (split by colon)
if [ -n "$GPUS" ]; then
    IFS=':' read -ra GPU_GROUPS <<< "$GPUS"
else
    GPU_GROUPS=()
fi

# Validate GPU count matches port count
if [ ${#GPU_GROUPS[@]} -gt 0 ] && [ ${#GPU_GROUPS[@]} -ne ${#PORT_ARRAY[@]} ]; then
    echo "Error: Number of GPU groups (${#GPU_GROUPS[@]}) must match number of ports (${#PORT_ARRAY[@]})"
    echo "Ports: ${PORT_ARRAY[*]}"
    echo "GPU groups: ${GPU_GROUPS[*]}"
    exit 1
fi

echo "Starting vLLM servers for model: $MODEL"
echo "Ports: ${PORT_ARRAY[*]}"
if [ ${#GPU_GROUPS[@]} -gt 0 ]; then
    echo "GPU assignments:"
    for i in "${!PORT_ARRAY[@]}"; do
        echo "  Port ${PORT_ARRAY[$i]} -> GPUs ${GPU_GROUPS[$i]}"
    done
else
    echo "GPU assignment: Using all available GPUs for single server"
fi
echo ""

# Start vLLM server for each port
PIDS=()
for i in "${!PORT_ARRAY[@]}"; do
    PORT="${PORT_ARRAY[$i]}"

    # Set CUDA_VISIBLE_DEVICES if GPU groups specified
    if [ ${#GPU_GROUPS[@]} -gt 0 ]; then
        GPU_IDS="${GPU_GROUPS[$i]}"
        echo "Starting vLLM server on port $PORT with GPUs $GPU_IDS..."
        CUDA_VISIBLE_DEVICES="$GPU_IDS" vllm serve "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --api-key="" \
            > "vllm_${PORT}.log" 2>&1 &
    else
        echo "Starting vLLM server on port $PORT..."
        vllm serve "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --api-key="" \
            > "vllm_${PORT}.log" 2>&1 &
    fi

    PID=$!
    PIDS+=($PID)
    echo "  Started with PID $PID (logs: vllm_${PORT}.log)"
done

echo ""
echo "All servers started!"
echo "PIDs: ${PIDS[*]}"
echo ""
echo "To stop all servers, run:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Or save PIDs to file and kill later:"
echo "${PIDS[*]}" > vllm_pids.txt
echo "  Saved PIDs to vllm_pids.txt"
echo "  To stop: kill \$(cat vllm_pids.txt)"
