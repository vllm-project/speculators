#!/bin/bash
#
# Run the complete vLLM pipeline: start servers, process dataset, stop servers
#
# Usage examples:
#   ./run_all.sh --dataset magpie --limit 100
#   ./run_all.sh --ports "8000,8001" --gpus "0,1:2,3" --dataset ultrachat
#   ./run_all.sh --ports "8000,8001,8002" --gpus "0,1:2,3:4,5" --dataset magpie
#   ./run_all.sh --dataset magpie --keep-servers  # Don't stop servers after
#

set -e  # Exit on error

# Default configuration
PORTS="8000"
GPUS=""
STOP_SERVERS_AFTER=true

# Parse arguments
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --ports)
            PORTS="$2"
            PYTHON_ARGS+=("--ports" "$2")
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --keep-servers)
            STOP_SERVERS_AFTER=false
            shift
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================="
echo "Starting vLLM Pipeline"
echo "========================================="
echo ""

# Start vLLM servers
echo "Step 1: Starting vLLM servers on ports: $PORTS"
if [ -n "$GPUS" ]; then
    echo "  GPU assignment: $GPUS"
    ./start_vllm_servers.sh --ports "$PORTS" --gpus "$GPUS"
else
    ./start_vllm_servers.sh --ports "$PORTS"
fi
echo ""

# Wait for servers to be ready
echo "Step 2: Waiting for servers to be ready..."
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
for PORT in "${PORT_ARRAY[@]}"; do
    PORT=$(echo "$PORT" | xargs)  # Trim whitespace
    ENDPOINT="http://127.0.0.1:$PORT/v1/models"
    echo "  Checking $ENDPOINT..."

    MAX_RETRIES=30
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s "$ENDPOINT" > /dev/null 2>&1; then
            echo "    ✓ Server on port $PORT is ready"
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo "    ✗ Server on port $PORT failed to start"
            ./stop_vllm_servers.sh
            exit 1
        fi
        sleep 2
    done
done
echo ""

# Run Python script
echo "Step 3: Running Python processing script..."
echo "Arguments: ${PYTHON_ARGS[*]}"
echo ""
python script.py "${PYTHON_ARGS[@]}"
PYTHON_EXIT_CODE=$?
echo ""

# Stop servers if requested
if [ "$STOP_SERVERS_AFTER" = true ]; then
    echo "Step 4: Stopping vLLM servers..."
    ./stop_vllm_servers.sh
else
    echo "Step 4: Keeping vLLM servers running (use ./stop_vllm_servers.sh to stop)"
fi

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="

exit $PYTHON_EXIT_CODE
