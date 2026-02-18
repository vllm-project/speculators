#!/bin/bash

# Stop vLLM servers using saved PIDs

PID_FILE="vllm_pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: $PID_FILE not found"
    echo "Servers may not be running or PIDs were not saved"
    exit 1
fi

PIDS=$(cat "$PID_FILE")

if [ -z "$PIDS" ]; then
    echo "No PIDs found in $PID_FILE"
    exit 1
fi

echo "Stopping vLLM servers..."
echo "PIDs: $PIDS"

for PID in $PIDS; do
    if kill -0 "$PID" 2>/dev/null; then
        echo "  Stopping PID $PID..."
        kill "$PID"
    else
        echo "  PID $PID not running"
    fi
done

echo "Done!"
rm "$PID_FILE"
