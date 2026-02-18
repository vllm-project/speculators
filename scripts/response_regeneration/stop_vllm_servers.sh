#!/bin/bash

# Stop vLLM servers using saved PIDs

# Get script directory for PID file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/vllm_pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file not found at $PID_FILE"
    echo ""
    echo "Servers may not be running or were not started with start_vllm_servers.sh"
    echo ""
    echo "To manually find and kill vLLM processes:"
    echo "  ps aux | grep 'vllm serve'"
    echo "  kill <PID>"
    exit 1
fi

PIDS=$(cat "$PID_FILE")

if [ -z "$PIDS" ]; then
    echo "Error: No PIDs found in $PID_FILE"
    rm "$PID_FILE"
    exit 1
fi

echo "Stopping vLLM servers..."
echo "PIDs: $PIDS"
echo ""

STOPPED_COUNT=0
NOT_RUNNING_COUNT=0

# First, try graceful shutdown with SIGTERM
for PID in $PIDS; do
    if kill -0 "$PID" 2>/dev/null; then
        echo "  Sending SIGTERM to PID $PID..."
        kill "$PID"
        STOPPED_COUNT=$((STOPPED_COUNT + 1))
    else
        echo "  PID $PID not running (already stopped)"
        NOT_RUNNING_COUNT=$((NOT_RUNNING_COUNT + 1))
    fi
done

# Wait a bit for graceful shutdown
if [ $STOPPED_COUNT -gt 0 ]; then
    echo ""
    echo "Waiting for processes to terminate..."
    sleep 3

    # Check if any processes are still running and force kill if needed
    FORCE_KILL_COUNT=0
    for PID in $PIDS; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Force killing PID $PID (SIGKILL)..."
            kill -9 "$PID"
            FORCE_KILL_COUNT=$((FORCE_KILL_COUNT + 1))
        fi
    done

    if [ $FORCE_KILL_COUNT -gt 0 ]; then
        echo "  Force killed $FORCE_KILL_COUNT process(es)"
        sleep 1
    fi
fi

echo ""
echo "Stopped $STOPPED_COUNT server(s)"
if [ $NOT_RUNNING_COUNT -gt 0 ]; then
    echo "$NOT_RUNNING_COUNT server(s) were already stopped"
fi

rm "$PID_FILE"
echo "Removed PID file: $PID_FILE"
echo "Done!"
