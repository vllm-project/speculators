#!/usr/bin/env bash
# Emergency cleanup: stop any leftover vllm-deepseek-* containers and remove lockfiles.
# Run this from any shell if run.sh was killed and left a container holding GPUs.
#
# Usage: bash kill_server.sh

set -euo pipefail

containers=$(docker ps --filter "name=vllm-deepseek-" --format "{{.Names}}" 2>/dev/null || true)

if [[ -z "${containers}" ]]; then
    echo "No vllm-deepseek containers running."
else
    for c in ${containers}; do
        echo "Stopping ${c}..."
        docker stop "${c}" > /dev/null && echo "  stopped."
    done
fi

locks=(/tmp/vllm_container_*.lock)
if [[ -e "${locks[0]}" ]]; then
    rm -f "${locks[@]}"
    echo "Removed lockfiles: ${locks[*]}"
fi

echo "Done."
