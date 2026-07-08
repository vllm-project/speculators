#!/bin/bash
# One-shot paper-to-implementation pipeline.
# Reads a paper, implements the method, runs smoke training, opens a draft PR.
#
# Usage:
#   ./scripts/oneshot.sh <paper_url>
#   ./scripts/oneshot.sh https://arxiv.org/abs/2407.11542
#
# Options (via env vars):
#   MAX_TURNS=50      Max agentic turns (default: 50)
#   MAX_BUDGET=10.00  Max API cost in USD (default: 10.00)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <paper_url>"
    echo "Example: $0 https://arxiv.org/abs/2407.11542"
    exit 1
fi

PAPER_URL="$1"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_BUDGET="${MAX_BUDGET:-10.00}"

cd "$(dirname "$0")/.."

echo "=== One-Shot Paper-to-Implementation ==="
echo "Paper: $PAPER_URL"
echo "Max turns: $MAX_TURNS"
echo "Max budget: \$$MAX_BUDGET"
echo "========================================="

claude --dangerously-skip-permissions \
    -p "/oneshot-paper $PAPER_URL" \
    --max-turns "$MAX_TURNS" \
    --max-budget-usd "$MAX_BUDGET"
