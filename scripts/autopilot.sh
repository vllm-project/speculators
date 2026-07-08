#!/bin/bash
# Autopilot: discover new speculative decoding papers and implement each one.
# Scans arxiv, HuggingFace, GitHub, then runs oneshot on each promising find.
#
# Usage:
#   ./scripts/autopilot.sh
#
# Run on a schedule (system cron, not Claude cron):
#   0 */8 * * * /workspace/speculators/scripts/autopilot.sh >> /workspace/speculators/autopilot.log 2>&1
#
# Options (via env vars):
#   MAX_TURNS=100     Max agentic turns (default: 100)
#   MAX_BUDGET=20.00  Max API cost in USD (default: 20.00)

set -euo pipefail

MAX_TURNS="${MAX_TURNS:-100}"
MAX_BUDGET="${MAX_BUDGET:-20.00}"

cd "$(dirname "$0")/.."

echo "=== Autopilot: $(date -Iseconds) ==="
echo "Max turns: $MAX_TURNS"
echo "Max budget: \$$MAX_BUDGET"
echo "======================================="

claude --dangerously-skip-permissions \
    -p "/autopilot" \
    --max-turns "$MAX_TURNS" \
    --max-budget-usd "$MAX_BUDGET"
