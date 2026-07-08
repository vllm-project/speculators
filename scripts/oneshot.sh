#!/bin/bash
# One-shot paper-to-implementation pipeline.
# Reads a paper, implements the method, runs smoke training, opens a draft PR.
#
# Usage:
#   ./scripts/oneshot.sh <paper_url>
#   ./scripts/oneshot.sh https://arxiv.org/abs/2407.11542
#
# Options (via env vars):
#   MAX_TURNS=50          Max agentic turns (default: 50)
#   MAX_BUDGET=10.00      Max API cost in USD (default: 10.00)
#   SLACK_WEBHOOK_URL     Slack incoming webhook URL for notifications (optional)

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

ALLOWED_TOOLS="WebSearch,WebFetch,Read,Edit,Write"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(git *),Bash(make *),Bash(find *),Bash(ls *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(mkdir *),Bash(mv *),Bash(cat *),Bash(head *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(tail *),Bash(grep *),Bash(wc *),Bash(curl *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(gh *),Bash(nvidia-smi*),Bash(python *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(/workspace/speculators/.venv/bin/python *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(torchrun *),Bash(CUDA_VISIBLE_DEVICES=* *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(ruff *),Bash(pytest *),Bash(uv *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(chmod *),Bash(rm *),Bash(cp *),Bash(diff *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(sort *),Bash(touch *),Bash(test *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,Bash(cd *),Bash(echo *),Bash(pip *)"
ALLOWED_TOOLS="$ALLOWED_TOOLS,TaskCreate,TaskUpdate,TaskList,TaskGet"

slack_notify() {
    local status="$1" detail="$2"
    if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then return; fi
    local emoji="white_check_mark"
    [ "$status" = "FAIL" ] && emoji="x"
    local payload
    payload=$(cat <<EOJSON
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": ":${emoji}: Speculators One-Shot: ${status}", "emoji": true}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Paper:*\n<${PAPER_URL}>"},
        {"type": "mrkdwn", "text": "*Status:*\n${status}"}
      ]
    },
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "${detail}"}
    }
  ]
}
EOJSON
)
    curl -sf -X POST -H 'Content-type: application/json' -d "$payload" "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
}

echo "=== One-Shot Paper-to-Implementation ==="
echo "Paper: $PAPER_URL"
echo "Max turns: $MAX_TURNS"
echo "Max budget: \$$MAX_BUDGET"
[ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "Slack: notifications enabled"
echo "========================================="

# Run as interactive session with prompt pre-filled.
# This gives full TUI output (progress, tool calls, streaming text)
# instead of buffered -p mode.
if echo "/oneshot-paper $PAPER_URL" | claude \
    --allowedTools "$ALLOWED_TOOLS" \
    --max-turns "$MAX_TURNS" \
    --max-budget-usd "$MAX_BUDGET"; then
    slack_notify "SUCCESS" "Implementation pipeline completed. Check for draft PRs on <https://github.com/vllm-project/speculators/pulls|speculators> and <https://github.com/vllm-project/vllm/pulls|vLLM>."
else
    slack_notify "FAIL" "Pipeline exited with code $?. Check the logs for details."
fi
