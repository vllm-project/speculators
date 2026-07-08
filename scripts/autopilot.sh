#!/bin/bash
# Autopilot: discover new speculative decoding papers and implement each one.
# Scans arxiv, HuggingFace, GitHub, then runs oneshot on each promising find.
#
# Usage:
#   ./scripts/autopilot.sh
#
# Run on a schedule (system cron):
#   0 */8 * * * /workspace/speculators/scripts/autopilot.sh >> /workspace/speculators/autopilot.log 2>&1
#
# Options (via env vars):
#   MAX_TURNS=100         Max agentic turns (default: 100)
#   MAX_BUDGET=20.00      Max API cost in USD (default: 20.00)
#   SLACK_WEBHOOK_URL     Slack incoming webhook URL for notifications (optional)

set -euo pipefail

MAX_TURNS="${MAX_TURNS:-100}"
MAX_BUDGET="${MAX_BUDGET:-20.00}"

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
    [ "$status" = "SCAN" ] && emoji="mag"
    local payload
    payload=$(cat <<EOJSON
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": ":${emoji}: Speculators Autopilot: ${status}", "emoji": true}
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

echo "=== Autopilot: $(date -Iseconds) ==="
echo "Max turns: $MAX_TURNS"
echo "Max budget: \$$MAX_BUDGET"
[ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "Slack: notifications enabled"
echo "======================================="

if echo "/autopilot" | claude \
    --allowedTools "$ALLOWED_TOOLS" \
    --max-turns "$MAX_TURNS" \
    --max-budget-usd "$MAX_BUDGET"; then
    slack_notify "SUCCESS" "Autopilot scan completed. Check for draft PRs on <https://github.com/vllm-project/speculators/pulls|speculators> and <https://github.com/vllm-project/vllm/pulls|vLLM>."
else
    slack_notify "FAIL" "Autopilot exited with code $?. Check the logs for details."
fi
