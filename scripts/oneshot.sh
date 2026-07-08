#!/bin/bash
# One-shot paper-to-implementation pipeline.
# Reads a paper, implements the method, runs smoke training, opens a draft PR.
#
# Usage:
#   ./scripts/oneshot.sh <paper_url>
#   ./scripts/oneshot.sh https://arxiv.org/abs/2407.11542
#
# Options (via env vars):
#   MAX_TURNS             Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL     Slack incoming webhook URL for notifications (optional)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <paper_url>"
    echo "Example: $0 https://arxiv.org/abs/2407.11542"
    exit 1
fi

PAPER_URL="$1"

cd "$(dirname "$0")/.."

# When running as root (containers), re-exec as a non-root user so we can
# use --dangerously-skip-permissions (which Claude blocks for root).
# sudo -E preserves all env vars (HF_TOKEN, CUDA, AWS, etc.).
if [ "$(id -u)" -eq 0 ]; then
    id claude-runner &>/dev/null || useradd -M -d /root -g 0 claude-runner
    chmod -R g+rwX /root /workspace 2>/dev/null || true
    export HOME=/root
    exec runuser -u claude-runner -- "$0" "$@"
fi

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

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== One-Shot Paper-to-Implementation ==="
echo "Paper: $PAPER_URL"
echo "Running as: $(whoami)"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
[ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "Slack: notifications enabled"
echo "========================================="

if echo "/oneshot-paper $PAPER_URL" | claude --dangerously-skip-permissions \
    "${EXTRA_ARGS[@]}"; then
    slack_notify "SUCCESS" "Implementation pipeline completed. Check for draft PRs on <https://github.com/vllm-project/speculators/pulls|speculators> and <https://github.com/vllm-project/vllm/pulls|vLLM>."
else
    slack_notify "FAIL" "Pipeline exited with code $?. Check the logs for details."
fi
