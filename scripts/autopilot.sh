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
#   MAX_TURNS             Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL     Slack incoming webhook URL for notifications (optional)

set -euo pipefail

cd "$(dirname "$0")/.."

# When running as root (containers), re-exec as a non-root user so we can
# use --dangerously-skip-permissions (which Claude blocks for root).
# sudo -E preserves all env vars (HF_TOKEN, CUDA, AWS, etc.).
if [ "$(id -u)" -eq 0 ]; then
    id claude-runner &>/dev/null || useradd -M -d /root -g 0 claude-runner
    chmod -R g+rwX /root /workspace 2>/dev/null || true
    exec sudo -E -u claude-runner "$0" "$@"
fi

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

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== Autopilot: $(date -Iseconds) ==="
echo "Running as: $(whoami)"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
[ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "Slack: notifications enabled"
echo "======================================="

if echo "/autopilot" | claude --dangerously-skip-permissions \
    "${EXTRA_ARGS[@]}"; then
    slack_notify "SUCCESS" "Autopilot scan completed. Check for draft PRs on <https://github.com/vllm-project/speculators/pulls|speculators> and <https://github.com/vllm-project/vllm/pulls|vLLM>."
else
    slack_notify "FAIL" "Autopilot exited with code $?. Check the logs for details."
fi
