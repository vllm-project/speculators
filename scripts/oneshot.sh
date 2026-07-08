#!/bin/bash
# One-shot paper-to-implementation pipeline.
# Reads a paper, implements the method, runs smoke training, opens a draft PR.
#
# Usage:
#   ./scripts/oneshot.sh <paper_url>
#   ./scripts/oneshot.sh https://arxiv.org/abs/2407.11542
#
# Options (via env vars):
#   MAX_TURNS                 Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL         Slack webhook for all notifications — logs channel (optional)
#   SLACK_WEBHOOK_URL_SUCCESS Slack webhook for successful runs only — results channel (optional)

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
if [ "$(id -u)" -eq 0 ]; then
    # Use the workspace's group as primary (OpenShift assigns a namespace GID),
    # and add root group (0) as supplementary for /root file access.
    WS_GID=$(stat -c '%g' /workspace/speculators)
    NEED_RECREATE=0
    if id claude-runner &>/dev/null; then
        if [ "$(id -g claude-runner)" != "$WS_GID" ]; then
            userdel claude-runner
            NEED_RECREATE=1
        fi
    else
        NEED_RECREATE=1
    fi
    if [ "$NEED_RECREATE" -eq 1 ]; then
        useradd -M -d /root -g "$WS_GID" -G 0 claude-runner
    fi
    find /root -not -path '/root/.ssh*' -exec chmod o+rX {} + 2>/dev/null || true
    exec runuser --preserve-environment -u claude-runner -- "$0" "$@"
fi

slack_post() {
    local webhook_url="$1" payload="$2"
    curl -sf -X POST -H 'Content-type: application/json' -d "$payload" "$webhook_url" >/dev/null 2>&1 || true
}

slack_notify() {
    local status="$1" detail="$2"
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
      "type": "context",
      "elements": [{"type": "mrkdwn", "text": "Skill: \`/oneshot-paper\` | Script: \`scripts/oneshot.sh\`"}]
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
    # Logs channel — all notifications
    [ -n "${SLACK_WEBHOOK_URL:-}" ] && slack_post "$SLACK_WEBHOOK_URL" "$payload"
    # Results channel — success only
    [ "$status" = "SUCCESS" ] && [ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ] && slack_post "$SLACK_WEBHOOK_URL_SUCCESS" "$payload"
}

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== One-Shot Paper-to-Implementation ==="
echo "Paper: $PAPER_URL"
echo "Running as: $(whoami)"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
if [ -n "${SLACK_WEBHOOK_URL:-}" ] || [ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ]; then
    echo "Slack logs channel: $([ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "enabled" || echo "disabled")"
    echo "Slack results channel: $([ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ] && echo "enabled" || echo "disabled")"
else
    echo "Slack: disabled (set SLACK_WEBHOOK_URL and/or SLACK_WEBHOOK_URL_SUCCESS)"
fi
echo "========================================="

# Trap Ctrl-C — don't post to Slack on manual abort
trap 'echo "Aborted."; exit 130' INT

PR_FILE=".claude/agent_state/last_run_prs.json"
rm -f "$PR_FILE"

echo "/oneshot-paper $PAPER_URL" | claude --dangerously-skip-permissions \
    "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    DETAIL="Implementation pipeline completed."
    # Read actual PR URLs written by the oneshot skill
    if [ -f "$PR_FILE" ]; then
        SPEC_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('speculators',''))" 2>/dev/null || true)
        VLLM_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('vllm',''))" 2>/dev/null || true)
        [ -n "$SPEC_PR" ] && DETAIL="$DETAIL\n*speculators PR:* <${SPEC_PR}>"
        [ -n "$VLLM_PR" ] && DETAIL="$DETAIL\n*vLLM PR:* <${VLLM_PR}>"
    fi
    slack_notify "SUCCESS" "$DETAIL"
else
    slack_notify "FAIL" "Pipeline exited with code $EXIT_CODE. Check the logs for details."
fi
