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
#   MAX_TURNS                 Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL         Slack webhook for all notifications — logs channel (optional)
#   SLACK_WEBHOOK_URL_SUCCESS Slack webhook for successful runs only — results channel (optional)

set -euo pipefail

cd "$(dirname "$0")/.."

# When running as root (containers), re-exec as a non-root user so we can
# use --dangerously-skip-permissions (which Claude blocks for root).
if [ "$(id -u)" -eq 0 ]; then
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
      "type": "context",
      "elements": [{"type": "mrkdwn", "text": "Skill: \`/autopilot\` | Script: \`scripts/autopilot.sh\`"}]
    },
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "${detail}"}
    }
  ]
}
EOJSON
)
    [ -n "${SLACK_WEBHOOK_URL:-}" ] && slack_post "$SLACK_WEBHOOK_URL" "$payload"
    [ "$status" = "SUCCESS" ] && [ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ] && slack_post "$SLACK_WEBHOOK_URL_SUCCESS" "$payload"
}

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== Autopilot: $(date -Iseconds) ==="
echo "Running as: $(whoami)"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
if [ -n "${SLACK_WEBHOOK_URL:-}" ] || [ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ]; then
    echo "Slack logs channel: $([ -n "${SLACK_WEBHOOK_URL:-}" ] && echo "enabled" || echo "disabled")"
    echo "Slack results channel: $([ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ] && echo "enabled" || echo "disabled")"
else
    echo "Slack: disabled (set SLACK_WEBHOOK_URL and/or SLACK_WEBHOOK_URL_SUCCESS)"
fi
echo "======================================="

# Trap Ctrl-C — don't post to Slack on manual abort
trap 'echo "Aborted."; exit 130' INT

PR_FILE=".claude/agent_state/last_run_prs.json"
rm -f "$PR_FILE"

echo "/autopilot" | claude --dangerously-skip-permissions \
    "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    DETAIL="Autopilot scan completed."
    if [ -f "$PR_FILE" ]; then
        SPEC_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('speculators',''))" 2>/dev/null || true)
        VLLM_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('vllm',''))" 2>/dev/null || true)
        [ -n "$SPEC_PR" ] && DETAIL="$DETAIL\n*speculators PR:* <${SPEC_PR}>"
        [ -n "$VLLM_PR" ] && DETAIL="$DETAIL\n*vLLM PR:* <${VLLM_PR}>"
    fi
    slack_notify "SUCCESS" "$DETAIL"
else
    slack_notify "FAIL" "Autopilot exited with code $EXIT_CODE. Check the logs for details."
fi
