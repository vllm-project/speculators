#!/bin/bash
# Autopilot: discover new speculative decoding papers and implement each one.
# Scans arxiv, HuggingFace, GitHub, then runs oneshot on each promising find.
#
# Usage:
#   ./scripts/autopilot.sh                          # headless, 60-day window
#   ./scripts/autopilot.sh --days 90                # custom window
#   ./scripts/autopilot.sh --interactive             # live TUI (debugging)
#   ./scripts/autopilot.sh --interactive --days 120  # both
#
# Run on a schedule (system cron):
#   0 */8 * * * /workspace/speculators/scripts/autopilot.sh >> /workspace/speculators/autopilot.log 2>&1
#
# Options (via env vars):
#   MAX_TURNS                 Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL         Slack webhook for all notifications — logs channel (optional)
#   SLACK_WEBHOOK_URL_SUCCESS Slack webhook for successful runs only — results channel (optional)

set -euo pipefail

export HOME=/root
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Claude authenticates via Vertex AI — cron doesn't inherit these
export CLAUDE_CODE_USE_VERTEX="${CLAUDE_CODE_USE_VERTEX:-1}"
export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-itpc-gcp-ai-eng-claude}"

cd "$(dirname "$0")/.."

# Claude blocks --dangerously-skip-permissions for root. The devenv entrypoint
# creates a claude-runner user with the right groups — just re-exec as it.
if [ "$(id -u)" -eq 0 ]; then
    WS_GID=$(stat -c '%g' /workspace)
    chmod -R g+rwX .claude 2>/dev/null || true
    chmod g+r /root/.claude/.credentials.json 2>/dev/null || true
    chown root:"$WS_GID" /root/.claude/.credentials.json 2>/dev/null || true
    # GCP/Vertex auth files
    chmod g+r /root/.config/gcloud/credentials.db /root/.config/gcloud/application_default_credentials.json 2>/dev/null || true
    chown root:"$WS_GID" /root/.config/gcloud/credentials.db /root/.config/gcloud/application_default_credentials.json 2>/dev/null || true
    chmod -R g+rwX /root/.config/gcloud/logs 2>/dev/null || true
    exec runuser -u claude-runner -- "$0" "$@"
fi

INTERACTIVE=false
DAYS=90
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive) INTERACTIVE=true; shift ;;
        --days) DAYS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
      "elements": [{"type": "mrkdwn", "text": "Skill: \`/autopilot --days $DAYS\` | Script: \`scripts/autopilot.sh\`"}]
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
    # Only post to success channel when something was actually implemented (PRs created)
    [ "$status" = "IMPLEMENTED" ] && [ -n "${SLACK_WEBHOOK_URL_SUCCESS:-}" ] && slack_post "$SLACK_WEBHOOK_URL_SUCCESS" "$payload"
}

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== Autopilot: $(date -Iseconds) ==="
echo "Running as: $(whoami)"
echo "Mode: $([ "$INTERACTIVE" = true ] && echo "interactive (live TUI)" || echo "headless")"
echo "Window: $DAYS days"
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

if [ "$INTERACTIVE" = true ]; then
    echo "/autopilot --days $DAYS" | claude --model claude-opus-4-6 --dangerously-skip-permissions \
        "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?
else
    claude -p "/autopilot --days $DAYS" --model claude-opus-4-6 --dangerously-skip-permissions \
        "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?
fi

# The agent posts success/implemented notifications to Slack directly.
# The script only handles failure (agent crash / non-zero exit).
if [ $EXIT_CODE -ne 0 ]; then
    slack_notify "FAIL" "Autopilot exited with code $EXIT_CODE. Check the logs for details."
fi
