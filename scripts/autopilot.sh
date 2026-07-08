#!/bin/bash
# Autopilot: discover new speculative decoding papers and implement each one.
# Scans arxiv, HuggingFace, GitHub, then runs oneshot on each promising find.
#
# Usage:
#   ./scripts/autopilot --days $DAYS.sh                          # headless, 60-day window
#   ./scripts/autopilot --days $DAYS.sh --days 90                # custom window
#   ./scripts/autopilot --days $DAYS.sh --interactive             # live TUI (debugging)
#   ./scripts/autopilot --days $DAYS.sh --interactive --days 120  # both
#
# Run on a schedule (system cron):
#   0 */8 * * * /workspace/speculators/scripts/autopilot --days $DAYS.sh >> /workspace/speculators/autopilot --days $DAYS.log 2>&1
#
# Options (via env vars):
#   MAX_TURNS                 Max agentic turns (default: unlimited)
#   SLACK_WEBHOOK_URL         Slack webhook for all notifications — logs channel (optional)
#   SLACK_WEBHOOK_URL_SUCCESS Slack webhook for successful runs only — results channel (optional)

set -euo pipefail

cd "$(dirname "$0")/.."

# Claude blocks --dangerously-skip-permissions for root. The devenv entrypoint
# creates a claude-runner user with the right groups — just re-exec as it.
if [ "$(id -u)" -eq 0 ]; then
    exec runuser --preserve-environment -u claude-runner -- "$0" "$@"
fi

INTERACTIVE=false
DAYS=60
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
      "elements": [{"type": "mrkdwn", "text": "Skill: \`/autopilot --days $DAYS\` | Script: \`scripts/autopilot --days $DAYS.sh\`"}]
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

STATE_DIR=".claude/agent_state"
PR_FILE="$STATE_DIR/last_run_prs.json"
REPORT_FILE="$STATE_DIR/last_run_report.md"
rm -f "$PR_FILE" "$REPORT_FILE"

if [ "$INTERACTIVE" = true ]; then
    echo "/autopilot --days $DAYS" | claude --dangerously-skip-permissions \
        "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?
else
    claude -p "/autopilot --days $DAYS" --dangerously-skip-permissions \
        "${EXTRA_ARGS[@]}" && EXIT_CODE=0 || EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    DETAIL="Autopilot scan completed."
    STATUS="SUCCESS"
    if [ -f "$PR_FILE" ]; then
        SPEC_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('speculators',''))" 2>/dev/null || true)
        VLLM_PR=$(python3 -c "import json; d=json.load(open('$PR_FILE')); print(d.get('vllm',''))" 2>/dev/null || true)
        if [ -n "$SPEC_PR" ] || [ -n "$VLLM_PR" ]; then
            STATUS="IMPLEMENTED"
            [ -n "$SPEC_PR" ] && DETAIL="$DETAIL\n*speculators PR:* <${SPEC_PR}>"
            [ -n "$VLLM_PR" ] && DETAIL="$DETAIL\n*vLLM PR:* <${VLLM_PR}>"
        fi
    fi
    if [ -f "$REPORT_FILE" ]; then
        REPORT=$(head -c 2800 "$REPORT_FILE" | sed 's/"/\\"/g; s/$/\\n/' | tr -d '\n')
        DETAIL="$DETAIL\n\n${REPORT}"
    fi
    slack_notify "$STATUS" "$DETAIL"
else
    slack_notify "FAIL" "Autopilot exited with code $EXIT_CODE. Check the logs for details."
fi
