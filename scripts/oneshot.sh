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
    if id claude-runner &>/dev/null; then
        # Recreate if not in root group
        if [ "$(id -g claude-runner)" != "0" ]; then
            userdel claude-runner
            useradd -M -d /root -g 0 claude-runner
        fi
    else
        useradd -M -d /root -g 0 claude-runner
    fi
    find /root -not -path '/root/.ssh*' -exec chmod g+rwX {} + 2>/dev/null || true
    chmod -R g+rwX /workspace 2>/dev/null || true
    exec runuser --preserve-environment -u claude-runner -- "$0" "$@"
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
    curl -sf -X POST -H 'Content-type: application/json' -d "$payload" "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
}

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== One-Shot Paper-to-Implementation ==="
echo "Paper: $PAPER_URL"
echo "Running as: $(whoami)"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then echo "Slack: enabled"; else echo "Slack: disabled (set SLACK_WEBHOOK_URL to enable)"; fi
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
