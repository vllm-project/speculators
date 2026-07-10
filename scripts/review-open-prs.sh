#!/bin/bash
# Review open PRs: fetch non-draft, unapproved PRs and post reviews.
#
# Usage:
#   ./scripts/review-open-prs.sh                  # headless (default)
#   ./scripts/review-open-prs.sh --interactive     # live TUI (debugging)
#
# Run on a schedule (system cron):
#   7 * * * * /workspace/speculators/scripts/review-open-prs.sh >> /workspace/speculators/pr-review.log 2>&1

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
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive) INTERACTIVE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=()
[ -n "${MAX_TURNS:-}" ] && EXTRA_ARGS+=(--max-turns "$MAX_TURNS")

echo "=== PR Review Sweep: $(date -Iseconds) ==="
echo "Running as: $(whoami)"
echo "Mode: $([ "$INTERACTIVE" = true ] && echo "interactive (live TUI)" || echo "headless")"
[ -n "${MAX_TURNS:-}" ] && echo "Max turns: $MAX_TURNS" || echo "Max turns: unlimited"
echo "======================================="

if [ "$INTERACTIVE" = true ]; then
    echo "/review-open-prs" | claude --dangerously-skip-permissions "${EXTRA_ARGS[@]}"
else
    claude -p "/review-open-prs" --dangerously-skip-permissions "${EXTRA_ARGS[@]}"
fi
