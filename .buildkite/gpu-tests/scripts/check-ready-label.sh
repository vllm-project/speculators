#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BUILDKITE_PULL_REQUEST:-}" ]] || [[ "${BUILDKITE_PULL_REQUEST}" == "false" ]]; then
  BRANCH="${BUILDKITE_BRANCH:-}"
  if [[ "$BRANCH" == "main" ]] || [[ "$BRANCH" == release* ]]; then
    echo "Push to ${BRANCH} — proceeding with GPU tests"
    exit 0
  else
    echo "Push to ${BRANCH} — skipping GPU tests (only main and release branches run tests)"
    exit 1
  fi
fi

REPO=$(echo "${BUILDKITE_REPO}" | sed -E 's#.*github\.com[:/](.*)\.git#\1#')

LABELS=$(curl -sf \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/${REPO}/pulls/${BUILDKITE_PULL_REQUEST}" \
  | jq -r '.labels[].name')

if echo "$LABELS" | grep -qx "ready"; then
  echo "PR has 'ready' label — proceeding with GPU tests"
else
  echo "PR does not have 'ready' label — skipping GPU tests"
  echo "Add the 'ready' label to trigger GPU test suite"
  exit 1
fi
