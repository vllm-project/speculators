# Review Open PRs

Scan all open pull requests on the speculators repo and run `/pr-review` on each one that needs attention. Designed for periodic cron execution.

## Step 1: Fetch reviewable PRs

Run:

```bash
gh pr list --repo vllm-project/speculators --state open --json number,title,isDraft,reviewDecision,updatedAt,url --limit 100
```

Filter for PRs that meet **ALL** criteria:

1. **Not a draft** — `isDraft` is `false`
2. **Not already approved** — `reviewDecision` is not `"APPROVED"`
3. **Recent activity** — `updatedAt` is within the last 30 days

If no PRs match, report "No reviewable PRs found." and stop.

## Step 2: Skip already-reviewed PRs

For each candidate PR, check whether you have already reviewed it since its last push:

```bash
gh api repos/vllm-project/speculators/pulls/<number>/reviews --jq '[.[] | select(.user.login == "orestis-z")] | last | .submitted_at'
```

Also get the last push timestamp:

```bash
gh pr view <number> --repo vllm-project/speculators --json commits --jq '.commits[-1].committedDate'
```

**Skip** the PR if your last review is more recent than the last push — there is nothing new to review. If you have never reviewed the PR, it qualifies.

## Step 3: Review each PR

For each qualifying PR, spawn an Agent (subagent) to run the review. Launch **all agents in a single message** so they run in parallel:

```
Agent({
  description: "Review PR #<number>",
  prompt: "Run /pr-review <number> on the vllm-project/speculators repo. Follow the skill instructions fully — gather context, review, and post."
})
```

Send all Agent tool calls in one response to maximize parallelism. Each agent independently fetches context, reviews, and posts — they don't share state, so parallel execution is safe.

If an agent fails or errors out, note the error in the summary and continue.

## Step 4: Summary

After all PRs are processed, present a summary:

```
## PR Review Sweep Complete

- PRs scanned: <total open>
- PRs filtered out: <count> (draft: N, approved: N, stale: N, already reviewed: N)
- PRs reviewed: <count>

### Reviewed
| PR | Title | Result |
|----|-------|--------|
| #<number> | <title> | <findings count> findings |

### Skipped (already reviewed, no new commits)
| PR | Title | Last reviewed |
|----|-------|---------------|
| #<number> | <title> | <date> |
```

## Cron Integration

This skill is designed to be called by a durable cron job. Recommended schedule:

```
7 * * * *   (hourly at :07)
```

The skip-already-reviewed logic in Step 2 ensures idempotent runs — re-running within the same hour is harmless.
