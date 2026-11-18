---
name: pr-review
description: Review a vllm-project/speculators pull request with design-first analysis. Drafts the review and shows it; posts it as a GitHub review only when given --post.
---

# PR Review

## Input

$ARGUMENTS — a PR number or URL, optionally followed by `--post`. If no number is given, detect it from the current branch (`gh pr view -R vllm-project/speculators --json number`).

Without `--post` this is a dry run: draft the review and show it, but post nothing to GitHub.

## Files

- [references/rubric.md](references/rubric.md): how to review. The same rubric is used by the automated reviewer, so keep repo-specific review knowledge there, not here.
- [references/output.md](references/output.md): the `review.json` format.
- `scripts/`: `pr_state.py` (should this PR be reviewed?), `gather_context.py` (fetch the PR and its discussion), `review_payload.py` (turn `review.json` into a GitHub review). They need only Python 3 and `curl`, and read `GITHUB_TOKEN` from the environment.

Run the commands below from the repository root. Note the time when you start.

## Steps

1. **Eligibility.** Run:

   ```bash
   GITHUB_TOKEN=$(gh auth token) python3 .claude/skills/pr-review/scripts/pr_state.py \
     vllm-project/speculators <number> "$(gh api user --jq .login)"
   ```

   If `eligible` is false, stop and tell the user the `reason`.

2. **Context.** Fetch everything the rubric reads into a fresh directory:

   ```bash
   CTX=$(mktemp -d)
   GITHUB_TOKEN=$(gh auth token) python3 .claude/skills/pr-review/scripts/gather_context.py \
     vllm-project/speculators <number> "$CTX"
   ```

3. **Review.** Follow [references/rubric.md](references/rubric.md) against `$CTX`. If the rubric says the PR isn't ready or nothing has changed since the last review, stop and tell the user.

4. **Write `$CTX/review.json`** in the format of [references/output.md](references/output.md).

5. **Build the payload.**

   ```bash
   # The footer links the skill at this commit, so it must exist on GitHub:
   # unpushed skill commits fall back to main's.
   SKILL_REF=$(git log -1 --format=%H -- .claude/skills/pr-review)
   git branch -r --contains "$SKILL_REF" | grep -q . || SKILL_REF=$(git rev-parse origin/main)
   python3 .claude/skills/pr-review/scripts/review_payload.py "$CTX/review.json" "$CTX" \
     --skill-ref "$SKILL_REF" \
     --byline "🤖 Generated with [Claude Code](https://claude.ai/code)" > "$CTX/payload.json"
   ```

   If your copy of the skill has changes that aren't on GitHub, tell the user the footer links main's version instead. Exit status 3 means no finding reached the confidence bar: tell the user and post nothing. Clean PRs don't need noise.

6. **Dry run (default).** Show the user the review body and each inline comment with its file and line. Do not post.

   **With `--post`**, submit it as one review:

   ```bash
   gh api repos/vllm-project/speculators/pulls/<number>/reviews -X POST \
     --input "$CTX/payload.json" --jq .html_url
   ```

   The payload is always a `COMMENT` review. Never approve or request changes; that is a maintainer's call.

7. **Report** the review URL (if posted), how many findings were posted inline, in the body, and dropped, and the wall-clock time since you started.
