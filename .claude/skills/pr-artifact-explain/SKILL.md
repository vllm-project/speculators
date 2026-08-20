---
name: pr-artifact-explain
description: Explain a PR from scratch for a newcomer — no repo or domain knowledge assumed — with a short review→response→fix digest, published as an Artifact.
allowed-tools: Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh issue view:*), Bash(gh api:*), Bash(git rev-parse:*), Bash(git status:*), Read, Grep, Glob, WebFetch, WebSearch, Skill, Write, Artifact
---

# PR Artifact Explain

Explain a GitHub pull request to someone brand new to this repository: no knowledge of the codebase, the domain, or the review thread. Bring them from zero to "I understand what this changes, why, and how", grounded in the actual code, plus a short digest of how review shaped the result. Deliver it as a self-contained Artifact.

## Input

`$ARGUMENTS` — PR number or URL. If empty, the current branch's PR.

## Hard constraints

- **Read-only.** Do not edit, stage, commit, push, or post. `gh api` is GET only. `Write` is only for the artifact HTML in the scratchpad, never a repository file.
- **No assumed background.** Explain the subsystem the PR touches from scratch. Define every repo-specific term, acronym, and abbreviation at first use.
- **Core explanation stands alone.** TL;DR, background, and walkthrough must read without the review thread — no "as the reviewer noted". The review arc gets its own section.
- **Explain from source, label inference.** Cite `file:line`, and make every citation refer to the revision you actually read. Mark anything the code does not state outright as inference.

## Steps

### 1. Resolve the PR

- URL: parse owner/repo/number. Bare number: current repo. Empty `$ARGUMENTS`: the current branch's PR (`gh pr view` with no target).
- If it does not resolve to a real PR, stop and say so.

### 2. Fetch (read-only)

Wave 1, in parallel:

- `gh pr view <target> --json number,title,body,state,isDraft,author,baseRefName,headRefOid,headRepositoryOwner,isCrossRepository,additions,deletions,files,labels,url,reviews,comments,closingIssuesReferences` — `reviews` carries review verdicts and summaries, `comments` the top-level thread, so neither needs its own API call.
- `gh pr diff <target>`
- Inline line comments — the most substantive feedback, and the only piece `gh pr view` cannot return: `gh api --paginate repos/{owner}/{repo}/pulls/{number}/comments --jq '[.[]|{id,reply_to:.in_reply_to_id,user:.user.login,path,line,commit:.commit_id,at:.created_at,body}]'`. `--paginate` is load-bearing: the endpoint caps at 30 per page and reports no error, silently dropping the tail of a heavily-reviewed PR. It emits one array per page (`--slurp` cannot combine with `--jq`). Keep `reply_to`/`id` — they are what pair a response to the concern it answers; `commit` tells you whether a comment still applies to the current head.

Wave 2, only if wave 1 found linked issues — for each entry in `closingIssuesReferences`, `gh issue view <n> --repo <its repo> --json number,title,body` for motivation.

If a call fails, report the error and stop. Do not proceed on partial data: an empty comments payload is indistinguishable from a quiet review and would publish a confidently wrong digest.

### 3. Understand it from zero

**Read at the PR's revision.** Compare `git rev-parse HEAD && git status --porcelain` against `headRefOid`:

- Match, and clean for the paths you need: use local `Read`/`Grep`/`Glob`.
- Otherwise read at the PR head with `gh api repos/{owner}/{repo}/contents/{path}?ref=<headRefOid> -H "Accept: application/vnd.github.raw"` — without that header the API returns base64-in-JSON. Fork head SHAs resolve against the base repo. Local `Grep`/`Glob` still work for locating paths; re-read the hits at the PR revision before quoting them.
- A 404 means the revision is gone (force-push, deleted fork): fall back to the diff hunks, and say in the artifact that surrounding context was unavailable.
- Deleted files: read at the base ref, or take them from the diff.

The diff already carries the changed hunks at head-revision line numbers. Read a full file only where surrounding context — callers, defaults, the enclosing class — is needed to explain the change, and issue a file's neighborhood reads as one parallel block.

- Identify the subsystem(s) the diff touches. For each meaningful file, read enough around it (module `__init__`, nearby `README`/docs, docstrings, the definitions the diff calls into) to explain what that area does and how the change fits.
- Trace the key code path the change adds or alters. Establish before-vs-after behavior.
- Separate load-bearing changes from mechanical ones (renames, formatting, generated files).
- Distill the review arc per the *Review → response → fix* spec in step 4. Pair each concern with its answer via `reply_to`/`id` rather than guessing from a shared file and line, and check `commit` before reporting something as unaddressed — it may predate a revision that fixed it. Drop reactions, nits, and resolved-without-change chatter.

### 3b. External references (only when the PR names one)

When the PR body, linked issue, or a review comment names a paper, RFC, upstream design doc, or documented library behavior, follow it one hop with `WebSearch`/`WebFetch` and use it for **background and glossary definitions only**. This hop can run concurrently with step 3's reading.

- One hop: do not chase the reference's own citations.
- No named source in the PR means no search. Never invent a reference.
- Where the source and the code disagree, the code wins; mention the divergence.
- Cite what you used, title and link, in *Further reading*.

### 4. Build the artifact

1. Invoke the `artifact-design` skill first.
2. Write the HTML to the scratchpad, then publish it with the `Artifact` tool. `<title>`: `PR #<n>: <short title> — Explained`. One-sentence `description`. Favicon `📖`.

These are the sections in order, not a quota. Any section may collapse to a single line when the PR doesn't warrant it, and Glossary and Further reading drop entirely when empty. Padding a section is worse than a one-line section.

- **Header** — PR number, title, repo, author, state (including draft), labels, +additions/−deletions, link. Metadata as small chips.
- **TL;DR** — one plain-language paragraph: what it does and why it matters.
- **Background you need** — the subsystem from zero: what this part of the codebase is responsible for, and how the moving parts worked before this PR.
- **The problem** — what was missing or wrong (from the linked issue if any, otherwise inferred from the diff and labeled as inference).
- **What changed** — a table of the load-bearing files (`path` · what changed · why); mechanical changes summarized in one line.
- **How it works** — the mechanism, walking the key code path with short snippets from the diff. No diagram.
- **Impact & risk** — who is affected, and behavior/compatibility/performance changes and edge cases.
- **How to verify** — tests added or changed and the command to run them; what a reviewer should eyeball.
- **Review → response → fix** (short) — the substantive concerns reviewers raised, how the author responded (agreed or rebutted), and what changed as a result. A compact table (concern · response · outcome) or a few bullets.
- **Glossary** — every repo or domain term, one line each.
- **Further reading** — one line per external source from 3b, title and link.

### 5. Report back

In chat, give a 2-3 sentence summary and the artifact link.
