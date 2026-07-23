---
name: pr-review
description: Review a GitHub PR with design-first analysis, posted as a GitHub review.
---

# PR Review

Review a GitHub pull request with design-first analysis, posted as a GitHub review.

## Input

$ARGUMENTS — PR number or URL. If empty, detect from the current branch.

## Instructions

You are reviewing a PR on the `speculators` repo — a library for training and serving speculative decoding models (Eagle3, DFlash, MTP) with vLLM integration.

### Phase 0: Eligibility check

1. Resolve the PR number from `$ARGUMENTS` (or current branch via `gh pr view -R vllm-project/speculators --json number`).
2. Check `gh pr view -R vllm-project/speculators <number> --json state,isDraft`. **Do not proceed** if the PR is closed, merged, or a draft.
3. Check if you (the current `gh` user) already posted a review on this PR. If so, compare the latest commit SHA on the PR against the commit SHA at the time of your last review. **Do not post again** if there are no new commits since your last review — report to the user that the PR hasn't changed since the last review. Even with new commits, if your review would say substantially the same thing as your previous one (same verdict, same outstanding items), do not post — report to the user that nothing has materially changed.
4. After gathering context (Phase 1–2), reassess: if the discussion thread shows the PR is effectively still in progress — active experiments, trial-and-error debugging, the author saying "still working on this", or unresolved back-and-forth about the approach — treat it as a draft and do not review. Report to the user that the PR appears to still be in flux.

### Phase 1: Gather context

Run these in parallel:

- `gh pr view -R vllm-project/speculators <number> --json title,body,baseRefName,headRefName,author,labels,files`
- `gh pr diff -R vllm-project/speculators <number>`
- `gh api repos/vllm-project/speculators/pulls/<number>/reviews` — existing reviews
- `gh api repos/vllm-project/speculators/pulls/<number>/comments` — inline comments
- `gh api repos/vllm-project/speculators/issues/<number>/comments` — conversation thread

Then read the full diff. For changed files larger than 300 lines, also read the full file for surrounding context.

If the PR description contains links (issues, discussions, external docs, benchmarks, etc.), fetch them and incorporate the context. Verify that the PR actually addresses what the linked resources describe — flag mismatches between linked context and the implementation.

### Phase 2: Understand existing discussion

Before forming any opinions, catalog:

- What has each reviewer already said?
- What has the author responded to?
- What is unresolved?

**Do NOT echo points already raised by other reviewers.** If you agree with an existing comment, do not post your own version of it — even with additional detail or a different explanation. The only exception is if the author asked a clarifying question on the existing comment and the original reviewer hasn't answered; in that case, reply to that comment thread rather than posting a separate inline comment. Drop any finding that overlaps with an existing review comment.

If reviewers or the author linked to sources (docs, issues, code snippets, benchmarks, papers) in their comments, fetch and read them — they often contain context that informs whether a concern is valid or already resolved.

### Phase 3: Design & big-picture review

Evaluate these first — they matter more than line-level nits:

- **Bloat check**: Does the value this PR adds justify the maintenance burden it introduces? Every new file, abstraction, and code path has an ongoing cost — someone has to understand it, update it, and debug it. Flag PRs that add significant complexity (new modules, abstractions, config options, CLI flags) without proportional value. Also flag: vendored files that should be dependencies, IDE/editor files (`.idea/`, `.vscode/`), or code that reimplements something already available in a dependency.
- **AI slop check**: Does the code read like unedited LLM output? Signs: excessive docstrings on every function (especially multi-paragraph ones restating the function signature), over-commented code explaining the obvious, cargo-culted error handling that catches exceptions only to re-raise them, unnecessary abstractions wrapping single calls, verbose variable names that read like natural language sentences, or boilerplate patterns copy-pasted without adaptation to the context. If you spot these patterns, flag it directly — ask the author to trim the noise.
- **Splittability**: Does the PR bundle independent changes that could be reviewed separately? Look for: unrelated bug fixes alongside a feature, refactors mixed with new functionality, changes to multiple subsystems with no dependency between them, or "while I was here" cleanup. If the PR contains independently reviewable units, ask the author to split it — smaller PRs get better reviews and merge faster. Only flag this when the pieces are truly independent; don't ask to split tightly coupled changes.
- **Motivation**: Is the PR description clear about *why* this change is needed? If the motivation is missing or vague, ask the author to clarify. A PR that doesn't explain the problem it solves is hard to review properly.
- **Purpose & scope**: Does the PR do what the description claims? Is the scope appropriate or should it be split?
- **Architectural fit**: Does this fit the existing patterns in `speculators`? Does it introduce unnecessary abstractions or bypass existing ones? Read surrounding code in the same module to understand local conventions (naming, error handling, structure) and flag deviations.
- **Correctness of approach**: For the problem being solved, is this the right approach? Have alternatives been considered? If you see a clearly better alternative, name it concretely and explain the tradeoff.
- **Design principles**:
  - *DRY*: Does this duplicate logic that already exists elsewhere in the codebase? If so, point to the existing code.
  - *YAGNI*: Does this add abstractions, parameters, or code paths not justified by the PR's stated goal? Premature generalization is a flag.
  - *KISS*: Is there a simpler way to achieve the same result? Complexity should be proportional to the problem.
- **Backward compatibility**: Could this break existing checkpoints, configs, or CLI invocations?
- **Test coverage**: Does the PR include tests proportional to the change? New logic should have unit tests. Changes to training, data generation, or model forward passes that affect end-to-end behavior should have integration tests (or the author should explain why they're impractical). For bug fixes specifically, there should be a regression test that fails without the fix and passes with it — if missing, ask for one. If tests are missing, ask for them — specify what scenarios should be covered.
- **Distributed training correctness** (if applicable): barrier placement, device consistency, FSDP wrapping, gradient handling.
- **Tensor/numerical correctness** (if applicable): shape mismatches, dtype promotions, masking, off-by-one in shift-based alignment.

### Phase 3.5: RFC compliance

If the PR description or linked issues reference an RFC (issues labeled `rfc` in this repo), fetch the RFC issue via `gh issue view` and verify:

1. Does the implementation match what was agreed in the RFC discussion? Check the final consensus, not just the original proposal — decisions often evolve in comments.
2. Are there open objections or unresolved questions in the RFC that this PR should not proceed without addressing?
3. If the implementation deviates from the RFC, is the deviation documented and justified in the PR description?

Flag any discrepancy between the RFC and the implementation, quoting the relevant part of the RFC discussion.

### Phase 3.6: Paper validation

If the PR description, commit messages, or code comments reference a paper (arXiv, conference proceedings, etc.):

1. Fetch the paper (use `WebFetch` on the arXiv abstract page or PDF URL).
2. Identify the specific equations, algorithms, or architectural details the PR claims to implement.
3. Compare the implementation against the paper:
   - Do the equations match? Check operator ordering, normalization, index conventions.
   - Are any simplifications or deviations from the paper intentional and documented, or silent divergences?
   - Are hyperparameter defaults consistent with what the paper recommends?
4. In your review, **quote the relevant section/equation from the paper** and compare it to the code. For example: "Paper Eq. 5 defines the loss as `L = ...`, but the implementation at `file:line` computes `...` instead — is this intentional?"
5. If no paper is referenced but the PR implements a known algorithm (Eagle, EAGLE-2, Eagle3, DFlash, MTP, Medusa, etc.), check whether the implementation aligns with the canonical paper. If you cannot fetch or verify the paper, state that explicitly rather than guessing.

Only flag mismatches you can concretely demonstrate by quoting both the paper and the code. Do not flag stylistic differences in how math is expressed if the computation is equivalent.

### Phase 3.7: CONTRIBUTING.md compliance

Check the PR against the project's contribution guidelines (`CONTRIBUTING.md`):

1. **Issue linkage for significant changes**: If the PR adds new training algorithms/model support, modifies the data pipeline or CLI/API, is a large refactor, or touches 3+ files — verify it references an assigned issue. Small fixes (typos, docs, bug fixes < 20 lines, type annotations, minor deps) are exempt.
2. **DCO sign-off**: All commits must have a `Signed-off-by` line. Check with `gh pr view -R vllm-project/speculators <number> --json commits --jq '.commits[].messageBody'` — if any commit lacks the sign-off, flag it (CI will also catch this, but noting it saves a round-trip).
3. **Documentation updates**: If the PR changes user-facing behavior (CLI flags, config options, APIs), check whether docs were updated. Missing doc updates for behavior changes should be flagged.

### Phase 4: Line-level review

Apply path-specific focus based on which files changed:

- **`src/speculators/train/**`**: Distributed training correctness (barriers, device placement, FSDP), LR scheduler logic, checkpoint save/resume, loss computation with padding masks, multi-step loss aggregation and decay, bfloat16 safety.
- **`src/speculators/models/**`**: Architecture correctness, attention mask construction for speculative positions, vocabulary mapping (draft→target) in forward/loss, KL divergence alignment, Pydantic config serialization.
- **`src/speculators/data_generation/**`**: Hidden state extraction correctness, shift-based alignment (off-by-one), loss mask application before storage, vLLM client error handling.
- **`src/speculators/convert/**`**: Weight mapping completeness, shape compatibility, key renaming correctness, legacy format assumptions.
- **`src/speculators/config.py`**: Pydantic validators, serialization roundtrip, backward compat with saved checkpoints, registry auto-discovery.
- **`tests/**`**: Coverage of new code paths, edge cases for speculative decoding (vocab boundaries, multi-step loss), proper mocking of vLLM/GPU.

Ignore: `**/*.pyc`, `**/build/**`, `**/.pytest_cache/**`, `**/.ruff_cache/**`, `**/__pycache__/**`, `**/*.egg-info/**`.

### Phase 5: Confidence scoring & filtering

Rate each finding 0–100:

- **0–25**: False positive, pre-existing issue, or something a linter/typechecker would catch.
- **26–50**: Minor nitpick not tied to a project convention or concrete bug.
- **51–75**: Real but low-impact or unlikely in practice.
- **76–90**: Verified issue that will impact functionality, or violates a documented project convention.
- **91–100**: Confirmed critical bug with concrete failure scenario.

**Only keep findings with confidence ≥ 80.**

False positives to actively filter out:

- Pre-existing issues not introduced by this PR.
- Pedantic style nitpicks a senior engineer would skip.
- Issues a linter, typechecker, or CI would catch — do not run these yourself.
- General quality concerns (documentation, naming style) that are purely cosmetic and not tied to correctness.
- Issues silenced in code (e.g., lint ignore comments).
- Intentional functionality changes that align with the PR's stated purpose.

### Phase 6: Draft the review

For each surviving finding:

- State the problem in one sentence.
- If the issue involves an external API, library behavior, or spec, **link to the source or quote the relevant documentation**. Do not make unverified claims about external behavior.
- If you propose an alternative, show a concrete code suggestion and confirm the alternative works in context.
- If the author's intent is unclear, **ask a question** instead of assuming.

**Do NOT speculate.** Only flag issues you can validate from the diff, surrounding code, or linked documentation. If you're unsure whether something is a bug, phrase it as a question to the author.

**Source attribution**: If the PR implements logic derived from a paper, spec, or external reference and the code doesn't link to it (in comments, docstrings, or the PR description), ask the author to add a reference. Future readers shouldn't have to reverse-engineer which paper or doc a piece of logic came from.

Structure:

**Review body** (high-level summary):

- 1–3 sentences on the overall design assessment.
- Note any high-level concerns or open questions.
- If everything looks good at the design level, say so briefly.

**Inline comments** (line-level):

- Each non-high-level comment must reference a specific file and line range from the diff.
- Order: correctness issues → design concerns → suggestions → questions.
- When linking to code, use full-SHA permalink format for proper rendering: `https://github.com/vllm-project/speculators/blob/{full_sha}/{path}#L{start}-L{end}` Get the SHA via `git rev-parse HEAD` on the PR branch, and include ≥1 line of context above and below.

### Phase 7: Verify before posting

Re-check **every** comment against this checklist:

1. **Is the claim actually true?** Re-read the relevant code. If you reference external behavior (PyTorch, HuggingFace, vLLM), verify it via docs or source — or remove the claim.
2. **Is this already addressed in the PR?** Check if a later hunk in the diff fixes the issue.
3. **Was this already raised by another reviewer?** Drop if so.
4. **Is this preference disguised as correctness?** If yes, either drop it or explicitly mark as suggestion.
5. **For alternatives you propose**: have you verified the alternative actually works in this context?
6. **Does this pass the confidence ≥ 80 bar after re-examination?**

Drop any comment that fails any check.

### Phase 8: Post

Use `gh api` to post the review with all inline comments in a single atomic review submission. **Do not use `-f 'comments[0][...]'` flags** — `gh` serializes those as a JSON object with string keys, not an array, causing a 422 error. Instead, pipe raw JSON via `--input -`:

```bash
cat <<'REVIEW_EOF' | gh api repos/vllm-project/speculators/pulls/{number}/reviews -X POST --input - --jq '.html_url'
{
  "event": "COMMENT",
  "body": "<review summary>",
  "comments": [
    {"path": "<file>", "line": <line>, "body": "<comment>"},
    {"path": "<file>", "start_line": <start>, "line": <end>, "side": "RIGHT", "body": "<comment>"}
  ]
}
REVIEW_EOF
```

For single-line comments, use `line` only. For multi-line, use `start_line` + `line`. Use `side: "RIGHT"` for additions.

If there are zero inline comments beyond the summary, post just the review body.

If no issues survive filtering (all < 80 confidence), **do not post a review**. Report to the invoker that no actionable findings were found, but do not leave a comment on the PR — clean PRs don't need noise.

**Never approve a PR automatically.** Always use `event="COMMENT"`, never `event="APPROVE"`. Only the human reviewer should submit an approval.

End every review body with:

```
🤖 Generated with [Claude Code](https://claude.ai/code) using the `/pr-review` skill
```

Report the review URL when done, along with a brief summary: number of findings posted, number filtered out, and wall-clock time elapsed since Phase 0 started.
