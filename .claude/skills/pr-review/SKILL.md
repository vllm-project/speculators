# PR Review

Review a GitHub pull request with design-first analysis, posted as a GitHub review.

## Input

$ARGUMENTS — PR number or URL. If empty, detect from the current branch.

## Instructions

You are reviewing a PR on the `speculators` repo — a library for training and serving speculative decoding models (Eagle3, DFlash, MTP) with vLLM integration.

### Phase 0: Eligibility check

1. Resolve the PR number from `$ARGUMENTS` (or current branch via `gh pr view --json number`).
2. Check `gh pr view <number> --json state,isDraft`. **Do not proceed** if the PR is closed, merged, or a draft.

### Phase 1: Gather context

Run these in parallel:
- `gh pr view <number> --json title,body,baseRefName,headRefName,author,labels,files`
- `gh pr diff <number>`
- `gh api repos/{owner}/{repo}/pulls/<number>/reviews` — existing reviews
- `gh api repos/{owner}/{repo}/pulls/<number>/comments` — inline comments
- `gh api repos/{owner}/{repo}/issues/<number>/comments` — conversation thread

Then read the full diff. For changed files larger than 300 lines, also read the full file for surrounding context.

### Phase 2: Understand existing discussion

Before forming any opinions, catalog:
- What has each reviewer already said?
- What has the author responded to?
- What is unresolved?

**Do NOT echo points already raised by other reviewers.** If you agree with an existing comment, you may reference it briefly ("I agree with @X's point on ...") but do not restate it. Drop any finding that duplicates an existing review comment.

### Phase 3: Design & big-picture review

Evaluate these first — they matter more than line-level nits:

- **Purpose & scope**: Does the PR do what the description claims? Is the scope appropriate or should it be split?
- **Architectural fit**: Does this fit the existing patterns in `speculators`? Does it introduce unnecessary abstractions or bypass existing ones?
- **Correctness of approach**: For the problem being solved, is this the right approach? Have alternatives been considered? If you see a clearly better alternative, name it concretely and explain the tradeoff.
- **Backward compatibility**: Could this break existing checkpoints, configs, or CLI invocations?
- **Distributed training correctness** (if applicable): barrier placement, device consistency, FSDP wrapping, gradient handling.
- **Tensor/numerical correctness** (if applicable): shape mismatches, dtype promotions, masking, off-by-one in shift-based alignment.

### Phase 3.5: Paper validation

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
- General quality concerns (test coverage, documentation) unless explicitly required by project conventions.
- Issues silenced in code (e.g., lint ignore comments).
- Intentional functionality changes that align with the PR's stated purpose.

### Phase 6: Draft the review

For each surviving finding:
- State the problem in one sentence.
- If the issue involves an external API, library behavior, or spec, **link to the source or quote the relevant documentation**. Do not make unverified claims about external behavior.
- If you propose an alternative, show a concrete code suggestion and confirm the alternative works in context.
- If the author's intent is unclear, **ask a question** instead of assuming.

**Do NOT speculate.** Only flag issues you can validate from the diff, surrounding code, or linked documentation. If you're unsure whether something is a bug, phrase it as a question to the author.

Structure:

**Review body** (high-level summary):
- 1–3 sentences on the overall design assessment.
- Note any high-level concerns or open questions.
- If everything looks good at the design level, say so briefly.

**Inline comments** (line-level):
- Each non-high-level comment must reference a specific file and line range from the diff.
- Order: correctness issues → design concerns → suggestions → questions.
- When linking to code, use full-SHA permalink format for proper rendering:
  `https://github.com/{owner}/{repo}/blob/{full_sha}/{path}#L{start}-L{end}`
  Get the SHA via `git rev-parse HEAD` on the PR branch, and include ≥1 line of context above and below.

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

Use `gh api` to post the review with all inline comments in a single atomic review submission:

```bash
gh api repos/{owner}/{repo}/pulls/{number}/reviews \
  -X POST \
  -f event="COMMENT" \
  -f body="<review summary>" \
  -f 'comments[0][path]=<file>' \
  -f 'comments[0][line]=<line>' \
  -f 'comments[0][body]=<comment>' \
  --jq '.html_url'
```

For multi-line comments, include both `start_line` and `line` (end). Use `side=RIGHT` for additions.

If there are zero inline comments beyond the summary, post just the review body.

If no issues survive filtering (all < 80 confidence), post:

> No issues found. Checked for bugs, design concerns, and project convention compliance.

Report the review URL when done.
