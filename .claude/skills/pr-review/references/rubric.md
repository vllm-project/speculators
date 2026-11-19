# PR review rubric

How to review a pull request on `vllm-project/speculators`, a library for training and serving speculative decoding models with vLLM integration. This rubric is independent of the tool running it: it reads the context files written by `scripts/gather_context.py` and produces a `review.json` as described in [output.md](output.md).

## What a review is for

Review like a maintainer of this repository who has little time and cares about its long-term health. Say only what should change the PR or the author's next step; every comment costs the author time to read and answer.

- **Be right.** Every claim is checked against the code, its documentation, or the source it cites. When you're not sure something is a bug, ask instead of asserting.
- **Be new.** Don't repeat what a reviewer, the author or an earlier review already said, here or on a PR this one is stacked on, even with more detail. A question the author asked in a thread that nobody answered is the exception: answer it, pointing to the thread.
- **Weigh it.** Design problems matter more than line-level ones, and a few findings that matter beat many that don't.
- **Be readable.** The author should get each finding in one read.

## Context

The context directory holds `pr.json`, `pr.diff`, `files.json`, `reviews.json`, `review_comments.json`, `issue_comments.json` and `stack.json`. Read the whole diff, and enough of the surrounding code to judge it in context.

If `stack.json` lists parents, this PR is stacked on other open PRs by the same author, and its diff includes theirs. Review only `own.diff`, its changes on top of the nearest parent; if there is none (the branch has merged other changes since, so they can't be separated), review `pr.diff` but leave what the parents change, in `stack/<number>/pr.diff`, to them. Read the parents' discussions under `stack/<number>/`: what was decided or said there is settled. Line numbers still refer to the PR head.

Read what the PR and its discussion link to (issues, RFCs, papers, docs, benchmarks), and check that the PR does what they describe. If something you need can't be fetched, say so rather than guess.

Code is often the way it is for a reason the diff doesn't show. Before suggesting that something be changed back, or calling a choice a mistake, read the history of those lines: the commits and the PRs that made them.

## Is the PR ready for review?

Don't review a PR that is still in flux (active experiments, trial-and-error fixes, the author saying it isn't done, unresolved debate about the approach), or one where your review would say substantially what an earlier review of yours already said, unless someone asked for another review. Report which of the two it is instead.

## What to look at

**Design.** These matter more than anything line-level:

- **Worth it?** Every new module, abstraction, config option or flag is something someone has to understand and maintain. Flag complexity out of proportion to the value, code that reimplements what the codebase or a dependency already has (point to it), vendored files that should be dependencies, and editor files.
- **Does what it says.** The description says why the change is needed, and the diff does what the description claims. Missing or vague motivation is a finding.
- **Fits the codebase.** Read the surrounding module and follow its conventions; flag abstractions nothing needs yet and parameters the PR's goal doesn't call for. If there is a clearly simpler or better approach, name it and the tradeoff.
- **One thing per PR.** Unrelated fixes alongside a feature, refactors mixed with new behaviour, or "while I was here" cleanup should be split. Only ask when the parts are truly independent.
- **Generated noise.** Docstrings that restate the signature, comments explaining the obvious, error handling that only re-raises, wrappers around a single call: ask the author to trim it.
- **Compatibility.** Could it break checkpoints, configs or CLI invocations that shipped in a release? Code that has never been released needs no backward compatibility.

**Tests.** New logic needs tests in proportion to the change; changes to training, data generation or forward passes that affect end-to-end behaviour need integration tests or a reason why not; a bug fix needs a regression test that fails without it. When asking for tests, say which scenarios they should cover.

**Numerical code** anywhere: shapes, dtype promotion, masking, and off-by-one errors in shift-based alignment.

**Where this repo breaks.** Look hardest at what the touched code is responsible for:

- `src/speculators/train/`: distributed correctness (barriers, devices, FSDP), LR schedules, checkpoint save and resume, masked and multi-step losses, bfloat16 safety.
- `src/speculators/models/`: attention masks for speculative positions, draft-to-target vocabulary mapping, loss alignment, config serialization.
- `src/speculators/data_generation/`: hidden-state extraction, shift-based alignment (off-by-one), loss masks before storage, vLLM client errors, deprecated code paths coming back.
- `src/speculators/convert/`: completeness of weight mappings, shapes, key renames, legacy formats.
- `src/speculators/config.py`: validators, serialization round trips, compatibility with saved checkpoints, registry discovery.
- `scripts/`: argument handling, progress logging, and safety on multiple GPUs. `scripts/launch_vllm.py` runs in the vLLM virtualenv, which doesn't have `speculators` installed, so any `speculators` import there is a bug.
- `examples/`: they run end to end against the current API and explain what a newcomer to speculative decoding needs.
- `docs/` and READMEs: code examples, install steps and vLLM integration steps match the current API and behaviour.
- `tests/`: whether the new paths and the speculative-decoding edge cases (vocab boundaries, multi-step loss, distributed vs single-GPU checkpoints) are covered, with vLLM and GPUs mocked where needed.
- `.claude/skills/`, `AGENTS.md` and other instructions for agents: guidance should give principles and the repository knowledge a model can't infer, not micromanage. Flag step-by-step rules for judgment calls, restatements of what a capable model already knows, arbitrary thresholds, and shouting. Scripts, commands and machine-checked formats are the exception: those should be exact.

**Contribution rules.** `CONTRIBUTING.md` and `AGENTS.md` say what a PR must carry (issue link, DCO sign-off, doc updates, the description). Flag what's missing as a finding about the PR as a whole, citing the rule. The human-review checkbox is not one of them: AGENTS.md tells the agent writing a PR to leave it for a person to tick, so a ticked box on a PR an agent wrote is the expected state.

**RFCs.** If the PR implements an RFC (issues labelled `rfc`), check it against the discussion's final consensus, not just the original proposal, and against objections still open. A deviation should be stated and justified in the PR.

**Papers.** If the PR implements a paper, or a known algorithm (Eagle, EAGLE-2, Eagle3, DFlash, MTP, Medusa), read the paper's full text and compare the specific equations or algorithm with the code: operator order, normalization, index conventions, and hyperparameter defaults. Flag only mismatches you can show by quoting both, e.g. "Eq. 5 defines the loss as `L = …`, but `file:line` computes `…`; is that intentional?". Equivalent math written differently is not a finding. Logic taken from a paper or spec should say where it comes from.

## Calibration

Score each finding 0–100 for how sure you are that it is real and matters:

- below 50: a false positive, a nitpick, or something pre-existing;
- 50–79: real but low-impact or unlikely in practice;
- 80–89: verified, affects behaviour or breaks a documented project rule;
- 90 and up: a confirmed bug with a concrete failure scenario.

Only findings at 80 or above are posted; `scripts/review_payload.py` drops the rest.

Below the bar, whatever their score would otherwise be:

- problems the PR didn't introduce, style a senior engineer would skip, and restatements of what a linter, type checker or CI already reports (explaining the cause of a failure is still in scope);
- the change the PR intends, or one a maintainer asked for (here or on a stacked parent), though not its unintended effects;
- failures that need a caller or input the code doesn't have yet. Read the code's documented contract and its real callers before claiming it is broken;
- preferences presented as correctness, and alternatives you haven't checked would work.

Before writing, go over each finding once more against the code: is it true, does a later hunk already handle it, has someone already said it, and does it still clear the bar?

## Writing the review

The format is in [output.md](output.md).

The **summary** holds only what no finding can: which finding to settle first when the others depend on it, a linked source (a paper, an issue) you needed and couldn't load, or why you didn't review. Most reviews have none of these, and the summary stays empty. The author wrote the PR, so saying what it does, what it gets right or what you checked gives them nothing to act on, and a PR that is fine needs no case made for it.

Each **finding** has a `body` and, when it needs one, a `detail`. The body is what a human reviewer would write: what is wrong and what to do, in a sentence or two. Checking a claim is your work, not the comment's content. The author knows the code and can verify a claim that is stated plainly, so the body gives the conclusion and leaves out the trail that led to it; don't compress the trail into the body either. If the author's intent is unclear, ask. A finding about the PR as a whole has no `file`.

The `detail` is posted collapsed under the body, for an author who doubts the claim or wants to act on it: the evidence, a reproduction, a source, suggested code (make sure it works in context). A finding the code makes obvious needs none.

**Sources** belong in the detail. When a finding rests on external behaviour (PyTorch, Transformers, vLLM), link or quote the documentation or source. Link the repository files you cite at the commit you read: code at the PR head, as `https://github.com/vllm-project/speculators/blob/{head.sha}/{path}#L{start}-L{end}` with a line of context either side, and repository rules at the base commit, down to the section, e.g. `[CONTRIBUTING.md › Code Style and Guidelines](https://github.com/vllm-project/speculators/blob/{base.sha}/CONTRIBUTING.md#code-style-and-guidelines)`.
