---
name: oneshot-paper
description: End-to-end autonomous pipeline — reads a speculative decoding paper, implements it in speculators + vLLM, and runs a smoke training. No human checkpoints.
---

# One-Shot Paper-to-Training Pipeline

You are running the full autonomous pipeline for a speculative decoding paper. The user provides a URL (arxiv, HF model, blog post) and you do everything end-to-end without asking for confirmation. Report results at the very end.

**IMPORTANT: Do not stop to ask the user questions. Make reasonable decisions and document them. If something is ambiguous, pick the most conservative option and note it in the final report.**

## Context Management

This pipeline is long. To avoid running out of context, **compact after each phase**:

1. After each phase completes, append a status summary to `.claude/agent_state/specs/<algo_name>.md` (the spec file is your persistent state across compactions):
   ```markdown
   ## Phase N Status
   - Result: PASS/FAIL
   - Key outputs: [files created, branch name, loss values, PR URLs, etc.]
   - Decisions: [any non-obvious choices made]
   ```
2. Then run `/compact` to free context. After compaction, re-read the spec file to recover state before continuing to the next phase.

## Phase 0: Duplicate & Blacklist Check

Before doing any work, check if this method is blacklisted, already implemented, or has an open PR:

1. **Check blacklist**: Read `.claude/agent_state/blacklist.txt`. If the paper URL, arxiv ID, or method name matches any entry (case-insensitive, ignoring comment lines starting with `#`), stop and report "blacklisted".
2. **Check existing models**: `ls src/speculators/models/` — if the algorithm already has a directory, stop and report "already implemented".
3. **Check open PRs on speculators**:
   ```bash
   gh pr list --repo vllm-project/speculators --state open --json title,url --limit 50
   ```
   If any PR title matches the method name (case-insensitive), stop and report "open PR already exists" with the PR URL.
4. **Check open PRs on vLLM**:
   ```bash
   gh pr list --repo vllm-project/vllm --state open --search "<algo_name>" --json title,url --limit 20
   ```
   Same — stop if a matching PR exists.

If a duplicate is found, write the reason to `.claude/agent_state/last_run_report.md` and exit cleanly (exit code 0). The wrapper script will still post to Slack with the report explaining why it was skipped.

## Phase 1: Analyze the Paper

Follow the instructions in `.claude/skills/analyze-paper.md` but skip the "Present to User" step. Write the spec to `.claude/agent_state/specs/<algo_name>.md` and continue immediately.

Key decisions to make autonomously:
- **Algorithm name**: Use the paper's own name (lowercased, hyphen-separated). If unclear, derive from the method name.
- **Inheritance vs standalone**: If the method is clearly a variant of eagle3/dflash/mtp, extend that class. Otherwise create standalone inheriting from `DraftVocabMixin, SpeculatorModel`.
- **vLLM pathway**: Default to mapping to the closest existing proposer (EagleProposer for autoregressive, DFlashProposer for parallel). Only create a new proposer if the inference method is fundamentally incompatible.

## Phase 2: Implement

Follow the instructions in `.claude/skills/implement-speculator.md` but skip the "Present Results" step. Continue immediately after validation passes.

Key autonomous decisions:
- Create a feature branch **from main**: `git checkout main && git pull origin main && git checkout -b feat/<algo_name>`
- After writing code, run `make style && make quality`. If lint fails, fix and retry up to 3 times.
- Run `pytest tests/unit/ -x` to verify registration works. If tests fail, fix and retry up to 3 times.
- If you hit a problem you truly cannot resolve after 3 attempts, log it and continue to training anyway (it will surface the real error).

## Phase 3: Smoke Training

Follow the instructions in `.claude/skills/train-speculator.md` but skip the "Report Results" step. Collect results for the final report.

Key autonomous decisions:
- **Verifier model**: Pick the smallest model available locally. Check cache first:
  ```bash
  ls /root/.cache/huggingface/hub/ 2>/dev/null | grep -E 'qwen|llama' | head -5
  ```
  If nothing cached, use `Qwen/Qwen3-0.6B` (smallest, fastest to download).
- **Training mode**: Use offline mode (no vLLM server) unless the algorithm specifically requires online generation. This is simpler and uses fewer GPUs.
- **Max steps**: 100 steps. Enough to see if loss decreases.
- **On failure**: If training crashes, read the traceback, attempt a fix, and retry once. If it fails again, include the traceback in the final report.

## Phase 4: Create Draft PR

**IMPORTANT — hard requirements for PRs:**
- The feature branch MUST be based on `main`. Do NOT branch from or include changes from any other feature branch (e.g. `feat/paper-to-impl-agent`). The PR should only contain the implementation files for this algorithm.
- The PR MUST be created as a **draft** (`--draft` flag). Never create a ready-for-review PR.
- The PR title MUST start with `[autopilot]`.
- Do NOT include `.claude/` or `scripts/` files in the PR — only implementation code.

After implementation and training, commit all changes and open a draft PR:

1. **Stage and commit** all new and modified files in `/workspace/speculators`:
   ```bash
   cd /workspace/speculators
   git add -A src/speculators/models/<algo_name>/ scripts/train.py src/speculators/models/__init__.py
   git commit -s -m "feat(<algo_name>): implement <algo_name> speculative decoding method"
   ```

2. **If vLLM files were changed**, commit those too:
   ```bash
   cd /workspace/vllm
   git checkout -b feat/<algo_name>
   git add -A vllm/transformers_utils/configs/speculators/ vllm/model_executor/models/ vllm/config/ vllm/v1/worker/
   git commit -s -m "feat(<algo_name>): add <algo_name> draft model support"
   ```

3. **Push and create draft PR** on the speculators repo. **The title MUST start with `[autopilot]`** — this identifies auto-generated PRs. **Capture the PR URL** from `gh pr create` output:
   ```bash
   cd /workspace/speculators
   git push -u my-fork feat/<algo_name>
   SPEC_PR_URL=$(gh pr create --draft --repo vllm-project/speculators \
     --title "[autopilot] feat(<algo_name>): implement <algo_name> speculative decoding" \
     --body "$(cat <<'PREOF'
   > **Note:** This PR was auto-generated by the [`/autopilot` skill](https://github.com/vllm-project/speculators/tree/main/.claude/skills/autopilot). It requires human review before merging.

   ## Summary
   - Implements the <algo_name> speculative decoding method from [paper title](paper_url)
   - Adds config, model, and training support in speculators
   - Adds vLLM integration (config translation + model registry)

   ## Paper
   <paper_url>

   ## Smoke Training Results
   - Verifier: <model>
   - Steps: <N>
   - Initial loss: X.XXX -> Final loss: X.XXX
   - Verdict: PASS/FAIL

   ## Test plan
   - [ ] `make quality` passes
   - [ ] Unit tests pass
   - [ ] Smoke training shows decreasing loss
   - [ ] Full training run on target model
   - [ ] vLLM inference validation

   🤖 Generated with [Claude Code](https://claude.ai/code) via the `/autopilot` skill
   PREOF
   )")
   ```

4. **If vLLM changes exist**, push and create a draft PR there too. **Capture the PR URL**:
   ```bash
   cd /workspace/vllm
   git push -u my-fork feat/<algo_name>
   VLLM_PR_URL=$(gh pr create --draft --repo vllm-project/vllm \
     --title "[autopilot] feat(<algo_name>): add <algo_name> draft model support" \
     --body "$(cat <<'PREOF'
   > **Note:** This PR was auto-generated by the [`/autopilot` skill](https://github.com/vllm-project/speculators/tree/main/.claude/skills/autopilot). It requires human review before merging.

   ## Summary
   - Adds <algo_name> draft model for speculative decoding
   - Companion to speculators PR: <speculators_pr_url>

   ## Test plan
   - [ ] Model loads and runs inference
   - [ ] Speculative decoding produces valid tokens

   🤖 Generated with [Claude Code](https://claude.ai/code) via the `/autopilot` skill
   PREOF
   )")
   ```

5. **Write PR URLs to a file** so the wrapper script can include them in Slack notifications:
   ```bash
   cd /workspace/speculators
   mkdir -p .claude/agent_state
   python3 -c "
   import json
   urls = {}
   speculators_pr = '${SPEC_PR_URL:-}'
   vllm_pr = '${VLLM_PR_URL:-}'
   if speculators_pr: urls['speculators'] = speculators_pr
   if vllm_pr: urls['vllm'] = vllm_pr
   with open('.claude/agent_state/last_run_prs.json', 'w') as f:
       json.dump(urls, f, indent=2)
   "
   ```

## Phase 5: Final Report

Write a concise report to `.claude/agent_state/last_run_report.md`. This gets embedded in a Slack message, so:
- **Use Slack mrkdwn**: `*bold*` for headers (NOT markdown `##` or `###`), `\`code\`` for inline code
- **Keep it under 1500 characters** — Slack blocks have a hard limit
- **Be terse**: one line per item, no verbose explanations

```bash
cd /workspace/speculators
cat > .claude/agent_state/last_run_report.md <<'REPORT'
<paste the report content here>
REPORT
```

Report template (follow this format exactly):

```
*<algo_name>* — <one-sentence method summary>

*Implementation*: branch `feat/<algo_name>` | lint PASS/FAIL | tests PASS/FAIL
*Training*: <verifier> | <N> steps | loss <X.XX> → <X.XX> | PASS/FAIL
*vLLM*: registered as `<method>`, maps to <proposer>

*Decisions*: <1-2 key choices, comma-separated>
*Risks*: <1-2 key risks, comma-separated>
```

## Error Recovery

If any phase fails catastrophically (not fixable in 3 retries):
1. Log what happened and what was tried
2. Skip to the final report
3. Mark the failed phase clearly
4. Include the error traceback
5. Suggest what the user should look at to fix it

The goal is to always produce a report, even if the implementation is incomplete. A partial report with clear failure notes is more useful than silence.
