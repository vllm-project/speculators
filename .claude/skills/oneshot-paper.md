---
name: oneshot-paper
description: End-to-end autonomous pipeline — reads a speculative decoding paper, implements it in speculators + vLLM, and runs a smoke training. No human checkpoints.
---

# One-Shot Paper-to-Training Pipeline

You are running the full autonomous pipeline for a speculative decoding paper. The user provides a URL (arxiv, HF model, blog post) and you do everything end-to-end without asking for confirmation. Report results at the very end.

**IMPORTANT: Do not stop to ask the user questions. Make reasonable decisions and document them. If something is ambiguous, pick the most conservative option and note it in the final report.**

## Phase 1: Analyze the Paper

Follow the instructions in `.claude/skills/analyze-paper.md` but skip the "Present to User" step. Write the spec to `.claude/agent_state/specs/<algo_name>.md` and continue immediately.

Key decisions to make autonomously:
- **Algorithm name**: Use the paper's own name (lowercased, hyphen-separated). If unclear, derive from the method name.
- **Inheritance vs standalone**: If the method is clearly a variant of eagle3/dflash/mtp, extend that class. Otherwise create standalone inheriting from `DraftVocabMixin, SpeculatorModel`.
- **vLLM pathway**: Default to mapping to the closest existing proposer (EagleProposer for autoregressive, DFlashProposer for parallel). Only create a new proposer if the inference method is fundamentally incompatible.

## Phase 2: Implement

Follow the instructions in `.claude/skills/implement-speculator.md` but skip the "Present Results" step. Continue immediately after validation passes.

Key autonomous decisions:
- Create a feature branch: `git checkout -b feat/<algo_name>`
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

## Phase 4: Final Report

Present a single comprehensive report:

```markdown
## One-Shot Implementation Report: <algo_name>

### Paper
- Title: ...
- URL: ...
- Method summary: <2-3 sentences>

### Classification
- Type: new-method | variant-of-<parent>
- Closest existing: <model>
- Strategy: standalone | extends <parent>

### Implementation
- Branch: feat/<algo_name>
- Files created: [list]
- Files modified: [list]
- Lint: PASS/FAIL
- Unit tests: PASS/FAIL

### vLLM Integration
- Registered as: <method_name>
- Maps to proposer: <proposer>
- Files touched: [list]

### Smoke Training
- Verifier: <model>
- Steps completed: N
- Initial loss: X.XXX
- Final loss: X.XXX
- Verdict: PASS (loss decreased) / FAIL (reason)

### Decisions Made
- [List any non-obvious decisions and why]

### Issues & Risks
- [List anything that needs human review]
- [List assumptions that may be wrong]

### What to Review
1. `git diff main..feat/<algo_name>` — full implementation diff
2. `.claude/agent_state/specs/<algo_name>.md` — the implementation spec
3. Training logs at `./output/checkpoints/<algo_name>_smoke/`
```

## Error Recovery

If any phase fails catastrophically (not fixable in 3 retries):
1. Log what happened and what was tried
2. Skip to the final report
3. Mark the failed phase clearly
4. Include the error traceback
5. Suggest what the user should look at to fix it

The goal is to always produce a report, even if the implementation is incomplete. A partial report with clear failure notes is more useful than silence.
