---
name: train-ab
description: "A/B a training-code change: train a speculative-decoding drafter on both the base and the head ref with the same config, then compare drafter quality (expected accepted length). Use before merging a branch that touches training, loss, data generation, or model code, to tell whether the change helped, hurt, or did nothing. Triggers on 'did my training change help', 'benchmark this against main', 'A/B the drafter', 'is my branch better than main', 'compare acceptance'. Infers base, head, and algorithm from the branch; defaults to on-policy online Eagle-3 on Qwen3-8B at 5K samples."
---

# Train A/B

Answer one question: **did this branch's training change make the drafter better, worse, or the same as the baseline?** Train the same config on both refs, then compare the quality metric each run writes to disk.

Bounded comparison — two arms, plus one extra seed only to settle a close call. Not an open-ended optimization loop.

## When to use

- Before merging a branch that touches `src/speculators/train/`, `src/speculators/models/`, the loss, or data generation, and you want evidence the change helped.
- "Is my branch better than main?" for drafter acceptance.

Not for pure refactors, inference-only changes, or anything a unit test already covers.

## Workflow

### 1. Infer the plan, then confirm it

Don't present a form. Read the context and propose a filled-in plan; ask only where you are genuinely unsure. A run is long (see Timing), so confirm before launching — but confirm a plan, not a questionnaire.

Infer:

- **Head** — the current branch.
- **Base** — `main`, or the branch's parent if it is stacked.
- **Algorithm** — from what the branch changes. Touches `models/dflash/` means benchmark DFlash; `models/eagle3/` means Eagle-3; a cross-cutting change (loss, trainer, data) means the default, Eagle-3.
- **GPUs** — from what is free (`nvidia-smi`). Online needs a vLLM server plus training, so several.

Default config, overridden only if the branch implies otherwise: on-policy **online** Eagle-3, **Qwen3-8B**, sharegpt, 5000 samples — mirroring `examples/train/eagle3_qwen3_8b_sharegpt_online_5k.sh`. Online because it is on-policy, the faithful training signal, and it keeps no hidden-state cache on disk.

Confirm plainly:

> Plan: base `main` vs head `my-branch`, on-policy online Eagle-3, Qwen3-8B, 5000 samples, vLLM on GPUs 0,1 and training on 2,3, one seed each. Roughly 35 minutes for the pair. Go?

### 2. Run each arm

Each arm must train its **own ref's code**: a clean tree then `git checkout <ref>` (an editable install picks up the source), or a `git worktree` per ref with `PYTHONPATH=<worktree>/src`.

Regenerate the dataset responses **once** (`scripts/response_regeneration/run_all.sh`) and reuse them for both arms — the target model is the same on both refs, so regenerating per arm only adds noise. The exception: if the change under test is in data generation or regeneration, regenerate per arm, because that code is the thing being tested.

For each arm, run the online pipeline from the example above: prepare the data, then train against a live vLLM server with `--on-missing generate --on-generate delete`. Hidden states are generated on the fly and discarded. Send `--save-path` to a per-arm directory and pass the chosen `--seed`. Every flag is identical across arms except the ref.

### 3. Read the metric

Training writes `<save-path>/checkpoint_best/val_metrics.json`. Read it from each arm. Keys carry an `_epoch` suffix:

- `loss_epoch` — always present.
- Eagle-3: `cond_acc_0_epoch`, `cond_acc_1_epoch`, `cond_acc_2_epoch` (per-depth conditional acceptance), plus `full_acc_*_epoch`.
- DFlash and DSpark: `eal_epoch` (expected accepted length, the headline), plus `position_{i}_acc_epoch`.

The headline metric is **expected accepted length (EAL)** — higher means more drafted tokens accepted per step. DFlash writes it directly. For Eagle-3, derive it from the conditional accuracies, using the same definition DFlash uses:

```
eal = 0
cum = 1
for k in 0, 1, 2, ...:        # each drafted depth present in the file
    cum = cum * cond_acc_k_epoch
    eal = eal + cum
```

### 4. Report and judge

Report base, head, and delta for **all three** of the following — never collapse the result to a single number:

1. **EAL** — the headline summary.
2. **The per-position acceptance profile** — `cond_acc_{k}_epoch` per depth for Eagle-3, `position_{i}_acc_epoch` for DFlash and DSpark. This is the checkpoint's real quality: two checkpoints can share an EAL yet accept very differently by depth, and the profile is what shows up at inference. Do not drop it.
3. **`loss_epoch`** — supporting. Flag it if it moves against EAL, but it is not the verdict.

Judge on EAL, honestly — no noise floor is measured by default:

- Delta at least ~5% of base: call it, improvement or regression.
- Smaller than that: "within what a second seed could flip." Do not declare a winner. Offer one more seed per arm and compare the paired deltas. Online generation adds run-to-run variation, so close calls genuinely need this.

```
online Eagle-3, Qwen3-8B, 5000 samples

                   base (main)   head (my-branch)   delta
EAL                   2.70            2.85          +0.15  (+5.6%)
cond_acc depth 0      0.78            0.79          +0.01
cond_acc depth 1      0.55            0.60          +0.05
cond_acc depth 2      0.31            0.39          +0.08
loss                  1.83            1.79          -0.04
=> improvement, concentrated at the deeper draft positions
```

Then emit a compact block the user can paste straight into the PR description — the setting (so the run is reproducible) followed by the result table. Fill in the short commit SHAs so the two refs are pinned:

```markdown
### Drafter A/B

Online Eagle-3, Qwen3-8B, sharegpt, 5000 samples, on-policy regen, seed 42.
base `main` (`abc1234`) vs head `my-branch` (`def5678`).

| metric      | base | head | Δ             |
| ----------- | ---- | ---- | ------------- |
| EAL         | 2.70 | 2.85 | +0.15 (+5.6%) |
| cond_acc d0 | 0.78 | 0.79 | +0.01         |
| cond_acc d1 | 0.55 | 0.60 | +0.05         |
| cond_acc d2 | 0.31 | 0.39 | +0.08         |
| loss        | 1.83 | 1.79 | -0.04         |

**Improvement**, concentrated at the deeper draft positions.
```

## Timing

A run is not quick — tens of minutes each, so a pair is the better part of an hour, and more for the heavier algorithms. Rough per-run end-to-end at the 5K setting:

| Algorithm        | Per run |
| ---------------- | ------- |
| Eagle-3 (online) | ~17 min |
| DFlash (online)  | ~25 min |
| P-EAGLE (online) | ~48 min |

A pair is two of these, plus one more per arm when a close call needs a confirming seed.

## Guardrails

- The arms differ by one thing: the git ref. Same config, same seed, and the same regenerated data — unless a data-generation change is the thing under test. Otherwise the delta is not attributable to the change.
- A small delta is probably run-to-run noise. Confirm a close result with a second seed before claiming a win.
- Report what actually ran. If an arm fell back or ran out of memory, say so — it may not be comparable to the other arm.
