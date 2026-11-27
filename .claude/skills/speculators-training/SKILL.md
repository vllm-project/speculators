---
name: speculators-training
description: Plan, launch, monitor, resume, and validate Speculators drafter training, including response regeneration, hidden-state extraction, and training ablations. Use for Eagle-3, DFlash, DFlash2, DSpark, or MTP; use speculators-evaluation for inference-only benchmarking.
---

# Speculators training

Use the selected checkout's [training tutorial](../../../docs/user_guide/tutorials/train.md), CLI help, and config schema as the source of truth for supported architectures, losses, connectors, and flags. Prefer the tutorial's separate Speculators and vLLM environments.

## Resolve the run

Reuse the user's recipe and established configuration. Ask only for unresolved choices that affect cost or the experiment:

- verifier and dataset revisions, response-generation policy, and train/held-out split;
- draft family, depth, target hidden-state count, initialization, vocabulary, and loss;
- online or offline hidden states, Speculators/vLLM revisions, assigned hosts/GPUs, and connector or storage location;
- epochs, sequence length, effective batch size, optimizer/LRs, pilot budget, output path, fresh versus resume, and tracker destination.

Check architecture support in both the trainer and serving runtime. MTP requires a verifier with native MTP layers and uses the final hidden state; do not apply multi-layer selection to it. For DFlash/DFlash2, recommend D-PACE where supported and check its compatible CE configuration. With Muon, resolve the matrix and AdamW learning rates separately. Reduced vocabulary requires mappings that survive training, serialization, and serving.

Planning requests produce a plan; execute pilots or full training only within the user's launch authorization. Delegating candidate selection alone does not authorize execution. Read-only monitoring does not imply recovery or restart.

## Prepare data and extraction

Assistant responses must come from the intended verifier and generation policy. Reuse them only with validated provenance; otherwise regenerate before preprocessing. Regeneration, chat rendering/tokenization, and hidden-state extraction are separate stages. Freeze validated artifacts before training, retaining source-row identities and generation settings.

Use `scripts/launch_vllm.py` for extraction and pass `--provenance-dir` as required by `AGENTS.md`. For online training, keep the extractor available and verify the selected connector works across trainer hosts. For offline training, generate complete compatible states first and check storage capacity and visibility.

Match extractor and trainer verifier revision, tokenizer/chat template, sequence length, layer IDs/order, hidden-state format, and dtype. Inspect the launcher's final-layer convention so the final state is neither omitted nor duplicated. If extraction and serving use different vLLM builds or model implementations, compare all consumed hidden states numerically on identical inputs; token IDs and tensor shapes alone do not establish parity.

## Select a configuration

For layer selection, follow the checkout's [layer-selection guide](../../../docs/user_guide/hyperparameter_selection/hidden_state_layer_id_selection.md):

1. Generate spaced candidate lists with `scripts/sample_layer_candidates.py`, comparing lists with the same hidden-state count.
2. Screen candidates with small runs, then train the strongest K candidates at medium scale before choosing a winner.
3. Use the guide's starting budgets of roughly one epoch on 4,000 samples, then one epoch on 64,000 samples; adjust to the available data and authorized compute. Rankings across different hidden-state counts may change with training scale.

For LR, loss, or other ablations, agree on a comparable budget rather than imposing the layer-selection budget universally. Hold data, seed, architecture, optimizer, and runtime settings fixed except for the variable under study. Response-policy ablations may use distinct validated response snapshots, but must use matching source rows and training budgets. Keep evaluation prompts separate from training data.

Use the sibling [Speculators evaluation skill](../speculators-evaluation/SKILL.md) for every trained candidate before selection: use a common held-out acceptance screen in the intended vLLM runtime, verify speculation is active, and retain raw results and provenance. Compare acceptance rate/length, per-position acceptance, failures, and training cost. Measure matched end-to-end speedup on finalists when deployment performance is the goal; acceptance length alone does not establish speedup.

Present the candidate settings, budgets, checkpoints, metrics, and recommendation in a comparison table. Let the user select the full-run configuration unless selection was delegated. State whether full training starts fresh or continues a pilot.

## Launch and continue

Before a full run, validate model construction, a forward/backward step, checkpoint save/reload, and loading in the intended vLLM runtime. Disclose necessary Speculators/vLLM patches before applying them and retain the diffs with run provenance.

Distinguish continuation modes explicitly:

- **Fresh or weight-only warm start:** use a clean destination or disable auto-resume; reset optimizer/scheduler intentionally.
- **Exact resume:** restore a complete compatible checkpoint, including optimizer, scheduler, global step/epoch, and distributed sampler cursors. Check code/config compatibility before recovery.

Share the actual tracker link after launch, or the durable local metrics/log path if no dashboard is available. Monitor advancing steps, finite metrics, extractor delivery/failures, and completed checkpoints together. Report sustained throughput and ETA with validation/checkpoint overhead distinguished.

Keep the resolved command/config, code revisions and patches, model/data identities, checkpoints, and evaluation artifacts with the run. Use the repository's built-in provenance outputs rather than inventing a parallel reporting system. Hand off the run location, progress, selected checkpoint, evaluation outcome, and any unresolved failures.
