# Training details and decision criteria

Use these criteria to design and compare runs. Derive final values from the selected model, current Speculators and vLLM checkouts, controlled pilots, and the user's goal.

## Draft-family selection

- **Eagle3 / P-EAGLE:** choose when the requested architecture or existing checkpoint uses that family. Pay attention to decoder architecture, normalization choices, recurrent/test-time-training depth, and target hidden-state selection.
- **DFlash:** parallel masked draft blocks. Confirm whether the anchor is a bonus token, the resulting number of proposed tokens, and whether the selected loss is compatible with D-PACE or another position-weighting scheme.
- **DFlash2:** DFlash plus local dynamic convolutions and a candidate selector; optional confidence-head training must match a serving runtime that consumes the same features and realized path.
- **DSpark:** DFlash-family draft with a sequential/Markov head and optional path-conditioned confidence. Confirm Markov rank/head type and sampling-from-anchor semantics.
- **MTP:** use the MTP-specific data/model path and avoid applying DFlash-family attention or block assumptions.

The target and draft architectures need not be identical, but the selected architecture must be supported by both training and the intended inference runtime.

## Hidden-state modes

### Online

Generate hidden states on demand and discard them after consumption. This minimizes persistent storage and stays close to the live extractor, but couples training availability and throughput to the extractor. Size retries conservatively and stop on sustained failures.

### Offline

Generate and validate hidden states before training, then train with missing-state behavior set to fail. This separates failure domains and is easiest to reproduce, but can require large storage and transfer bandwidth. Record exact tensor format, dtype, layer list, and checksums.

### Cached or hybrid

Generate missing states and retain them for later epochs. Use only after projecting capacity for all cached states and proving that concurrent writers and cleanup cannot corrupt the cache. Do not describe a partially filled cache as an offline dataset.

For multi-node extraction, keep transport mechanics explicit: extractor replicas, routing policy, endpoint list, storage visibility, TTL/lease behavior, staging size, and per-rank consumption cursor.

## Response regeneration

Draft training assumes assistant responses are on-policy for the verifier. Reuse response data only when its provenance proves the exact verifier revision, tokenizer/chat template, reasoning policy, sampling configuration, output limit, and source-row mapping. Otherwise regenerate it before preprocessing.

Keep source data, regenerated conversations, rendered/tokenized training data, and hidden-state artifacts as separate immutable stages with separate manifests. This separation makes a changed generation policy or tokenizer visible and prevents stale hidden states from being paired with new text.

For every ablation, reuse the same validated regenerated snapshot unless response-data policy is itself the variable under study. If it is, evaluate each data-policy candidate like any other ablation and disclose that the training examples differ.

## Vocabulary choices

Full vocabulary reduces mapping risk. Reduced vocabulary can save draft memory and compute but requires a token-frequency artifact plus verified draft-to-target and target-to-draft mappings. Verify special tokens, mask token, serialization, training, conversion, and serving together.

Absence of a token-frequency file may be intentional for full-vocabulary training; do not “fix” it without checking the config.

## Losses and optimization

- Record every loss component and weight in the resolved config. A lower aggregate loss can conceal a degraded acceptance-related component.
- Recommend D-PACE for DFlash and DFlash2; verify supported loss components and configuration in the chosen revision. D-PACE requires the compatible CE path in checkouts that enforce this.
- Select loss components and weights from the current implementation and a controlled pilot. Do not inherit a mixture from another model or run without testing it.
- With Muon, distinguish the AdamW/non-matrix learning rate from the Muon matrix learning rate. Some versions default Muon LR to a multiple of `lr`; set both explicitly when reproducibility matters.
- Clarify scientific notation. `1e3` and `1e-3` differ by six orders of magnitude; do not launch through an ambiguous instruction.
- Preserve the current trainer's master-weight and hidden-state dtype contract. Do not select float16 if the implementation requires gradient scaling that it does not provide.

## Model and layer configuration

Auxiliary layer selection is an experimental input. Prefer evenly distributed layers as a starting point only when no validated selection exists. For a new target:

1. identify the actual number and type of target layers;
2. confirm the extractor's layer-index convention;
3. test a small, evenly spaced set and architecture-informed candidates;
4. compare pilots under the same data, seed, optimizer, and evaluation;
5. promote based on held-out and serving metrics, not training loss alone.

The launcher may append the final target state. Inspect the current implementation and pass the auxiliary IDs according to its contract so the final state is neither omitted nor duplicated.

## Common ablation protocol

Apply this protocol to layer, learning-rate, loss, data, vocabulary, architecture, head, initialization, and other learning-based comparisons before committing to a full production allocation.

### Define candidates

Build a small, interpretable candidate matrix and state the hypothesis for each candidate. For layers, read the [current layer-selection guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/hyperparameter_selection/hidden_state_layer_id_selection/#recommended-layer-selection-strategy), inspect `scripts/sample_layer_candidates.py` when available, and consider uniform, deep, shallow, tails, center, and architecture-aware sets. Validate bounds, uniqueness, ordering, index semantics, and architectural restrictions. Do not select top individual layers independently and concatenate them.

Keep model capacity constant during a primary comparison when possible. If candidate parameter counts or extractor costs differ, record those differences and do not attribute the entire result to the named variable.

### Freeze the screen

Use the same 30,000 unique, representative training samples for one complete epoch for every comparable candidate. Use deterministic sample IDs and a separate frozen held-out evaluation set. Hold constant:

- verifier snapshot, tokenizer, chat template, regenerated-data snapshot, and vLLM build;
- draft family, layer count, hidden size, vocabulary, block size, and optional heads unless one is the ablated variable;
- data order, train/validation policy, seed, sequence length, and hidden-state mode;
- optimizer, learning rates, scheduler, batch construction, anchors, loss weights, and world size unless one is the ablated variable;
- serving and evaluator configuration.

Run the configuration and one-batch gates before each candidate. Use independent save paths and explicit run names. Capture held-out loss components, acceptance proxies, per-position accuracy, step time, peak memory, and extractor throughput.

The 30,000-sample/one-epoch budget is a comparative screen, not evidence that a checkpoint is production-ready. Keep the configuration dry run and one-batch sanity check separate and cheaper. If this pilot is infeasible, stop and ask the user to approve a common alternative budget rather than silently changing it.

### Evaluate every candidate

After the epoch completes, launch a clean instance of the intended vLLM runtime with the selected checkpoint. Use the sibling [Speculators evaluation skill](../../speculators-evaluation/SKILL.md) for every candidate under the same acceptance-only dataset matrix, prompt set, sampling, speculative-token count, and serving settings.

A candidate passes only if:

- target and draft load without unexplained weight, shape, architecture, or fallback errors;
- health and model-list endpoints identify the intended model;
- a real completion succeeds and acceptance counters prove speculative decoding is active;
- pooled acceptance rate, mean acceptance length, per-position acceptance, and request failures are present;
- exact serving and evaluator provenance is retained.

Define “expected acceptance” before reading results. Prefer a same-target, same-runtime, same-prompt baseline or an agreed threshold. For a new target with no trusted baseline, require clear improvement over an untrained or existing reference, non-collapsed later positions, correct outputs, and repeatable results, then set a provisional gate for a larger confirmation run.

Run the evaluation skill's matched speedup workflow on finalists when deployment performance matters. Do not infer speedup from acceptance length.

### Promote deliberately

Reject candidates that cannot serve or lack complete evaluation artifacts, even if validation loss is best. Rank passing candidates using held-out and serving acceptance first, then measured end-to-end throughput and operational cost. Run a larger confirmation before finalizing when candidates are close or the screen ordering is noisy.

Write a comparison table containing the exact ablated values, controlled variables, pilot sample count, checkpoint selected, held-out metrics, acceptance metrics, measured throughput where available, memory, provenance links, and pass/fail reason. Preserve failed candidates as labeled diagnostics rather than overwriting them. Present results and a recommendation to the user, and wait for their choice before full training unless they explicitly delegated selection. State whether full training starts fresh or resumes the selected pilot; never carry over pilot optimizer state implicitly.

## Common failure patterns

- Training and serving can emit materially different hidden states when their verifier builds or model implementations differ. Matching token IDs and tensor shapes does not prove parity; use exact-input numerical comparison.
- Checkpoint conventions can change across code revisions. Do not reuse weights or optimizer state across incompatible schemas merely because most tensor names match.
- Ambiguous scientific notation or implicit optimizer defaults can change the effective learning rate by orders of magnitude. Record resolved values explicitly.
- Distributed extraction can appear healthy while leases expire, transfers fail, or samples are dropped. Monitor delivered training progress and bounded failure counters, not only endpoint health.
- A resumed distributed run can duplicate or skip data without checkpointed, consistent per-rank sampler cursors. Verify cursor state before recovery.
- Training loss can improve while serving acceptance or throughput regresses. Never complete an ablation without the common evaluation.

Translate these patterns into checks for the current run without importing another run's thresholds, topology, or configuration.
