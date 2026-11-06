# Kickoff and operations playbook

Use this reference for concrete training operations. Adapt paths and orchestration to the environment; do not copy example hostnames or ports.

## 1. Inventory before changing state

Inspect the exact target hosts and checkout:

- scheduler allocations and remaining wall-time;
- GPU model, count, topology, VRAM, utilization, and compute PIDs;
- free disk space and inodes on data, checkpoint, cache, and temporary filesystems;
- listening ports, running containers, terminal sessions, and launchers;
- repository branch, commit, dirty state, submodules, active worktrees, and installed package source;
- Python, PyTorch, CUDA, vLLM, Transformers, Speculators, and logging-backend versions;
- model snapshot availability and whether network access will be needed after launch.

Do not assume an idle-looking GPU is available; correlate the scheduler, GPU processes, containers, and user ownership. Never use broad process termination, global container cleanup, or another run's session name.

Assign a unique run ID. Map each component to explicit hosts, GPU IDs, ports, directories, and process/container/session names. Verify every selected port is unused.

## 2. Freeze inputs and provenance

Create the run root before launch, but fail if a supposedly new run root already contains checkpoints. Capture:

- Speculators and vLLM commit SHAs plus dirty status or patch files;
- model repository ID and immutable snapshot/revision;
- source and regenerated dataset revisions, generation policy, merge manifest, and checksums;
- launch scripts/config hashes;
- container image ID/digest and bind mounts;
- package inventory;
- GPU topology, allocation ID, and hostnames;
- the resolved configuration, not only the shorthand command.

Use strict error handling in shell launchers. Put logs on durable storage and set unbuffered output. Keep caches and temporary hidden states on filesystems sized for them.

## 3. Regenerate and validate response data

Decide explicitly whether the training responses are regenerated or reused. Regenerate by default unless the existing artifact proves that every assistant response came from the exact verifier revision under the intended chat template, reasoning policy, sampling parameters, token limit, and seed policy. A dataset with unknown or mismatched response provenance is off-policy even if its schema is valid.

Regeneration is distinct from chat rendering/tokenization and hidden-state extraction. Record the source row identity before generation, keep a deterministic source-to-output mapping, and write each shard to a temporary location before publishing a frozen manifest. Never let training read shards that are still being generated or merged.

Validation must reject:

- malformed or missing required fields;
- conflicting duplicates and unexpected identical duplicates;
- token/input/loss-mask length mismatches;
- out-of-range source indices or nondeterministic shard assignment;
- tokenizer or chat-template mismatch;
- non-finite values or corrupt Arrow/JSONL records;
- overlap with held-out evaluation data when that matters.

Record conversation count, total tokens, trainable/generated tokens, length distribution, per-shard checksums, generator provenance, and a durable completion marker. Prepare data with the exact verifier tokenizer and intended sequence length.

## 4. Validate extractor parity

Start the extractor only on assigned GPUs. Make its virtual-environment tools discoverable to worker processes when extensions compile at startup. Wait for a real health endpoint and fail if the process exits before becoming healthy.

Confirm the endpoint exposes the exact auxiliary layers expected by training. If a launcher appends the final verifier layer automatically, verify the current implementation and keep the user-specified list consistent on both sides.

When training and serving use different machines, containers, vLLM versions, patches, or optimized model implementations, run an exact-input hidden-state parity test. Compare every state consumed by the draft with cosine similarity, relative error, and maximum error. Material divergence invalidates model-quality comparisons even if token IDs and shapes match.

For distributed hidden-state transport, validate with a disposable canary service before production. Check routing, TTL, staging capacity, writer concurrency, request timeouts, and recovery behavior. Do not probe a live production service with a temporary client that can consume leases or mutate state.

## 5. Run gates in increasing cost order

### Configuration dry run

Build the draft and save initialized weights without training when the checkout supports it. Verify resolved architecture, auxiliary-layer order, vocabulary mapping, attention semantics, optional heads, and that vLLM can load the initialized checkpoint.

### One-batch training dry run

Use one step and a new scratch save path. Require a successful forward/backward pass on every rank, finite metrics and gradients, healthy hidden-state delivery, a checkpoint that saves/reloads, a serving smoke request, and expected memory headroom.

### Extended smoke and ablation pilot

After the configuration and one-batch gates pass, use the same frozen 30,000 unique training samples for one complete epoch for layer selection and other learning-based ablations. Keep a separate frozen held-out prompt set; do not report training-set acceptance as validation. Every candidate must use the same data order, seed, tokenizer, sequence length, optimizer, learning rates, vocabulary, draft architecture, and evaluation setup. Change only the variable under study.

Record held-out metrics at the end of the epoch and enough training metrics during the run to detect divergence, instability, or a misleading final point. Then start the intended vLLM build from a clean process with the candidate checkpoint. Require successful model loading, health and model-list endpoints, a real completion request, and proof that the speculative path is active.

Invoke the sibling [Speculators evaluation skill](../../speculators-evaluation/SKILL.md) for every candidate. Use a predeclared acceptance-only matrix for the common ablation screen, and preserve its raw artifacts and provenance. A candidate that trains successfully but cannot serve, silently falls back, lacks evaluation artifacts, or misses its acceptance gate does not advance. Run matched speedup evaluation on finalists when speedup is part of the promotion goal.

If an intermediate checkpoint is selected instead of the end-of-epoch checkpoint, evaluate both under the same serving harness and state which predeclared selection rule chose the promoted checkpoint. Do not tune against a single noisy sample or promote a pilot merely because training loss decreased.

## 6. Launch the full run

Before launch, recheck that inputs still match their hashes and resources remain free. Make process ownership unambiguous:

- use a scheduler job, container name, or terminal session unique to the run;
- redirect output to durable, timestamped logs;
- use a supervisor arrangement that observes the real training process;
- add cleanup traps only for child processes started by this launcher;
- write `STARTED` only after prerequisites pass and `COMPLETE` only after the final checkpoint is verified;
- preserve a failure marker and exit code on error.

For online training, verify both progress and extractor traffic. A healthy HTTP endpoint with no completed training steps is not sufficient. Bound retries and use a circuit breaker for repeated generation failures; do not silently drop an unbounded portion of the dataset.

## 7. Monitor without perturbing the run

Check these together:

- live trainer, rank, extractor, transport, and logger processes;
- latest global step, epoch, denominator, and change since the prior observation;
- finite total loss and component losses, acceptance rate/length, and per-position metrics;
- sustained seconds per optimizer step over a substantial timestamped window;
- GPU utilization/memory and CPU/dataloader pressure;
- extractor health, request throughput, retries, expired leases, transfer failures, dropped samples/batches, OOMs, and tracebacks;
- newest completed checkpoint, age, stable files, training state, and per-rank sampler cursors;
- logger freshness and allocation time remaining.

Estimate training-only ETA separately from validation, checkpoint, queue, and downtime overhead. Recompute the step denominator after epoch transitions or resume; do not reuse stale assumptions.

Do not declare a checkpoint complete while files are changing. Require a completion log and stable, nonempty model/config/optimizer/scheduler/training-state files as applicable.

## 8. Resume or recover

Classify the requested continuation:

- **Fresh:** new initialization and optimizer/scheduler; use a clean save path or disable auto-resume.
- **Weight-only warm start:** load compatible draft weights, intentionally reset optimizer/scheduler, and disable auto-resume from the destination.
- **Exact resume:** restore model, optimizer, scheduler, epoch/global step, and distributed sampler cursors from one complete checkpoint.

Before resuming, inspect the command/config embedded in the checkpoint and compare code revision, architecture, vocabulary, layer IDs, sequence/block settings, world-size assumptions, and checkpoint format. Never coerce checkpoints across known-incompatible conventions just because tensor names mostly match.

If a distributed transport or extractor failed, first preserve logs and state, then identify whether replay is safe. Do not restart automatically when doing so could duplicate or skip samples. Require an explicit recovery plan when cursor consistency is uncertain.
