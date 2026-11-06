---
name: speculators-training
description: Plan, preflight, launch, monitor, resume, and validate training for vllm-project/speculators draft models, including response regeneration, hidden-state extraction, distributed training, checkpoints, ablations, and serving parity. Use for Eagle, DFlash, DFlash2, DSpark, P-EAGLE, or MTP training operations; do not use for inference-only benchmarking unless it validates a trained checkpoint.
---

# Speculators training

Treat a training run as a reproducible pipeline, not a command. Establish the exact data, verifier, extractor, trainer, checkpoint, and evaluation identities before consuming GPUs.

## Route the task

- For planning, launching, resuming, or monitoring a run, read [references/kickoff-playbook.md](references/kickoff-playbook.md).
- For model-family choices, data modes, hyperparameters, and layer-selection methodology, read [references/training-details.md](references/training-details.md).
- For evaluation, promotion, or post-run handoff, read [references/evaluation-and-provenance.md](references/evaluation-and-provenance.md).
- Every learning-based ablation includes evaluation. Use the sibling [Speculators evaluation skill](../speculators-evaluation/SKILL.md) to construct and execute that evaluation rather than defining a second benchmark workflow here.

Read the current checkout's `AGENTS.md`, training documentation, `speculators train --help`, and config schema before composing flags. Speculators evolves quickly; the checkout is authoritative when it conflicts with this skill's generic guidance.

## Establish the run contract

For a new run, prompt for every unresolved item below in a concise, grouped intake. Reuse explicit answers already supplied in the request or established run configuration; do not ask again. Do not start this questionnaire for a read-only monitoring task or a narrow resume request whose configuration is already known.

01. Verifier name/path: Hugging Face repository or local checkpoint, with revision when available.
02. Training algorithm: discover supported algorithms in the selected checkout's `src/speculators/models` and registry. Explain relevant compatibility, experimental status, or deprecation; do not treat every subdirectory as an algorithm.
03. Loss function: recommend D-PACE for DFlash and DFlash2, while checking the selected revision's supported loss configuration and compatible CE path.
04. Training dataset: Hugging Face repository or local JSONL, including revision/split when available.
05. Response-data policy: regenerate with the verifier or reuse an existing artifact whose on-policy provenance can be validated. For regeneration, resolve the generator revision, sampling/reasoning policy, maximum tokens, seed, source mapping, and output location.
06. Number of full-training epochs.
07. Machine(s) to use and assigned GPU resources.
08. Speculators version/branch; resolve it to a commit for the run.
09. Number of transformer layers in the drafter.
10. Number of target-layer hidden states consumed by the drafter. Clarify whether this count includes an automatically appended final state; validate algorithm restrictions such as MTP's final-state input.
11. Online or offline hidden-state training. Discuss cached/hybrid only if requested or relevant.
12. Monitoring backend, such as Trackio, W&B, TensorBoard, or MLflow, plus project/run destination.

For offline training, also ask for the hidden-state storage path and whether compatible states already exist or must be generated. Check capacity and access from every trainer host. For online training, also ask for the vLLM version/branch and connector backend; recommend Mooncake for multi-node transport after checking support and network prerequisites. Record vLLM provenance for offline extraction and evaluation too.

Resolve sequence length, effective batch size/gradient accumulation, optimizer, draft initialization/vocabulary, train/held-out split, pilot budget, checkpoint/output directory, fresh versus resume semantics, and wall-time/stop condition before launching. Offer supported defaults where appropriate and ask for missing choices that affect cost or scientific meaning.

Do not silently infer a high-cost choice such as epoch count, full-vocabulary training, regeneration sampling policy, public upload, or extra hosts. If the user supplied a launch recipe, verify it rather than redesigning it without evidence.

## Required training workflow

Follow the selected revision's [training tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/) and the operations references:

1. Set up environments first using [Step 0](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/#step-0-setup-your-environment). Prefer separate Speculators and vLLM environments; install the requested tracker in the trainer environment and verify its configuration.
2. Resolve response regeneration before preprocessing or ablations. Default to regeneration when the existing assistant responses cannot be proven to come from the exact verifier revision under the intended generation policy. Regeneration, chat rendering/tokenization, and hidden-state extraction are different phases; completing one does not substitute for another. Freeze and validate the regenerated artifact before downstream work.
3. Run preprocessing using [Step 1](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/#step-1-prepare-your-data). Validate an existing prepared artifact before reuse and reject off-policy or provenance-unknown responses.
4. Use the selected checkout's `scripts/launch_vllm.py` for the extractor, following [Step 2](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/#step-2-launch-vllm-server). Inspect its help and layer conventions rather than guessing flags. In offline mode, generate and validate the required hidden states before pilots; online mode keeps the extractor available during training.
5. Suggest several valid layer-ID combinations for the user's requested hidden-state count, following the [layer-selection guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/hyperparameter_selection/hidden_state_layer_id_selection/#recommended-layer-selection-strategy). Run each on the same frozen 30,000 samples for one epoch, then evaluate every candidate through the sibling evaluation skill in the intended vLLM runtime.
6. Run comparable learning-rate and other ablations under the same 30,000-sample, one-epoch protocol. Preserve controlled conditions and evaluate every candidate; an ablation without evaluation artifacts is incomplete.
7. Summarize all candidates in a comparison table and recommend a configuration. Let the user choose the layer/LR combination before full training; wait for their selection unless they explicitly delegated that choice. Then launch the requested epochs following [Step 4](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/#step-4-train), after the existing validation gates pass.

Always tell the user before patching Speculators or vLLM: explain the reason and expected effect, and preserve the diff and code identity. Disclose pre-existing patches in the run summary. After each pilot or full training run starts and the tracker registers it, share the actual monitoring dashboard/run link. If no dashboard is available, state why and provide the durable local metrics/log location; never invent a URL or silently switch backends.

## Preserve authorization boundaries

A request to launch on a named machine authorizes normal setup and launch work on that machine. It does not authorize killing unrelated processes, taking another user's allocation, destructive cleanup, changing shared branches, publishing artifacts, or expanding to additional machines. Inspect first. Stop if the requested resources are occupied and no safe assigned alternative exists.

For read-only monitoring, never restart, retune, finalize, or create test clients against production services. Report evidence and the safest next decision.

## Non-negotiable invariants

1. **Train/serve hidden-state parity.** Pin the verifier snapshot, tokenizer, vLLM commit/build, model implementation path, auxiliary layer IDs and order, final-layer convention, and hidden-state dtype. Run identical-input parity when the training extractor and serving runtime are not literally the same build and code path. Matching token IDs and tensor shapes is insufficient.
2. **Fresh and resumed runs are distinct.** Use a new save path or explicitly disable auto-resume for a fresh or weight-only warm start. Resume only from a complete compatible checkpoint with the intended optimizer, scheduler, global step, epoch, and sampler state.
3. **Regenerated and prepared data are complete before training.** Require a durable completion marker, strict schema validation, deterministic deduplication, conflict rejection, checksums, token/count statistics, and generator provenance. Never train against manifests that are still being appended.
4. **Extractor and trainer agree exactly.** The layer IDs, tokenizer, verifier revision, maximum length, hidden-state format, and endpoint/storage protocol must match. Remember that some launchers append the verifier's final layer; pass the training-side list according to the current checkout's contract.
5. **Dry-run, one-batch, pilot, and evaluation gates precede the full run.** Validate construction, forward/backward, checkpoint save/reload, finite metrics, hidden-state delivery, and serving compatibility before spending the full allocation. For layer selection and other learning-based smoke tests, use the common 30,000-sample, one-epoch protocol, then require a clean vLLM load and controlled evaluation for every candidate before promotion.
6. **Every long run is observable and recoverable.** Write logs and resolved configuration to durable storage, use unique process/container/session names, create stable checkpoints, and record enough state to distinguish progress from a hung pipeline.

## Decision standards

- Prefer immutable revisions and local snapshot paths over floating model or dataset names.
- Prefer full vocabulary unless reduced-vocabulary memory or speed is essential and the mapping artifacts are verified end to end.
- Use FSDP only when the draft does not fit per rank or measured scaling justifies it; it adds checkpoint and recovery complexity.
- Treat Muon and AdamW learning rates as separate values. Record both explicitly when using Muon; do not rely on a changing default ratio.
- Choose the hidden-state mode based on storage, network, and repeatability. Online generation minimizes storage; offline generation isolates training and is most reproducible; cached/hybrid mode trades storage for reuse. Do not call one universally best.
- Use sustained windows of hundreds of timestamped steps for throughput and ETA. Include validation and checkpoint overhead separately.

## Output of a kickoff task

Leave behind, in the run root or a user-approved staging directory:

- an immutable or hashed launch script/config;
- a preflight report and resource map;
- provenance for code, models, regenerated/prepared data, runtime, container, and hardware;
- the exact resolved training command;
- logs, metrics location, checkpoint location, evaluation artifacts, and completion/failure markers;
- a concise handoff stating what launched, where, how it was validated, current progress, expected completion, and which actions still require authorization.

Do not call a run healthy merely because a process exists. Confirm steps advance, metrics are finite, GPUs are doing expected work, the extractor is serving real requests, checkpoints complete, and the logger is fresh.
