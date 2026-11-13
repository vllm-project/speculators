# Evaluation, promotion, and handoff

Use the sibling [Speculators evaluation skill](../../speculators-evaluation/SKILL.md) for evaluator intake, vLLM serving, AIPerf execution, metric definitions, and report formatting. This reference defines how training consumes that workflow.

## Checkpoint acceptance

Treat a checkpoint as usable only after all applicable artifacts are present, stable, and nonempty:

- model weights and configuration;
- optimizer and scheduler state for exact resume;
- training state with global step, epoch/local step, and distributed sampler cursors;
- copied resolved command/config;
- validation metrics;
- a completed-save log after the save-start log.

Load the checkpoint through the same Speculators revision that wrote it. For promotion, also load it through the intended vLLM revision and run a real completion request.

## Every ablation requires evaluation

Evaluation is part of an ablation, not an optional follow-up. This applies to layer, learning-rate, loss, data, vocabulary, architecture, head, initialization, and training-policy comparisons.

For every candidate:

1. use the same frozen held-out prompts and evaluator configuration;
2. start the intended vLLM build from a clean process and prove the speculative path is active;
3. run at least the acceptance-only scope from the evaluation skill;
4. preserve request-level artifacts, commands, server provenance, failures, and summaries;
5. reject comparisons whose serving or evaluation configuration is unmatched.

The acceptance screen must report pooled acceptance rate, mean acceptance length, per-position acceptance, successful/error counts, and enough latency/throughput telemetry to detect a broken serving path. Run the evaluation skill's matched speedup workflow for finalists before promotion when deployment performance is part of the goal. Do not run the evaluation skill's broad default matrix for every early candidate unless the user requests it; resolve a smaller common dataset matrix for the ablation and keep it identical across candidates.

An ablation table with training loss but no evaluation artifacts is incomplete and cannot select a winner.

## Evaluation ladder

Evaluate candidates in increasing cost:

1. config and weight-load smoke test;
2. deterministic small prompt set for correctness and acceptance telemetry;
3. held-out acceptance rate, accepted length, per-position accuracy, and loss components;
4. end-to-end latency and throughput against the same target without speculation;
5. comparison to relevant alternatives such as native MTP under identical serving settings;
6. representative quality suites, tool-call/coding tests, and long-context tests when relevant;
7. concurrency and request-rate sweeps at the deployment regime.

Acceptance alone is not the deployment objective. A larger speculative-token setting can raise accepted tokens per request while reducing throughput or increasing tail latency. Keep prompts, sampling policy, max tokens, concurrency, tensor parallelism, cache dtype, and measurement windows controlled.

Define the expected acceptance range before inspecting results. Derive it from a same-target, same-runtime, same-prompt baseline or an explicit project target. Acceptance from a different model, sampling temperature, speculative-token count, or vLLM path is not a valid gate. If no trusted baseline exists, label the threshold provisional and compare against an untrained/current draft plus target-only behavior before running a larger confirmation pilot.

Exclude warmup and failed requests according to a predeclared rule. Preserve raw results and the exact server/evaluator commands. If a result used a patched runtime, save the patch and commit SHA.

## Promotion rules

Define the gate before viewing the final result. Typical gates include:

- no quality regression beyond an agreed tolerance;
- finite, stable held-out metrics;
- acceptance and per-position behavior at least as good as the selected baseline;
- meaningful end-to-end improvement in the intended concurrency regime;
- runtime stability without OOMs, transfer failures, or silent fallback;
- reproducible model load and smoke results from a clean process.

Do not upload or label a checkpoint “final” until the gate passes. Keep failed or mismatched runs clearly marked as diagnostic so they cannot be mistaken for canonical results.

## Provenance bundle

Retain:

- target and draft model IDs/revisions and checkpoint hash;
- Speculators and vLLM SHAs, dirty status, and patches;
- dataset IDs/revisions, regeneration config, merge manifest, validation summary, and checksums;
- resolved training config and command;
- package versions, container digest, CUDA/driver, hardware, and topology;
- run logs, metrics database/export, checkpoint manifest, and timestamps;
- serving command, evaluator command, raw outputs, and summaries;
- known limitations or invalid comparisons.

Prefer machine-readable JSON/YAML plus a concise Markdown handoff. Never embed credentials, access tokens, private SSH material, or temporary signed URLs.

## Final handoff

State:

- what was trained and why this configuration was chosen;
- exact run location and process/allocation identity;
- current or final step, epochs, checkpoint, and validation result;
- sustained speed and ETA, with validation overhead separated;
- extractor/transport/logger health and any observed failures;
- evaluation outcome and whether the checkpoint is promoted, diagnostic, or still pending;
- actions that require fresh authorization, such as stopping resources, uploading publicly, or deleting large artifacts.

For a monitor, distinguish live observations from last-known state. If access fails, say the run could not be verified instead of presenting stale metrics as current.
