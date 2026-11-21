---
name: speculators-evaluation
description: Evaluate draft models produced for vllm-project/speculators by serving them with vLLM, measuring acceptance and speedup with AIPerf, comparing against MTP or the base model, and preparing model-card-ready results. Use for drafter acceptance rates, acceptance length, long-context behavior, performance sweeps, plots, and evaluation reports. Does not train models.
---

# Speculators evaluation

Evaluate a Speculators drafter as a deployable speculative-decoding system. The primary outcomes are:

- pooled acceptance rate, mean acceptance length, and per-position acceptance rates;
- long-context acceptance behavior when MRCR is selected;
- matched throughput-versus-interactivity curves and speedup conclusions;
- a reproducible README-style evaluation report.

Use the evaluator in the selected Speculators checkout. The current implementation is AIPerf-based; inspect the checkout's help and `docs/user_guide/tutorials/evaluating_performance.md` before forming commands. Do not apply these instructions to an older evaluator without reconciling its interface.

## Intake

At the start, ask once for any of these that the user has not already supplied:

1. **Drafter** — Hugging Face name or local checkpoint path.
2. **Scope** — acceptance only, speedup only, or the full evaluation.
3. **Datasets** — for example MT-Bench, GSM8K, MBPP, HumanEval, Math500, MRCR, or SPEED-Bench qualitative.

Do not re-ask for information already present. The drafter is required. Infer the base/verifier model, speculative method, supported draft length, and other model settings from the drafter configuration when possible. Ask separately for a host, vLLM environment, or hardware allocation only when execution was requested and those details cannot be discovered safely.

If the user leaves scope open, use the full evaluation. If the user leaves datasets open, use these defaults for the selected modes:

- **Acceptance:** SPEED-Bench qualitative, including individual category runs when model-card matrices are required, plus MT-Bench, GSM8K, HumanEval, MBPP, Math500, and MRCR.
- **Speedup:** MT-Bench, GSM8K, HumanEval, math500 and MBPP.
- For every speedup dataset, benchmark the drafter with adaptive verification disabled and enabled when supported.
- Use native MTP as the comparator when it is present and supported; otherwise use the same base model without speculative decoding.
- Produce comparison plots for every speedup dataset.

An explicit user selection overrides this matrix. Acceptance-only means do not launch speedup sweeps. Speedup-only means do not add acceptance runs. If adaptive verification is unsupported, record it as unavailable rather than fabricating that condition.

## Plan before execution

Summarize the resolved evaluation matrix before launching costly work: drafter, inferred verifier, modes, datasets, server configurations, comparator, sampling settings, budgets, hardware, and output location. Distinguish user choices, inferred values, and evaluator defaults.

A request to plan, provide commands, inspect artifacts, or write a report does not authorize launches. When execution is requested, keep all work on the selected hosts and allocations. Preserve failed measurements; do not silently change settings, substitute checkpoints, or retry indefinitely to obtain a favorable result.

## Serve with vLLM

Launch controlled vLLM instances for the configurations required by the resolved matrix. Prefer a fresh server per configuration so commands, logs, and artifacts are attributable. Reuse an existing endpoint only when its effective configuration is verified to match.

Before each workload:

- inspect the drafter and verifier configs and record immutable revisions;
- use the selected vLLM checkout/environment and record its exact version;
- match GPU type/count, tensor parallelism, precision, context/RoPE, scheduler limits, sampling, output cap, and prefix-caching state across comparisons;
- enable per-request speculative metrics for acceptance runs using the supported vLLM interface;
- keep server metrics enabled and verify `/metrics` when native SPEED-Bench category reports are required;
- verify `/health` and `/v1/models`, then run a small request that confirms serving and speculative metrics;
- keep commands, ports, logs, and effective settings with the run artifacts.

For speedup comparisons, create matched configurations for drafter/adaptive-off, drafter/adaptive-on when supported, and MTP-or-base. Do not treat adaptive verification as a separate drafter checkpoint. Serialize workloads per endpoint; separate authorized GPU allocations may run concurrently. Stop only processes owned by this evaluation.

Read [references/aiperf-workflow.md](references/aiperf-workflow.md) before constructing commands, selecting budgets, interpreting AIPerf outputs, or plotting results.

## Evaluate and validate

Run the selected acceptance workloads first when performing the default full evaluation, then the matched speedup sweeps. Use fresh output directories and retain AIPerf's native exports plus command and server provenance.

Acceptance results must include request-derived pooled acceptance length, pooled acceptance rate, and per-position rates. SPEED-Bench category matrices are separate server-metric reports: preserve their native aggregation and do not present their arithmetic category mean as the pooled overall result. Report successful/error counts and missing speculative metrics; never assume the requested budget completed. MRCR results also require prompt-length bins, maximum successful prompt length, and context-limit failures.

Speedup conclusions must use matched measurement points. State the exact x/y statistics in every graph and do not conflate acceptance length, task accuracy, output throughput, or latency-derived interactivity. A plot existing is not sufficient validation: inspect its statistic, grouping, completed requests, and configuration labels.

## Deliver a model-card-ready report

Read [references/report-format.md](references/report-format.md) when preparing or reviewing the final README. Follow its structure and disclosure checklist, modeled on the requested Red Hat drafter model card without copying run-specific values.

The final report must make it possible to reproduce and audit the comparison. Link summaries, plots, native archives, commands, logs, and source/checkpoint provenance. Publish only configurations actually measured. Keep restricted benchmark prompts and raw exports private when their licenses or access terms require it.

Saving artifacts does not authorize commits, pushes, pull requests, model-card edits, uploads, or publication.
