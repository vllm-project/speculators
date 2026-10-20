---
name: speculators-evaluation
description: Run Speculators acceptance evaluations and vLLM concurrency sweeps, start the required servers, and produce acceptance-rate/length Markdown sections and throughput-versus-interactivity graphs. Use for dataset selection, benchmark runs, and baseline/speculative comparisons. Does not perform model training.
---

# Speculators evaluation

Produce an acceptance-rate/length Markdown section and speedup curve graphs, with the settings and commands needed to reproduce them. A request for commands or a partial run only needs the requested subset.

Use the existing two-mode evaluator in `scripts/evaluate/`. Keep acceptance collection separate from speedup measurement; delegate timed speedup scheduling to GuideLLM rather than implementing another benchmark client.

## Workspace and current interface

Read the checkout's `AGENTS.md` and inspect current mode help before constructing commands:

```bash
speculators evaluate acceptance --help
speculators evaluate speedup --help
```

The CLI requires an editable source-checkout installation. Use the existing `.venv/bin/speculators`, or `.venv/bin/python scripts/evaluate/evaluate.py`, if the environment is not activated. The repository's help and implementation take precedence over this skill if they change.

Use the requested server and served model ID; the server's `/v1/models` can resolve the latter. An evaluation request does not require launching another server if a suitable one already exists. Check resource availability before launching a requested server. When using `scripts/launch_vllm.py`, pass `--provenance-dir` as required by the repository.

## Choose datasets

When the requested mode's dataset is unspecified in the request or session, ask which dataset to target. Combine this with any other unanswered setup choices in one short prompt. Reuse an existing selection; do not ask again or replace it with a default.

Offer these recommendations:

- **Acceptance:** SPEED-Bench qualitative, using `--dataset speedbench --dataset-config qualitative`, for its category breakdown. Always include the configuration: bare `--dataset speedbench` selects all six configurations.
- **Speedup:** HumanEval (`--dataset humaneval`) for an initial coding curve. Offer MATH-500 (`--dataset math500`) for reasoning or MT-Bench (`--dataset mtbench`) for chat if the user wants broader coverage. Run each selected dataset separately so curves have a clear workload.

If the user has no preference or delegates dataset selection, use SPEED-Bench qualitative for acceptance and HumanEval for speedup. State the selection before running. These are skill workflow defaults; the CLI still requires `--dataset`. Do not expand the fallback into the full `basic` blend, all SPEED-Bench configurations, or multiple speedup sweeps. Apply the 128-sample per-dataset default and existing generation settings; increase the cap when the user requests more coverage.

## Start vLLM instances

Before starting or benchmarking a speculative configuration, ask for any choices the user has not already supplied:

- How many speculative tokens should we use?
- Should adaptive verification be enabled?

Reuse answers already given in the task. Set `SPEC_TOKENS` to the chosen integer and `ADAPTIVE_VERIFICATION` to JSON `true` or `false`; do not assume 8 tokens or silently choose an adaptive-verification setting. Check the installed build and checkpoint support the requested configuration. Adaptive verification is currently supported for DSpark; for other methods, explain the capability limit rather than enabling an unsupported option. With adaptive verification, `SPEC_TOKENS` is the maximum draft length.

Acceptance targets the vLLM 0.30+ experimental `metrics.speculative_decoding` schema and requires `--per-request-spec-decode-metrics summary`. Pin the server commit/version; later schema changes may need an adapter. Speedup requires `--no-enable-prefix-caching` on every compared server, including the baseline. A speculative instance with both flags supports both evaluation modes.

Use the existing vLLM environment. Set `VLLM_BIN` to its executable, `BASE_MODEL` to the verifier checkpoint, `DRAFT_MODEL` to the speculator, and `MODEL` to the served alias. Set `TARGET=http://localhost:9123`. Choose available GPUs with `CUDA_VISIBLE_DEVICES` and set `TP` to the intended tensor-parallel size. Keep hardware, TP, precision, context length, and scheduler limits consistent across comparisons.

From the Speculators checkout, start the speculative server in a separate terminal:

```bash
"$VLLM_BIN" serve "$BASE_MODEL" \
  --served-model-name "$MODEL" --host 127.0.0.1 --port 9123 \
  --tensor-parallel-size "$TP" --max-num-seqs 128 \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS,\"enable_adaptive_verification\":$ADAPTIVE_VERIFICATION}" \
  --per-request-spec-decode-metrics summary \
  --no-enable-prefix-caching
```

This example uses DSpark with the chosen draft limit and adaptive-verification setting; choose the method supported by the actual checkpoint. Set model-specific context, memory, and attention options as needed. `--max-num-seqs 128` matches the example sweep's largest concurrency; it does not guarantee that the KV cache can hold 128 requests.

When the chosen comparator is a plain baseline, stop the speculative process you started and launch the same verifier on the same hardware:

```bash
"$VLLM_BIN" serve "$BASE_MODEL" \
  --served-model-name "$MODEL" --host 127.0.0.1 --port 9123 \
  --tensor-parallel-size "$TP" --max-num-seqs 128 \
  --no-enable-prefix-caching
```

For the native MTP comparator, use the checkpoint's supported MTP speculative configuration with prefix caching disabled. Verify and record its effective draft length and method; do not apply DSpark-only adaptive settings to MTP. Acceptance on an MTP instance also needs the per-request metrics flag. Do not enable speculative decoding on a plain baseline.

Before evaluation, wait for `/health` and verify the served alias through `/v1/models`:

```bash
curl --fail "$TARGET/health"
curl --fail "$TARGET/v1/models"
```

Use separate ports and GPU allocations if configurations run simultaneously; point each evaluation at its matching server. Save the serve command and logs beside the results. If using `scripts/launch_vllm.py` instead, run it in the vLLM environment with `--provenance-dir` as required by `AGENTS.md`.

## Shared settings and datasets

- Both modes default to **8192 maximum output tokens**. `--max-tokens N` overrides it; this argument also overrides token budgets in `--body`.
- Both modes default to `{"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}`, as requested by the user. An explicit `--body` replaces that JSON; use it to override sampling or disable thinking. Keep settings identical across compared runs.
- `basic` contains GSM8K, MATH-500, HumanEval, MBPP, AIME26, MT-Bench, and Arena-Hard. The per-dataset sample cap applies to each; smaller datasets keep all their rows. Presets supply their source/configuration/split/mapping. Source-specific overrides require a single dataset, excluding blends; run differently configured sources separately. SPEED-Bench supports configuration/files, not HF split/revision/mapping overrides.
- `--max-samples N` defaults to 128 and selects up to the first N source rows from each named dataset, including each expanded `basic` dataset and each selected SPEED-Bench configuration. Smaller datasets keep all their rows; later datasets are still visited. `--max-requests` remains a total ordered acceptance request limit, cycling the selection when needed. A short limit may visit only the first dataset; overall is only the requests actually sent. Prefer the per-dataset sample cap for breadth.
- Raw HF IDs and local JSON, JSONL, CSV, or Parquet files use `--prompt-column`, `--messages-column`, or `--prompt-template`. Plain text becomes a user message; message lists preserve conversation roles. HF datasets do not share one universal prompt schema. Use `--dataset-config` only when that source requires it; MATH-500 uses its default, not GSM8K's `main`.
- `mrcr`/`openai/mrcr` is **acceptance-only**, defaults to 8 needles, and preserves the full conversation including historical assistant replies. Select `--needles 2 4 8` or explicit `--data-files`. MRCR uses the serving `/tokenize` endpoint with chat-template options, then filters by inclusive `--min-prompt-tokens`/`--max-prompt-tokens` bounds before the sample cap. The default upper bound is `/v1/models` context minus the output budget. Never truncate histories or silently change the generation budget. Record scanned/filtered/selected counts from `source_selection`; selection is source-ordered, not balanced by length. Speedup rejects MRCR before loading.
- `speedbench` automatically prepares NVIDIA's source files locally and caches them under `$HF_HOME/speculators/speedbench`. With no `--dataset-config`, it prepares all six configurations: `qualitative` and `throughput_1k`, `throughput_2k`, `throughput_8k`, `throughput_16k`, `throughput_32k`. `--data-files` reuses prepared files and bypasses preparation. First preparation can require substantial downloads and gated-source access; `--max-samples` does not reduce preparation. Keep prepared data local; redistribution is restricted.

## Acceptance

The server must support `--per-request-spec-decode-metrics summary`. This mode expects speculative metrics and exits unsuccessfully for missing metrics or request errors while retaining reports.

Set `TARGET` to the server base URL and `MODEL` to its served model ID. After dataset selection, the recommended acceptance command is:

```bash
speculators evaluate acceptance \
  --dataset speedbench --dataset-config qualitative \
  --target "$TARGET" --model "$MODEL" \
  --output-dir results/speedbench_qualitative_acceptance
```

By default, this sends one pass over the full selection at concurrency 64 (`--concurrency` overrides it), completing each dataset before starting the next. The total wall-clock timeout is 900 seconds per request; explicitly choose a longer timeout for long context/thinking. Individual request errors do not stop remaining requests; exit 1 marks errors or missing metrics after reports are saved. Streamed HTTP errors retain the server response body. Requests retain source order. `--max-requests` remains a total run limit; a short run may cover only the first dataset. Use `--dataset basic` when the user requests the seven-dataset blend, adjusting the sample cap to the requested coverage.

`summary.csv` is updated after each completed dataset, with counts, acceptance rate/length, numeric `acceptance_rate_pos_1`, `acceptance_rate_pos_2`, etc. columns, and elapsed seconds. Unavailable values are blank. A final `overall` row uses summed counters. `requests.jsonl` preserves raw response metrics; `summary.json` is updated after each dataset with overall, per-dataset, prompt-length, and category summaries. `summary_by_prompt_length.csv` exports the existing successful-response length buckets with weighted counters and numeric per-position columns, updating after each dataset. Source `category` values are retained through the loader and raw request records. `by_category` and `summary_by_category.csv` aggregate across selected configurations using summed counters, including errors and missing metrics. Categories are not inferred from text, and uncategorized rows remain in overall/dataset results:

- Acceptance rate = accepted draft tokens / proposed draft tokens.
- Acceptance length = 1 + accepted draft tokens / verification steps.
- Per-position acceptance rate = fraction of verification steps accepting at least that draft position, starting at 1. Their sum is acceptance length minus 1.

Counters are summed before division; do not average per-request rates or treat SSE chunks as tokens. Prompt-length buckets use actual response `prompt_tokens`, with upper bounds 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, plus overflow. Successful-response bucketing needs no client tokenization. MRCR selection separately uses server-side tokenization before sending generation requests.

## Speedup and graphing

For a requested speedup comparison without an explicitly chosen comparator, compare the draft configuration against the verifier checkpoint's **native MTP** when that checkpoint includes the MTP layers and weights. Otherwise compare against the same verifier with **speculative decoding disabled** (plain baseline). Verify checkpoint contents and installed vLLM support; do not infer MTP availability from a model name alone. If MTP is expected but unsupported or fails to start, report the limitation or failure rather than silently substituting a baseline. Reuse an explicitly requested comparator.

The skill selects and runs the comparison configurations; the `speedup` CLI measures only its supplied endpoint and does not automatically detect MTP or launch a comparator. A request for one endpoint or commands alone does not authorize additional benchmark runs. Label curves `Native MTP` or `Baseline` as appropriate, and record the MTP settings. The plotting example below uses a plain baseline; substitute the native MTP results and label when selected.

Use GuideLLM 0.8.0 in the existing client environment. The speedup wrapper delegates concurrency and duration to GuideLLM's native scheduler. Do not replace timed windows with fixed request-count batches. `--guidellm-bin` selects an existing client executable; `--tokenizer` selects the verifier tokenizer or local snapshot when the served alias is not a tokenizer repository.

The server must have prefix caching disabled (`--no-enable-prefix-caching`), because prompts repeat. Speedup checks this once through `/metrics`. Speculative metrics are optional.

```bash
speculators evaluate speedup \
  --dataset humaneval --target "$TARGET" --model "$MODEL" \
  --tokenizer "$BASE_MODEL" --streams 1,4,8,16,32,64,128 \
  --warmup-seconds 30 --max-seconds 100 --repeats 3 \
  --label dspark --output-dir results/humaneval_dspark
```

Defaults are 30 seconds warmup plus 100 measured seconds per point, repeated three times. The default concurrencies are 1, 4, 8, 16, 32, 64, and 128. These seven concurrencies budget 45.5 minutes of scheduled load plus setup/teardown. Prompt replay supplies the timed run; it is not a completion quota. The runner automatically repeats the full source selection internally to supply at least 5,000 input rows. There is no user-facing replay-count setting. Fail if the pool exhausts before the deadline; do not silently accept a shortened window. A timed run need not visit every source row. `--max-requests` remains acceptance-only. `--warmup-seconds` must be 0 or at least 1.

Collect baseline and speculative runs with identical workload, tokenizer, generation settings, time budgets, concurrency, and hardware. Use a fresh output directory for a new run. `--resume` checks identical inputs/configuration, locks the journal and runs missing points; it skips recorded failures/interrupted attempts too. Only explicitly requested `--resume --retry-failed` retries those points, retaining earlier attempts. Old archives without configuration metadata remain plottable but are not resumable. Do not silently retry or change measurement settings. Keep one `benchmarks.jsonl` at the run root: its first record preserves the input selection, replay count and configuration; subsequent start/outcome records preserve point identity, command, report, failures and diagnostics. Resolve the last attempt per point when reading it. Per-point JSON files and the replay pool are temporary and cleaned up on exit. Do not create an `artifacts/` folder. The runner updates `summary.csv` after points, one row per concurrency averaged across valid repeats, with expected/failed repeat counts and partial status. It includes completion counts, TTFT, repeat CV, sampled waiting maximum and pooled whole-point acceptance length. Do not add a per-repeat summary CSV. Plotting excludes failed attempts and incomplete concurrency groups with warnings.

```bash
python scripts/evaluate/throughput_interactivity.py plot \
  --series results/humaneval_baseline:Baseline \
  --series results/humaneval_dspark:DSpark \
  --out results/humaneval_speedup.png
```

Y is GuideLLM's native total output-token throughput over its measurement window. X is reciprocal mean per-request TPOT for successful requests completed inside that window, excluding responses with at most one token. GuideLLM's per-request `inter_token_latency_ms` is token-normalized decode time, not raw chunk latency. Average X and Y per run across repeats. Warmup and completions after the window are excluded from X. Deadline-incomplete requests are expected and recorded separately from failures; their available token timings can contribute to native window throughput. Record a failed point for actual request errors, early termination or no usable completions; continue the sweep and return exit 1 if failures remain. Warn on fewer than ten usable completions and repeat CV above 10%; choose a longer new measurement explicitly rather than extending it automatically. Use client timing consistently.

Legacy vLLM batch results can be plotted separately, but must not be mixed with timed-window runs on one chart. Do not claim numerical equivalence with the old custom prorating collector. Keep lightweight per-point diagnostics: sample `/metrics` during a point because waiting gauges may be zero at both boundaries; difference accepted-token and verification-step counters to obtain acceptance length. These counters span the whole point including warmup/setup and require an isolated endpoint; they are not exactly the GuideLLM measurement window. Missing/reset counters stay unavailable. Samples can miss brief queueing or delayed metric updates. Review queueing against scheduler/KV limits. Keep GuideLLM's native axes; do not add a custom window estimator or claim steady state from these diagnostics alone.

## Write the results

Write an `Acceptance Rates` Markdown section and a `Speedup Curves` section with relative links to generated PNGs. Add them to the requested results document or provide them as a standalone evaluation README. Keep unrelated content intact when updating an existing document.

- Build acceptance table rows from the dataset rows in `summary.csv` or `by_dataset` in `summary.json`, showing `acceptance_rate`, `acceptance_length`, and `acceptance_rate_by_position`. Use `Pos 1`, `Pos 2`, etc. to match the JSON keys. Include successful, errored, and missing-metric counts alongside the results. Render unavailable metrics as unavailable, not zero.
- Label the actual dataset selection and sampling configuration. Use separate tables for temperatures/configurations only when each was measured. Acceptance is a decoding statistic, not task accuracy; calculate it from summed counters rather than averaging per-request rates.
- For measured MRCR acceptance, add a table using the actual `by_prompt_length` labels and successful counts. Describe the needle configuration and files. Call counts unique only after deduplicating successful `sample_id` values in `requests.jsonl`. Report context-limit failures and the maximum observed successful prompt length.
- Produce one speedup chart per measured dataset, comparing only configurations actually run. State GPU type/count, TP/DP, verifier precision, software commits, draft length, sampling/thinking settings, maximum output tokens, prefix caching, measured window, warmup seconds, observed request counts, concurrency, and repeats. Explain X as reciprocal mean TPOT and Y as native window throughput. Quote speedup ratios only at explicitly matched comparison points.
- Include the actual serve/evaluation/plot commands and links to raw results and provenance. Keep `eval_command.txt` and available server-command, checkpoint-hash, and patch artifacts with the results. Do not copy restricted prepared datasets into a shareable report.

Check displayed numbers against the summaries, image links, units, position numbering, and effective generation budgets before handing over the sections and graphs. Omit unrun experiments. Fake-server checks verify plumbing and cannot supply model-performance claims. Saving evaluation artifacts does not authorize publishing them.

## Results and implementation

Default output directories use served-model name plus timestamp. Both modes write `eval_command.txt`; keep server provenance with results. Speedup saves `summary.csv` at the run root. A single `benchmarks.jsonl` preserves inputs, commands, and native GuideLLM reports at the run root. Plotting reads this layout and older per-point reports, including older `artifacts/` directories. Acceptance writes request records incrementally and flushes CSV/JSON summaries after each dataset. Report real test/benchmark outcomes, including whether a check used a fake server or GPUs.

For changes, consult the checkout's `docs/user_guide/tutorials/evaluating_performance.md` and these implementation owners:

- `evaluate.py`: CLI and provenance; `loader.py`: selection and shared API bodies.
- `eval_datasets/basic.py` and `speedbench.py`: dataset preparation and mappings.
- `acceptance.py` and `request_runner.py`: per-request acceptance and SSE collection.
- `speedup.py`: native benchmark dispatch; `throughput_interactivity.py`: offline graphs.

Prefer the smallest change in the owning component. Keep dataset transformations out of runners, retain the known response schema, and keep point validity checks, queue/completion diagnostics and failure/resume handling close to collection. Avoid fallback response schemas and custom timing estimators; retain original failed reports for review.
