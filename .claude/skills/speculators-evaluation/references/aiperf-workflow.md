# AIPerf evaluation workflow

Use this reference for execution, dataset handling, metrics, and plots. The checked-out evaluator and its current `--help` output are authoritative when they differ from this reference.

## Interface and setup

From the Speculators repository root, inspect:

```bash
python scripts/evaluate/evaluate.py acceptance --help
python scripts/evaluate/evaluate.py speedup --help
```

Install the evaluation client only when needed and authorized:

```bash
pip install -r scripts/evaluate/requirements.txt
```

The updated adapter uses AIPerf for requests, scheduling, metrics, native exports, and plotting. Its direct entrypoint is `scripts/evaluate/evaluate.py`; do not assume a `speculators evaluate` command exists. In the inspected implementation, the adapter pins AIPerf 0.13.0 and supports Python 3.11–3.13. Record the actual installed versions.

Acceptance serving requires vLLM support for per-request speculative-decode summary metrics with a response schema understood by the installed AIPerf adapter. Check the current vLLM help rather than inferring a minimum release from a previously successful run.

## Defaults and native configuration

Without an AIPerf config file, both modes use chat endpoints, sequential sampling, server token counts, raw exports, and this generation object:

```json
{"temperature":1,"chat_template_kwargs":{"enable_thinking":true},"max_completion_tokens":32768}
```

`--extra-inputs` replaces the complete object, so include every desired field when overriding it.

| Mode       | Wrapper defaults                                                                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Acceptance | Nonstreaming, concurrency 64, request timeout 900 seconds.                                                                                                            |
| Speedup    | Streaming; concurrency 1, 4, 8, 16, 32, 64, 128; three profile runs; 30 seconds warmup per point; 100 seconds measurement; zero grace period; automatic native plots. |

When `--dataset` is omitted, the evaluator defaults acceptance to SPEED-Bench qualitative and speedup to HumanEval. The skill's resolved evaluation matrix may deliberately supply a broader dataset list.

Other canonical long options pass through to `aiperf profile`. Supplying `--config FILE.yaml` or `-f FILE.yaml` disables the wrapper's execution defaults, automatic SPEED-Bench preparation, and automatic conversation budgets. The YAML must therefore define the complete intended profile. Explicit native CLI options can override YAML.

Use a fresh `--output-artifact-dir`; the wrapper has no resume or overwrite policy.

## Datasets and budgets

Use `--dataset NAME [NAME ...]`. Multiple datasets run sequentially with the same profile flags and budget.

- `mtbench`, `gsm8k`, `humaneval`, `mbpp`, and `math500` map to AIPerf's speculative-acceptance loaders. Without an explicit budget, native public datasets use AIPerf's default request count.
- `qualitative` aliases SPEED-Bench qualitative and runs the aggregate split in one native profile. It does not expand into category runs. Select named categories such as `speed_bench_coding` separately when category matrices or category-specific performance measurements are required.
- `mrcr` uses the registered MRCR loader and preserves each source row's complete conversation history. `needles` accepts 2, 4, or 8; `max_samples` caps source rows; `revision` can pin a dataset commit.
- `--request-count` limits model calls and can stop a multi-turn conversation early. `--num-conversations` lets selected conversations complete their turns. A source-row cap and a request budget are different controls.

When SPEED-Bench needs preparation, use the adapter's pinned preparation path and cache rather than creating a second converter. Preserve the prepared input for category joins and reproducibility. Keep prepared prompts local when redistribution is restricted.

For MRCR, retain and report over-context errors. Do not truncate prompts or regenerate historical assistant turns. This measures speculative-decoding behavior, not MRCR answer accuracy.

## Acceptance execution

Set the served model alias and endpoint, then run the selected datasets. A single qualitative run is:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset qualitative \
  --model "$MODEL" \
  --url "$TARGET" \
  --output-artifact-dir output/acceptance-qualitative
```

A multi-dataset run is:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset mtbench gsm8k humaneval mbpp math500 \
  --model "$MODEL" \
  --url "$TARGET" \
  --output-artifact-dir output/acceptance-public
```

Choose MRCR budgets deliberately for the verifier's context window and available runtime:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset mrcr \
  --dataset-filter needles=8 \
  --model "$MODEL" \
  --url "$TARGET" \
  --output-artifact-dir output/acceptance-mrcr
```

Native files include aggregate `profile_export_aiperf.json`/`.csv`, request records in `profile_export.jsonl`, and raw payloads/responses in `profile_export_raw.jsonl`. Retain `eval_command.txt` and `aiperf_command.txt`.

The request-record summarizer writes:

- `acceptance.csv` for pooled metrics by native run;
- `acceptance_by_prompt_length.csv` for prompt-length bins and request counts.

Pool counters before division:

- acceptance length = `1 + accepted_draft_tokens / num_spec_steps`;
- acceptance rate = `100 * accepted_draft_tokens / proposed_draft_tokens`;
- position k rate = `100 * steps_accepting_at_least_k / num_spec_steps`.

Exclude errored requests from acceptance counters but include them in disclosed request/error totals. Native per-request averages can differ from pooled values. A successful process exit alone is insufficient: confirm complete native summaries, successful requests, speculative metrics for every success, and usable request records.

For SPEED-Bench category matrices, run the desired categories separately and use AIPerf's native reporter:

```bash
aiperf speed-bench-report output/categories/speed_bench_* \
  --metric accept_length --format csv \
  --output output/categories/acceptance_length_by_category.csv

aiperf speed-bench-report output/categories/speed_bench_* \
  --metric accept_rate --format csv \
  --output output/categories/acceptance_rate_by_category.csv
```

These commands read each category run's `server_metrics_export.json`. Keep server metrics enabled and ensure the vLLM `/metrics` endpoint is reachable; collection is enabled by default. Native category acceptance rates are fractions from 0 to 1. The matrix's `Overall` column is the arithmetic mean of category values, not a request-weighted pooled statistic. It can therefore differ from the request-derived pooled results in `acceptance.csv`. Per-position rates for each category remain in that category run's `acceptance.csv` row. Do not invent a custom category join or relabel the category mean as the pooled overall result.

## Speedup execution and comparators

Run every requested dataset separately for each matched server configuration. Example for one configuration:

```bash
python scripts/evaluate/evaluate.py speedup \
  --dataset humaneval \
  --model "$MODEL" \
  --url "$TARGET" \
  --output-artifact-dir output/speedup-humaneval-drafter
```

Do not change workload, sampling, output cap, hardware, tensor parallelism, precision, context, warmup, measurement window, grace period, or concurrency grid between comparator runs. Keep each configuration in a distinct directory.

Use the drafter configuration as defined by its model config. Benchmark adaptive verification off and on only when the method and installed vLLM support both. Prefer native MTP as the default comparator when the base model has compatible MTP weights and the installed vLLM supports them; otherwise benchmark the base model with speculative decoding disabled. Record the reason for the fallback.

Speedup mode collects performance curves; it does not itself launch servers, choose a comparator, or compute baseline-relative ratios. Compute ratios only at explicitly matched points and state the definition used.

## Plotting and interactivity

Native plotting can aggregate saved runs:

```bash
aiperf plot output/speedup-runs --output output/plots
```

PNG export requires Chrome or Chromium and may need `BROWSER_PATH`. Plotting errors are nonfatal unless the run requests `--plot-required`.

Choose and disclose the interactivity definition:

- AIPerf's `output_token_throughput_per_user.avg` is the mean of per-request reciprocal inter-token latencies.
- The referenced model-card convention is `1000 / inter_token_latency.avg` when ITL is measured in milliseconds.

These are not the same statistic. If the report must follow the model-card convention exactly, compute the reciprocal-mean value offline from verified native exports and label it. Otherwise label the native per-user statistic precisely. The y-axis is native aggregate `output_token_throughput.avg`.

In AIPerf 0.13.0, static multi-run PNG export can select p50 even when a preset requests avg, and native grouping can merge distinct deployments at the same concurrency. Inspect plotted statistics and group labels. Produce an offline plot from verified native exports when necessary rather than presenting a misleading native image.

## Failure and provenance rules

Preserve incomplete or failed run artifacts. Report request errors, incomplete summaries, missing metrics, and skipped configurations. Do not auto-retry with changed settings or silently discard failed concurrency points.

For each server/configuration, retain:

- drafter, verifier, and dataset revisions;
- exact vLLM and AIPerf versions;
- launch and evaluation commands;
- server logs and health/model checks;
- hardware, GPU count, tensor parallelism, precision, context, speculative method/draft length, adaptive state, prefix caching, and sampling;
- requested and completed budgets, warmup/window/grace, and profile repetitions.
