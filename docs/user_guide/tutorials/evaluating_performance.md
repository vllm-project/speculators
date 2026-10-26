# Evaluating Model Performance

The evaluator uses AIPerf 0.13.0 for dataset loading, requests, concurrency sweeps, native reports, and plotting. Our adapters prepare SPEED-Bench, preserve MRCR histories, and summarize acceptance statistics.

## Install

From the repository root, use a client environment with Python 3.11–3.13:

```bash
pip install -r scripts/evaluate/requirements.txt
```

## Serve the model

All examples use [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) with the `Qwen/Qwen3.8-27B` verifier, 7 speculative tokens, and adaptive verification disabled.

```bash
vllm serve Qwen/Qwen3.8-27B \
  --port 9145 \
  --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":7,"method":"dspark","enable_adaptive_verification":false}' \
  --per-request-spec-decode-metrics summary \
  --no-enable-prefix-caching
```

The evaluation commands use `--model Qwen/Qwen3.8-27B` to match the server's model name; the speculator is selected by `--speculative-config`. The experimental per-request response format must match AIPerf's adapter. The wrapper does not validate prefix-cache counters.

## Acceptance

Use `--dataset` to select datasets. When omitted, acceptance defaults to SPEED-Bench qualitative and speedup defaults to HumanEval.

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset qualitative \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/qualitative
```

This prepares qualitative once and runs the aggregate split in one native profile. The current source contains 880 conversations across 11 categories and 1,138 user turns. Follow-up turns are generated in the same conversation. For category matrices, use the native sweep below.

Defaults are concurrency 64, timeout 900 seconds, sequential sampling, server-reported token counts, and raw exports. Generation defaults are temperature 1, thinking enabled, and `max_completion_tokens=32768`. Replace that settings object with `--extra-inputs '{"temperature":0,"max_completion_tokens":1024}'`.

Run several datasets sequentially:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset gsm8k math500 qualitative \
  --request-count 8 --concurrency 64 \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/acceptance-smoke
```

The budget applies separately to each dataset. Each dataset keeps native reports under `<output-dir>/<dataset>/`, with combined acceptance CSVs at the root. A native run failure retains its results, subsequent datasets run, and the command returns nonzero.

`gsm8k`, `math500`, `humaneval`, `mbpp`, and `mtbench` select the corresponding native `spec_al_*` public loaders. Native public dataset names also work. Their prompt templates and conversation handling are AIPerf's; MT-Bench includes both turns. Without an explicit budget, these public datasets use AIPerf's default ten requests. AIME26 and Arena-Hard have no equivalent registered presets in this version.

## SPEED-Bench selection

`qualitative` aliases `speed_bench_qualitative`. The wrapper does not expand it into categories. Select individual categories with native names, for example:

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset speed_bench_coding speed_bench_math \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/coding-math
```

Each selected qualitative category defaults to all its conversations (currently 80) unless a budget is supplied. Explicitly selecting multiple categories runs them separately, which also supports category-specific speedup measurements.

Generate category reports from those runs:

```bash
aiperf speed-bench-report output/coding-math/speed_bench_* \
  --metric accept_length --format csv \
  --output output/coding-math/acceptance_length_by_category.csv
aiperf speed-bench-report output/coding-math/speed_bench_* \
  --metric accept_rate --format csv \
  --output output/coding-math/acceptance_rate_by_category.csv
```

Throughput selections use native names such as `speed_bench_throughput_8k_high_entropy`. They use native request budgets in acceptance mode and timed sweeps in speedup mode. Select the desired entropy tiers explicitly when separate reports are needed.

## MRCR

```bash
python scripts/evaluate/evaluate.py acceptance \
  --dataset mrcr \
  --dataset-filter needles=8 --dataset-filter max_samples=800 \
  --request-count 800 --concurrency 64 --request-timeout-seconds 3600 \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/mrcr
```

The registered MRCR plugin streams `openai/mrcr` from Hugging Face. Each complete history becomes one request, preserving system messages, user turns, and historical assistant responses. Earlier turns are not regenerated. No client tokenization, truncation, or context filtering occurs.

Select `needles=2`, `4`, or `8` and optionally `revision=<HF-commit>` with `--dataset-filter`. Defaults are eight needles and at most 100 source rows. `max_samples` limits source rows; AIPerf's request budget is separate and can replay them. The normal generation cap and `--extra-inputs` overrides also apply to MRCR. Over-context responses remain errors; this evaluates decoding statistics rather than answer accuracy.

## Acceptance results

- `acceptance.csv`: one row per native run, with pooled acceptance length, acceptance rate, and numeric position rates.
- `acceptance_length_by_category.csv` and `acceptance_rate_by_category.csv`: produced by the native `speed-bench-report` commands above.
- `acceptance_by_prompt_length.csv`: those metrics and request counts grouped by actual prompt-token length. Boundaries start at 8,192 and double through 1,048,576, with overflow and `unknown` bins.

Each profiling record is read once. Warmup and duplicate phase exports are excluded; follow-up turns and repeated conversations count as separate requests. Acceptance counters include successful responses with speculative metrics. Prompt-bin request counts also retain exported failures; missing token lengths appear under `unknown`. Unexported incomplete requests cannot be counted.

Counters are summed before division: acceptance length is `1 + accepted draft tokens / verification steps`; acceptance rate is `accepted draft tokens / proposed draft tokens`. Position k is the fraction of steps accepting at least k draft tokens. Rates are percentages and unavailable values are blank. These pooled values can differ from native means across requests.

Category matrices are generated by AIPerf 0.13.0's `speed-bench-report`, using each category run's `server_metrics_export.json`. Keep server metrics enabled and expose the server's `/metrics` endpoint (collection is enabled by default). Native acceptance rates are fractions from 0 to 1; the `Overall` column is the arithmetic mean of category values. These server-based matrices can differ from the request-based pooled statistics in `acceptance.csv`. Per-position rates for each category remain in that run's row of `acceptance.csv`. There is no custom category aggregation or request-ID lookup.

## Speedup and plotting

```bash
# SPEED-Bench qualitative
python scripts/evaluate/evaluate.py speedup \
  --dataset qualitative \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-qualitative

# Individual categories
python scripts/evaluate/evaluate.py speedup \
  --dataset speed_bench_coding speed_bench_math \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-categories

# HumanEval
python scripts/evaluate/evaluate.py speedup \
  --dataset humaneval \
  --model Qwen/Qwen3.8-27B --url http://localhost:9145 \
  --output-artifact-dir output/speedup-humaneval

# Generate plots
aiperf plot output/speedup-qualitative --output output/plots
```
