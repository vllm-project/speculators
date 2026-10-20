# Evaluating Model Performance

The evaluator has two modes: `acceptance` collects speculative acceptance counters; `speedup` runs GuideLLM timed concurrency sweeps. Collect each comparator separately and plot the results offline.

## Setup

Run from an editable source checkout and install the client dependencies:

```bash
uv pip install -r scripts/evaluate/requirements.txt
```

Acceptance supports the vLLM 0.30+ `metrics.speculative_decoding` response schema and requires `--per-request-spec-decode-metrics summary` with `n=1`. This is an [experimental vLLM interface](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/observability.py): pin the server version/commit and retain the serving command. A version number alone does not guarantee an unchanged response schema.

Speedup uses GuideLLM 0.8.0. Disable prefix caching on every compared server (`--no-enable-prefix-caching`). Use `--guidellm-bin` for an existing client executable and `--tokenizer` for the verifier tokenizer when the served alias is not a tokenizer repository. Keep hardware, TP, generation settings and time budgets matched.

## Dataset inputs

`--dataset` accepts presets, Hugging Face IDs and local JSON/JSONL/CSV/Parquet files. Presets supply their source, split and column mapping. `basic` expands GSM8K, MATH-500, HumanEval, MBPP, AIME26, MT-Bench and Arena-Hard; MRCR is separate and acceptance-only.

`--max-samples` defaults to 128 **per dataset**, keeping all rows of smaller datasets. Generation defaults are 8192 output tokens, temperature 0 and thinking enabled. `--body` replaces the generation defaults; `--max-tokens` overrides the body's output budget.

Source-specific flags (`--dataset-config`, `--data-files`, `--split`, `--revision` and column mappings) require a single dataset, excluding blends such as `basic`. They override that source's preset. Run differently configured datasets separately so flags cannot accidentally apply to another dataset.

Acceptance's `--max-requests` is a **total ordered limit**, not a per-dataset cap. A short limit may visit only the first dataset; the overall row describes only requests actually sent. Use `--max-samples` for coverage across datasets. All requests and datasets are attempted despite individual request errors. Reports are retained; exit 1 indicates request errors or missing speculative metrics. The default total timeout is 900 seconds per request, including streaming; increase it for long thinking or long context.

## Acceptance

```bash
speculators evaluate acceptance \
  --dataset speedbench --dataset-config qualitative \
  --target http://localhost:9123 --model my-model \
  --output-dir results/acceptance
```

```bash
# MRCR: 8 needles, with prompts rendered by the serving tokenizer.
speculators evaluate acceptance \
  --dataset mrcr --needles 8 --concurrency 1 --timeout 3600 \
  --target http://localhost:9123 --model my-model \
  --output-dir results/mrcr_8needle

# Choose a rendered prompt-length range (inclusive), e.g. 8K–128K.
speculators evaluate acceptance \
  --dataset mrcr --needles 8 \
  --min-prompt-tokens 8192 --max-prompt-tokens 131072 \
  --concurrency 1 --timeout 3600 \
  --target http://localhost:9123 --model my-model \
  --output-dir results/mrcr_8needle_8k_128k
```

MRCR defaults to 8 needles; `--needles 2 4 8` selects each separately. `--data-files '8needle/*.parquet'` can select files instead. The loader preserves complete message histories, uses `/tokenize` with the generation chat-template options, and filters **before** the sample cap. It discovers the context limit through `/v1/models` and reserves the output-token budget. Length bounds must fit that budget; prompts are never truncated. `summary.json` records scanned, filtered and selected counts under `source_selection`. Selection is in source order, not balanced across length buckets. A range filter is selection, not an over-context failure. Server errors, including their response bodies, remain in `requests.jsonl`. Successful response token counts drive `summary_by_prompt_length.csv` automatically.

## Speedup

```bash
# Baseline server.
speculators evaluate speedup \
  --dataset humaneval --target http://localhost:9124 --model my-model \
  --tokenizer my-verifier-model --label baseline \
  --output-dir results/humaneval_baseline

# Speculative server.
speculators evaluate speedup \
  --dataset humaneval --target http://localhost:9123 --model my-model \
  --tokenizer my-verifier-model --label dspark \
  --output-dir results/humaneval_dspark
```

## Graph

```bash
python scripts/evaluate/throughput_interactivity.py plot \
  --series results/humaneval_baseline:Baseline \
  --series results/humaneval_dspark:DSpark \
  --out results/humaneval_speedup.png
```
