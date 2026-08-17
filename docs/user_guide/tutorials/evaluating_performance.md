# Evaluating Model Performance

## Prerequisites

```bash
cd scripts/evaluate
pip install -r requirements.txt
```

## Quick Start

Run the full benchmark pipeline (output-length estimation → performance sweep → CSV):

```bash
python evaluate.py sweep --target http://localhost:8000/v1
```

This runs all 9 subsets from `RedHatAI/speculator_benchmarks` and produces `perf_results_<timestamp>/perf_results.csv`.

For acceptance rates only (skips the sweep):

```bash
python evaluate.py throughput --target http://localhost:8000/v1
```

See [`examples/evaluate/`](https://github.com/vllm-project/speculators/tree/main/examples/evaluate) for end-to-end examples that launch a vLLM server and run the pipeline.

## Options

Both `throughput` and `sweep` share the same options:

```
  --target URL               vLLM server endpoint (required)
  --dataset DATASET          HF ID, local path, or prepared benchmark selector
                             (default: RedHatAI/speculator_benchmarks)
  --dataset-dir DIR          Directory containing prepared benchmark data
  --subsets LIST             Comma-separated HF subset names (default: all 9)
  --output-dir DIR           Output directory (default: perf_results_TIMESTAMP)
  --max-concurrency N        Max concurrent requests (default: 128)
  --max-requests N           Max requests per sweep point (default: 200)
  --gen-len-rate N           Request rate for gen-len estimation (default: 128)
  --sweep-rate N             Number of sweep rate points (default: 10)
  --gen-kwargs JSON          Generation kwargs, e.g. '{"temperature":0.6}'
  --data-column-mapper TEXT  Column mapping for guidellm in typed key=value format
                             (default: kind=generative_column_mapper,column_mappings.text_column=prompt)
```

A prepared benchmark selector identifies a supported benchmark, such as `speedbench/qualitative/coding` or `longbench-v2`. Pass its prepared files separately with `--dataset-dir`. The existing `--speedbench-data-dir` option remains accepted as a compatibility alias.

## SPEED-Bench

[NVIDIA SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) provides structured evaluation across qualitative categories (coding, math, reasoning, multilingual, …) and throughput splits with varying input sequence lengths (1 k–32 k tokens).

### One-time data preparation

SPEED-Bench prompts are fetched from external sources and cannot be redistributed directly. Run the preparation step once to materialise them locally:

```bash
# Fetch and materialise prompts, then split into per-category files (all in one command)
python scripts/evaluate/prepare_speedbench.py \
    --data-dir ./speedbench_data \
    --download

# Or run the two steps separately if you already have the flat files:
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
    | python3 - --output_dir ./speedbench_data
python scripts/evaluate/prepare_speedbench.py --data-dir ./speedbench_data
```

> **Note:** `prepare_speedbench.py` reads from the URL above to fetch NVIDIA's `prepare.py`. Save a local copy (`--download` does this implicitly) if you anticipate running data preparation again. The materialised files contain data from third-party sources — do not redistribute them.

### Running evaluations

Pass a `speedbench/<config>` selector to `--dataset` together with `--dataset-dir`:

```bash
# All 11 qualitative categories
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/qualitative \
    --dataset-dir ./speedbench_data

# Single category
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/qualitative/coding \
    --dataset-dir ./speedbench_data

# All throughput_1k subcategories
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/throughput_1k \
    --dataset-dir ./speedbench_data

# One entropy tier only
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/throughput_1k/high_entropy \
    --dataset-dir ./speedbench_data
```

Available configs: `qualitative`, `throughput_1k`, `throughput_2k`, `throughput_8k`, `throughput_32k`.

Results are written to `acceptance.csv` in the output directory with per-category acceptance lengths and per-position acceptance rates, identical in format to the `RedHatAI/speculator_benchmarks` output. Each row also carries `requests_successful`, `requests_errored` and `requests_incomplete`, so a result computed from a partly failed run stays recognisable as one.

## LongBench-v2

[LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) contains 503 multiple-choice questions with contexts ranging from 8K to 2M words. Prepare one local prompt-only file using the benchmark's official zero-shot template:

```bash
python scripts/evaluate/prepare_longbench_v2.py \
    --data-dir ./longbench_v2_data \
    --download
```

This downloads the pinned official JSON file (about 465 MB) and uses only the Python standard library. The prepared file keeps the full contexts, so configure the server's context window for the requests you intend to run.

```bash
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset longbench-v2 \
    --dataset-dir ./longbench_v2_data
```

This measures speculative-decoding throughput and acceptance; it does not compute LongBench-v2 answer accuracy or truncate prompts to a model-specific context window. Use the [official LongBench evaluation pipeline](https://github.com/THUDM/LongBench) for accuracy comparisons and model-specific truncation.

Rejections are routine rather than an edge case: measured with the Qwen3 tokenizer and 512 tokens reserved for generation, the median prompt is about 100K tokens, roughly 40% exceed a 128K window and about 77% exceed a 32K one. Rejections do not fail the run, so check `requests_errored` in `acceptance.csv` before comparing numbers — the longest prompts are the ones that drop out, biasing whatever survives toward shorter contexts.

## Visualization

```bash
# Compare multiple versions
python plot.py compare \
    --source "No Spec=nospec/perf_results.csv" \
    --source "DFlash=dflash/perf_results.csv" \
    --metric latency --output-dir ./plots

# Pairwise speedup (blue = faster, red = regression)
python plot.py speedup \
    --baseline "No Spec=nospec/perf_results.csv" \
    --target "DFlash=dflash/perf_results.csv" \
    --metric latency --title "Qwen3-8B" --output-dir ./plots
```

Both accept CSVs or raw GuideLLM sweep JSONs. Available metrics: `latency`, `itl`, `ttft`, `output_tps`.
