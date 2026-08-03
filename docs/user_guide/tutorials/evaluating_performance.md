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
  --dataset DATASET          HF dataset ID or local dir (default: RedHatAI/speculator_benchmarks)
  --subsets LIST             Comma-separated subset names (default: all 9)
  --output-dir DIR           Output directory (default: perf_results_TIMESTAMP)
  --max-concurrency N        Max concurrent requests (default: 128)
  --max-requests N           Max requests per sweep point (default: 200)
  --gen-len-rate N           Request rate for gen-len estimation (default: 128)
  --sweep-rate N             Number of sweep rate points (default: 10)
  --gen-kwargs JSON          Generation kwargs, e.g. '{"temperature":0.6}'
  --data-column-mapper TEXT  Column mapping for guidellm in typed key=value format
                             (default: kind=generative_column_mapper,column_mappings.text_column=prompt)
```

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

Pass a `speedbench/<config>` spec to `--dataset` together with `--speedbench-data-dir`:

```bash
# All 11 qualitative categories
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/qualitative \
    --speedbench-data-dir ./speedbench_data

# Single category
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/qualitative/coding \
    --speedbench-data-dir ./speedbench_data

# All throughput_1k subcategories
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/throughput_1k \
    --speedbench-data-dir ./speedbench_data

# One entropy tier only
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset speedbench/throughput_1k/high_entropy \
    --speedbench-data-dir ./speedbench_data
```

Available configs: `qualitative`, `throughput_1k`, `throughput_2k`, `throughput_8k`, `throughput_32k`.

Results are written to `acceptance.csv` in the output directory with per-category acceptance lengths and per-position acceptance rates, identical in format to the `RedHatAI/speculator_benchmarks` output.

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
