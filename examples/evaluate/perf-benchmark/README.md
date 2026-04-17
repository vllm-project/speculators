# Performance Benchmarking with GuideLLM

Noise-reduced serving performance benchmarks using [GuideLLM](https://github.com/neuralmagic/guidellm).

## Motivation

GuideLLM sweeps a range of request rates to build a performance profile. Because generation uses real prompts with sampling, output lengths can vary between runs of the same prompt. A few very long outliers can contaminate latency and throughput numbers, making benchmarks unreliable.

To mitigate this, this pipeline first runs a quick throughput pass over all prompts to record the distribution of generated output lengths. The **median** output length is then rounded up to the next power of 2 and used as `max_tokens` during the sweep. This caps outliers while keeping the truncation point representative of the typical output for that prompt distribution.

## Prerequisites

```bash
pip install -r requirements.txt
```

A running vLLM server is required. The pipeline does **not** start or stop the server -- manage it separately (see `../eval-guidellm/scripts/vllm_serve.sh`).

## Quick Start

```bash
./run_perf_benchmark.sh --target http://localhost:8000/v1
```

This runs all 9 subsets from `RedHatAI/speculator_benchmarks` through the full pipeline and produces a CSV at `perf_results_<timestamp>/perf_results.csv`.

## Usage

```
./run_perf_benchmark.sh --target URL [OPTIONS]

Required:
  --target URL               vLLM server endpoint (e.g. http://localhost:8000/v1)

Optional:
  --dataset DATASET          HF dataset ID or local dir
                             (default: RedHatAI/speculator_benchmarks)
  --subsets LIST             Comma-separated subset names
                             (default: HumanEval,math_reasoning,qa,question,
                              rag,summarization,tool_call,translation,writing)
  --output-dir DIR           Output directory (default: perf_results_TIMESTAMP)
  --max-concurrency N        Max concurrent requests for guidellm (default: 128)
  --max-requests N           Max requests per sweep point (default: 200)
  --gen-len-rate N           Request rate for gen-len estimation (default: 128)
  --gen-kwargs JSON          Flat JSON with generation kwargs, e.g.
                             '{"temperature":0.6, "top_p":0.95, "top_k":20}'
  --data-column-mapper JSON  Column mapping for guidellm
                             (default: '{"text_column":"prompt"}')
```

### Examples

Run only two subsets with custom sampling parameters:

```bash
./run_perf_benchmark.sh \
    --target http://localhost:8000/v1 \
    --subsets "HumanEval,qa" \
    --gen-kwargs '{"temperature":0.6, "top_p":0.95, "top_k":20}'
```

Run with a local dataset directory and custom column mapping:

```bash
./run_perf_benchmark.sh \
    --target http://localhost:8000/v1 \
    --dataset ./my_prompts/ \
    --subsets "code,chat" \
    --data-column-mapper '{"text_column":"input"}'
```

## Pipeline Steps

### Step 1: Output Length Estimation

For each subset, runs `guidellm benchmark --profile throughput` to generate responses for all prompts and record output token counts.

Script: `scripts/run_gen_len_estimation.sh`

### Step 2: Compute max_tokens

Parses the gen-len output JSONs, computes the median `output_tokens` per subset, and derives `max_tokens = 2^ceil(log2(median))`.

Script: `scripts/parse_gen_len.py`

Output: `max_tokens.json` mapping subset names to their max_tokens values.

### Step 3: Performance Sweep

For each subset, runs `guidellm benchmark --profile sweep` with the computed `max_tokens` injected into `--backend-args`.

Script: `scripts/run_sweep.sh`

### Step 4: Extract Metrics to CSV

Parses all sweep JSONs and extracts median metrics for each sweep point. Throughput-mode entries are excluded (they represent max-load saturation).

Script: `scripts/parse_sweep_results.py`

### CSV Output Format

| Column              | Description                                 |
| ------------------- | ------------------------------------------- |
| `subset`            | Dataset subset name                         |
| `strategy`          | `synchronous` or `constant`                 |
| `target_rate`       | Target request rate (empty for synchronous) |
| `rps_median`        | Median requests per second                  |
| `latency_median_s`  | Median request latency in seconds           |
| `itl_median_ms`     | Median inter-token latency in milliseconds  |
| `ttft_median_ms`    | Median time to first token in milliseconds  |
| `output_tps_median` | Median output tokens per second             |

## Output Directory Structure

```
perf_results_YYYYMMDD_HHMMSS/
├── gen_len/
│   ├── gen_len_HumanEval.json
│   ├── gen_len_qa.json
│   └── ...
├── sweeps/
│   ├── sweep_HumanEval.json
│   ├── sweep_qa.json
│   └── ...
├── max_tokens.json
└── perf_results.csv
```

## Using Individual Scripts

Each script can be used standalone:

```bash
# Estimate output lengths for a single subset
scripts/run_gen_len_estimation.sh \
    --target http://localhost:8000/v1 \
    --dataset RedHatAI/speculator_benchmarks \
    --subset HumanEval

# Parse gen-len results
python scripts/parse_gen_len.py --output max_tokens.json gen_len_*.json

# Run a single sweep
scripts/run_sweep.sh \
    --target http://localhost:8000/v1 \
    --dataset RedHatAI/speculator_benchmarks \
    --subset HumanEval \
    --max-tokens 4096

# Parse sweep results
python scripts/parse_sweep_results.py --output results.csv sweep_*.json
```

## Visualization

Two plotting scripts are available for visualizing performance results. Both accept CSV files (from `parse_sweep_results.py`) or raw GuideLLM sweep JSONs.

Data is noisy, so both scripts apply a smoothing pipeline: binning, isotonic regression, and PCHIP interpolation. Raw data points are shown as scatter markers alongside the smooth curves.

### Multi-version comparison (`plot_perf_compare.py`)

Overlays performance curves for multiple model versions on the same plot.

```bash
python scripts/plot_perf_compare.py \
    --source "No Spec=nospec/perf_results.csv" \
    --source "DFlash=dflash/perf_results.csv" \
    --source "Eagle3 k3=eagle3/sweep_HumanEval.json" \
    --metric latency \
    --output-dir ./plots
```

| Argument              | Description                                                                             |
| --------------------- | --------------------------------------------------------------------------------------- |
| `--source LABEL=PATH` | Version data source (repeatable; same label pools repetitions)                          |
| `--metric METRIC`     | Metric to plot: `latency`, `itl`, `ttft`, `output_tps` (repeatable, default: `latency`) |
| `--output-dir DIR`    | Output directory for PNGs (default: current directory)                                  |
| `--subsets LIST`      | Comma-separated subset filter (default: all)                                            |

Output: one PNG per (subset, metric), e.g. `compare_HumanEval_latency.png`.

### Pairwise speedup (`plot_perf_speedup.py`)

Compares exactly two versions with a gradient-shaded region between the curves showing local speedup intensity. Uses the `bwr` colormap: blue = target is faster (speedup > 1), red = target is slower (regression).

```bash
python scripts/plot_perf_speedup.py \
    --baseline "No Spec=nospec/perf_results.csv" \
    --target "DFlash=dflash/perf_results.csv" \
    --metric latency \
    --title "Qwen3-8B" \
    --output-dir ./plots
```

| Argument                | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `--baseline LABEL=PATH` | Baseline version (repeatable for pooling reps)               |
| `--target LABEL=PATH`   | Target version being evaluated (repeatable for pooling reps) |
| `--metric METRIC`       | Metric to plot (repeatable, default: `latency`)              |
| `--title TEXT`          | Optional title prefix (e.g. model name)                      |
| `--output-dir DIR`      | Output directory for PNGs (default: current directory)       |
| `--subsets LIST`        | Comma-separated subset filter (default: all)                 |

Output: one PNG per (subset, metric), e.g. `speedup_HumanEval_latency.png`.
