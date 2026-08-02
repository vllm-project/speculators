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
  --dataset DATASET          HF dataset ID, local path, or benchmark spec
  --dataset-dir DIR          Prepared data for a benchmark spec
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

Prepared benchmarks use a common syntax: `--dataset <adapter>[/<selection>] --dataset-dir <path>`. The existing `--speedbench-data-dir` option remains accepted as a compatibility alias.

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

Pass a `speedbench/<config>` spec to `--dataset` together with `--dataset-dir`:

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

Results are written to `acceptance.csv` in the output directory with per-category acceptance lengths and per-position acceptance rates, identical in format to the `RedHatAI/speculator_benchmarks` output.

## RULER v2

[RULER v2](https://github.com/NVIDIA/RULER/tree/rulerv2-ns) covers 12 long-context tasks across multi-key retrieval, multi-value retrieval, and multi-document QA. Its prompts depend on the model tokenizer and target context length, so generate them with [NeMo Skills](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset/ruler2):

```bash
# Run in a NeMo Skills environment with a configured local cluster.
ns prepare_data ruler2 \
    --cluster=local \
    --data_dir=./ruler2_data \
    --setup=qwen3-8b-32k \
    --tokenizer_path=Qwen/Qwen3-8B \
    --max_seq_length=32768
```

NeMo Skills writes the tasks under `./ruler2_data/ruler2/qwen3-8b-32k`. Pass that directory and setup name to `evaluate.py`:

```bash
# All 12 tasks
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset ruler2/qwen3-8b-32k \
    --dataset-dir ./ruler2_data/ruler2

# One task
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset ruler2/qwen3-8b-32k/mv_niah_hard \
    --dataset-dir ./ruler2_data/ruler2
```

This measures speculative-decoding performance and acceptance per RULER v2 task. Use NeMo Skills' `ns eval` pipeline when you also need the benchmark's answer-quality scores.

## LongBench

[LongBench](https://huggingface.co/datasets/zai-org/LongBench) contains 21 English, Chinese, and code tasks. Its rows keep the long `context` and task `input` in separate columns, and the Hugging Face repository uses a legacy dataset loading script. Prepare local prompt-only files once so the evaluator can use the benchmark's official per-task prompt templates with current `datasets` releases:

```bash
python scripts/evaluate/prepare_longbench.py \
    --data-dir ./longbench_data \
    --download
```

The download includes the official data archive (about 114 MB) and prompt-template configuration. The component datasets retain their original licenses; do not redistribute the prepared files without checking them.

```bash
# All 21 tasks
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset longbench \
    --dataset-dir ./longbench_data

# One task
python evaluate.py throughput \
    --target http://localhost:8000/v1 \
    --dataset longbench/hotpotqa \
    --dataset-dir ./longbench_data
```

This path measures speculative-decoding throughput and acceptance per task; it does not compute LongBench answer-quality scores. Use the official LongBench evaluation pipeline for accuracy comparisons.

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
