# Evaluating Speculator Models with GuideLLM

Automate evaluation of speculator models and extract acceptance length metrics.

## Quick Start

**1. Install dependencies:**
```bash
bash setup.sh
```

**2. Run evaluation:**
```bash
# With built-in emulated dataset
./run_eval.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d emulated

# With HuggingFace dataset (automatically downloaded)
./run_eval.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d "RedHatAI/speculator_benchmarks"
```

## Command Line Options

```bash
./run_eval.sh -m MODEL -d DATASET [OPTIONS]
```

### Required Arguments
- `-m MODEL` - Speculator model path or HuggingFace model ID
- `-d DATASET` - Dataset for guidellm benchmarking
  - Built-in datasets: `emulated` (included with guidellm)
  - HuggingFace datasets: `org/dataset-name` (e.g., `RedHatAI/speculator_benchmarks`)
    - Automatically downloaded using HuggingFace CLI
    - First data file (.json/.jsonl) is used automatically
  - Local files: Path to a local .json or .jsonl file

### Optional Arguments
- `--tensor-parallel-size SIZE` - Number of GPUs (default: 2)
- `--gpu-memory-utilization UTIL` - GPU memory fraction (default: 0.8)
- `--port PORT` - Server port (default: 8000)
- `--max-seconds SECONDS` - Benchmark duration (default: 600)
- `-o OUTPUT_DIR` - Output directory (default: eval_results_TIMESTAMP)

## Output Files

All results are saved to `eval_results_TIMESTAMP/`:

| File | Description |
|------|-------------|
| `vllm_server.log` | vLLM server logs with SpecDecoding metrics |
| `guidellm_output.log` | GuideLLM console output |
| `guidellm_results.json` | GuideLLM benchmark results |
| `acceptance_analysis.txt` | Parsed acceptance rate statistics |

## Understanding the Results

The `acceptance_analysis.txt` file contains two key metrics:

**1. Weighted per-position acceptance rates:**
- Average acceptance rate at each draft position
- Weighted by number of tokens drafted

**2. Conditional acceptance rates:**
- Probability of accepting position i given position i-1 was accepted
- Formula: P(accept_i | accept_{i-1})

### Example Output
```
======================================================================
Speculative Decoding Acceptance Analysis
======================================================================

Total samples: 150
Total drafted tokens: 750
Average drafted tokens: 5.00

Weighted per-position acceptance rates:
[0.95  0.85  0.72  0.58  0.42]

Conditional acceptance rates:
[0.947 0.894 0.847 0.806 0.724]
======================================================================
```

**Interpretation:**
- Position 0: 95% of first draft tokens are accepted
- Position 1: 85% of second draft tokens are accepted
- Conditional rate at position 1: 89.4% acceptance given position 0 accepted
