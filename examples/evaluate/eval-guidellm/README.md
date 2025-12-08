# Speculator Model Evaluation with GuideLLM

Evaluate speculator models using vLLM and GuideLLM, and extract acceptance length metrics.

## Quick Start

**1. Install dependencies:**
```bash
bash setup.sh  # or: bash setup.sh --use-uv for faster installation
```

**2. Run evaluation with a pre-configured model:**
```bash
# Llama-3.1-8B EAGLE3 on math_reasoning dataset
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env

# Llama-3.3-70B EAGLE3 on math_reasoning dataset
./run_evaluation.sh -c configs/llama-3.3-70b-eagle3.env

# GPT-OSS-20B EAGLE3 on math_reasoning dataset
./run_evaluation.sh -c configs/gpt-oss-20b-eagle3.env

# Qwen3-8B EAGLE3 on math_reasoning dataset
./run_evaluation.sh -c configs/qwen3-8b-eagle3.env

# Qwen3-32B EAGLE3 on math_reasoning dataset
./run_evaluation.sh -c configs/qwen3-32b-eagle3.env
```

**Or run with custom parameters:**
```bash
./run_evaluation.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d "emulated"
```

Results will be in a timestamped directory like `eval_results_20251203_165432/`.

## Architecture

This framework consists of modular scripts organized in a clean directory structure:

```
eval-guidellm/
├── run_evaluation.sh              # Main controller
├── configs/                       # Pre-configured evaluations
│   ├── llama-3.1-8b-eagle3.env    # Llama-3.1-8B
│   ├── llama-3.3-70b-eagle3.env   # Llama-3.3-70B
│   ├── gpt-oss-20b-eagle3.env     # GPT-OSS-20B
│   ├── qwen3-8b-eagle3.env        # Qwen3-8B
│   └── qwen3-32b-eagle3.env       # Qwen3-32B
├── scripts/                       # Utility scripts
│   ├── vllm_serve.sh
│   ├── vllm_stop.sh
│   ├── run_guidellm.sh
│   └── parse_logs.py
└── setup.sh                       # Install dependencies
```

## Configuration

### Pre-configured Models

The framework includes configs for common models:

```bash
# Llama-3.1-8B EAGLE3 on math_reasoning
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env

# Llama-3.3-70B EAGLE3 on math_reasoning
./run_evaluation.sh -c configs/llama-3.3-70b-eagle3.env

# GPT-OSS-20B EAGLE3 on math_reasoning
./run_evaluation.sh -c configs/gpt-oss-20b-eagle3.env

# Qwen3-8B EAGLE3 on math_reasoning
./run_evaluation.sh -c configs/qwen3-8b-eagle3.env

# Qwen3-32B EAGLE3 on math_reasoning
./run_evaluation.sh -c configs/qwen3-32b-eagle3.env
```

### Command Line Usage

```bash
./run_evaluation.sh -m MODEL -d DATASET [OPTIONS]

Required:
  -m MODEL      Speculator model (e.g., "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3")
  -d DATASET    Dataset for benchmarking (see Dataset Options below)

Optional:
  -c FILE       Config file to use (e.g., configs/llama-eagle3.env)
  -o DIR        Output directory (default: eval_results_TIMESTAMP)
```

### Creating Custom Configs

Create a new config file in `configs/`:

```bash
# configs/my-model.env
MODEL="my-org/my-model"
DATASET="RedHatAI/speculator_benchmarks:math_reasoning.jsonl"
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.8
PORT=8000
HEALTH_CHECK_TIMEOUT=300

# Sampling parameters
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20

OUTPUT_DIR="eval_results_$(date +%Y%m%d_%H%M%S)"
```

Then run:
```bash
./run_evaluation.sh -c configs/my-model.env
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `MODEL` | Speculator model path or HuggingFace ID | (required) |
| `DATASET` | Dataset for benchmarking (emulated, HF dataset, or file path) | (required) |
| `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | 2 |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction to use | 0.8 |
| `PORT` | Server port | 8000 |
| `HEALTH_CHECK_TIMEOUT` | Server startup timeout (seconds) | 300 |
| `TEMPERATURE` | Sampling temperature | 0.6 |
| `TOP_P` | Top-p (nucleus) sampling parameter | 0.95 |
| `TOP_K` | Top-k sampling parameter | 20 |
| `OUTPUT_DIR` | Output directory | `eval_results_TIMESTAMP` |

### Dataset Options

The framework supports five types of dataset inputs:

1. **Built-in datasets**: `emulated` (included with guidellm)
   - Example: `DATASET="emulated"`

2. **HuggingFace datasets (all files)**: `org/dataset-name`
   - Automatically downloaded using HuggingFace CLI
   - Runs benchmarks on **all** .jsonl files in the dataset
   - Example: `DATASET="RedHatAI/speculator_benchmarks"`

3. **HuggingFace datasets (specific file)**: `org/dataset-name:filename.jsonl`
   - Downloads the dataset and uses only the specified file
   - Use colon (`:`) to separate dataset from filename
   - Example: `DATASET="RedHatAI/speculator_benchmarks:math_reasoning.jsonl"`

4. **Local directory**: Path to a folder containing .jsonl files
   - Runs benchmarks on **all** .jsonl files in the directory
   - Results are saved with dataset-specific filenames
   - Example: `DATASET="./my_datasets/"`

5. **Local file**: Path to a single .jsonl file
   - Runs benchmark on that specific file
   - Example: `DATASET="./my_data.jsonl"`

## Advanced Usage

### Manual Workflow

For debugging or running multiple benchmarks against the same server:

```bash
# Terminal 1: Start server
./scripts/vllm_serve.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --log-file server.log \
  --pid-file server.pid

# Terminal 2: Run benchmarks
./scripts/run_guidellm.sh -d "dataset1.jsonl" --output-file results1.json
./scripts/run_guidellm.sh -d "dataset2.jsonl" --output-file results2.json

# Parse acceptance metrics
python scripts/parse_logs.py server.log -o acceptance_stats.txt

# Terminal 1: Stop server
./scripts/vllm_stop.sh --pid-file server.pid
```

## Output Files

All results are saved in a timestamped output directory.

### Single Dataset

```
eval_results_20251203_165432/
├── vllm_server.log          # vLLM server output (used for parsing)
├── guidellm_output.log      # GuideLLM benchmark progress
├── guidellm_results.json    # GuideLLM performance metrics
└── acceptance_analysis.txt  # Acceptance length statistics
```

### Multiple Datasets (Directory or HuggingFace)

When using a directory or HuggingFace dataset with multiple .jsonl files:

```
eval_results_20251203_165432/
├── vllm_server.log                    # vLLM server output (all benchmarks)
├── guidellm_output_dataset1.log       # Benchmark progress for dataset1
├── guidellm_output_dataset2.log       # Benchmark progress for dataset2
├── guidellm_results_dataset1.json     # Performance metrics for dataset1
├── guidellm_results_dataset2.json     # Performance metrics for dataset2
└── acceptance_analysis.txt            # Combined acceptance statistics
```

### Acceptance Metrics

The `acceptance_analysis.txt` contains:
- **Weighted acceptance rates**: Per-position acceptance rates weighted by draft tokens
- **Conditional acceptance rates**: Probability of accepting position N given position N-1 was accepted

These metrics help evaluate the effectiveness of speculative decoding.

## Examples

### Using Pre-configured Models
```bash
./run_evaluation.sh -c configs/llama-3.1-8b-eagle3.env
./run_evaluation.sh -c configs/llama-3.3-70b-eagle3.env
./run_evaluation.sh -c configs/gpt-oss-20b-eagle3.env
./run_evaluation.sh -c configs/qwen3-8b-eagle3.env
./run_evaluation.sh -c configs/qwen3-32b-eagle3.env
```

### Quick Test with Emulated Dataset
```bash
./run_evaluation.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d "emulated"
```

### HuggingFace Dataset (Specific File)
```bash
./run_evaluation.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d "RedHatAI/speculator_benchmarks:math_reasoning.jsonl"
```

### HuggingFace Dataset (All Files)
```bash
./run_evaluation.sh \
  -m "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" \
  -d "RedHatAI/speculator_benchmarks"
```

### Local File or Directory
```bash
# Single file
./run_evaluation.sh -m MODEL -d "./my_data.jsonl"

# All .jsonl files in directory
./run_evaluation.sh -m MODEL -d "./my_datasets/"
```


## Troubleshooting

**Server won't start:**
```bash
tail -n 50 eval_results_*/vllm_server.log  # Check logs
nvidia-smi                                  # Verify GPU availability
```

**Dataset not found:**
```bash
hf download DATASET --repo-type dataset  # Test HF dataset download
./run_evaluation.sh -m MODEL -d emulated # Use built-in dataset
```

**Server cleanup:**
```bash
./scripts/vllm_stop.sh                   # Graceful shutdown
pkill -9 -f "vllm serve"                 # Force kill if needed
```

## Dependencies

Required: Python 3.9+, vLLM, GuideLLM, HuggingFace CLI, curl

```bash
bash setup.sh              # Install with pip
bash setup.sh --use-uv     # Install with uv (faster)
```
