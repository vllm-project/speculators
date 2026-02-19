# Response Regeneration Pipeline

Regenerate assistant responses in existing datasets using a vLLM-served model. Given a dataset containing user prompts (e.g., Magpie, UltraChat), this pipeline extracts the prompts, sends them to a vLLM server, and produces a new dataset with the original prompts paired with freshly generated responses from the target model. This is useful for creating training data where you want a specific model's outputs in place of the original assistant responses.

Uses vLLM's built-in data parallelism (`--data-parallel-size`) for multi-GPU scaling with automatic load balancing.

## Scripts Overview

### `run_all.sh` - Complete Pipeline Runner
Orchestrates the entire pipeline: starts a vLLM server (with optional data/tensor parallelism), regenerates responses for the dataset, and stops the server.

**Usage:**
```bash
# Basic usage
./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie

# Process Magpie dataset with limit
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset magpie --limit 1000

# Select specific GPUs
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --gpus 0,1,2,4 --tp-size 4 --dataset magpie

# Use data parallelism (4 replicas, each with TP=2, uses 8 GPUs)
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dp-size 4 --tp-size 2 --dataset magpie

# Keep server running after processing
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset ultrachat --keep-server

# All script.py arguments work (output: magpie_Llama-3.3-70B-Instruct.jsonl)
./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie --limit 500 --concurrency 128 --max-tokens 4096

# Custom output filename
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset ultrachat --outfile my_custom_output.jsonl
```

**Arguments:**
- `--model`: Model to serve (required)
- `--gpus`: Comma-separated GPU IDs (sets `CUDA_VISIBLE_DEVICES`)
- `--port`: Server port (default: 8000)
- `--dp-size`: Number of data parallel replicas (maps to `--data-parallel-size`)
- `--tp-size`: Tensor parallel size per replica (maps to `--tensor-parallel-size`)
- `--keep-server`: Don't stop the server after processing
- All other arguments are passed through to `script.py`

### `script.py` - Response Regeneration Script
Extracts user prompts from a dataset, sends them to a vLLM chat completion endpoint, and writes out new prompt-response pairs with the target model's generated responses.

**Features:**
- Auto-detects model from vLLM server (no need to specify `--model`)
- Supports multiple datasets (Magpie and UltraChat)
- Resume capability to skip already-processed rows
- Async processing with configurable concurrency

**Usage:**
```bash
# Basic usage (assumes server already running)
python script.py

# Specify dataset
python script.py --dataset magpie
python script.py --dataset ultrachat

# Limit number of rows
python script.py --dataset magpie --limit 1000

# Resume from previous run
python script.py --resume

# Custom concurrency and output
python script.py --concurrency 128 --outfile my_results.jsonl

# Custom endpoint
python script.py --endpoint http://127.0.0.1:9000/v1/chat/completions
```

**Arguments:**
- `--dataset`: Choose `magpie` or `ultrachat` (default: ultrachat)
- `--model`: Model name (auto-detected from vLLM server if not specified)
- `--endpoint`: vLLM chat completions endpoint (default: `http://127.0.0.1:8000/v1/chat/completions`)
- `--split`: Dataset split (defaults to `train` for magpie, `train_sft` for ultrachat)
- `--limit`: Stop after N rows
- `--concurrency`: Max concurrent requests (default: 64)
- `--max-tokens`: Max tokens for generation (default: 8192)
- `--outfile`: Output JSONL file (auto-generated as `{dataset}_{model}.jsonl` if not specified)
- `--resume`: Skip already processed rows
- `--language-filter`: Only process specific language (e.g., EN)

## GPU Configuration Examples

### Llama 3.3 70B on 8 GPUs (4 data-parallel replicas with TP=2)
```bash
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 4 --tp-size 2 \
  --dataset magpie
```

### Llama 3.3 70B on 4 GPUs (2 data-parallel replicas with TP=2)
```bash
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 2 --tp-size 2 \
  --dataset magpie
```

### Qwen 235B on 8 GPUs (2 data-parallel replicas with TP=4)
```bash
./run_all.sh \
  --model "Qwen/Qwen3-VL-235B-A22B-Instruct" \
  --dp-size 2 --tp-size 4 \
  --dataset ultrachat
```

### Single replica using all available GPUs
```bash
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset magpie
```

## Supported Datasets

### Magpie
- Dataset ID: `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered`
- Prompt field: `instruction`
- Default split: `train`

### UltraChat
- Dataset ID: `HuggingFaceH4/ultrachat_200k`
- Prompt field: `prompt`
- Default split: `train_sft`

## Workflow Examples

### Quick Start (All-in-One)
```bash
# Process 100 rows from Magpie dataset
./run_all.sh --model "Qwen/Qwen2.5-72B-Instruct" --dataset magpie --limit 100

# Process with specific model and data parallelism
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 2 --tp-size 2 \
  --dataset magpie \
  --limit 1000
```

### Manual Control
```bash
# 1. Start server manually with data parallelism
vllm serve "meta-llama/Llama-3.3-70B-Instruct" \
  --data-parallel-size 4 --tensor-parallel-size 2 \
  --port 8000

# 2. Run regeneration (model auto-detected from server)
python script.py --dataset magpie --limit 1000

# 3. Stop server (Ctrl+C or kill the process)
```

### Resume Interrupted Processing
```bash
# If processing was interrupted, resume from where it left off
python script.py --dataset magpie --resume

# Or with explicit output file
python script.py --dataset magpie --outfile magpie_Llama-3.3-70B-Instruct.jsonl --resume
```

## Output Format

Each row in the output pairs the original user prompt with the newly generated response from the target model, saved as JSONL in a conversations format compatible with fine-tuning. The `id` field uses the dataset's UUID if available, otherwise falls back to `sample_{idx}`.

Each line contains:
```json
{
  "id": "sample_0",
  "conversations": [
    {
      "from": "human",
      "value": "What is the capital of France?"
    },
    {
      "from": "gpt",
      "value": "The capital of France is Paris."
    }
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "latency_s": 1.234,
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
    "reasoning_content": "..." // Only included if model provides reasoning
  }
}
```

Note: The `reasoning_content` field in metadata is only included when the model actually provides reasoning content (e.g., with reasoning models). For standard models, this field will not be present.

**Output Filenames:**

If you don't specify `--outfile`, the filename is auto-generated based on dataset and model:
- `magpie_Llama-3.3-70B-Instruct.jsonl`
- `ultrachat_Qwen3-VL-235B-A22B-Instruct.jsonl`
- `magpie_Qwen2.5-72B-Instruct.jsonl`

You can override with `--outfile custom_name.jsonl`.

Errors are logged as:
```json
{
  "id": "sample_0",
  "conversations": [
    {
      "from": "human",
      "value": "What is the capital of France?"
    }
  ],
  "metadata": {
    "idx": 0,
    "error": "ConnectionError(...)",
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```
