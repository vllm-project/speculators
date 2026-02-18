# vLLM Dataset Processing Pipeline

Scripts for processing datasets through vLLM servers with automatic model detection and multi-server support.

## Scripts Overview

### `run_all.sh` - Complete Pipeline Runner
Orchestrates the entire pipeline: starts vLLM servers, processes the dataset, and stops servers.

**Usage:**
```bash
# Process UltraChat dataset with single server (auto-detect model)
./run_all.sh

# Specify model explicitly
./run_all.sh --model "meta-llama/Llama-3.3-70B-Instruct" --dataset magpie

# Process Magpie dataset with limit
./run_all.sh --dataset magpie --limit 1000

# Use multiple servers with GPU assignment (one model per GPU pair)
./run_all.sh --ports "8000,8001" --gpus "0,1:2,3" --dataset magpie

# Run Llama 3.3 70B on 4 GPU pairs with explicit model
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001,8002,8003" \
  --gpus "0,1:2,3:4,5:6,7" \
  --dataset magpie

# Keep servers running after processing
./run_all.sh --dataset ultrachat --keep-servers

# All script.py arguments work (output: magpie_Llama-3.3-70B-Instruct.jsonl)
./run_all.sh --dataset magpie --limit 500 --concurrency 128 --max-tokens 4096

# Custom output filename
./run_all.sh --dataset ultrachat --outfile my_custom_output.jsonl
```

### `script.py` - Dataset Processing Script
Processes datasets through vLLM chat completion endpoints with automatic model detection.

**Features:**
- Auto-detects model from vLLM server (no need to specify `--model`)
- Supports multiple datasets (Magpie and UltraChat)
- Multi-server load balancing
- Resume capability to skip already-processed rows
- Async processing with configurable concurrency

**Usage:**
```bash
# Basic usage (assumes servers already running)
python script.py

# Specify dataset
python script.py --dataset magpie
python script.py --dataset ultrachat

# Use multiple servers
python script.py --ports "8000,8001,8002"

# Limit number of rows
python script.py --dataset magpie --limit 1000

# Resume from previous run
python script.py --resume

# Custom concurrency and output
python script.py --concurrency 128 --outfile my_results.jsonl
```

**Arguments:**
- `--dataset`: Choose `magpie` or `ultrachat` (default: ultrachat)
- `--model`: Model name (auto-detected from vLLM server if not specified)
- `--ports`: Comma-separated list of ports for multiple servers
- `--host`: Base host for servers (default: http://127.0.0.1)
- `--endpoint`: Single endpoint (used if --ports not set)
- `--split`: Dataset split (defaults to `train` for magpie, `train_sft` for ultrachat)
- `--limit`: Stop after N rows
- `--concurrency`: Max concurrent requests (default: 64)
- `--max-tokens`: Max tokens for generation (default: 8192)
- `--outfile`: Output JSONL file (auto-generated as `{dataset}_{model}.jsonl` if not specified)
- `--resume`: Skip already processed rows
- `--language-filter`: Only process specific language (e.g., EN)

### `start_vllm_servers.sh` - Start vLLM Servers
Starts one or more vLLM servers on specified ports with GPU assignment.

**Usage:**
```bash
# Start single server on port 8000 (uses all GPUs)
./start_vllm_servers.sh

# Start single server with specific GPUs
./start_vllm_servers.sh --ports "8000" --gpus "0,1,2,3"

# Start multiple servers with GPU assignment (TP auto-configured)
./start_vllm_servers.sh --ports "8000,8001" --gpus "0,1:2,3"
# Server on 8000: GPUs 0,1 with TP=2
# Server on 8001: GPUs 2,3 with TP=2

# Run Llama 3.3 70B on GPU pairs [0,1], [2,3], [4,5], [6,7]
./start_vllm_servers.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001,8002,8003" \
  --gpus "0,1:2,3:4,5:6,7"
# Each server gets TP=2

# Use custom model
./start_vllm_servers.sh --ports "8000" --gpus "0,1" --model "meta-llama/Llama-3.3-70B-Instruct"
```

**GPU Assignment Format:**
- Use `:` to separate GPU groups for different servers
- Use `,` to list GPUs within a group
- Each GPU group is assigned to the corresponding port in order
- Example: `--gpus "0,1:2,3:4,5"` assigns GPUs 0,1 to first port, 2,3 to second port, 4,5 to third port
- **Tensor parallelism is automatically configured** based on the number of GPUs per group
  - `"0,1"` → `--tensor-parallel-size 2`
  - `"0,1,2,3"` → `--tensor-parallel-size 4`
  - `"0,1,2,3,4,5,6,7"` → `--tensor-parallel-size 8`

**Output files:**
- Logs: `vllm_{port}.log` (one per server)
- PIDs: `vllm_pids.txt` (used by `stop_vllm_servers.sh`)

### `stop_vllm_servers.sh` - Stop vLLM Servers
Stops all vLLM servers started with `start_vllm_servers.sh` using the saved PID file.

**Usage:**
```bash
./stop_vllm_servers.sh
```

**Note:** This script requires the `vllm_pids.txt` file created by `start_vllm_servers.sh`. If you need to manually stop vLLM processes:
```bash
ps aux | grep 'vllm serve'
kill <PID>
```

## GPU Configuration Examples

### Llama 3.3 70B on 8 GPUs (4 servers with 2 GPUs each)
```bash
# Each server gets 2 GPUs with TP=2 (tensor parallel size auto-set)
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001,8002,8003" \
  --gpus "0,1:2,3:4,5:6,7" \
  --dataset magpie
```

### Llama 3.3 70B on 4 GPUs (2 servers with 2 GPUs each)
```bash
# Each server gets 2 GPUs with TP=2
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001" \
  --gpus "0,1:2,3" \
  --dataset magpie
```

### Qwen 235B on multiple GPU sets (4 GPUs per server)
```bash
# Each server gets 4 GPUs with TP=4
./run_all.sh \
  --model "Qwen/Qwen3-VL-235B-A22B-Instruct" \
  --ports "8000,8001" \
  --gpus "0,1,2,3:4,5,6,7" \
  --dataset ultrachat
```

### Single server using all available GPUs (model auto-detected)
```bash
./run_all.sh --dataset magpie
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
# Process 100 rows from Magpie dataset (model auto-detected)
./run_all.sh --dataset magpie --limit 100

# Process with specific model and GPU assignment
./run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001" \
  --gpus "0,1:2,3" \
  --dataset magpie \
  --limit 1000
```

### Manual Control
```bash
# 1. Start servers with model and GPU assignment
./start_vllm_servers.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001,8002" \
  --gpus "0,1:2,3:4,5"

# 2. Process dataset (model auto-detected from servers)
python script.py --ports "8000,8001,8002" --dataset magpie --limit 1000

# Or specify model explicitly
python script.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --ports "8000,8001,8002" \
  --dataset magpie \
  --limit 1000

# 3. Stop servers
./stop_vllm_servers.sh
```

### Resume Interrupted Processing
```bash
# If processing was interrupted, resume from where it left off
# Make sure to use the same output file (auto-generated or specified)
python script.py --dataset magpie --resume

# Or with explicit output file
python script.py --dataset magpie --outfile magpie_Llama-3.3-70B-Instruct.jsonl --resume
```

## Output Format

Results are saved as JSONL in a conversations format compatible with fine-tuning datasets. The `id` field uses the dataset's UUID if available, otherwise falls back to `sample_{idx}`.

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
