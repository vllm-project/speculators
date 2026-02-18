# vLLM Dataset Processing Pipeline

Scripts for processing datasets through vLLM servers with automatic model detection and multi-server support.

## Scripts Overview

### `run_all.sh` - Complete Pipeline Runner
Orchestrates the entire pipeline: starts vLLM servers, processes the dataset, and stops servers.

**Usage:**
```bash
# Process UltraChat dataset with single server
./run_all.sh

# Process Magpie dataset with limit
./run_all.sh --dataset magpie --limit 1000

# Use multiple servers with GPU assignment (one model per GPU pair)
./run_all.sh --ports "8000,8001" --gpus "0,1:2,3" --dataset magpie

# Run Llama 3.3 70B on 4 GPU pairs
./run_all.sh --ports "8000,8001,8002,8003" --gpus "0,1:2,3:4,5:6,7" --dataset magpie

# Keep servers running after processing
./run_all.sh --dataset ultrachat --keep-servers

# All script.py arguments work
./run_all.sh --dataset magpie --limit 500 --concurrency 128 --max-tokens 4096
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
- `--ports`: Comma-separated list of ports for multiple servers
- `--host`: Base host for servers (default: http://127.0.0.1)
- `--endpoint`: Single endpoint (used if --ports not set)
- `--model`: Model name (auto-detected if not specified)
- `--split`: Dataset split (default: train_sft)
- `--limit`: Stop after N rows
- `--concurrency`: Max concurrent requests (default: 64)
- `--max-tokens`: Max tokens for generation (default: 8192)
- `--outfile`: Output JSONL file (default: ultrachat_qwen3_vl.jsonl)
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

# Start multiple servers with GPU assignment
./start_vllm_servers.sh --ports "8000,8001" --gpus "0,1:2,3"

# Run Llama 3.3 70B on GPU pairs [0,1], [2,3], [4,5], [6,7]
./start_vllm_servers.sh --ports "8000,8001,8002,8003" --gpus "0,1:2,3:4,5:6,7"

# Use custom model
./start_vllm_servers.sh --ports "8000" --gpus "0,1" --model "meta-llama/Llama-3.3-70B-Instruct"
```

**GPU Assignment Format:**
- Use `:` to separate GPU groups for different servers
- Use `,` to list GPUs within a group
- Each GPU group is assigned to the corresponding port in order
- Example: `--gpus "0,1:2,3:4,5"` assigns GPUs 0,1 to first port, 2,3 to second port, 4,5 to third port

Logs are saved to `vllm_{port}.log` and PIDs to `vllm_pids.txt`.

### `stop_vllm_servers.sh` - Stop vLLM Servers
Stops all running vLLM servers.

**Usage:**
```bash
./stop_vllm_servers.sh
```

## GPU Configuration Examples

### Llama 3.3 70B on 8 GPUs (4 servers with 2 GPUs each)
```bash
./run_all.sh \
  --ports "8000,8001,8002,8003" \
  --gpus "0,1:2,3:4,5:6,7" \
  --dataset magpie
```

### Llama 3.3 70B on 4 GPUs (2 servers with 2 GPUs each)
```bash
./run_all.sh \
  --ports "8000,8001" \
  --gpus "0,1:2,3" \
  --dataset magpie
```

### Single server using all available GPUs
```bash
./run_all.sh --dataset magpie
```

## Supported Datasets

### Magpie
- Dataset ID: `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered`
- Prompt field: `instruction`

### UltraChat
- Dataset ID: `HuggingFaceH4/ultrachat_200k`
- Prompt field: `prompt`

## Workflow Examples

### Quick Start (All-in-One)
```bash
# Process 100 rows from Magpie dataset
./run_all.sh --dataset magpie --limit 100

# Process with GPU assignment for multiple servers
./run_all.sh \
  --ports "8000,8001" \
  --gpus "0,1:2,3" \
  --dataset magpie \
  --limit 1000
```

### Manual Control
```bash
# 1. Start servers with GPU assignment
./start_vllm_servers.sh --ports "8000,8001,8002" --gpus "0,1:2,3:4,5"

# 2. Process dataset
python script.py --ports "8000,8001,8002" --dataset magpie --limit 1000

# 3. Stop servers
./stop_vllm_servers.sh
```

### Resume Interrupted Processing
```bash
# If processing was interrupted, resume from where it left off
python script.py --dataset magpie --resume
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
