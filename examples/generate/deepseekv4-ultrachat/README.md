# DeepSeek-V4-Flash UltraChat Response Regeneration

Regenerates UltraChat (or Magpie) responses using DeepSeek-V4-Flash served via vLLM.
Handles GPU reservation via `chg`, server lifecycle, retries, and deadline tracking.

Reuses [`scripts/response_regeneration/script.py`](../../../scripts/response_regeneration/script.py) for the actual generation.

## Prerequisites

- `chg` available in PATH (GPU reservation CLI)
- `vllm` installed in the active Python environment
- `HF_HUB_CACHE` set to a directory with the DeepSeek-V4-Flash weights

## Quick start

```bash
cd examples/generate/deepseekv4-ultrachat

# Test run — 50 samples to check response quality
bash run.sh --limit 50

# Full UltraChat run
bash run.sh
```

## All options

```
bash run.sh [OPTIONS]

  --dataset STR         Dataset to process: ultrachat | magpie  (default: ultrachat)
  --limit N             Stop after N rows. Omit to process the entire dataset.
  --concurrency N       Concurrent HTTP requests to the vLLM server  (default: 64)
  --max-tokens N        Max tokens per generated response            (default: 8192)
  --resume              Skip rows already written to the output file
  --output-dir DIR      Directory for results and server log         (default: results_YYYYMMDD_HHMMSS)
  --max-runtime N       Global deadline in hours before the script exits  (default: 12)
  --reserve-duration S  Duration string passed to chg reserve        (default: 2h)
```

## Common invocations

```bash
# Test run — check a few responses before committing to the full dataset
bash run.sh --limit 50

# Full UltraChat run
bash run.sh

# Magpie instead of UltraChat
bash run.sh --dataset magpie

# Higher concurrency for faster throughput
bash run.sh --concurrency 128

# Resume a run that was interrupted (point at the previous output directory)
bash run.sh --output-dir results_20260429_120000 --resume

# Longer reservation for a big run
bash run.sh --reserve-duration 4h --max-runtime 20

# Custom output directory
bash run.sh --output-dir /data/regen/deepseekv4_ultrachat
```

## Output

Results are written to `<output-dir>/ultrachat_DeepSeek-V4-Flash.jsonl` as one JSON object per line:

```json
{
  "id": "sample_0",
  "conversations": [
    {"from": "human", "value": "...original prompt..."},
    {"from": "gpt",   "value": "...generated response..."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "latency_s": 2.341,
    "usage": {"prompt_tokens": 42, "completion_tokens": 187, "total_tokens": 229},
    "endpoint": "http://localhost:8000/v1/chat/completions"
  }
}
```

Failed requests are also written with an `"error"` key in metadata so the file stays appendable for `--resume`.

The server log is saved to `<output-dir>/server.log`.

## vLLM serve command

`serve.sh` runs the following (with `CUDA_VISIBLE_DEVICES` set from the reserved GPU IDs):

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --host 127.0.0.1 \
    --port <PORT> \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --attention_config.use_fp4_indexer_cache=True \
    --moe-backend deep_gemm_mega_moe \
    --speculative_config '{"method":"mtp","num_speculative_tokens":2}'
```

## Workflow

```
run.sh
 ├── acquire 8 GPUs via chg reserve
 ├── start serve.sh (background process)
 ├── poll /health every 30s (up to 3600s)
 ├── run scripts/response_regeneration/script.py
 └── cleanup: kill server, release GPUs  ← runs on EXIT / Ctrl-C / SIGTERM
```

GPU reservation is renewed automatically if < 30 minutes remain on the current reservation.
