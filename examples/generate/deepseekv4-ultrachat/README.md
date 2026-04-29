# DeepSeek-V4-Flash UltraChat Response Regeneration

Regenerates UltraChat (or Magpie) responses using DeepSeek-V4-Flash served via vLLM.
Handles GPU reservation via `chg`, server lifecycle, and graceful cleanup.

Reuses [`scripts/response_regeneration/script.py`](../../../scripts/response_regeneration/script.py) for the actual generation.

## Prerequisites

- `chg` available in PATH (GPU reservation CLI)
- **Native mode**: `vllm` installed in the active Python environment
- **Docker mode**: Docker with NVIDIA runtime; `HF_HUB_CACHE` set to the model weights directory
- `HF_HUB_CACHE` set (Docker mode always requires it; native mode uses it implicitly from the environment)

## Quick start

```bash
cd examples/generate/deepseekv4-ultrachat

# Test run — native vllm, 50 samples to check response quality
bash run.sh --limit 50

# Test run — Docker on H100, 50 samples
bash run.sh --mode docker --hardware h100 --limit 50

# Full UltraChat run — native
bash run.sh

# Full UltraChat run — Docker on B200
bash run.sh --mode docker --hardware b200
```

## All options

```
bash run.sh [OPTIONS]

  --mode STR            How to launch the server: native | docker   (default: native)
  --hardware STR        GPU hardware profile: h100 | b200           (default: h100; docker mode only)
  --dataset STR         Dataset to process: ultrachat | magpie      (default: ultrachat)
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
# Test runs
bash run.sh --limit 50
bash run.sh --mode docker --hardware h100 --limit 50
bash run.sh --mode docker --hardware b200 --limit 50

# Full dataset runs
bash run.sh
bash run.sh --mode docker --hardware h100
bash run.sh --mode docker --hardware b200

# Magpie instead of UltraChat
bash run.sh --dataset magpie
bash run.sh --mode docker --hardware h100 --dataset magpie

# Higher concurrency for faster throughput
bash run.sh --concurrency 128

# Resume a run that was interrupted (point at the previous output directory)
bash run.sh --output-dir results_20260429_120000 --resume
bash run.sh --mode docker --hardware h100 --output-dir results_20260429_120000 --resume

# Longer reservation and deadline for a big run
bash run.sh --reserve-duration 4h --max-runtime 20

# Custom output directory
bash run.sh --output-dir /data/regen/deepseekv4_ultrachat
```

## Server modes

### Native (`--mode native`, default)

Runs `vllm serve` directly in the current Python environment with TP=8 on 8 GPUs.
Graceful shutdown: SIGTERM → 30s grace period → SIGKILL.

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
CUDA_VISIBLE_DEVICES=<reserved-gpu-ids> vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --host 127.0.0.1 --port <PORT> \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --enable-expert-parallel --tensor-parallel-size 8 \
    --attention_config.use_fp4_indexer_cache=True \
    --moe-backend deep_gemm_mega_moe \
    --speculative_config '{"method":"mtp","num_speculative_tokens":2}'
```

### Docker (`--mode docker`)

Runs vLLM inside `docker run --rm` using a pre-built image. Requires `HF_HUB_CACHE` to mount weights.
Graceful shutdown: kills the bash wrapper, then `docker stop` (SIGTERM → 10s → SIGKILL).

| `--hardware` | Image | GPUs | vLLM args |
|---|---|---|---|
| `h100` | `vllm/vllm-openai:deepseekv4-cu129` | 8 | `--data-parallel-size 4`, cudagraph compilation |
| `b200` | `vllm/vllm-openai:deepseekv4-cu130` | 4 | `--data-parallel-size 4`, fp4 indexer cache |

Both hardware variants add `--speculative_config '{"method":"mtp","num_speculative_tokens":2}'`.

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

## Workflow

```
run.sh
 ├── validate --mode and --hardware
 ├── acquire N GPUs via chg reserve  (8 for native/h100, 4 for b200)
 ├── start serve.sh or serve_docker.sh (background)
 ├── poll /health every 30s (up to 3600s)
 ├── run scripts/response_regeneration/script.py
 └── cleanup on EXIT / Ctrl-C / SIGTERM
      ├── native:  SIGTERM → 30s grace → SIGKILL
      └── docker:  kill wrapper + docker stop (SIGTERM → 10s → SIGKILL)
```

GPU reservation is renewed automatically if less than 30 minutes remain.
