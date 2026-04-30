# DeepSeek-V4-Flash UltraChat Response Regeneration

Regenerates UltraChat (or Magpie) responses using DeepSeek-V4-Flash served via vLLM.
Handles GPU reservation via `chg`, server lifecycle, and graceful cleanup.

Reuses [`scripts/response_regeneration/script.py`](../../../scripts/response_regeneration/script.py) for the actual generation.

## Prerequisites

- `chg` available in PATH (GPU reservation CLI)
- **Native mode**: `vllm` installed in the active Python environment
- **Docker mode**: Docker (H100/B200) or Podman (H200) with NVIDIA runtime; `HF_HUB_CACHE` set to the model weights directory
- `HF_HUB_CACHE` set (Docker mode always requires it; native mode uses it implicitly from the environment)

## Quick start

```bash
cd examples/generate/deepseekv4-ultrachat

# Test run — native vllm, 50 samples per dataset (ultrachat + magpie)
bash run.sh --limit 50

# Test run — Docker on H100, 50 samples per dataset
bash run.sh --mode docker --hardware h100 --limit 50

# Test run — Podman on H200, 50 samples per dataset
bash run.sh --mode docker --hardware h200 --limit 50

# Full run — both datasets, all samples
bash run.sh

# Full run — Docker on B200
bash run.sh --mode docker --hardware b200

# Full run — Podman on H200
bash run.sh --mode docker --hardware h200
```

## All options

```
bash run.sh [OPTIONS]

  --mode STR            How to launch the server: native | docker   (default: native)
  --hardware STR        GPU hardware profile: h100 | b200 | h200    (default: h100; docker mode only)
  --limit N             Stop after N rows per dataset. Omit to process entire datasets.
  --concurrency N       Concurrent HTTP requests to the vLLM server  (default: 64)
  --max-tokens N        Max tokens per generated response            (default: 8192)
  --resume              Skip rows already written to the output files
  --output-dir DIR      Directory for results and server log         (default: /mnt/data/engine/rahul-tuli/deepseekv4-{hardware}-{mode}-{timestamp})
  --max-runtime N       Global deadline in hours before the script exits  (default: 12)
  --reserve-duration S  Duration string passed to chg reserve        (default: 2h)
```

**Note:** Both ultrachat and magpie datasets are processed sequentially in every run. Use `--limit` for quick test runs of both datasets.

## Common invocations

```bash
# Test runs (50 samples per dataset, both ultrachat and magpie)
bash run.sh --limit 50
bash run.sh --mode docker --hardware h100 --limit 50
bash run.sh --mode docker --hardware h200 --limit 50

# Full dataset runs (both ultrachat and magpie, all samples)
bash run.sh
bash run.sh --mode docker --hardware h100
bash run.sh --mode docker --hardware b200
bash run.sh --mode docker --hardware h200

# Higher concurrency for faster throughput
bash run.sh --concurrency 128

# Resume a run that was interrupted
bash run.sh --output-dir /mnt/data/engine/rahul-tuli/deepseekv4-h200-docker-20260429_120000 --resume

# Longer reservation and deadline for a big run
bash run.sh --reserve-duration 4h --max-runtime 20

# Custom output directory
bash run.sh --output-dir /mnt/data/engine/rahul-tuli/my-custom-run
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

### Docker/Podman (`--mode docker`)

Runs vLLM inside a container using docker (H100/B200) or podman (H200). Requires `HF_HUB_CACHE` to mount weights.
Graceful shutdown: kills the bash wrapper, then `docker stop` or `podman stop` (SIGTERM → 10s → SIGKILL).

| `--hardware` | Runtime | Image | GPUs | vLLM args | Spec tokens |
|---|---|---|---|---|---|
| `h100` | docker | `vllm/vllm-openai:deepseekv4-cu129` | 8 | `--data-parallel-size 4` | 2 |
| `b200` | docker | `vllm/vllm-openai:deepseekv4-cu130` | 4 | `--data-parallel-size 4`, fp4 indexer cache | 2 |
| `h200` | **podman** | `vllm/vllm-openai:latest` | 8 | minimal (no DP, no compilation) | **1** |

**H200-specific configuration:**
- Uses podman instead of docker
- Sets HF_HUB_CACHE as environment variable inside container
- Uses 1 speculative token (vs 2 for H100/B200)
- Minimal vLLM configuration (no data-parallel-size or compilation config)

## Output

**Default location:** `/mnt/data/engine/rahul-tuli/deepseekv4-{hardware}-{mode}-{timestamp}/`

**Directory structure:**
```
deepseekv4-h200-docker-20260429_143022/
├── server.log                              (vLLM server log, shared)
├── ultrachat/
│   ├── ultrachat_DeepSeek-V4-Flash.jsonl  (generated responses)
│   └── generation.log                      (regeneration script output)
└── magpie/
    ├── magpie_DeepSeek-V4-Flash.jsonl     (generated responses)
    └── generation.log                      (regeneration script output)
```

Each JSONL file contains one JSON object per line:

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

Failed requests are also written with an `"error"` key in metadata so files stay appendable for `--resume`.

## Workflow

```
run.sh
 ├── validate --mode and --hardware
 ├── create output directory: /mnt/data/rahul-tuli/deepseekv4-{hardware}-{mode}-{timestamp}/
 ├── acquire N GPUs via chg reserve  (8 for native/h100/h200, 4 for b200)
 ├── start serve.sh or serve_docker.sh (background)
 ├── poll /health every 30s (up to 3600s)
 ├── for dataset in ultrachat magpie:
 │    ├── create dataset subdirectory
 │    ├── run scripts/response_regeneration/script.py
 │    └── save results to {dataset}/{dataset}_DeepSeek-V4-Flash.jsonl
 └── cleanup on EXIT / Ctrl-C / SIGTERM
      ├── native:  SIGTERM → 30s grace → SIGKILL
      └── docker/podman:  kill wrapper + container stop (SIGTERM → 10s → SIGKILL)
```

GPU reservation is renewed automatically if less than 30 minutes remain.

Both ultrachat and magpie are processed sequentially in a single run. The server stays running between datasets to avoid startup overhead.
