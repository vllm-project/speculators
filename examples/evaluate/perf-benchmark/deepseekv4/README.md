# DeepSeek-V4-Flash Acceptance Rate Evaluation

Modular system for measuring speculative decoding acceptance rates on a shared GPU server.

## Quick Start

```bash
# H100 — all 9 subsets, 8h deadline, 2h GPU reservations
bash run.sh --hardware h100

# B200 — all 9 subsets
bash run.sh --hardware b200

# Specific subsets only
bash run.sh --hardware h100 --subsets "HumanEval,qa,math_reasoning"

# Shorter deadline, more requests per subset
bash run.sh --hardware h100 --max-runtime 4 --max-requests 500
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--hardware` | *required* | `h100` (8 GPUs) or `b200` (4 GPUs) |
| `--subsets` | all 9 | Comma-separated subset names |
| `--output-dir` | `results_TIMESTAMP` | Output directory |
| `--max-requests` | 200 | Requests per subset |
| `--max-concurrency` | 128 | Concurrent requests |
| `--max-runtime` | 8 | Global deadline in hours |
| `--reserve-duration` | 2h | `chg reserve` duration per block |
| `--gen-kwargs` | | JSON generation params, e.g. `'{"temperature":0.6}'` |

## Standalone Scripts

### Start a server manually

```bash
# H100
GPU_IDS=0,1,2,3,4,5,6,7 PORT=8000 bash serve_h100.sh

# B200
GPU_IDS=0,1,2,3 PORT=8000 bash serve_b200.sh
```

### Run acceptance eval for one subset (server must be running)

```bash
bash eval_acceptance.sh \
    --endpoint http://localhost:8000 \
    --subset HumanEval \
    --output results/after_HumanEval.json \
    --max-requests 200
```

## Output

```
results_YYYYMMDD_HHMMSS/
├── acceptance/
│   └── after_SUBSET.json       # Per-subset Prometheus snapshot
├── acceptance_rates.json       # Aggregated rates
└── server_SUBSET_attempt1.log  # Server logs per attempt
```

## Available Subsets

`HumanEval`, `math_reasoning`, `qa`, `question`, `rag`, `summarization`, `tool_call`, `translation`, `writing`
