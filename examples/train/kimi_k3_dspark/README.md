# Kimi-K3 DSpark training (multi-node, online extraction)

Kimi-K3 does not fit on a single node, so this example keeps hidden-state extraction and draft training on disjoint node groups and streams hidden states between them through a Mooncake store:

```text
extraction (2 nodes)                 training (1 node)
Kimi-K3 vLLM TP8 + expert parallel -> torchrun DSpark DDP, 4 ranks
                 \                    /
                  Mooncake streaming store
```

This is a worked example for one 4-GPU-per-node NVL72 rack, not a portable harness. Values are hardcoded so the commands stay readable; adapt them rather than configuring them.

`k3_draft_layer_config.json` is the draft **decoder** config and nothing else — a plain `Qwen3Config` that `--draft-config` loads as the speculator's `transformer_layer_config`. Everything above the decoder (block size, vocab size, mask token, target layers, the Markov and confidence heads) is set by the flags in `launch_train_node.sh`, not by this file.

## Requirements

- Two environments: one with vLLM plus `hs_connectors` and `mooncake-transfer-engine` (extraction), one with `speculators` plus `mooncake-transfer-engine` (training).
- Verifier weights at `/mnt/shared/weights/kimi-k3`, or set `MODEL_PATH`.

## The components

### 1. Prepare data

```bash
SOURCE_JSONL=/path/to/data.jsonl \
  sbatch --export=ALL,SOURCE_JSONL examples/train/kimi_k3_dspark/prepare_data.sbatch
```

Writes the Arrow dataset, token frequencies, and vocabulary mapping to `runs/kimi_k3_dspark/data`. Everything below expects it to exist.

### 2. Mooncake master

One per run, anywhere both groups can reach:

```bash
mooncake_master --rpc_port 50051 --metrics_port 9003 \
  --rpc_thread_num 8 --enable_disk_eviction=false --logtostderr=true
```

### 3. Extractor (both nodes of the TP8 pair)

```bash
NODE_RANK=0 EXTRACT_ADDR=<head-fabric-ip> MOONCAKE_MASTER=<host>:50051 \
  FABRIC_SUBNET=<subnet-prefix> \
  bash examples/train/kimi_k3_dspark/launch_extractor_node.sh
```

`NODE_RANK=0` on the head, `1` on the other; `EXTRACT_ADDR` is the head's fabric IP on both. Rank 1 runs `--headless`. Wait for `/health` on port 8000 before starting training.

### 4. Trainer

```bash
MOONCAKE_MASTER=<host>:50051 VLLM_ENDPOINT=http://<head-fabric-ip>:8000/v1 \
  FABRIC_SUBNET=<subnet-prefix> \
  bash examples/train/kimi_k3_dspark/launch_train_node.sh
```

Add `MAX_STEPS=20` for a smoke run. Other useful overrides: `MODEL_PATH`, `DATA_DIR`, `RUN_NAME`.

## Running all four under Slurm

`run.sbatch` does exactly the above on a 3-node allocation: nodes 0-1 extract, node 2 trains, the Mooncake master shares node 0. It resolves the extractor head's fabric address, waits for `/health`, then launches training, and stops the extractors and master when training exits.

```bash
FABRIC_SUBNET=<subnet-prefix> sbatch --export=ALL examples/train/kimi_k3_dspark/run.sbatch
```

Logs and checkpoints land in `runs/kimi_k3_dspark/<run-name>/`.

## Integrity and failure handling

Every Mooncake sample carries a versioned shape/dtype/CRC32 manifest. The producer rejects non-finite tensors before publication; the consumer verifies checksum and finiteness before training. Invalid round trips are regenerated twice, independently of the HTTP retry budget. An exhausted sample is dropped; if that leaves a rank with no valid samples it runs a locally empty zero-loss batch and still participates in DDP. Twenty consecutive failed round trips in one worker trip a synchronized circuit breaker.
