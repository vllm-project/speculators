# Multi-Node Training

By default, Speculators runs the target model (vLLM) and the draft trainer on the same node, transferring hidden states through the local filesystem. This works well when the target model fits on a subset of a single node's GPUs, but breaks down when:

- **The target model is too large** for one node (e.g. Kimi-K3 requires TP8 across two nodes).
- **You want to scale extraction and training independently** on separate machines.

Multi-node training solves this by replacing the filesystem transfer with a network-based backend that streams hidden states between nodes.

## Architecture

Hidden-state transfer is handled by the `hs_connectors` package, which provides a pluggable backend system:

```text
vLLM (extraction nodes)              Trainer (training nodes)
  launch_vllm.py                       speculators train
       |                                  |
       v                                  v
  HiddenStatesBackend              HiddenStatesBackend
       |                                  |
       +--------- backend store ----------+
```

Both sides use `--hidden-states-backend <name>` to select the same backend.

### Built-in Backends

| Backend        | Flag                                     | Transfer method                              | Use case                                                     |
| -------------- | ---------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| **Filesystem** | `--hidden-states-backend file` (default) | Safetensors files on a shared filesystem     | Single node, or multi-node with a shared NFS/Lustre mount    |
| **Mooncake**   | `--hidden-states-backend mooncake`       | Distributed key-value store over TCP or RDMA | Multi-node without shared storage; high-throughput streaming |

## Mooncake Backend

[Mooncake](https://github.com/kvcache-ai/Mooncake) is a distributed transfer engine. The Mooncake backend streams hidden states from vLLM to the trainer through a Mooncake key-value store, without requiring a shared filesystem. The store entries can then be retrieved from other nodes, using tcp or rdma to transfer the data directly to the requesting process.

### Prerequisites

Install the Mooncake transfer engine in **both** the vLLM and Speculators environments:

```bash
# Standard CUDA
pip install mooncake-transfer-engine

# CUDA 13.x
pip install mooncake-transfer-engine-cuda13
```

Or install Speculators with the mooncake extra:

```bash
pip install speculators[mooncake-cuda]
# or
pip install speculators[mooncake-cuda13]
```

### Step-by-Step Setup

#### 1. Start the Mooncake Master

The master coordinates the distributed store. Run it once, on any node reachable by both the extraction and training nodes:

```bash
mooncake_master --rpc_port 50051
```

See the [Mooncake documentation](https://github.com/kvcache-ai/Mooncake) for additional `mooncake_master` options.

#### 2. Launch the Extractor (vLLM)

On the extraction node(s), launch vLLM with the Mooncake backend. The key addition compared to single-node is `--hidden-states-backend mooncake` and the `--mooncake-*` flags:

```bash
# in vLLM venv
python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --hidden-states-backend mooncake \
  --mooncake-master <master-ip>:50051 \
  --mooncake-protocol tcp \
  -- --tensor-parallel-size 4 --port 8000
```

For multi-node tensor parallelism (when the model spans multiple nodes), add vLLM's distributed arguments after `--`:

```bash
python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --hidden-states-backend mooncake \
  --mooncake-master <master-ip>:50051 \
  --mooncake-protocol tcp \
  -- --tensor-parallel-size 8 --nnodes 2 \
     --node-rank $NODE_RANK --master-addr <head-ip> \
     --port 8000
```

Wait for `Application startup complete` before starting training.

#### 3. Launch Training

On the training node, point the trainer at the same Mooncake master and the vLLM endpoint:

```bash
# in speculators venv
torchrun --standalone --nproc_per_node 4 \
  -m speculators.train \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --total-seq-len 8192 \
  --hidden-states-backend mooncake \
  --mooncake-master <master-ip>:50051 \
  --mooncake-protocol tcp \
  --vllm-endpoint http://<extractor-ip>:8000/v1 \
  --on-missing generate \
  --on-generate delete
```

This is the same as single-node online training, but with `--hidden-states-backend mooncake` and `--mooncake-*` flags replacing the default filesystem backend.

### Mooncake CLI Arguments

These flags are available on both `launch_vllm.py` and `speculators train` (`torchrun -m speculators.train`) when using `--hidden-states-backend mooncake`:

| Flag                            | Default           | Description                                                                                       |
| ------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------- |
| `--mooncake-master`             | `127.0.0.1:50051` | Mooncake master server address                                                                    |
| `--mooncake-metadata-server`    | `P2PHANDSHAKE`    | Metadata server address, or `P2PHANDSHAKE` for peer-to-peer                                       |
| `--mooncake-protocol`           | `tcp`             | Transport protocol: `tcp` or `rdma`                                                               |
| `--mooncake-global-segment-gib` | `4.0`             | Memory registered for globally visible objects (GiB). Increase for many concurrent long sequences |
| `--mooncake-local-buffer-gib`   | `2.0`             | Local staging buffer size (GiB)                                                                   |
| `--mooncake-writer-threads`     | `4`               | Async writer threads on the vLLM side (`launch_vllm.py` only)                                     |

### Networking

Both the extractor and trainer must be able to reach the Mooncake master and each other over the chosen protocol. By default, `hs_connectors` resolves the local hostname via `socket.gethostbyname(socket.gethostname())`. On multi-NIC clusters where that returns the wrong interface, set the `MOONCAKE_LOCAL_HOSTNAME` environment variable to the IP on the correct network interface.

## Worked Example: Kimi-K3

For a complete multi-node setup including Slurm orchestration, see the [Kimi-K3 DSpark training example](https://github.com/vllm-project/speculators/tree/main/examples/train/kimi_k3_dspark). It demonstrates:

- Two-node TP8 extraction with expert parallelism
- Single-node DDP training with 4 ranks
- Mooncake master co-located with the extraction head
- Slurm batch scripts that wire everything together

## Writing a Custom Backend

The `hs_connectors` package uses a plugin registry. If you would like to use a different transfer interface, you can simply register a new backend. To add a new backend:

1. Subclass `HiddenStatesBackend` and decorate it with `@HiddenStatesBackend.register("my_backend")`.
2. Implement the four required hooks: `add_train_args`, `add_launch_args`, `from_train_args`, and `build_kv_transfer_config`.
3. Subclass `HiddenStatesTransfer` to implement the actual data transfer logic (`get_cached`, `get_generated`, `cache`, `delete`).

The new backend is automatically discovered by both `speculators train` and `launch_vllm.py` and can be selected with `--hidden-states-backend my_backend`.
