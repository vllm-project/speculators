# train.py

Trains speculator models using either online or offline hidden states. Supports single-GPU and multi-GPU distributed training.

## Basic Usage

**Single-GPU:**

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**Multi-GPU (DDP):**

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**Multi-GPU (FSDP sharded):**

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --fsdp-shard
```

## Arguments

### Model Arguments

- **`--verifier-name-or-path`** (str, required) HuggingFace model ID or local path for the verifier/target model.

- **`--trust-remote-code`** (flag) Allow executing code from HF Hub when loading the verifier's tokenizer.

- **`--speculator-type`** (str, default: `"eagle3"`) Type of speculator model to train. Options: `eagle3`, `dflash`, `dspark`, `peagle`, `mtp`

- **`--from-pretrained`** (str, default: `""`) Path or HF id of an existing draft checkpoint to load weights from and train — either a previously trained draft or the initialized-but-untrained checkpoint produced by `--dry-run`. May also point to a local directory containing only a `config.json`, in which case a fresh draft is initialized from that full speculator config. Takes precedence over all other model-definition options: it is mutually exclusive with `--draft-config` and the decoder-shaping flags (`--num-layers`, `--draft-arch`, `--draft-hidden-act`, `--sliding-window`, `--full-attention-indices`).

- **`--draft-config`** (str, default: `""`) HF id, directory, or JSON path of a decoder config (`LlamaConfig` for eagle3/peagle, `Qwen3Config` for dflash) used as the draft `transformer_layer_config`; the rest of the speculator is built from the other CLI args. The draft `hidden_size` must match the verifier (mismatch is not yet supported). If a full speculator config is passed, its nested `transformer_layer_config` is extracted. Mutually exclusive with `--from-pretrained` and with the decoder-shaping flags (`--num-layers`, `--draft-arch`, `--draft-hidden-act`, `--sliding-window`, `--full-attention-indices`).

- **`--dry-run`** (flag) Build the speculator, initialize weights, save a checkpoint to `--save-path`, then exit before training. Useful to validate the config/weights in vLLM before launching a full run; the saved checkpoint can be fed straight back via `--from-pretrained`.

- **`--num-layers`** (int, default: `1`) Number of transformer layers in the draft model.

- **`--draft-arch`** (str, default: `"llama"`) Architecture for the synthesized draft decoder layers. Options: `llama`, `qwen3`. Used by Eagle3 and P-EAGLE, which select the decoder layer class from this value; DFlash always uses a Qwen3-style decoder regardless. Both are supported in vLLM for inference, and the target and draft architectures do not have to match.

- **`--draft-hidden-act`** (str, default: `"silu"`) Activation function for draft decoder layers. Setting as `None` will inherit activation function from the verifier model.

### Data Arguments

- **`--data-path`** (str, default: `"./data"`) Path to the processed training data directory.

- **`--on-missing`** (choice: `generate`|`skip`|`warn`|`raise`, default: `generate`) Behavior when cached hidden states are missing:

  - `generate`: Generate hidden states on-demand using vLLM endpoint
  - `skip`: Skip the sample silently, pads to fill batch.
  - `warn`: Skip the sample with a warning, pads to fill batch.
  - `raise`: Raise an error

- **`--on-generate`** (choice: `cache`|`delete`, default: `"delete"`) Behavior after generating new hidden states (only applies if `--on-missing=generate`):

  - `delete`: Delete hidden states after loading (pure online training)
  - `cache`: Store hidden states for reuse in future epochs (hybrid training)

- **`--hidden-states-path`** (str, default: `{data-path}/hidden_states`) Path where cached hidden states files are stored (or will be stored if generating).

- **`--vllm-endpoint`** (str, default: `"http://localhost:8000/v1"`) vLLM endpoint address for generating hidden states on-demand (online training). Ignored if `--on-missing` is not set to `generate`.

- **`--request-timeout`** (float, default: `180.0`) Timeout in seconds for each individual vLLM request.

- **`--max-retries`** (int, default: `3`) Maximum number of retry attempts per vLLM request on failure.

- **`--shared-hidden-states-path`** (str, default: `None`) Optional filesystem cache that coalesces identical online hidden-state requests across independent trainers. All participating trainers must use the same path. This does not change `--hidden-states-path`, which remains the per-dataset indexed cache used by `--on-generate cache`.

- **`--shared-hidden-states-namespace`** (str, default: `None`) Optional additional request-identity namespace for producer extraction settings not represented by the model or target layer IDs. Target layer IDs are fingerprinted automatically. Use the same value for trainers sharing artifacts and a different value for any other extraction semantic that changes the resulting tensors.

- **`--shared-hidden-states-ttl`** (float, default: `3600.0`) Seconds to retain shared artifacts before regenerating them. Set to `0` to disable expiration.

- **`--shared-hidden-states-lock-timeout`** (float, default: `300.0`) Maximum seconds to wait while another trainer generates and atomically publishes the same artifact.

- **`--shared-hidden-states-consumer-id`** (str, default: `None`) Stable logical trainer identity. Setting this option enables bounded asynchronous windows and requires `--on-missing generate`. Training and validation append separate stream suffixes automatically.

- **`--shared-hidden-states-lookbehind`** (int, default: `2`) Committed stream positions retained behind this consumer's cursor.

- **`--shared-hidden-states-lookahead`** (int, default: `40`) Stream positions retained ahead of this consumer's committed cursor.

- **`--shared-hidden-states-max-prefetch-per-consumer`** (int, default: `8`) Maximum PREFETCH artifacts for one consumer that may be queued or generating at once. Demand requests bypass this bound. This value cannot exceed `lookahead + 1`.

- **`--shared-hidden-states-capture-batch-size`** (int, default: `8`) Global maximum number of hidden-state captures in flight across all trainer dispatchers sharing the coordinator.

- **`--shared-hidden-states-capture-batch-wait`** (float, default: `0.002`) Seconds each dispatcher waits before claiming work so newly queued requests can coalesce into a producer batch.

- **`--shared-hidden-states-max-inflight`** (int, default: `32`) Maximum waiting or leased positions per consumer. Once the first sample of a packed batch is admitted, the rest of that batch may complete atomically so a batch larger than this value cannot deadlock before trainer ACK.

- **`--shared-hidden-states-consumer-timeout`** (float, default: `120.0`) Heartbeat timeout before a dead consumer's window and leases are released.

- **`--shared-hidden-states-claim-timeout`** (float, default: `300.0`) Timeout before an interrupted producer claim can be reassigned.

- **`--shared-hidden-states-generation-attempts`** (int, default: `3`) Maximum coordinated generation attempts, including expired producer claims.

  The shared cache is a filesystem data plane, not Mooncake or GPU-direct transport. Its directory must provide reliable POSIX `flock`, same-filesystem atomic rename, and directory `fsync` semantics to every trainer. Do not assume an arbitrary NFS mount is safe unless those guarantees have been verified.

  Without `--shared-hidden-states-consumer-id`, this remains the legacy TTL cache: setting the TTL to zero retains one artifact per unique request, and a finite TTL does not by itself bound a pass over unseen samples.

  With a consumer ID, SQLite tracks deterministic sampler positions, independent consumer cursors, generation claims, read leases, and the union of live windows. Window retention and active prefetch are separate bounds: the full lookahead remains reusable while only the nearest configured prefetches consume producer capacity. DataLoader workers only acquire and materialize authorized artifacts. The trainer main process advances the cursor after a successful training optimizer boundary or validation forward. Artifacts outside every live window are removed only after all read leases are released. In this mode TTL expiration is disabled; retention is controlled by windows and explicit leases.

- **`--legacy-data`** (flag) **DEPRECATED.** Use the old data format which stores hidden states alongside token_ids.

- **`--total-seq-len`** (int, default: `8192`) Maximum total sequence length for training batches. Note: samples will be packed into batches with total combined sequence length `{total-seq-len}`.

### Vocabulary Mapping Arguments

- **`--draft-vocab-size`** (int, default: `None`) Vocabulary size for the draft model. If not specified and no vocab mapping files are provided, uses full verifier vocabulary.

- **`--token-freq-path`** (str, default: `{data-path}/token_freq.pt`) Path to token frequency distribution file. This is used to determine which tokens to include in the reduced draft vocab.

- **`--d2t-path`** (str, default: `None`) Path to draft-to-target vocabulary mapping file (`.npy`). Must be provided with `--t2d-path`.

- **`--t2d-path`** (str, default: `None`) Path to target-to-draft vocabulary mapping file (`.npy`). Must be provided with `--d2t-path`.

- **`--mask-token-id`** (int, default: auto-detect) Token ID to use as mask token (for DFlash). Auto-detected if not provided.

- **`--target-layer-ids`** (int list, default: auto-select) Space-separated list of layer IDs used for hidden states. Default: `[2, num_layers//2, num_layers-3]` **Must match the values used when launching vLLM if custom layers were specified.**

### Distributed Training Arguments

- **`--fsdp-shard`** (flag) Shard model parameters across GPUs with FSDP. By default, parameters are fully replicated (DDP-like). Enable this when the model does not fit in a single GPU's memory.

### Training Arguments

- **`--save-path`** (str, default: `"./checkpoints"`) Directory to save model checkpoints.

- **`--epochs`** (int, default: `20`) Number of training epochs.

- **`--lr`** (float, default: `1e-4`) Learning rate.

- **`--train-data-ratio`** (float, default: `0.9`) Ratio of data to use for training. The rest is used for validation; set this to `1.0` for a train-only run with no validation loader.

- **`--no-resume-from-checkpoint`** (flag) Disable automatic checkpoint resumption. Without this flag, this script will automatically load the latest checkpoint in `{save-path}` if one exists.

- **`--logger`** (str, default: `""`) Metric logging backend(s). Options: `trackio`, `wandb`, `tensorboard`, `mlflow` Can specify multiple comma-separated: `--logger tensorboard,wandb`. **Warning:** backend must be pip installed before using.

- **`--log-dir`** (str, default: `"./logs"`) Directory to save training logs. Only applies to some logging backends (e.g. `tensorboard`)

- **`--run-name`** (str, default: `None`) Name for the training run (used by logging backends).

- **`--seed`** (int, default: `42`) Random seed for reproducibility.

- **`--hidden-states-dtype`** (str, default: `"bfloat16"`) Data type for dataloader hidden states and autocast compute. Model master weights are always kept in fp32. Options: `float32` (full precision, for debugging), `bfloat16` (recommended for mixed precision training). Note: `float16` is not supported as it requires gradient scaling to prevent underflow.

- **`--deterministic-cuda`** (flag) Enable deterministic CUDA operations. May impact performance.

### Optimizer Arguments

- **`--optimizer`** (str, default: `"muon"`) Optimizer to use. Options: `adamw`, `muon`. The `muon` option applies the Muon optimizer to 2D weight matrices and AdamW to the remaining parameters (norms, biases, embeddings, lm_head).

- **`--adamw-backend`** (str, default: `"auto"`) AdamW execution backend. Options: `auto`, `foreach`, `fused`. This also applies to the AdamW parameter group in Muon mode. The fused backend requires CUDA parameters.

- **`--gradient-clip-backend`** (str, default: `"torch"`) Gradient clipping implementation. Options: `torch`, `fused_adamw`. The fused option avoids a separate gradient-scaling kernel and requires both `--optimizer adamw` and `--adamw-backend fused`.

- **`--max-grad-norm`** (float, default: `1.0`) Maximum gradient norm used by either clipping backend.

- **`--weight-decay`** (float, default: `0.01`) Weight decay for the AdamW optimizer (and the AdamW group in muon mode).

- **`--muon-lr`** (float, default: `10*lr`) Learning rate for the Muon (2D weights) group. Only used with `--optimizer muon`. Defaults to 10× the `--lr` value.

- **`--muon-momentum`** (float, default: `0.95`) Momentum for the Muon optimizer. Only used with `--optimizer muon`.

- **`--muon-weight-decay`** (float, default: `0.1`) Weight decay for the Muon optimizer. Only used with `--optimizer muon`.

- **`--muon-ns-steps`** (int, default: `5`) Number of Newton-Schulz steps for Muon. Only used with `--optimizer muon`.

- **`--muon-adjust-lr-fn`** (str, default: `"match_rms_adamw"`) Muon LR adjustment strategy. Options: `original`, `match_rms_adamw`. Only used with `--optimizer muon`.

### Eagle3-Specific Arguments

- **`--use-off-policy-tokens`** (flag) Use off-policy tokens during training (required for [regenerated data](response_regeneration.md)).

- **`--norm-before-residual` / `--no-norm-before-residual`** (flag, default: `True`) Toggle normalization before residual connections.

- **`--embed-requires-grad` / `--no-embed-requires-grad`** (flag, default: `False`) Whether to train embedding layer weights.

- **`--norm-before-fc` / `--no-norm-before-fc`** (flag, default: `True` for eagle3, `False` otherwise) Apply a single RMSNorm to the concatenated auxiliary hidden states before the FC projection (gpt-oss style). See `--fc-norm` for the per-layer alternative from the Eagle 3.1 paper.

- **`--fc-norm`** (flag, default: `False`) Apply per-layer RMSNorm to each auxiliary hidden state before concatenation and FC projection (Eagle 3.1 paper approach).

- **`--norm-output` / `--no-norm-output`** (flag, default: `True` for eagle3, `False` otherwise) Feed post-norm hidden states back across TTT steps to stabilize magnitude drift across speculation depths.

- **`--ttt-steps`** (int, default: `3`) Number of test-time training steps

- **`--ttt-step-loss-decay`** (float, default: `1.0`) Loss decay factor for test-time training steps.

### Attention Backend Arguments

- **`--draft-attn-impl`** (str, default: `"simple_flex_attention"`) Attention implementation for draft layers. Options: `simple_flex_attention`, `sdpa`, `eager`. Use `sdpa` or `eager` on hardware where flex attention is unavailable (e.g. Ascend NPU). Applies to Eagle3, P-EAGLE, and DFlash. Not supported for MTP.

### DFlash-Specific Arguments

- **`--block-size`** (int, default: `8`) Block size for DFlash model.

- **`--sample-from-anchor`** / **`--no-sample-from-anchor`** (bool, default: algorithm-specific) Whether to sample from the anchor position. `True`: sample from anchor and all mask positions (default for dspark, produces block_size tokens). `False`: anchor is bonus token (default for dflash, produces block_size-1 tokens).

- **`--max-anchors`** (int, default: `3072`) Maximum anchor positions for DFlash training.

- **`--dflash-decay-gamma`** (float, default: `4.0`) Decay gamma for DFlash loss weighting.

- **`--dflash-linear-cross-entropy-backend`** (str, default: `"torch"`) DFlash cross-entropy backend. Options: `torch`, `liger`. Liger avoids materializing draft logits and requires an exactly-CE loss configuration plus the optional `speculators[liger]` dependency.

- **`--dflash-compact-zero-weight-ce-rows`** / **`--no-dflash-compact-zero-weight-ce-rows`** (bool, default: `False`) Exclude masked, zero-weight rows before the fused Liger CE kernel. Requires `--dflash-linear-cross-entropy-backend liger`.

- **`--dflash-label-source`** (str, default: `"verifier_argmax"`) Hard-label source for the opt-in Liger CE path. Options: `verifier_argmax`, `input_ids`. The default preserves DFlash verifier-target semantics; `input_ids` is an explicitly different training target and requires the full verifier vocabulary.

- **`--dflash-verifier-argmax-chunk-size`** (int, default: `0`) Number of verifier LM-head rows processed per argmax chunk in the Liger CE path. `0` materializes the complete verifier logits; a positive value reduces their peak memory. Requires the Liger CE backend.

### Sliding Window Attention Arguments

All speculator types (except `mtp`) use sliding window attention on all draft layers by default.

- **`--sliding-window`** (int, default: `2048`) Sliding window size for sliding window attention layers.

- **`--full-attention-indices`** (int list, default: none) Space-separated draft layer indices that should use full attention instead of sliding window. Example: `--full-attention-indices 0 2` makes layers 0 and 2 use full attention; the rest use sliding window.

- **`--sliding-window-non-causal`** (flag) Use non-causal (bidirectional) masking within draft blocks for sliding window attention layers. Full attention layers are always bidirectional. Note: vLLM currently doesn't support these models.

### Dataloader Arguments

- **`--num-workers`** (int, default: `12`) Number of dataloader worker processes.

- **`--prefetch-factor`** (int, default: `4`) Number of batches to prefetch per worker.

- **`--noise-std`** (float, default: `0.05`) Standard deviation for noise augmentation on hidden states.

### Checkpoint Arguments

- **`--checkpoint-freq`** (int, default: `1`) Save a checkpoint every N epochs. Must be ≥ 1.

- **`--save-best`** (flag) Save a symbolic link to the checkpoint with the lowest validation loss.

### Learning Rate Scheduler Arguments

- **`--scheduler-type`** (str, default: `"linear"`) Type of learning rate scheduler. Options: `linear`, `cosine`, `none`

- **`--scheduler-warmup-steps`** (int, default: `None`) Number of warmup steps for the scheduler.

- **`--scheduler-warmup-ratio`** (float, default: `None`) Warmup as a fraction of total scheduler steps, in `[0, 1]`. Ignored (with a warning) when `--scheduler-warmup-steps` is also set.

- **`--scheduler-total-steps`** (int, default: `None`) Total number of training steps for the scheduler.

- **`--scheduler-num-cosine-cycles`** (float, default: `0.5`) Number of cosine cycles for cosine scheduler.

## Examples

### Online Training

```bash
# First, start vLLM server
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --port 8000

# Then train with on-demand hidden states generation
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Offline Training

```bash
# Train using pre-generated hidden states
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Hybrid Training (Cache on First Epoch)

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate cache \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Multi-GPU Training with WandB Logging

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-70B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 20 \
  --lr 1e-4 \
  --logger wandb \
  --run-name eagle3-llama-70b \
  --scheduler-type cosine \
  --scheduler-warmup-steps 100 \
  --checkpoint-freq 2 \
  --save-best \
  --fsdp-shard
```

### Fine-tuning a Pretrained Model

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --from-pretrained ./pretrained_speculator \
  --data-path ./new_training_data \
  --hidden-states-path ./hidden_states \
  --save-path ./finetuned_checkpoints \
  --epochs 5 \
  --lr 5e-6
```

### Initializing From a Decoder Config (with Dry-Run Validation)

```bash
# Build the speculator from a plain decoder config, initialize weights, save a
# checkpoint, and exit before training so it can be validated in vLLM first.
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --speculator-type dflash \
  --draft-config ./qwen3_draft_decoder_config.json \
  --draft-vocab-size 32000 \
  --save-path ./draft_init \
  --dry-run

# After validating ./draft_init in vLLM, train starting from it:
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --speculator-type dflash \
  --from-pretrained ./draft_init \
  --data-path ./training_data \
  --epochs 5 \
  --lr 5e-6
```
