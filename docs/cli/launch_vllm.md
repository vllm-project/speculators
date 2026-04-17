# launch_vllm.py

Launches a vLLM server configured for hidden states extraction, used for online training or on-demand hidden states generation.

## Basic Usage

```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --hidden-states-path /tmp/hidden_states \
  -- --port 8000 --tensor-parallel-size 2
```

## Arguments

### Positional Arguments

- **`model`** (str, required)
  Model name or path to extract hidden states from.

### Speculators Arguments

- **`--hidden-states-path`** (str, default: `/tmp/hidden_states`)
  The directory to save hidden states to.

- **`--target-layer-ids`** (int list, default: auto-select)
  Space-separated list of integer layer IDs from which to capture hidden states.
  Default: `[2, num_layers//2, num_layers-3, num_layers]`

  **Important:** If set, you must also pass the same value to the training script using `--target-layer-ids`.

- **`--include-last-layer` / `--no-include-last-layer`** (flag, default: `True`)
  For DFlash models, append the last layer (`num_hidden_layers`) to `target_layer_ids` for verifier hidden states extraction.

- **`--dry-run`** (flag)
  Print the command that would be executed without running it.

### vLLM Arguments

All arguments after `--` are passed directly to vLLM. Common vLLM arguments include:

- `--port`: Server port (default: `8000`)
- `--host`: Server host (default: `0.0.0.0`)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--gpu-memory-utilization`: GPU memory utilization (0.0 to 1.0)
- `--max-model-len`: Maximum model context length
- `--trust-remote-code`: Allow custom model code execution

See [vLLM CLI documentation](https://docs.vllm.ai/en/latest/cli/) for full list of options.

## Examples

**Basic Launch:**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --port 8000
```

**Multi-GPU with Custom Layers:**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-70B-Instruct \
  --hidden-states-path /data/hidden_states \
  --target-layer-ids 5 20 40 80 \
  -- --tensor-parallel-size 4 --port 8000
```

**Dry Run (Preview Command):**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --dry-run \
  -- --port 8000
```

## See Also

- [CLI Reference Overview](README.md)
- [Previous Step: Prepare Data](prepare_data.md)
- [Next Step: Train Model](train.md)
- [vLLM CLI Reference](https://docs.vllm.ai/en/latest/cli/)
- [Online Training Tutorial](../user_guide/tutorials/train_eagle3_online.md)
