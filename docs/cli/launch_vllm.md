# launch_vllm.py

Launches a vLLM server configured for hidden states extraction, used for online training or offline hidden states generation.

## Basic Usage

```bash
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct 
```

## Arguments

### Positional Arguments

- **`model`** (str, required) Model name or path to extract hidden states from.

### Speculators Arguments

- **`--hidden-states-path`** (str, default: `/tmp/hidden_states`) The directory to initially cache hidden states to. Note: hidden states may then be moved or deleted by training/offline data generation.

- **`--target-layer-ids`** (int list, default: auto-select) Space-separated list of integer layer IDs from which to capture hidden states. Note: if `--include-last-layer` is enabled (default), the model's last layer will be appended to this list. Default: `[2, num_layers//2, num_layers-3]`

  **Important:** If set, you must also pass the same layer ids to the training script using `--target-layer-ids`.

- **`--include-last-layer` / `--no-include-last-layer`** (flag, default: `True`) For DFlash models, append the last layer (`num_hidden_layers`) to `target_layer_ids` for verifier hidden states extraction.

- **`--dry-run`** (flag) Print the command that would be executed without running it.

### vLLM Arguments

All arguments after `--` are passed directly to vLLM. Common vLLM arguments include:

- `--port`: Server port (default: `8000`)
- `--data-parallel-size`: Number of data parallel instances
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--gpu-memory-utilization`: GPU memory utilization (0.0 to 1.0)
- `--max-model-len`: Maximum model context length
- `--trust-remote-code`: Allow custom model code execution

See [vLLM CLI documentation](https://docs.vllm.ai/en/latest/cli/) for full list of options.

## Full Example

```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-70B-Instruct \
  --hidden-states-path /data/hidden_states \
  --target-layer-ids 5 20 40 80 \
  -- --data-parallel-size 2 --tensor-parallel-size 4 \
  --port 8000
```
