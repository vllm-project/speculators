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

  **Important:** If set, you must also pass the same layer ids to the training script using `--target-layer-ids`, **excluding** the final layer — training takes the auxiliary layers only. For the [full example](#full-example) below, that is `--target-layer-ids 5 20 40`.

- **`--include-last-layer` / `--no-include-last-layer`** (flag, default: `True`) Append the last layer (`num_hidden_layers`) to `target_layer_ids` for verifier hidden states extraction.

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

### Render throughput defaults

[prepare_data.py](prepare_data.md) `--render-endpoint` points at this server and drives its `/v1/chat/completions/render` endpoint. To keep that stage from serializing in vLLM's stock single-process, single-renderer front end, `launch_vllm.py` derives `--api-server-count` from the CPUs available to the process and defaults `--renderer-num-workers` to `2`.

On the standard 384-CPU H100 node, this resolves to `--api-server-count 18 --renderer-num-workers 2`, paired with 72 preprocessing workers. The default intentionally leaves 25% of the CPU budget for native runtime threads and other application work; smaller hosts scale down automatically. Arguments passed after `--` follow the defaults, so an explicit value still wins. No frontend defaults are added with `--headless`.

For non-headless launches, the script also defaults `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1`, and `RAYON_NUM_THREADS` to `2`. These defaults prevent each API-server process from creating a host-sized native thread pool; values already present in the environment are preserved.

The [training tutorial](../user_guide/tutorials/train.md) pins vLLM 0.27.1 for this path. These flags scale only the HTTP front end (chat-template application and tokenization); the engine and hidden-states connector are unaffected.

## Full Example

```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-70B-Instruct \
  --hidden-states-path /data/hidden_states \
  --target-layer-ids 5 20 40 80 \
  -- --data-parallel-size 2 --tensor-parallel-size 4 \
  --port 8000
```
