# vLLM dynamic hidden-states plugin

Swap **which** target-model hidden layers are captured during speculator
training-data generation (`method="extract_hidden_states"`) at runtime, over
HTTP, without restarting the engine.

Only the *set* of layer indices can change; the *count* is fixed at launch
(baked into proposer buffers, the dummy-proposer KV-cache shape and the
on-disk safetensors layout). Count mismatches are rejected (fail closed).

## Works under `torch.compile` (no `--enforce-eager` needed)

The stock capture uses a Python membership test —
`if layer_idx in self.aux_hidden_state_layers` — which `torch.compile`
constant-folds into the compiled forward, freezing the captured layer set. So a
naive attribute swap is a silent no-op unless the engine runs `--enforce-eager`.

This plugin ships a **general plugin** (`graph_patch.install`, auto-loaded via
`VLLM_PLUGINS`) that replaces that membership test *before compilation* with a
**masked accumulate**: it keeps `N` fixed accumulator buffers and folds every
candidate layer's residual into them, weighted by a one-hot selection mask held
in a registered buffer. The compiled graph is now selection-agnostic — which
layers land in which slot is a function of the mask *contents*, not the traced
code — so swapping is an in-place `mask.copy_(...)`. Same storage address, so
the compiled graph *and* a captured CUDA graph read the new selection on the
next forward. **No recompile, no `--enforce-eager`.**

Cost: memory stays O(N) (N accumulators, not O(num_layers)); the price is a
little compute — a fused multiply-add over every candidate layer's residual per
output slot, negligible next to the attention/MLP matmuls.

Verified empirically with Qwen3-8B (default `torch.compile` +
`FULL_AND_PIECEWISE` CUDA graphs, no `--enforce-eager`): swapping only the first
captured layer (2→6) changes exactly column 0 (`max|Δ|=21.125`) while columns
for the unchanged layers 18/34 stay **bit-identical** (`Δ=0`), and the compiled
masked path reproduces the eager baked path's values exactly. No recompilation
is triggered by the swap.

| Engine mode | Swap effective? |
|---|---|
| default `torch.compile` + CUDA graphs, plugin enabled | ✅ yes (masked accumulate) |
| `--enforce-eager` (with or without the plugin) | ✅ yes |
| default `torch.compile`, plugin **not** enabled | ❌ no (baked, silent no-op) |

**Scope:** the patch covers models that capture through
`EagleModelMixin._maybe_add_hidden_state` (Qwen2/Qwen3, Llama, and the other
generic dense/MoE paths). A few models (e.g. `deepseek_v2`, `qwen3_next`) inline
the membership test in their own forward and are **not** covered — those still
need `--enforce-eager` for a live swap. The worker logs a warning if you swap
when neither eager mode nor the patch is active.

## Components

- `dynamic_hidden_states.graph_patch.install` — pre-compile monkeypatch
  (masked-accumulate) that makes layer selection a runtime buffer input, wired
  via the `vllm.general_plugins` entry point (runs in every process, incl.
  workers, before model load). This is what makes swaps work without eager mode.
- `dynamic_hidden_states.worker_extension.AuxLayerWorkerExtension` — engine-side
  worker RPC (`set_/get_aux_hidden_state_layers_rpc`), added to every worker via
  `--worker-extension-cls`.
- `dynamic_hidden_states.endpoint.AuxLayerEndpoint` — API-server-side endpoint
  plugin registering `GET`/`POST /aux_hidden_state_layers`, wired via the
  `vllm.endpoint_plugins` entry point.

## Install

```bash
/workspace/vllm/.venv/bin/python -m pip install -e /workspace/speculators/vllm_plugins
# or: uv pip install -e /workspace/speculators/vllm_plugins   (with the vLLM venv active)
```

## Serve

Endpoint plugins are strict opt-in — they load only when named in
`VLLM_PLUGINS`. The same `VLLM_PLUGINS=dynamic_hidden_states` also enables the
general plugin (masked-accumulate patch), so no `--enforce-eager` is needed.

```bash
VLLM_PLUGINS=dynamic_hidden_states \
vllm serve Qwen/Qwen3-8B \
  --worker-extension-cls dynamic_hidden_states.worker_extension.AuxLayerWorkerExtension \
  --speculative_config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[2,18,34]}}}' \
  --kv_transfer_config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/dev/shm/hidden_states"}}'
```

## Use

```bash
# read current layers
curl -s localhost:8000/aux_hidden_state_layers

# swap (same count) — takes effect on the next forward, across all TP/PP workers
curl -s -X POST localhost:8000/aux_hidden_state_layers \
  -H 'content-type: application/json' \
  -d '{"layers": [4, 16, 28]}'
```

The safetensors files written by the hidden-states connector carry no
layer-set metadata, so a controller that swaps mid-run must record the
`{timestamp/req-range -> layer set}` mapping itself.
