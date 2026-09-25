"""Pre-compile monkeypatch making aux-layer selection a runtime tensor input.

This is "Option B" (masked accumulate). The stock capture does::

    if layer_idx in self.aux_hidden_state_layers:   # Python control flow
        aux.append(hidden_states + residual)

``torch.compile`` constant-folds that membership test into the compiled
forward, freezing the captured layer set — which is why the naive
attribute-swap only takes effect under ``--enforce-eager``.

Instead we keep ``N`` fixed accumulator buffers and, at *every* candidate
capture point, fold each layer's residual into them weighted by a one-hot
selection mask held in a registered buffer::

    value = hidden_states + residual
    for slot in range(N):
        aux[slot] += mask[layer_idx, slot] * value

The graph is now selection-agnostic: which layers land in which slot is a
function of the mask *contents*, not the traced code. Swapping layers is an
in-place ``mask.copy_(...)`` — same storage address, so both the compiled
graph and a captured CUDA graph read the new selection on the next forward.
No recompile, no ``--enforce-eager``.

Cost model: memory stays O(N) (N accumulators, not O(num_layers)); the price
is a little compute — a fused multiply-add over every candidate layer's
residual per output slot (negligible next to the attention/MLP matmuls).

Installed via a ``vllm.general_plugins`` entry point so it patches
``EagleModelMixin`` in the worker process *before* the target model is loaded
and compiled.

Scope: covers models that capture through
``EagleModelMixin._maybe_add_hidden_state`` (Qwen2/Qwen3, Llama, and the other
generic dense/MoE paths). A handful of models (e.g. deepseek_v2, qwen3_next)
inline the ``layer_idx in self.aux_hidden_state_layers`` test in their own
forward and are not covered by this patch — those still require
``--enforce-eager`` for a live swap.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import EagleModelMixin

# Namespace under "vllm." so vLLM's logging config (which only attaches a
# handler to the "vllm" logger) surfaces these INFO lines.
logger = init_logger(f"vllm.plugins.{__name__}")

MASK_ATTR = "aux_capture_mask"
_PATCHED_FLAG = "_dyn_hidden_states_mask_patched"


def _num_capture_points(model: object) -> int:
    """Candidate capture points: embeddings (index 0) + one per decoder layer.

    Matches the generic forward, which calls ``_maybe_add_hidden_state`` with
    ``layer_idx=0`` for the embedding output and ``idx + 1`` for each layer.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0
    return len(layers) + 1


def _build_mask(
    layers: tuple[int, ...],
    num_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One-hot ``[num_points, N]`` mask; column j = j-th smallest layer index.

    Ascending column order reproduces the stock append order (the loop appends
    in layer order regardless of the requested tuple order), so downstream
    column identity is unchanged.
    """
    ordered = sorted(int(x) for x in layers)
    mask = torch.zeros(num_points, len(ordered), device=device, dtype=dtype)
    for col, layer in enumerate(ordered):
        if 0 <= layer < num_points:
            mask[layer, col] = 1.0
    return mask


def _refresh_mask(model: object, layers: tuple[int, ...]) -> None:
    """Register (first call, pre-compile) or in-place update the mask buffer.

    After the first registration the buffer's storage is reused via ``copy_``
    so the compiled/CUDA graph — which captured that storage address — reads
    the new selection without a recompile. The layer *count* is fixed at
    launch, so the shape never changes after the first call.
    """
    num_points = _num_capture_points(model)
    n = len(layers)
    if num_points == 0 or n == 0:
        return

    ref = next(model.parameters())  # match model compute dtype/device
    mask = _build_mask(layers, num_points, ref.device, ref.dtype)

    existing = getattr(model, MASK_ATTR, None)
    same_shape = isinstance(existing, torch.Tensor) and existing.shape == mask.shape
    if same_shape:
        existing.copy_(mask)  # in-place: preserves storage address
    else:
        # First install, before compile. Non-persistent: derived from the
        # layer set, not a trained weight, so keep it out of the state dict.
        model.register_buffer(MASK_ATTR, mask, persistent=False)


def _patched_set_aux_hidden_state_layers(
    self: EagleModelMixin, layers: tuple[int, ...]
) -> None:
    self.aux_hidden_state_layers = tuple(int(x) for x in layers)
    _refresh_mask(self, self.aux_hidden_state_layers)


def _patched_maybe_add_hidden_state(
    self: EagleModelMixin,
    aux_hidden_states: list[torch.Tensor],
    layer_idx: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> list[torch.Tensor]:
    mask = getattr(self, MASK_ATTR, None)
    if mask is None:
        return aux_hidden_states
    n = mask.shape[1]
    if n == 0 or layer_idx >= mask.shape[0]:
        return aux_hidden_states

    value = hidden_states + residual if residual is not None else hidden_states
    # First candidate point of this forward: allocate the N accumulators.
    if len(aux_hidden_states) == 0:
        for _ in range(n):
            aux_hidden_states.append(torch.zeros_like(value))

    row = mask[layer_idx]  # [N], live buffer contents (not baked at trace time)
    for j in range(n):
        aux_hidden_states[j] = aux_hidden_states[j] + row[j] * value
    return aux_hidden_states


def install() -> None:
    """Entry point for the ``vllm.general_plugins`` group (called with no args)."""
    if getattr(EagleModelMixin, _PATCHED_FLAG, False):
        return
    EagleModelMixin._set_aux_hidden_state_layers = _patched_set_aux_hidden_state_layers
    EagleModelMixin._maybe_add_hidden_state = _patched_maybe_add_hidden_state
    setattr(EagleModelMixin, _PATCHED_FLAG, True)
    logger.info(
        "dynamic_hidden_states: installed masked-accumulate aux-capture patch; "
        "layer swaps take effect at runtime without --enforce-eager "
        "(for models using EagleModelMixin._maybe_add_hidden_state)."
    )
