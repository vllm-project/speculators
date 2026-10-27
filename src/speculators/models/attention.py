"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

import json
import os
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.attention.flex_attention as _flex_attention_mod
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)
from torch.nn.attention.flex_attention import (
    create_mask as _create_mask,
)
from transformers.modeling_utils import AttentionInterface

from speculators.models.utils import conditional_torch_compile, npu_available


# On NPU, force single-threaded inductor compilation so that monkey-patches
# applied below are visible to the compiler (subprocess workers would not
# inherit them).
if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "0") != "1":
    try:
        import torch_npu  # noqa: F401, PLC0415
        if torch.npu.is_available():
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    except ImportError:
        pass


# --- NPU flex_attention support -------------------------------------------------
# flex_attention is a HigherOrderOperator: it must be called inside a
# torch.compile graph so that dynamo traces it and inductor lowers it to a
# Triton kernel (via triton-ascend on Ascend NPU).  Without compilation the
# HOP has no registered kernel for the NPU dispatch key and raises
# NotImplementedError.
#
# Additionally, PyTorch's _validate_device hard-codes a device allowlist that
# does not include "npu", so we monkey-patch it before any call.
#
# torch_npu._inductor registers fallback lowerings for *all* HigherOrderOperator
# ops, including flex_attention.  The fallback handler does not understand
# Subgraph arguments and crashes with "AttributeError: 'Subgraph' object has no
# attribute 'dtype'".  We work around this by saving the upstream lowering
# before importing torch_npu._inductor and restoring it afterward.
#
# Two more upstream hard-codings must be patched:
#   1. flex_decoding.get_split_k calls torch.cuda.get_device_properties()
#   2. NPUTritonScheduling.define_kernel expects an extra traced_graph_hash arg

def _patch_flex_attention_device_check() -> None:
    """Allow flex_attention on NPU by extending the device allowlist."""
    original = _flex_attention_mod._validate_device

    def _patched_validate_device(query, key, value):  # noqa: ANN001
        if query.device.type == "npu":
            return
        return original(query, key, value)

    _flex_attention_mod._validate_device = _patched_validate_device


_patch_flex_attention_device_check()


def _setup_npu_flex_attention() -> None:
    """Apply all monkey-patches needed for flex_attention on Ascend NPU."""
    import torch_npu  # noqa: F401, PLC0415

    # 1. Save upstream flex_attention lowerings before torch_npu._inductor
    #    replaces them with crash-prone fallbacks.
    from torch._inductor.lowering import lowerings, needs_realized_inputs

    flex_op = torch.ops.higher_order.flex_attention
    flex_bwd_op = torch.ops.higher_order.flex_attention_backward
    orig_flex_fwd = lowerings.get(flex_op)
    orig_flex_bwd = lowerings.get(flex_bwd_op)

    # 2. Import torch_npu._inductor — applies triton-ascend patches that make
    #    inductor recognise NPU as a triton backend.
    import torch_npu._inductor  # noqa: F401, PLC0415

    # 3. Restore the upstream flex_attention lowerings.
    lowerings[flex_op] = orig_flex_fwd
    lowerings[flex_bwd_op] = orig_flex_bwd
    needs_realized_inputs.discard(flex_op)
    needs_realized_inputs.discard(flex_bwd_op)

    # 4. Patch flex_decoding.get_split_k — upstream hard-codes
    #    torch.cuda.get_device_properties() to obtain SM count.
    import torch._inductor.kernel.flex.flex_decoding as _flex_decoding_mod

    def _patched_get_split_k(B, H, Mk):  # noqa: ANN001, N803, ARG001
        if torch.npu.is_available():
            num_SM = torch.npu.get_device_properties("npu").multi_processor_count
        elif torch.xpu.is_available():
            num_SM = torch.xpu.get_device_properties("xpu").gpu_subslice_count
        else:
            num_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
        bh = max(B * H, 1)
        split_k = num_SM // bh * 2
        return max(split_k, 1)

    _flex_decoding_mod.get_split_k = _patched_get_split_k

    # 5. Patch NPUTritonScheduling.define_kernel — the NPU version has two
    #    incompatibilities with upstream inductor:
    #    (a) it expects an extra ``traced_graph_hash`` argument
    #    (b) it returns ``(kernel_name, src_code)`` instead of just ``kernel_name``
    #    NPU's own codegen_node_schedule passes traced_graph_hash and expects
    #    a tuple; upstream callers pass 3 args and expect a string.  A sentinel
    #    default distinguishes the two callers.
    from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling

    _orig_define_kernel = NPUTritonScheduling.define_kernel

    _sentinel = object()

    def _patched_define_kernel(self, src_code, node_schedule, kernel, traced_graph_hash=_sentinel):  # noqa: ANN001
        if traced_graph_hash is _sentinel:
            # Upstream caller — pass empty string, return only kernel_name
            result = _orig_define_kernel(self, src_code, node_schedule, kernel, "")
            if isinstance(result, tuple):
                return result[0]
            return result
        # NPU caller — pass through, return tuple as expected
        return _orig_define_kernel(self, src_code, node_schedule, kernel, traced_graph_hash)

    NPUTritonScheduling.define_kernel = _patched_define_kernel


if npu_available():
    _setup_npu_flex_attention()


def _compile_flex_attention():
    """Compile flex_attention on CUDA and NPU; leave uncompiled elsewhere."""
    if not hasattr(torch, "compile"):
        return flex_attention
    if torch.cuda.is_available() or npu_available():
        return torch.compile(flex_attention)
    return flex_attention


# Compile flex_attention on CUDA/NPU to generate fused Triton kernels.
_compiled_flex_attention = _compile_flex_attention()

# ~99 KB per block on sm_120 (RTX PRO 6000 / RTX 5090) and sm_89 (L40S / 4090);
# inductor's default flex backward tiles for head_dim > 128 need ~112 KB.
_LOW_SHARED_MEMORY_BYTES = 128 * 1024
_MAX_DEFAULT_TILE_HEAD_DIM = 128
_LOW_SHARED_MEMORY_KERNEL_OPTIONS = {
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
    "bwd_num_stages": 1,
}
# JSON dict, e.g. '{"bwd_BLOCK_M1": 16}'. Parsed once at import so the compiled
# forward only sees a constant: reading os.environ inside the traced function
# causes a dynamo graph break, which drops flex_attention out of the compiled
# graph and back to the dense-materializing eager path.
_ENV_KERNEL_OPTIONS = json.loads(
    os.environ.get("SPECULATORS_FLEX_KERNEL_OPTIONS", "null")
)


def flex_kernel_options(
    device: torch.device, head_dim: int
) -> FlexKernelOptions | None:
    """Flex ``kernel_options`` for this device, or None for inductor defaults.

    The low-shared-memory defaults assume bf16/fp16 training; fp32 needs
    smaller tiles still (e.g. 16x16), which can be forced via the
    ``SPECULATORS_FLEX_KERNEL_OPTIONS`` env var (a JSON dict,
    e.g. ``'{"bwd_BLOCK_M1": 16}'``) set before this module is imported.
    """
    if _ENV_KERNEL_OPTIONS is not None:
        return cast("FlexKernelOptions", dict(_ENV_KERNEL_OPTIONS))
    if device.type != "cuda" or head_dim <= _MAX_DEFAULT_TILE_HEAD_DIM:
        return None
    if torch.version.hip is not None:
        # ROCm reports device.type == "cuda", but torch binds
        # `shared_memory_per_block_optin` only on CUDA builds (it sits inside
        # `#ifndef USE_ROCM`), so there is nothing to probe here.
        return None
    props = torch.cuda.get_device_properties(device)
    # `getattr` because the property is NVIDIA-only and torch's stubs know
    # only the smaller static `shared_memory_per_block`.
    optin = getattr(props, "shared_memory_per_block_optin", None)
    if optin is not None and optin < _LOW_SHARED_MEMORY_BYTES:
        return cast("FlexKernelOptions", dict(_LOW_SHARED_MEMORY_KERNEL_OPTIONS))
    return None


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shared flex attention forward implementation.

    This function is used by both EAGLE3 and DFlash attention mechanisms to avoid
    code duplication and ensure consistent behavior.

    Args:
        module: The attention module (unused but required for interface compatibility).
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim).
        attention_mask: BlockMask for flex attention.
        scaling: Optional scaling factor for attention scores.
        **_kwargs: Additional unused kwargs for interface compatibility.

    Returns:
        Tuple of (attention_output, None) where attention_output has shape
        (batch, seq_len, num_heads, head_dim) and None represents no attention weights.
    """
    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    _flex_opts = flex_kernel_options(query.device, query.shape[-1])
    try:
        flex_attention_output = _compiled_flex_attention(
            query,
            key,
            value,
            score_mod=None,
            block_mask=attention_mask,
            enable_gqa=enable_gqa,
            scale=scaling,
            kernel_options=_flex_opts,
        )
    except NotImplementedError:
        # On NPU, gradient checkpointing disables dynamo via
        # @torch._disable_dynamo on torch.utils.checkpoint.checkpoint.
        # When dynamo is disabled, torch.compile(flex_attention) cannot
        # trace the HOP, so it falls through to the eager flex_attention_hop
        # which has no registered kernel on NPU.  Fall back to SDPA which
        # is mathematically equivalent for causal/block-sparse masks.
        flex_attention_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=enable_gqa,
            scale=scaling,
        )
    attention_output: torch.Tensor = flex_attention_output
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def create_float_mask(
    mask_mod: Callable,
    B: int | None = None,  # noqa: N803
    H: int | None = None,  # noqa: N803
    Q_LEN: int = 0,  # noqa: N803
    KV_LEN: int = 0,  # noqa: N803
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Wrap ``create_mask`` and convert the boolean result to a float mask.

    Non-flex attention backends (eager, SDPA) add the mask numerically
    (``scores + mask``) and need 0 for attended and ``-inf`` for masked.
    """
    bool_mask = _create_mask(
        mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    float_mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    float_mask.masked_fill_(~bool_mask, float("-inf"))
    return float_mask


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype
):
    attention_mask = torch.ones(block_mask.shape, device=device, dtype=dtype)

    for q_idx in range(attention_mask.shape[2]):
        attention_mask[0, 0, q_idx, :] = block_mask.mask_mod(
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.ones(1, device=device, dtype=torch.long) * q_idx,
            torch.arange(attention_mask.shape[3], device=device, dtype=torch.long),
        )
    return attention_mask


# Singleton registry for attention functions (shared across all models)
ALL_ATTENTION_FUNCTIONS = AttentionInterface()
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
