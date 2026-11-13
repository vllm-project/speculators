"""Rotary position embedding for the DSV4 DSpark draft (interleaved, YaRN-capable).

Two pieces:

* :func:`precompute_freqs_cis` builds complex exponentials ``e^{i·t·θ_k}`` with
  optional YaRN frequency interpolation (a smooth linear ramp between the
  ``beta_fast`` / ``beta_slow`` correction dims). The draft's sliding-window
  attention runs YaRN **off** (pass ``original_seq_len=0``) with the base
  ``rope_theta`` — matching the reference, which disables YaRN on the pure
  sliding path.
* :func:`apply_rotary_emb` rotates the trailing ``rope_head_dim`` slice of a
  ``[..., D]`` tensor using the **interleaved** pairing ``(x0,x1),(x2,x3),…``.
  ``inverse=True`` conjugates the rotation to de-rotate the attention output's
  rope slice (needed because DSV4 shares K=V, so V carried the rotation).

The result is applied out-of-place (returns a new tensor) rather than the reference's
in-place ``copy_`` so it composes cleanly with autograd.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: str = "cpu",
) -> torch.Tensor:
    """Complex rotary frequencies ``[seqlen, dim//2]`` with optional YaRN.

    ``dim`` is the rope slice width (``rope_head_dim``). With
    ``original_seq_len == 0`` YaRN is disabled and plain ``1/base^(2k/dim)``
    frequencies are used (the draft's sliding-window path).
    """

    def correction_dim(num_rotations: float) -> float:
        return (
            dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def correction_range(low_rot: float, high_rot: float) -> tuple[int, int]:
        low = math.floor(correction_dim(low_rot))
        high = math.ceil(correction_dim(high_rot))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp(lo: float, hi: float, n: int) -> torch.Tensor:
        if lo == hi:
            hi += 0.001
        ramp = (torch.arange(n, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(ramp, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = correction_range(beta_fast, beta_slow)
        smooth = 1 - linear_ramp(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis.to(device)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Rotate ``x`` (``[B, S, ..., rope_dim]``) with REAL interleaved cos/sin.

    ``freqs_cis`` is the interleaved rotary cache as a REAL tensor ``[S, rope_dim//2,
    2]`` (``[...,0]``=cos, ``[...,1]``=sin); a complex ``[S, rope_dim//2]`` is accepted
    and converted via ``view_as_real``. The rotation is applied as ``x·cos +
    rotate_half(x)·sin`` rather than a complex multiply, for two reasons: a complex
    cache cast to bf16 silently loses its imaginary part, which turns the rotation into
    a scale and is not an error anyone sees; and complex indexing and multiplication are
    not available on every accelerator backend. Interleaved:
    ``out_2k = x_2k·cos_k - x_2k+1·sin_k``, ``out_2k+1 = x_2k·sin_k + x_2k+1·cos_k``.
    ``inverse`` negates sin (de-rotation, K=V shared rotation).
    """
    if freqs_cis.is_complex():
        freqs_cis = torch.view_as_real(freqs_cis)
    cos = freqs_cis[..., 0].repeat_interleave(2, dim=-1)  # [S, rope_dim]
    sin = freqs_cis[..., 1].repeat_interleave(2, dim=-1)
    if inverse:
        sin = -sin
    # Broadcast cos/sin over batch + any head dims: shape [1, S, (1,)*extra, rope_dim].
    seq_len = x.shape[1]
    extra = x.ndim - 2  # dims between the seq axis and the rope axis
    view_shape = (
        (1, seq_len, *([1] * (extra - 1)), x.shape[-1])
        if extra >= 1
        else (1, seq_len, x.shape[-1])
    )
    cos = cos.reshape(*view_shape).float()
    sin = sin.reshape(*view_shape).float()
    xf = x.float()
    pair = xf.unflatten(-1, (-1, 2))  # [..., rope//2, 2]
    rotate_half = torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).flatten(
        -2
    )  # interleaved
    return (xf * cos + rotate_half * sin).to(x.dtype)
