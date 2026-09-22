"""Unit tests for FP8 hidden-states quantization utilities."""

import pytest
import torch
from hs_connectors.fp8_utils import (
    SCALES_KEY,
    dequantize_fp8_tensor,
    quantize_tensor_to_fp8,
)


def test_quantize_round_trip_is_close():
    torch.manual_seed(0)
    hidden_states = torch.randn(8, 3, 128, dtype=torch.bfloat16) * 5.0

    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    restored = dequantize_fp8_tensor(fp8_hs, scales, dtype=torch.bfloat16)

    assert fp8_hs.dtype == torch.float8_e4m3fn
    assert restored.shape == hidden_states.shape
    assert restored.dtype == torch.bfloat16
    # FP8 e4m3 has ~2 decimal digits of precision; allow generous tolerance.
    torch.testing.assert_close(
        restored.float(), hidden_states.float(), atol=0.5, rtol=0.1
    )


def test_scale_shape_has_trailing_singletons():
    hidden_states = torch.randn(4, 3, 64)
    _, scales = quantize_tensor_to_fp8(hidden_states)
    assert scales.shape == (4, 1, 1)


def test_scale_is_per_token_not_per_layer():
    """A single scale is shared across all layers of a given token, but
    differs across tokens."""
    hidden_states = torch.zeros(2, 2, 16)
    hidden_states[0, 0] = 1.0
    hidden_states[0, 1] = 100.0
    hidden_states[1, 0] = 0.01
    hidden_states[1, 1] = 50.0

    _, scales = quantize_tensor_to_fp8(hidden_states)
    # One scale per token (dims 1, 2 are singleton), derived from the max
    # magnitude across *both* layers -- i.e. token 0's scale reflects its
    # layer-1 value of 100.0, not just its layer-0 value of 1.0.
    assert scales.shape == (2, 1, 1)
    assert scales[0].item() == pytest.approx(
        100.0 / torch.finfo(torch.float8_e4m3fn).max
    )
    # ...and scales differ across tokens (different magnitudes).
    assert scales[0].item() != scales[1].item()


def test_all_zero_tensor_does_not_produce_nan_or_inf():
    hidden_states = torch.zeros(4, 3, 32)
    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    restored = dequantize_fp8_tensor(fp8_hs, scales)
    assert torch.isfinite(restored).all()
    assert torch.equal(restored, torch.zeros_like(restored))


def test_negative_values_round_trip():
    hidden_states = torch.tensor([[[-10.0, 3.0, -0.5, 7.0]]])
    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    restored = dequantize_fp8_tensor(fp8_hs, scales, dtype=torch.float32)
    torch.testing.assert_close(restored, hidden_states, atol=0.5, rtol=0.1)


def test_dequantize_default_dtype_is_bfloat16():
    hidden_states = torch.randn(2, 1, 16)
    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    restored = dequantize_fp8_tensor(fp8_hs, scales)
    assert restored.dtype == torch.bfloat16


def test_scales_key_constant():
    assert SCALES_KEY == "hidden_states_scales"


@pytest.mark.parametrize("shape", [(1, 1, 1), (16, 3, 4096), (5, 1, 8)])
def test_various_shapes(shape):
    hidden_states = torch.randn(*shape)
    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    assert fp8_hs.shape == hidden_states.shape
    assert scales.shape == (shape[0], *([1] * (len(shape) - 1)))
    restored = dequantize_fp8_tensor(fp8_hs, scales, dtype=torch.float32)
    assert restored.shape == hidden_states.shape


def test_scale_is_per_token_over_2d_input():
    """For a plain 2-D input, behavior matches a simple per-row scale."""
    hidden_states = torch.tensor([[1.0, 2.0, -4.0], [10.0, -20.0, 5.0]])
    _, scales = quantize_tensor_to_fp8(hidden_states)
    assert scales.shape == (2, 1)
    assert scales[0].item() != scales[1].item()
