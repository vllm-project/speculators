"""Unit tests for FP8Transfer (the read side paired with FP8HiddenStatesConnector)."""

import torch
from hs_connectors.fp8_utils import SCALES_KEY, quantize_tensor_to_fp8
from hs_connectors.transfer import FP8Transfer
from safetensors.torch import save_file


def _write_quantized_sample(path, hidden_states: torch.Tensor, token_ids: torch.Tensor):
    fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
    save_file(
        {"hidden_states": fp8_hs, SCALES_KEY: scales, "token_ids": token_ids}, path
    )


def test_get_generated_dequantizes_transparently(tmp_path):
    hidden_states = torch.randn(6, 3, 32, dtype=torch.bfloat16) * 4.0
    token_ids = torch.arange(6, dtype=torch.long)

    handle = tmp_path / "req-1.safetensors"
    _write_quantized_sample(handle, hidden_states, token_ids)

    transfer = FP8Transfer(tmp_path)
    sample = transfer.get_generated(str(handle))

    assert sample is not None
    assert SCALES_KEY not in sample
    assert sample["hidden_states"].dtype == torch.bfloat16
    assert sample["hidden_states"].shape == hidden_states.shape
    torch.testing.assert_close(
        sample["hidden_states"].float(), hidden_states.float(), atol=0.5, rtol=0.1
    )
    torch.testing.assert_close(sample["token_ids"], token_ids)


def test_get_generated_missing_file_returns_none(tmp_path):
    transfer = FP8Transfer(tmp_path)
    assert transfer.get_generated(str(tmp_path / "missing.safetensors")) is None


def test_get_cached_dequantizes_transparently(tmp_path):
    hidden_states = torch.randn(4, 2, 16, dtype=torch.bfloat16)
    token_ids = torch.arange(4, dtype=torch.long)

    hs_dir = tmp_path / "hidden_states"
    hs_dir.mkdir()
    _write_quantized_sample(hs_dir / "hs_7.safetensors", hidden_states, token_ids)

    transfer = FP8Transfer(hs_dir)
    sample = transfer.get_cached(7)

    assert sample is not None
    assert SCALES_KEY not in sample
    torch.testing.assert_close(
        sample["hidden_states"].float(), hidden_states.float(), atol=0.5, rtol=0.1
    )


def test_passthrough_when_scales_absent(tmp_path):
    """A plain (unquantized) payload without a scales tensor is returned as-is."""
    hidden_states = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    token_ids = torch.arange(3, dtype=torch.long)

    handle = tmp_path / "req-plain.safetensors"
    save_file({"hidden_states": hidden_states, "token_ids": token_ids}, handle)

    transfer = FP8Transfer(tmp_path)
    sample = transfer.get_generated(str(handle))

    assert sample is not None
    torch.testing.assert_close(sample["hidden_states"], hidden_states)


def test_dequantize_dtype_is_configurable(tmp_path):
    hidden_states = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    token_ids = torch.arange(2, dtype=torch.long)

    handle = tmp_path / "req-fp32.safetensors"
    _write_quantized_sample(handle, hidden_states, token_ids)

    transfer = FP8Transfer(tmp_path, dequantize_dtype=torch.float32)
    sample = transfer.get_generated(str(handle))

    assert sample is not None
    assert sample["hidden_states"].dtype == torch.float32
