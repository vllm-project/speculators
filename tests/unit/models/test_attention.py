"""Tests for the FA4 (FlashAttention-4) flex attention backend."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from speculators.models.attention import (
    _should_use_fa4,
    configure_fa4,
    fa4_is_available,
    flex_attention_forward,
)
from tests.conftest import requires_cuda

requires_fa4 = pytest.mark.skipif(not fa4_is_available(), reason="FA4 not available")


@requires_cuda
@requires_fa4
class TestFA4ForwardPass:
    """Run flex_attention_forward with the FA4 backend on real hardware."""

    @pytest.fixture(autouse=True)
    def _enable_fa4(self):
        configure_fa4("on")
        yield
        configure_fa4("auto")

    def _make_inputs(self, batch, heads_q, heads_kv, seq_len, head_dim):
        q = torch.randn(
            batch, heads_q, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            batch, heads_kv, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn(
            batch, heads_kv, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        mask = create_block_mask(
            causal, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device="cuda"
        )
        return q, k, v, mask

    def test_basic_forward(self):
        assert _should_use_fa4()
        q, k, v, mask = self._make_inputs(
            batch=2, heads_q=8, heads_kv=8, seq_len=128, head_dim=64
        )
        out, weights = flex_attention_forward(
            module=torch.nn.Identity(), query=q, key=k, value=v, attention_mask=mask
        )
        assert out.shape == (2, 128, 8, 64)
        assert weights is None
        assert out.isfinite().all()

    def test_gqa(self):
        q, k, v, mask = self._make_inputs(
            batch=2, heads_q=8, heads_kv=2, seq_len=128, head_dim=64
        )
        out, _ = flex_attention_forward(
            module=torch.nn.Identity(), query=q, key=k, value=v, attention_mask=mask
        )
        assert out.shape == (2, 128, 8, 64)
        assert out.isfinite().all()

    def test_matches_triton_backend(self):
        """FA4 and Triton backends should produce close results."""
        q, k, v, mask = self._make_inputs(
            batch=2, heads_q=8, heads_kv=8, seq_len=128, head_dim=64
        )

        configure_fa4("on")
        fa4_out, _ = flex_attention_forward(
            module=torch.nn.Identity(), query=q, key=k, value=v, attention_mask=mask
        )

        configure_fa4("off")
        triton_out, _ = flex_attention_forward(
            module=torch.nn.Identity(), query=q, key=k, value=v, attention_mask=mask
        )

        torch.testing.assert_close(fa4_out, triton_out, atol=1e-2, rtol=1e-2)
