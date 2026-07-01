"""DFlash attention backend parity tests.

Verifies that all attention backends (eager, SDPA, flex_attention) produce
equivalent outputs for the DFlash draft model, and that anchor-block masks
are consistent between speculators and vLLM-style masking.

See: https://github.com/vllm-project/speculators/issues/686
"""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from tests.conftest import requires_cuda
from tests.integration.conftest import (
    HIDDEN_SIZE,
    VOCAB_SIZE,
    make_batch,
    make_dflash_model,
    make_sample,
)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def _build_anchor_block_mask(
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    document_ids: torch.Tensor,
    non_causal: bool = True,
) -> torch.Tensor:
    """Materialize the anchor-block mask as a dense boolean tensor."""
    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=anchor_positions.device)

    for q_idx in range(q_len):
        anchor_idx = q_idx // block_size
        q_anchor = int(anchor_positions[anchor_idx].item())
        q_doc = int(document_ids[q_anchor].item())

        for kv_idx in range(kv_len):
            if kv_idx < total_seq_len:
                kv_doc = document_ids[kv_idx].item()
                if q_doc == kv_doc and q_doc != -1 and kv_idx < q_anchor:
                    mask[q_idx, kv_idx] = True
            else:
                q_block = q_idx // block_size
                kv_block = (kv_idx - total_seq_len) // block_size
                if q_block == kv_block and (
                    non_causal or kv_idx <= q_idx + total_seq_len
                ):
                    mask[q_idx, kv_idx] = True

    return mask


@requires_cuda
class TestDFlashAttentionParity:
    """Verify attention backend parity for DFlash."""

    @pytest.mark.parametrize(
        ("backend_a", "backend_b"),
        [("eager", "sdpa"), ("eager", "simple_flex_attention")],
    )
    def test_attention_backends_agree(self, backend_a, backend_b):
        """All attention backends must produce near-identical outputs.

        Regression test for the boolean-mask bug where create_mask returned
        a bool tensor but eager_attention_forward added it numerically,
        effectively ignoring the mask.
        """
        torch.manual_seed(42)
        ref = make_dflash_model(block_size=4, max_anchors=4, draft_attn_impl="eager")
        samples = [
            make_sample(
                seq_len=64,
                hidden_size=HIDDEN_SIZE,
                hidden_multiplier=3,
                vocab_size=VOCAB_SIZE,
                loss_mask_pattern="all",
            )
        ]
        batch = make_batch(max_len=64, samples=samples, hidden_size=HIDDEN_SIZE)
        state = ref.state_dict()

        outputs = {}
        for backend in (backend_a, backend_b):
            torch.manual_seed(42)
            m = make_dflash_model(block_size=4, max_anchors=4, draft_attn_impl=backend)
            m.load_state_dict(state)
            with torch.no_grad():
                _, loss, _ = m(**batch)
            outputs[backend] = loss.item()
            del m
            torch.cuda.empty_cache()

        assert abs(outputs[backend_a] - outputs[backend_b]) < 0.01, (
            f"Backend divergence: {backend_a} loss={outputs[backend_a]:.4f}, "
            f"{backend_b} loss={outputs[backend_b]:.4f}"
        )

    def test_single_doc_mask_matches_vllm_semantics(self):
        """For single-document input, the anchor-block mask matches vLLM semantics.

        vLLM implements DFlash attention via KV-cache with a causal/non-causal
        flag rather than an explicit mask.  The reference mask below is a
        hand-written reimplementation of those semantics (prefix-causal for
        context KV + within-block for draft KV), NOT code copied from vLLM.
        This lets us verify mask equivalence without depending on vLLM.
        """
        device = "cuda:0"
        seq_len = 32
        block_size = 4
        document_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
        anchor_positions = torch.tensor([8, 16, 24], device=device)

        spec_mask = _build_anchor_block_mask(
            seq_len, anchor_positions, block_size, document_ids
        )

        # Reference mask reimplementing vLLM's effective attention pattern:
        #   - context KV (kv_idx < seq_len): attend to positions before anchor
        #   - draft KV (kv_idx >= seq_len): attend within same block only
        n_anchors = anchor_positions.numel()
        q_len = n_anchors * block_size
        kv_len = seq_len + q_len
        ref_mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        for q_idx in range(q_len):
            anchor_idx = q_idx // block_size
            q_anchor = anchor_positions[anchor_idx].item()
            for kv_idx in range(kv_len):
                if kv_idx < seq_len:
                    if kv_idx < q_anchor:
                        ref_mask[q_idx, kv_idx] = True
                else:
                    q_block = q_idx // block_size
                    kv_block = (kv_idx - seq_len) // block_size
                    if q_block == kv_block:
                        ref_mask[q_idx, kv_idx] = True

        assert torch.equal(spec_mask, ref_mask)

    def test_mask_applied_correctly_by_eager(self):
        """Float mask (0/-inf) produces identical results to masked_fill.

        ``eager_attention_forward`` computes ``scores + mask``, so the mask
        must use 0 (attend) and -inf (masked).  This test verifies that
        approach is numerically equivalent to ``masked_fill(~bool, -inf)``.
        """
        torch.manual_seed(42)
        device = "cuda:0"
        dtype = torch.bfloat16

        bsz, num_heads, head_dim = 1, 4, 16
        seq_len, block_size = 32, 4
        anchor_positions = torch.tensor([8, 16, 24], device=device)
        n_anchors = anchor_positions.numel()
        q_len = n_anchors * block_size
        kv_len = seq_len + q_len
        scale = head_dim**-0.5

        q = torch.randn(bsz, num_heads, q_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(bsz, num_heads, kv_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(bsz, num_heads, kv_len, head_dim, device=device, dtype=dtype)

        document_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
        bool_mask = _build_anchor_block_mask(
            seq_len, anchor_positions, block_size, document_ids
        )

        # Reference: correct mask application
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_w_ref = attn_w.masked_fill(~bool_mask, float("-inf"))
        out_ref = torch.matmul(
            F.softmax(attn_w_ref, dim=-1, dtype=torch.float32).to(dtype), v
        )

        # Float mask: what create_float_mask produces
        float_mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
        float_mask.masked_fill_(~bool_mask, float("-inf"))
        attn_w_float = attn_w + float_mask
        out_float = torch.matmul(
            F.softmax(attn_w_float, dim=-1, dtype=torch.float32).to(dtype), v
        )

        assert _cosine_sim(out_ref, out_float) > 0.9999
