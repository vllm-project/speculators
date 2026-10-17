"""LiLiCorr head parity between speculators training and vLLM serving.

A LiLiCorr checkpoint trained by speculators must score the candidate
lattice in vLLM exactly as it did in training. This test loads a randomly
initialized speculators ``LiLiCorrHead`` state dict into vLLM's serving
head (``vllm/model_executor/models/lilicorr.py``) and compares the
logit-scaled lattice scores on identical inputs, pinning the checkpoint
contract: submodule names, candidate feature order, attention bias
geometry, the factor-input split, and the logit scale.
"""

from types import SimpleNamespace

import pytest
import torch

from speculators.models.lilicorr.model_definitions import LiLiCorrHead
from tests.conftest import requires_cuda

try:
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import parallel_state
    from vllm.model_executor.models.lilicorr import (
        LiLiCorrHead as VllmLiLiCorrHead,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

requires_vllm = pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")

MODEL_HIDDEN = 16
BLOCK_SIZE = 8
TOPK = 4


def _spec_head(head_width: int, seed: int = 7) -> LiLiCorrHead:
    """Build a speculators head with nondegenerate weights everywhere.

    ``reset_parameters`` leaves the lattice embeddings and attention biases
    at zero and the QKV at xavier; re-randomize everything so the parity
    check cannot pass on a degenerate slice of the math.
    """
    torch.manual_seed(seed)
    head = LiLiCorrHead(
        model_hidden_size=MODEL_HIDDEN,
        block_size=BLOCK_SIZE,
        candidate_topk=TOPK,
        hidden_size=head_width,
        factor_dim=12,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        logit_scale=8.0,
        vector_eps=1e-6,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        for param in head.parameters():
            param.normal_(std=0.1)
    return head


def _vllm_head(
    state: dict[str, torch.Tensor], head_width: int, device: torch.device
) -> "VllmLiLiCorrHead":
    # Single-rank construction only needs TP metadata.
    old_tp = parallel_state._TP
    parallel_state._TP = SimpleNamespace(rank_in_group=0, world_size=1)  # type: ignore[assignment]
    try:
        with (
            torch.device(device),
            set_current_vllm_config(
                VllmConfig(
                    device_config=DeviceConfig("cuda"),
                    compilation_config=CompilationConfig(mode=CompilationMode.NONE),
                )
            ),
        ):
            head = VllmLiLiCorrHead(
                model_hidden_size=MODEL_HIDDEN,
                block_size=BLOCK_SIZE,
                rms_norm_eps=1e-6,
                config={
                    "lilicorr_candidate_topk": TOPK,
                    "lilicorr_hidden_size": head_width,
                    "lilicorr_factor_dim": 12,
                    "lilicorr_num_layers": 2,
                    "lilicorr_num_heads": 2,
                    "lilicorr_mlp_ratio": 2.0,
                    "lilicorr_logit_scale": 8.0,
                    "lilicorr_vector_eps": 1e-6,
                },
            )
        head.load_state_dict(state)
        head.materialize_inference_buffers(device, torch.float32)
    finally:
        parallel_state._TP = old_tp
    return head


def _inputs(device: torch.device, batch: int = 2, seed: int = 123):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    num_slots = BLOCK_SIZE - 1
    # Top-k values of a random log-softmax give descending-order log probs,
    # the rank order both heads are trained and served with.
    log_probs = randn(batch, num_slots, 23).log_softmax(-1).topk(TOPK).values
    return (
        randn(batch, num_slots, TOPK, MODEL_HIDDEN),  # token_embeddings
        log_probs,  # candidate_log_probs
        randn(batch, num_slots, MODEL_HIDDEN),  # pass_hidden
        randn(batch, MODEL_HIDDEN),  # anchor_hidden
        torch.tensor([True, False], device=device),  # anchor_valid
    )


@requires_cuda
class TestLiLiCorrHeadVllmParity:
    """Verify the speculators-trained head scores identically in vLLM."""

    @requires_vllm
    @pytest.mark.parametrize(
        "head_width", [8, 16], ids=["projected", "identity-token-proj"]
    )
    def test_scores_match_vllm_serving_head(self, head_width):
        device = torch.device("cuda")
        speculators_head = _spec_head(head_width).to(device)
        vllm_head = _vllm_head(speculators_head.state_dict(), head_width, device)

        inputs = _inputs(device)
        with torch.no_grad():
            log_start, log_pair = speculators_head(*inputs)
            actual = vllm_head(*inputs)

        # vLLM's forward returns the logit-scaled [B, S, K, K] lattice with
        # the anchor start row broadcast over predecessor index 0.
        expected = torch.cat(
            [log_start[:, None, None].expand(-1, 1, TOPK, -1), log_pair], dim=1
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @requires_vllm
    def test_state_dict_loads_strictly(self):
        """Every vLLM head parameter is covered by the speculators state dict."""
        device = torch.device("cuda")
        speculators_head = _spec_head(8)
        vllm_head = _vllm_head(
            {k: v.to(device) for k, v in speculators_head.state_dict().items()},
            8,
            device,
        )
        speculators_head = speculators_head.to(device)
        for name, param in vllm_head.named_parameters():
            assert name in speculators_head.state_dict(), f"missing {name}"
            torch.testing.assert_close(param, speculators_head.state_dict()[name])
