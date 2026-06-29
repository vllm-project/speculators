"""Integration test for P-EAGLE with StreamingLLM (attention sinks)."""

import copy

import pytest
import torch

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3.data import shift_batch
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.data import create_collate_fn
from tests.conftest import requires_cuda

_TINY_LLAMA_CONFIG = pytest.importorskip(
    "transformers.models.llama.configuration_llama"
).LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    _attn_implementation="simple_flex_attention",
)


def _make_peagle_model(
    sink_size=None,
    max_context_window=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    transformer_config = copy.deepcopy(_TINY_LLAMA_CONFIG)
    config = PEagleSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        norm_before_residual=False,
        embed_requires_grad=True,
        num_depths=4,
        down_sample_ratio=0.7,
        down_sample_ratio_min=0.2,
        mask_token_id=0,
        sink_size=sink_size,
        max_context_window=max_context_window,
        speculators_config=SpeculatorsConfig(
            algorithm="peagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )
    model = PEagleDraftModel(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buf in model.buffers():
            if buf.is_floating_point() and buf.isnan().any():
                buf.zero_()
    return model.to(device=device, dtype=dtype)


def _make_batch(seq_lengths, hidden_size=64, max_len=128, device="cuda:0"):
    samples = []
    for sl in seq_lengths:
        samples.append(
            {
                "hidden_states": torch.randn(sl, 3 * hidden_size, dtype=torch.bfloat16),
                "input_ids": torch.randint(0, 128, (sl,)),
                "verifier_last_hidden_states": torch.randn(
                    sl, hidden_size, dtype=torch.bfloat16
                ),
                "loss_mask": torch.ones(sl, dtype=torch.bfloat16),
                "lengths": torch.tensor([sl], dtype=torch.long),
                "position_ids": torch.arange(sl, dtype=torch.long),
            }
        )
    collate_fn = create_collate_fn(max_len, hidden_size, preprocess=shift_batch)
    batch = collate_fn(samples)
    return {k: v.to(device) for k, v in batch.items()}


@requires_cuda
class TestPEagleStreaming:
    def test_forward_backward_streaming(self):
        model = _make_peagle_model(sink_size=4, max_context_window=32)
        batch = _make_batch([128])
        _, loss, metrics = model(**batch)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert "loss_sum" in metrics
        loss.backward()

    def test_forward_backward_no_streaming(self):
        """Baseline: verify forward+backward works without streaming too."""
        model = _make_peagle_model()
        batch = _make_batch([128])
        _, loss, metrics = model(**batch)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        loss.backward()

    def test_forward_backward_streaming_multi_doc(self):
        model = _make_peagle_model(sink_size=4, max_context_window=16)
        batch = _make_batch([64, 64])
        _, loss, metrics = model(**batch)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        loss.backward()

    def test_streaming_config_roundtrip(self, tmp_path):
        """Config with streaming params survives save/load."""
        model = _make_peagle_model(sink_size=8, max_context_window=64)
        model.save_pretrained(str(tmp_path))

        loaded_config = PEagleSpeculatorConfig.from_pretrained(str(tmp_path))
        assert loaded_config.sink_size == 8
        assert loaded_config.max_context_window == 64

    def test_config_validation_both_or_neither(self):
        """Setting only one of sink_size/max_context_window raises."""
        with pytest.raises(ValueError, match="both be set or both be None"):
            PEagleSpeculatorConfig(
                transformer_layer_config=copy.deepcopy(_TINY_LLAMA_CONFIG),
                sink_size=4,
                max_context_window=None,
            )
        with pytest.raises(ValueError, match="both be set or both be None"):
            PEagleSpeculatorConfig(
                transformer_layer_config=copy.deepcopy(_TINY_LLAMA_CONFIG),
                sink_size=None,
                max_context_window=32,
            )
