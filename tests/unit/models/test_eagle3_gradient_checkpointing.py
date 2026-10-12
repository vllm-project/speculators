"""Gradient equivalence of EAGLE-3 training with and without checkpointing.

Gradient checkpointing replays each decoder layer's forward during the
backward pass. EAGLE-3 unrolls ``ttt_steps`` passes over the same sequence
and step ``s`` attends over the K/V of steps ``0..s`` through a shared
``DynamicCache``, so the replay must see exactly the prefix the original
forward saw. Two things have to hold for that:

1. every decoder layer opts in via ``_can_checkpoint_with_cache``, otherwise
   transformers' ``GradientCheckpointingLayer`` nulls ``past_key_values`` and
   the layer is left with ``kv_len == q_len`` while the attention mask was
   built for the grown cache;
2. each layer crops **only its own** cache entry back to ``cache_position``,
   since autograd gives no ordering guarantee across the layers' replays.

``num_hidden_layers=2`` is the case that matters: the first layer is
``{Llama,Qwen3}DecoderEagle3FirstLayer`` while layers ``1..N-1`` take a
separate code path, so a 1-layer draft never exercises it.

Runs in fp32 with eager attention on CPU: the comparison is about cache
bookkeeping, not about the attention backend, and fp32 keeps rounding noise
from masking a real replay bug.
"""

import copy

import pytest
import torch
from transformers import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.eagle3.config import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.proposals import GreedyTokenProposalConfig

SEQ_LEN = 32
HIDDEN_SIZE = 32
VOCAB_SIZE = 64
NUM_AUX = 3
TTT_STEPS = 2


def _build_model(num_hidden_layers: int) -> Eagle3DraftModel:
    transformer_layer_config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
    )
    # Eager attention keeps the test on CPU; flex attention requires a GPU.
    transformer_layer_config._attn_implementation = "eager"
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=transformer_layer_config,
        draft_vocab_size=VOCAB_SIZE,
        norm_before_residual=True,
        eagle_aux_hidden_state_layer_ids=list(range(NUM_AUX)),
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=TTT_STEPS),
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )

    torch.manual_seed(0)
    model = Eagle3DraftModel(config).float()
    # `_init_vocab` fills the verifier-copied weights with NaN so a missing
    # load is obvious; give them real values instead of loading a verifier.
    for param in model.parameters():
        if torch.isnan(param).any():
            torch.nn.init.normal_(param, std=0.02)
    return model.train()


def _make_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        "hidden_states": torch.randn(
            1, SEQ_LEN, NUM_AUX * HIDDEN_SIZE, generator=generator
        ),
        "verifier_last_hidden_states": torch.randn(
            1, SEQ_LEN, HIDDEN_SIZE, generator=generator
        ),
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), generator=generator),
        # Four documents so the document-boundary mask is exercised.
        "document_ids": torch.arange(SEQ_LEN)
        .div(SEQ_LEN // 4, rounding_mode="floor")
        .clamp(max=3)
        .unsqueeze(0),
        "loss_mask": torch.ones(1, SEQ_LEN, dtype=torch.bool),
    }


def _forward_backward(
    model: Eagle3DraftModel, batch: dict[str, torch.Tensor]
) -> tuple[float, dict[str, torch.Tensor]]:
    for param in model.parameters():
        param.grad = None
    # The default "fused" loss implementation needs a Triton GPU kernel.
    _, loss, _ = model(
        **copy.deepcopy(batch),
        ttt_steps=TTT_STEPS,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss.item(), grads


@pytest.mark.parametrize("num_hidden_layers", [1, 2])
def test_gradient_checkpointing_matches_plain_backward(num_hidden_layers):
    batch = _make_batch()
    model = _build_model(num_hidden_layers)

    loss_off, grads_off = _forward_backward(model, batch)
    model.gradient_checkpointing_enable()
    loss_on, grads_on = _forward_backward(model, batch)

    assert grads_off.keys() == grads_on.keys()
    assert grads_off, "no parameter received a gradient"
    assert loss_off == pytest.approx(loss_on, abs=1e-6)

    for name in grads_off:
        off, on = grads_off[name], grads_on[name]
        scale = off.abs().max().item() or 1.0
        rel_diff = (off - on).abs().max().item() / scale
        assert rel_diff < 1e-6, (
            f"{name}: checkpointed gradient differs by {rel_diff:.3e} "
            f"(relative). The checkpointed backward replayed the TTT steps "
            f"against a different KV prefix than the original forward."
        )


def test_every_decoder_layer_can_checkpoint_with_cache():
    """All layers must opt in, or transformers drops the shared TTT cache."""
    model = _build_model(num_hidden_layers=2)

    assert len(model.layers) == 2
    for layer in model.layers:
        assert layer._can_checkpoint_with_cache, (
            f"{type(layer).__name__} does not set `_can_checkpoint_with_cache`, "
            f"so `past_key_values` is nulled during checkpointed training."
        )
