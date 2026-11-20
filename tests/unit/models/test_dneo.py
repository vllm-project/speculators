"""Unit tests for the DNeo drafter's components and training forward."""

from typing import Any

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from speculators.models.dneo import DNeoDraftModel, DNeoSpeculatorConfig
from speculators.models.dneo.model_definitions import (
    DNeoCausalHead,
    FeatureEncoder,
    Qwen3DNeoAttention,
    create_prefix_only_mask_mod,
    dneo_target_layer_ids,
)
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.config.schema import TrainConfig
from speculators.train.optimizers import split_named_params_for_muon


@pytest.mark.parametrize("mode", ["mofa", "diagonal"])
def test_feature_encoder_matches_its_dense_matrix(mode):
    torch.manual_seed(0)
    encoder = FeatureEncoder(3, 8, mode=mode)
    with torch.no_grad():
        for p in encoder.parameters():
            p.normal_()
    x = torch.randn(5, 3 * 8)
    torch.testing.assert_close(encoder(x), x @ encoder.to_dense().T)


def test_mofa_weights_are_a_distribution_over_layers():
    encoder = FeatureEncoder(4, 8, mode="mofa")
    x = torch.randn(2, 4, 8)
    # equal logits: the mean of the layers
    torch.testing.assert_close(encoder(x.view(2, 32)), x.mean(1))


def test_from_dense_keeps_only_within_layer_diagonals():
    torch.manual_seed(0)
    dense = torch.nn.Linear(3 * 8, 8, bias=False)
    encoder = FeatureEncoder.from_dense(dense)
    for i in range(3):
        torch.testing.assert_close(
            encoder.layer_weights[i], dense.weight[:, i * 8 : (i + 1) * 8].diagonal()
        )
    restricted = dense.weight.detach() * (encoder.to_dense() != 0)
    torch.testing.assert_close(encoder.to_dense(), restricted)


def test_dneo_protocol_ids():
    # 36 layers: the first layer, then ids 19..36 (0-based layers 18..35)
    assert dneo_target_layer_ids(36, 6) == [1, 19, 23, 28, 32, 36]
    ids = dneo_target_layer_ids(93, 6)
    assert ids[0] == 1
    assert ids[1] == 47
    assert ids[-1] == 93
    assert len(ids) == 6
    assert ids == sorted(set(ids))
    with pytest.raises(ValueError, match="second half"):
        dneo_target_layer_ids(4, 6)
    with pytest.raises(ValueError, match="num_features"):
        dneo_target_layer_ids(36, 1)


def test_prefix_only_mask_sees_only_context_before_the_anchor():
    document_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    anchors = torch.tensor([2, 6])
    mask_mod, q_len, kv_len = create_prefix_only_mask_mod(
        document_ids, 8, anchors, block_size=2
    )
    assert (q_len, kv_len) == (4, 8)
    allowed = mask_mod(0, 0, torch.arange(q_len)[:, None], torch.arange(kv_len)[None])
    expected = torch.zeros(4, 8, dtype=torch.bool)
    expected[0:2, 0:2] = True  # anchor 2, document 0: positions 0 and 1
    expected[2:4, 4:6] = True  # anchor 6, document 1: positions 4 and 5
    assert torch.equal(allowed, expected)


def test_prefix_only_mask_has_no_empty_rows():
    # anchor 0 (a padding slot or a document start) and anchor 4 (document start)
    document_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    anchors = torch.tensor([0, 4, 6])
    mask_mod, q_len, kv_len = create_prefix_only_mask_mod(
        document_ids, 8, anchors, block_size=2
    )
    allowed = mask_mod(0, 0, torch.arange(q_len)[:, None], torch.arange(kv_len)[None])
    assert allowed.any(-1).all()
    assert allowed[0].nonzero().flatten().tolist() == [0]
    assert allowed[2].nonzero().flatten().tolist() == [4]
    assert allowed[4].nonzero().flatten().tolist() == [4, 5]


def test_prefix_only_attention_equals_dflash_attention_without_draft_keys():
    torch.manual_seed(0)
    config = _transformer_config()
    dflash = Qwen3DFlashAttention(config, layer_idx=0)
    dneo = Qwen3DNeoAttention(config, layer_idx=0)
    dneo.load_state_dict(dflash.state_dict())
    ctx_len, q_len, head_dim = 6, 4, config.head_dim
    context = torch.randn(1, ctx_len, config.hidden_size)
    draft = torch.randn(1, q_len, config.hidden_size)
    positions = torch.randn(1, ctx_len + q_len, head_dim)
    embeddings = (positions.cos(), positions.sin())
    visible = torch.rand(q_len, ctx_len) > 0.3
    visible[:, 0] = True
    dneo_mask = torch.zeros(1, 1, q_len, ctx_len).masked_fill(~visible, float("-inf"))
    dflash_mask = torch.cat(
        [dneo_mask, torch.full((1, 1, q_len, q_len), float("-inf"))], dim=-1
    )
    want, _ = dflash(draft, context, embeddings, dflash_mask)
    got, _ = dneo(draft, context, embeddings, dneo_mask)
    torch.testing.assert_close(got, want)


def test_causal_head_latent_fuses_both_terms():
    torch.manual_seed(0)
    head = DNeoCausalHead(
        verifier_vocab_size=10, draft_vocab_size=10, hidden_size=6, rank=4, block_size=3
    )
    with torch.no_grad():
        head.markov_scale.copy_(torch.tensor([1.0, 2.0, 0.5]))
        head.context_scale.copy_(torch.tensor([0.0, 1.0, 3.0]))
    prev = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 6)
    markov = head.markov_w1(prev)
    context = head.predecessor_codebook(prev) * head.hidden_projection(hidden)
    want = (
        torch.tensor([1.0, 2.0, 0.5])[:, None] * markov
        + torch.tensor([0.0, 1.0, 3.0])[:, None] * context
    )
    torch.testing.assert_close(head.latent(prev, hidden), want)
    assert head(prev, hidden).shape == (1, 3, 10)


def _transformer_config(hidden_size=16, head_dim=8) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )


def _tiny_config(hidden_size=16, head_dim=8, **overrides) -> DNeoSpeculatorConfig:
    values: dict[str, Any] = {
        "transformer_layer_config": _transformer_config(hidden_size, head_dim),
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [1, 2, 3],
        "mask_token_id": 0,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "causal_head_rank": 8,
    }
    values.update(overrides)
    return DNeoSpeculatorConfig(**values)


def test_checkpoint_keys_and_config_round_trip(tmp_path):
    config = _tiny_config(
        feature_encoder="diagonal",
        speculators_config=SpeculatorsConfig(
            algorithm="dneo",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B", architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    state = DNeoDraftModel(config).state_dict()
    # not DFlash's dense fc.weight key: a DFlash loader must not take it
    assert "fc.weight" not in state
    assert state["fc.layer_weights"].shape == (3, 16)
    assert "layers.0.attention_conv.base_kernel" in state
    assert "causal_head.markov_w2.weight" in state
    assert "causal_head.markov_scale" in state
    assert "fc.layer_logits" in DNeoDraftModel(_tiny_config()).state_dict()

    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, DNeoSpeculatorConfig)
    assert loaded.architectures == ["DNeoDraftModel"]
    assert loaded.feature_encoder == "diagonal"
    assert loaded.prefix_only_attention
    assert loaded.causal_head_rank == 8


def test_convolutions_start_as_identity():
    model = DNeoDraftModel(_tiny_config())
    for layer in model.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            assert torch.count_nonzero(conv.kernel_projection.weight) == 0
            first_tap = conv.base_kernel[:, 0]
            assert torch.equal(first_tap, torch.ones_like(first_tap))
            assert torch.count_nonzero(conv.base_kernel[:, 1:]) == 0


@pytest.mark.parametrize("feature_encoder", ["mofa", "diagonal"])
def test_muon_takes_only_linear_maps(feature_encoder):
    model = DNeoDraftModel(_tiny_config(feature_encoder=feature_encoder))
    muon, adamw = split_named_params_for_muon(model)
    muon_names = {n for n, _ in muon}
    adamw_names = {n for n, _ in adamw}
    for name in (
        "causal_head.markov_w1.weight",
        "causal_head.markov_w2.weight",
        "causal_head.predecessor_codebook.weight",
    ):
        assert name in adamw_names
    assert "causal_head.hidden_projection.weight" in muon_names
    assert not any(n.startswith("fc.") for n in muon_names)


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_previous_token_ids(sample_from_anchor):
    model = DNeoDraftModel(_tiny_config(sample_from_anchor=sample_from_anchor))
    input_ids = torch.arange(100, 116).unsqueeze(0)
    indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])  # anchors 2 and 8, block 4
    prev = model._previous_token_ids(input_ids, indices)  # noqa: SLF001
    if sample_from_anchor:
        # slot k predicts token p+k+1, after token p+k
        assert prev.tolist() == [[102, 103, 104, 105], [108, 109, 110, 111]]
    else:
        # slot k predicts token p+k, after token p+k-1 (slot 0 is the anchor)
        assert prev.tolist() == [[102, 102, 103, 104], [108, 108, 109, 110]]


def test_train_config_registers_dneo():
    flat = TrainConfig(speculator_type="dneo").flatten()
    assert flat["feature_encoder"] == "mofa"
    assert flat["prefix_only_attention"] is True
    assert flat["causal_head_rank"] == 256
    assert flat["dneo_conv_kernel_size"] == 2
    assert flat["dneo_conv_group_size"] == 16
    assert flat["num_layers"] == 5


def _inputs(model, device, dtype, seq_len=32, documents=1):
    hidden_size = model.config.transformer_layer_config.hidden_size
    positions = torch.arange(seq_len, device=device)
    document_ids = (positions * documents // seq_len).view(1, -1)
    return {
        "hidden_states": torch.randn(
            1, seq_len, 3 * hidden_size, device=device, dtype=dtype
        ),
        "input_ids": torch.randint(
            0, model.verifier_vocab_size, (1, seq_len), device=device
        ),
        "loss_mask": torch.ones(1, seq_len, device=device),
        "verifier_last_hidden_states": torch.randn(
            1, seq_len, hidden_size, device=device, dtype=dtype
        ),
        "document_ids": document_ids,
    }


def _fake_verifier_weights(model):
    """Stand-ins for the tensors ``load_verifier_weights`` copies from the verifier
    (allocated uninitialized); every other parameter must initialize finite."""
    verifier_owned = (
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in verifier_owned:
                torch.nn.init.normal_(p, std=0.02)
            else:
                assert torch.isfinite(p).all(), f"non-finite init of {name}"
    return model


def _assert_new_parameters_train(model):
    fragments = ("fc.", "attention_conv", "mlp_conv", "causal_head", "self_attn")
    new = {
        n: p
        for n, p in model.named_parameters()
        if any(f in n for f in fragments) and p.requires_grad
    }
    assert new
    for name, p in new.items():
        assert p.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    # prefix-only attention computes keys and values from the context only, so
    # every projection still trains
    assert torch.count_nonzero(model.layers[0].self_attn.k_proj.weight.grad)
    assert torch.count_nonzero(model.causal_head.context_scale.grad)


@pytest.mark.parametrize("feature_encoder", ["mofa", "diagonal"])
@pytest.mark.parametrize("prefix_only", [True, False])
def test_cpu_forward_backward(feature_encoder, prefix_only):
    torch.manual_seed(0)
    model = _fake_verifier_weights(
        DNeoDraftModel(
            _tiny_config(
                feature_encoder=feature_encoder, prefix_only_attention=prefix_only
            )
        )
    )
    # every position a candidate anchor (document starts included) and more anchor
    # slots than positions to fill (padding anchors)
    _, loss, _ = model(
        **_inputs(model, "cpu", torch.float32, documents=3),
        max_anchors=40,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    _assert_new_parameters_train(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_gpu_forward_backward_with_flex_attention():
    torch.manual_seed(0)
    # flex attention needs a head dimension of at least 16
    config = _tiny_config(hidden_size=32, head_dim=16)
    layer_config = config.transformer_layer_config
    layer_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001
    model = _fake_verifier_weights(DNeoDraftModel(config)).to(
        device="cuda", dtype=torch.bfloat16
    )
    _, loss, _ = model(
        **_inputs(model, "cuda", torch.bfloat16, documents=3),
        max_anchors=40,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    _assert_new_parameters_train(model)
