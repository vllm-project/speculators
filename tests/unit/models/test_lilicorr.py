import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.lilicorr import LiLiCorrDraftModel, LiLiCorrSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

VOCAB = 64
HIDDEN = 16
BLOCK = 4
CALL_KWARGS = {
    "max_anchors": 6,
    "loss_config": resolve_loss_config("ce", "eager"),
}


def _tiny_model(**overrides) -> LiLiCorrDraftModel:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = LiLiCorrSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="lilicorr",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
        **{
            "lilicorr_candidate_topk": 4,
            "lilicorr_hidden_size": 8,
            "lilicorr_factor_dim": 8,
            "lilicorr_num_heads": 2,
            **overrides,
        },
    )
    model = LiLiCorrDraftModel(config)
    for param in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(param)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def _batch(seq_len: int = 32) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * HIDDEN),
        "input_ids": torch.randint(0, VOCAB, (1, seq_len)),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, HIDDEN),
        # Two packed documents, to exercise the same-document checks.
        "document_ids": torch.cat(
            [torch.zeros(1, seq_len // 2), torch.ones(1, seq_len // 2)], dim=1
        ).long(),
    }


def test_forward_loss_metrics_and_gradients():
    torch.manual_seed(0)
    model = _tiny_model()
    _, loss, metrics = model(**_batch(), **CALL_KWARGS)

    assert torch.isfinite(loss)
    lattice = 0.25 * metrics["lilicorr_ce_sum"] + 0.25 * metrics["lilicorr_penalty_sum"]
    torch.testing.assert_close(
        metrics["loss_sum"], metrics["dflash_loss_sum"] + lattice
    )
    for name in ("lilicorr_eal", "lilicorr_top1_eal", "lilicorr_oracle_eal"):
        assert f"{name}_sum" in metrics
        assert f"{name}_total" in metrics

    loss.backward()
    assert all(
        p.grad is not None
        for n, p in model.named_parameters()
        if n.startswith("lilicorr.")
    )
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_lattice_terms_alone_reach_the_backbone():
    torch.manual_seed(0)
    model = _tiny_model()
    batch = _batch()
    # Make the target agree with the draft's own top-1 so the lattice covers
    # every slot and the lattice terms are guaranteed to be active.
    model.verifier_lm_head.weight.data.copy_(model.lm_head.weight.data)
    model.verifier_norm.weight.data.copy_(model.norm.weight.data)
    _, _, metrics = model(**batch, **CALL_KWARGS)
    assert metrics["lilicorr_ce_sum"] > 0

    model.zero_grad()
    _, loss, metrics = model(**batch, **CALL_KWARGS)
    (loss - metrics["dflash_loss_sum"]).backward()
    assert model.fc.weight.grad is not None
    assert model.fc.weight.grad.abs().sum() > 0
    layer_grads = [p.grad for p in model.layers.parameters() if p.grad is not None]
    assert layer_grads
    assert any(g.abs().sum() > 0 for g in layer_grads)


def test_rejects_unservable_configs():
    with pytest.raises(ValueError, match="power of two"):
        _tiny_model(lilicorr_candidate_topk=3)
    with pytest.raises(ValueError, match="sample_from_anchor"):
        _tiny_model(sample_from_anchor=True)


def test_saved_head_keys_match_vllm_layout(tmp_path):
    model = _tiny_model()
    model.save_pretrained(tmp_path)
    keys = set(load_file(tmp_path / "model.safetensors"))
    head = {k for k in keys if k.startswith("lilicorr.")}
    expected = {
        "lilicorr.token_proj.weight",
        "lilicorr.token_proj.bias",
        "lilicorr.pass_hidden_proj.weight",
        "lilicorr.pass_hidden_proj.bias",
        "lilicorr.feature_norm.weight",
        "lilicorr.feature_norm.bias",
        "lilicorr.feature_mlp.up_proj.weight",
        "lilicorr.feature_mlp.up_proj.bias",
        "lilicorr.feature_mlp.down_proj.weight",
        "lilicorr.feature_mlp.down_proj.bias",
        "lilicorr.slot_embedding",
        "lilicorr.rank_embedding",
        "lilicorr.relative_slot_bias",
        "lilicorr.same_slot_bias",
        "lilicorr.context_proj.weight",
        "lilicorr.context_proj.bias",
        "lilicorr.output_norm.weight",
        "lilicorr.anchor_norm.weight",
        "lilicorr.factor_input_proj.weight",
        "lilicorr.factor_input_proj.bias",
        "lilicorr.out_head.weight",
        "lilicorr.out_head.bias",
        "lilicorr.in_head.weight",
        "lilicorr.in_head.bias",
        "lilicorr.anchor_out_head.weight",
        "lilicorr.anchor_out_head.bias",
    }
    for i in range(model.config.lilicorr_num_layers):
        prefix = f"lilicorr.layers.{i}."
        expected |= {
            prefix + "attn_norm.weight",
            prefix + "attn.in_proj_weight",
            prefix + "attn.in_proj_bias",
            prefix + "attn.out_proj.weight",
            prefix + "attn.out_proj.bias",
            prefix + "mlp_norm.weight",
            prefix + "mlp.up_proj.weight",
            prefix + "mlp.up_proj.bias",
            prefix + "mlp.down_proj.weight",
            prefix + "mlp.down_proj.bias",
        }
    assert head == expected
    assert "embed_tokens.weight" not in keys
    assert "lm_head.weight" not in keys

    reloaded = _tiny_model()
    result = reloaded.load_state_dict(
        load_file(tmp_path / "model.safetensors"), strict=False
    )
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if k.startswith("lilicorr.")]
    torch.testing.assert_close(
        reloaded.lilicorr.out_head.weight, model.lilicorr.out_head.weight
    )
