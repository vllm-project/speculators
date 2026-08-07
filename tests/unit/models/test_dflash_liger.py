"""Tests for opt-in, instance-local DFlash Liger kernels."""

import argparse
import builtins

import pytest
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP, Qwen3RMSNorm

from scripts.train import _resolve_dflash_kernels
from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.kernels import load_liger_dflash_kernels
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _config() -> DFlashSpeculatorConfig:
    transformer_config = Qwen3Config(  # type: ignore[call-arg]
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        _attn_implementation="eager",
    )
    return DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )


def test_disabled_resolver_does_not_load_liger(monkeypatch):
    def fail_if_called():
        raise AssertionError("disabled Liger path must not load the optional extra")

    monkeypatch.setattr("scripts.train.load_liger_dflash_kernels", fail_if_called)

    assert _resolve_dflash_kernels(argparse.Namespace(use_liger_kernel=False)) is None


def test_missing_liger_extra_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def missing_liger(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "liger_kernel.transformers":
            raise ModuleNotFoundError(
                "No module named 'liger_kernel'", name="liger_kernel"
            )
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_liger)

    with pytest.raises(ImportError, match=r"speculators\[liger\]"):
        load_liger_dflash_kernels()


def test_liger_kernels_are_instance_local_and_cover_dflash_backbone():
    pytest.importorskip("liger_kernel")
    from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP  # noqa: PLC0415

    native_before = DFlashDraftModel(_config())
    liger = DFlashDraftModel(_config(), dflash_kernels=load_liger_dflash_kernels())
    native_after = DFlashDraftModel(_config())

    assert type(native_before.norm) is Qwen3RMSNorm
    assert type(native_after.norm) is Qwen3RMSNorm
    assert type(native_before.layers[0].mlp) is Qwen3MLP
    assert type(native_after.layers[0].mlp) is Qwen3MLP

    layer = liger.layers[0]
    assert all(
        isinstance(module, LigerRMSNorm)
        for module in (
            liger.norm,
            liger.hidden_norm,
            liger.verifier_norm,
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_norm,  # type: ignore[union-attr]
            layer.self_attn.k_norm,  # type: ignore[union-attr]
        )
    )
    assert isinstance(layer.mlp, LigerSwiGLUMLP)


def test_liger_and_native_dflash_checkpoints_are_compatible(tmp_path):
    pytest.importorskip("liger_kernel")
    from liger_kernel.transformers import LigerRMSNorm  # noqa: PLC0415

    native = DFlashDraftModel(_config())
    liger = DFlashDraftModel(_config(), dflash_kernels=load_liger_dflash_kernels())

    assert native.state_dict().keys() == liger.state_dict().keys()
    result = liger.load_state_dict(native.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys

    native.save_pretrained(tmp_path)
    reloaded = DFlashDraftModel.from_pretrained(
        tmp_path,
        dflash_kernels=load_liger_dflash_kernels(),
    )
    assert isinstance(reloaded.norm, LigerRMSNorm)
