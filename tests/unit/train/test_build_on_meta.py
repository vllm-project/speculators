"""Tests for build_on_meta (meta-device construction for --init-on-meta)."""

import copy

import torch
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.models.eagle3 import Eagle3DraftModel
from speculators.train.distributed import build_on_meta

_TINY = LlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,  # type: ignore[arg-type]
    tie_word_embeddings=False,
    _attn_implementation="eager",  # type: ignore[call-arg]
)


def test_build_on_meta_params_meta_buffers_real():
    """Parameters land on meta; buffers keep their real storage and values."""
    with build_on_meta():
        m = nn.Linear(4, 4)
        m.register_buffer("buf", torch.ones(4))
    assert m.weight.is_meta and m.bias.is_meta
    assert not m.buf.is_meta
    assert m.buf.sum().item() == 4.0


def test_build_on_meta_restores_register_parameter():
    """The register_parameter monkeypatch is undone on exit (no global leak)."""
    orig = nn.Module.register_parameter
    with build_on_meta():
        assert nn.Module.register_parameter is not orig
    assert nn.Module.register_parameter is orig
    assert not nn.Linear(2, 2).weight.is_meta  # params are real again after exit


def test_from_training_args_under_build_on_meta():
    """A real speculator builds under build_on_meta: all params land on meta and
    buffers stay real. Also exercises the base-class ``is_meta`` guard: with meta
    params, ``load_verifier_weights`` must early-return -- it neither touches the
    (nonexistent) verifier path nor calls ``isnan()`` on a meta tensor."""
    with build_on_meta():
        model = Eagle3DraftModel.from_training_args(
            verifier_config=copy.deepcopy(_TINY),
            t2d=None,
            d2t=None,
            draft_vocab_size=64,
            norm_before_residual=False,
            ttt_steps=1,
            # never fetched: the meta guard returns before the verifier load
            verifier_name_or_path="does-not-exist/verifier",
        )
    assert all(p.is_meta for p in model.parameters())
    assert all(not b.is_meta for b in model.buffers())
