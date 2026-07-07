"""Tests for MRoPE handling in ``create_transformer_layer_config``.

These tests verify that the draft config retains MRoPE parameters so the
Eagle3 MRoPE-aware rotary embedding can be enabled during training, while
still dropping the legacy ``type`` alias that breaks transformers/vLLM
config validation.
"""

import types

import pytest

import scripts.train as train_module
from scripts.train import create_transformer_layer_config


def _make_verifier_config(**overrides) -> types.SimpleNamespace:
    """Build a minimal fake verifier config for the config builder."""
    base = dict(
        vocab_size=1000,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        head_dim=128,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.fixture
def patch_verifier(monkeypatch):
    """Patch ``AutoConfig.from_pretrained`` and the transformers version."""

    def _apply(verifier_config, transformers_version: str = "5.0.0"):
        class _FakeAutoConfig:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return verifier_config

        monkeypatch.setattr(train_module, "AutoConfig", _FakeAutoConfig)
        monkeypatch.setattr(
            train_module.transformers, "__version__", transformers_version
        )

    return _apply


def _build(**kwargs):
    defaults = dict(
        verifier_name_or_path="dummy",
        num_layers=1,
        draft_arch="llama",
        hidden_act="silu",
        sliding_window=0,
        sliding_window_indices=[],
    )
    defaults.update(kwargs)
    return create_transformer_layer_config(**defaults)


# ---------------------------------------------------------------------------
# transformers >= 5.0 (rope_parameters) path
# ---------------------------------------------------------------------------


def test_mrope_section_preserved_and_type_alias_dropped(patch_verifier):
    """``mrope_section`` is kept; the legacy ``type`` alias is removed."""
    vc = _make_verifier_config(
        rope_parameters={
            "rope_type": "default",
            "type": "mrope",
            "mrope_section": [16, 24, 24],
            "rope_theta": 1000000.0,
        }
    )
    patch_verifier(vc, "5.0.0")

    config = _build()

    assert config.rope_parameters["mrope_section"] == [16, 24, 24]
    assert config.rope_parameters["rope_theta"] == 1000000.0
    assert config.rope_parameters["rope_type"] == "default"
    # Legacy alias must be stripped so it doesn't break vLLM config checks.
    assert "type" not in config.rope_parameters


def test_full_head_hack_rescales_partial_mrope(patch_verifier):
    """partial_rotary_factor < 1 is rescaled to full-head MRoPE semantics."""
    vc = _make_verifier_config(
        rope_parameters={
            "rope_type": "default",
            # sum=32; head_dim=128 -> scale=2 -> 2*sum(new)=128
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.5,
            "rope_theta": 1000000.0,
        }
    )
    patch_verifier(vc, "5.0.0")

    config = _build(mrope_full_head_hack=True)

    assert config.rope_parameters["mrope_section"] == [16, 24, 24]
    assert config.rope_parameters["partial_rotary_factor"] == 1.0


def test_full_head_hack_disabled_keeps_partial(patch_verifier):
    """With the hack disabled, native partial MRoPE values are preserved."""
    vc = _make_verifier_config(
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.5,
            "rope_theta": 1000000.0,
        }
    )
    patch_verifier(vc, "5.0.0")

    config = _build(mrope_full_head_hack=False)

    assert config.rope_parameters["mrope_section"] == [8, 12, 12]
    assert config.rope_parameters["partial_rotary_factor"] == 0.5


def test_partial_rotary_factor_not_leaked(patch_verifier):
    """Regression for #613: partial_rotary_factor must not cause
    a train/inference rotary dimension mismatch."""
    vc = _make_verifier_config(
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
        head_dim=256,
    )
    patch_verifier(vc, "5.0.0")

    config = _build()

    partial = config.rope_parameters.get("partial_rotary_factor", 1.0)
    inference_rotary_dim = int(config.head_dim * partial)

    assert inference_rotary_dim == config.head_dim, (
        f"Train/inference rotary dim mismatch: training uses {config.head_dim}, "
        f"inference would use {inference_rotary_dim} (partial_rotary_factor={partial})"
    )


def test_full_head_hack_non_integer_inverse_raises(patch_verifier):
    """Non-integer 1/partial_rotary_factor cannot be rescaled cleanly."""
    vc = _make_verifier_config(
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.75,
            "rope_theta": 1000000.0,
        }
    )
    patch_verifier(vc, "5.0.0")

    with pytest.raises(ValueError, match="not an integer"):
        _build(mrope_full_head_hack=True)


def test_non_mrope_model_has_no_mrope_section(patch_verifier):
    """Dense models keep their rope params without an MRoPE section."""
    vc = _make_verifier_config(
        rope_theta=500000.0,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
        },
    )
    patch_verifier(vc, "5.0.0")

    config = _build()

    assert "mrope_section" not in config.rope_parameters
    assert config.rope_parameters["rope_theta"] == 500000.0
    assert config.rope_parameters["rope_type"] == "llama3"


# ---------------------------------------------------------------------------
# transformers < 5.0 (rope_scaling) path
# ---------------------------------------------------------------------------


def test_pre_transformers_5_preserves_mrope_in_rope_scaling(patch_verifier):
    """On legacy transformers, MRoPE lives in ``rope_scaling``."""
    vc = _make_verifier_config(
        rope_theta=1000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [16, 24, 24]},
    )
    patch_verifier(vc, "4.46.0")

    config = _build()

    assert config.rope_scaling["mrope_section"] == [16, 24, 24]
    assert config.rope_theta == 1000000.0
