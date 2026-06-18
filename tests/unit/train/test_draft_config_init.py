"""Tests for the draft-model initialization sources in ``scripts/train.py``.

Covers the three mutually exclusive init paths and their guard rails:
- ``--draft-config``: decoder ``transformer_layer_config`` loaded from a file,
  reconciled against the verifier (hidden-size match, vocab-size alignment).
- ``--from-pretrained`` pointing at a config-only directory: fresh weights
  initialized from a full saved speculator config.
- ``--dry-run`` flag parsing.
- CLI validation: ``--from-pretrained`` takes precedence over and is mutually
  exclusive with ``--draft-config`` and the decoder-shaping flags; and
  ``--draft-config`` is mutually exclusive with the decoder-shaping flags.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts.train import (
    DECODER_SHAPING_FLAGS,
    _build_from_config_only,
    build_draft_model,
    create_transformer_layer_config,
    load_draft_transformer_layer_config,
    parse_args,
)
from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import is_config_only_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_LLAMA_KWARGS = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 8,
    "max_position_embeddings": 32,
    "tie_word_embeddings": False,
}


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


def _make_eagle3_config(verifier_name_or_path: str | None = "some-verifier"):
    return Eagle3SpeculatorConfig(
        transformer_layer_config=LlamaConfig(
            _attn_implementation="eager", **TINY_LLAMA_KWARGS
        ),
        draft_vocab_size=64,
        norm_before_residual=False,
        embed_requires_grad=False,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=verifier_name_or_path,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )


def _save_config_only_dir(
    tmp_path: Path, verifier_name_or_path="some-verifier"
) -> Path:
    """Save a full speculator checkpoint then strip the weight files, leaving a
    config-only directory."""
    model = Eagle3DraftModel(_make_eagle3_config(verifier_name_or_path))
    model_dir = tmp_path / "config_only"
    model.save_pretrained(str(model_dir))
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    for weights in weight_files:
        weights.unlink()
    return model_dir


# ---------------------------------------------------------------------------
# CLI validation: --draft-config exclusivity
# ---------------------------------------------------------------------------


def test_draft_config_alone_parses(monkeypatch):
    args = _parse(monkeypatch, ["--draft-config", "some/decoder/config"])
    assert args.draft_config == "some/decoder/config"
    assert args.from_pretrained == ""


def test_dry_run_flag_parses(monkeypatch):
    assert _parse(monkeypatch, []).dry_run is False
    assert _parse(monkeypatch, ["--dry-run"]).dry_run is True


def test_draft_config_with_from_pretrained_errors(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--draft-config", "c", "--from-pretrained", "p"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-layers", "5"],
        ["--draft-arch", "qwen3"],
        ["--draft-hidden-act", "gelu"],
        ["--sliding-window", "1024"],
        ["--sliding-window-indices", "0", "1"],
    ],
)
def test_draft_config_with_decoder_flag_errors(monkeypatch, extra):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--draft-config", "c", *extra])


def test_draft_config_with_explicit_default_flag_errors(monkeypatch):
    """Explicitly passing a decoder-shaping flag conflicts with --draft-config even
    when its value equals the argparse default (--num-layers default is 1):
    detection is based on what was provided, not on the resulting value."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--draft-config", "c", "--num-layers", "1"])


def test_decoder_shaping_flags_dests_exist(monkeypatch):
    """Every dest in DECODER_SHAPING_FLAGS is a real parsed attribute."""
    args = _parse(monkeypatch, [])
    for dest in DECODER_SHAPING_FLAGS:
        assert hasattr(args, dest), dest


# ---------------------------------------------------------------------------
# CLI validation: --from-pretrained precedence
# ---------------------------------------------------------------------------


def test_from_pretrained_alone_parses(monkeypatch):
    args = _parse(monkeypatch, ["--from-pretrained", "some/checkpoint"])
    assert args.from_pretrained == "some/checkpoint"
    assert args.draft_config == ""


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-layers", "5"],
        ["--draft-arch", "qwen3"],
        ["--draft-hidden-act", "gelu"],
        ["--sliding-window", "1024"],
        ["--sliding-window-indices", "0", "1"],
        ["--draft-config", "c"],
    ],
)
def test_from_pretrained_takes_precedence_over_model_flags(monkeypatch, extra):
    """--from-pretrained defines the whole draft and takes precedence over every
    other model-definition option, erroring if combined with any of them."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--from-pretrained", "p", *extra])


def test_from_pretrained_with_explicit_default_flag_errors(monkeypatch):
    """--from-pretrained conflicts with an explicitly-passed decoder-shaping flag
    even when its value equals the argparse default (--num-layers default is 1)."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--from-pretrained", "p", "--num-layers", "1"])


# ---------------------------------------------------------------------------
# load_draft_transformer_layer_config
# ---------------------------------------------------------------------------


def _patch_verifier(monkeypatch, hidden_size: int, vocab_size: int):
    monkeypatch.setattr(
        "scripts.train.get_verifier_config",
        lambda _path: SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size),
    )


def test_load_draft_config_from_dir(tmp_path, monkeypatch):
    Qwen3Config(hidden_size=64, num_hidden_layers=3, vocab_size=100).save_pretrained(
        tmp_path
    )
    _patch_verifier(monkeypatch, hidden_size=64, vocab_size=200)

    out = load_draft_transformer_layer_config(str(tmp_path), "dummy-verifier")

    assert isinstance(out, Qwen3Config)
    assert out.num_hidden_layers == 3
    # vocab_size is aligned to the verifier's target vocabulary
    assert out.vocab_size == 200


def test_load_draft_config_hidden_size_mismatch_raises(tmp_path, monkeypatch):
    Qwen3Config(hidden_size=64, num_hidden_layers=2, vocab_size=100).save_pretrained(
        tmp_path
    )
    _patch_verifier(monkeypatch, hidden_size=128, vocab_size=100)

    with pytest.raises(ValueError, match="hidden_size"):
        load_draft_transformer_layer_config(str(tmp_path), "dummy-verifier")


def test_load_draft_config_extracts_nested_from_full_config(tmp_path, monkeypatch):
    """A full speculator config (with nested transformer_layer_config) is accepted;
    only the decoder definition is used."""
    nested = Qwen3Config(hidden_size=64, num_hidden_layers=5, vocab_size=100).to_dict()
    full = {"speculators_model_type": "dflash", "transformer_layer_config": nested}
    (tmp_path / "config.json").write_text(json.dumps(full))
    _patch_verifier(monkeypatch, hidden_size=64, vocab_size=64)

    out = load_draft_transformer_layer_config(str(tmp_path), "dummy-verifier")

    assert isinstance(out, Qwen3Config)
    assert out.num_hidden_layers == 5


# ---------------------------------------------------------------------------
# _build_from_config_only
# ---------------------------------------------------------------------------


def test_build_from_config_only(tmp_path):
    model_dir = _save_config_only_dir(tmp_path)
    assert is_config_only_dir(str(model_dir))

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        built = _build_from_config_only(Eagle3DraftModel, str(model_dir), None, None)

    assert isinstance(built, Eagle3DraftModel)
    # decoder (trainable) weights are freshly/randomly initialized, not NaN
    assert not built.fc.weight.isnan().any()


def test_build_from_config_only_fills_missing_verifier_name(tmp_path):
    model_dir = _save_config_only_dir(tmp_path, verifier_name_or_path=None)

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        built = _build_from_config_only(
            Eagle3DraftModel,
            str(model_dir),
            None,
            None,
            verifier_name_or_path="fallback-verifier",
        )

    assert built.config.speculators_config.verifier.name_or_path == "fallback-verifier"


@pytest.mark.parametrize("blanked", ["", None])
def test_build_from_config_only_fills_blank_verifier_name(tmp_path, blanked):
    # Manually blanking name_or_path in config.json yields "" (not null), so the
    # fallback must treat any empty value -- not just None -- as "use the CLI arg".
    model_dir = _save_config_only_dir(tmp_path, verifier_name_or_path=blanked)

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        built = _build_from_config_only(
            Eagle3DraftModel,
            str(model_dir),
            None,
            None,
            verifier_name_or_path="fallback-verifier",
        )

    assert built.config.speculators_config.verifier.name_or_path == "fallback-verifier"


def test_build_from_config_only_preserves_existing_verifier_name(tmp_path):
    model_dir = _save_config_only_dir(tmp_path, verifier_name_or_path="real-verifier")

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        built = _build_from_config_only(
            Eagle3DraftModel,
            str(model_dir),
            None,
            None,
            verifier_name_or_path="fallback-verifier",
        )

    assert built.config.speculators_config.verifier.name_or_path == "real-verifier"


# ---------------------------------------------------------------------------
# build_draft_model: MTP-from-scratch routing
# ---------------------------------------------------------------------------


def test_build_draft_model_mtp_from_scratch_uses_verifier_decoder(monkeypatch):
    """MTP without --from-pretrained reuses the verifier's own decoder config and
    must not synthesize a decoder or resolve a draft mask token."""
    verifier_cfg = object()
    monkeypatch.setattr("scripts.train.get_verifier_config", lambda _p: verifier_cfg)

    def _must_not_call(*_a, **_k):
        raise AssertionError("not expected for MTP-from-scratch")

    monkeypatch.setattr("scripts.train.create_transformer_layer_config", _must_not_call)
    monkeypatch.setattr("scripts.train.resolve_mask_token_id", _must_not_call)

    captured = {}

    class _FakeMTP:
        @classmethod
        def from_training_args(cls, *, verifier_config, t2d, d2t, **kwargs):
            captured["verifier_config"] = verifier_config
            captured["num_speculative_steps"] = kwargs.get("num_speculative_steps")
            return "MTP_MODEL"

    args = SimpleNamespace(
        speculator_type="mtp",
        from_pretrained="",
        draft_config="",
        verifier_name_or_path="some-verifier",
        mask_token_id=None,
        num_speculative_steps=3,
    )

    built = build_draft_model(args, _FakeMTP, None, None, None)

    assert built == "MTP_MODEL"
    assert captured["verifier_config"] is verifier_cfg
    assert captured["num_speculative_steps"] == 3
    # mask token stays unset for MTP (resolve_mask_token_id was never called)
    assert args.mask_token_id is None


# ---------------------------------------------------------------------------
# intermediate_size resolution (dense + MoE verifiers)
# ---------------------------------------------------------------------------


def _make_verifier_namespace(**overrides) -> SimpleNamespace:
    """A minimal stand-in for a verifier PretrainedConfig as consumed by
    create_transformer_layer_config (no text_config / rope fields)."""
    base = {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "head_dim": 8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _create_layer_config_for(verifier: SimpleNamespace):
    with patch("scripts.train.AutoConfig.from_pretrained", return_value=verifier):
        return create_transformer_layer_config(
            "target",
            num_layers=2,
            draft_arch="qwen3",
            hidden_act=None,
            sliding_window=None,
            sliding_window_indices=[],
        )


def test_create_layer_config_uses_dense_intermediate_size():
    verifier = _make_verifier_namespace(intermediate_size=48)

    config = _create_layer_config_for(verifier)

    assert config.intermediate_size == 48


def test_create_layer_config_infers_moe_intermediate_size():
    # MoE verifier (no dense intermediate_size): the draft MLP width falls back to
    # 3 * hidden_size. Detailed resolver behavior is covered in
    # tests/unit/models/test_utils.py.
    verifier = _make_verifier_namespace(
        moe_intermediate_size=768,  # present but irrelevant to the fallback
        num_experts_per_tok=8,
        num_experts=128,
    )

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        config = _create_layer_config_for(verifier)

    assert config.intermediate_size == 3 * 32
