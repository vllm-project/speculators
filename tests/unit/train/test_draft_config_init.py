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
- Cross-type warm start: ``--from-pretrained`` with a different
  ``--speculator-type`` transfers backbone weights.
"""

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts.train import (
    DECODER_SHAPING_FLAGS,
    _build_from_config_only,
    _warm_start_from_checkpoint,
    build_draft_model,
    create_transformer_layer_config,
    load_draft_transformer_layer_config,
    parse_args,
)
from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import is_config_only_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_LLAMA_KWARGS: dict[str, Any] = {
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
            **{"_attn_implementation": "eager", **TINY_LLAMA_KWARGS}
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
        ["--full-attention-indices", "0", "1"],
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
# --full-attention-indices: CLI parsing and layer-type synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cli_args", "expected"),
    [
        ([], []),
        (["--full-attention-indices", "0", "2"], [0, 2]),
    ],
)
def test_full_attention_indices_parsing(monkeypatch, cli_args, expected):
    """Defaults to empty (all layers sliding window) and parses explicit values."""
    assert _parse(monkeypatch, cli_args).full_attention_indices == expected


@pytest.mark.parametrize(
    (
        "num_layers",
        "full_attention_indices",
        "expected_layer_types",
        "expected_use_sliding_window",
    ),
    [
        (3, [], ["sliding_attention"] * 3, True),
        (
            3,
            [1],
            ["sliding_attention", "full_attention", "sliding_attention"],
            True,
        ),
        (2, [0, 1], ["full_attention", "full_attention"], False),
    ],
)
def test_create_layer_config_layer_types(
    num_layers,
    full_attention_indices,
    expected_layer_types,
    expected_use_sliding_window,
):
    """full_attention_indices selects per-layer attention; sliding window stays
    enabled unless every layer opts into full attention."""
    verifier = _make_verifier_namespace()
    with patch("scripts.train.AutoConfig.from_pretrained", return_value=verifier):
        config = create_transformer_layer_config(
            "target",
            num_layers=num_layers,
            draft_arch="llama",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=full_attention_indices,
        )
    assert config.layer_types == expected_layer_types
    assert config.use_sliding_window is expected_use_sliding_window


@pytest.mark.parametrize("bad_indices", [[-1], [3], [0, 3]])
def test_create_layer_config_rejects_out_of_range_indices(bad_indices):
    """full_attention_indices outside [0, num_layers) is a hard error."""
    verifier = _make_verifier_namespace()
    with (
        patch("scripts.train.AutoConfig.from_pretrained", return_value=verifier),
        pytest.raises(ValueError, match="valid draft layer ids"),
    ):
        create_transformer_layer_config(
            "target",
            num_layers=3,
            draft_arch="llama",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=bad_indices,
        )


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
        ["--full-attention-indices", "0", "1"],
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
# CLI validation: MTP-from-scratch rejects inapplicable draft-definition flags
# ---------------------------------------------------------------------------


def test_mtp_from_scratch_alone_parses(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "mtp"])
    assert args.speculator_type == "mtp"
    assert args.draft_config == ""


@pytest.mark.parametrize(
    "extra",
    [
        ["--draft-config", "c"],
        ["--num-layers", "5"],
        ["--draft-arch", "qwen3"],
        ["--sliding-window", "1024"],
    ],
)
def test_mtp_from_scratch_rejects_inapplicable_flags(monkeypatch, extra):
    """MTP-from-scratch reuses the verifier decoder config, so --draft-config and
    decoder-shaping flags do not apply and must error instead of being ignored."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--speculator-type", "mtp", *extra])


def test_mtp_with_from_pretrained_parses(monkeypatch):
    """MTP + --from-pretrained (a converted checkpoint) parses; the MTP-from-scratch
    rejection only applies when not loading from a checkpoint."""
    args = _parse(
        monkeypatch, ["--speculator-type", "mtp", "--from-pretrained", "ckpt"]
    )
    assert args.from_pretrained == "ckpt"


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


def test_load_draft_config_missing_model_type_raises(tmp_path, monkeypatch):
    """A --draft-config without a model_type fails loudly rather than silently
    defaulting to a particular decoder class."""
    cfg = {"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    _patch_verifier(monkeypatch, hidden_size=64, vocab_size=100)

    with pytest.raises(ValueError, match="model_type"):
        load_draft_transformer_layer_config(str(tmp_path), "dummy-verifier")


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


def test_config_roundtrip_drops_attn_implementation(tmp_path):
    # Precondition of the --from-pretrained bug: HF configs never serialize
    # _attn_implementation, so the field does not survive a save/load round-trip
    # and must be re-applied from the CLI selection.
    config = _make_eagle3_config()
    config.transformer_layer_config._attn_implementation = "sdpa"
    save_dir = tmp_path / "roundtrip"
    config.save_pretrained(str(save_dir))

    reloaded = Eagle3SpeculatorConfig.from_pretrained(str(save_dir))

    assert reloaded.transformer_layer_config._attn_implementation != "sdpa"


def test_build_from_config_only_reapplies_draft_attn_impl(tmp_path):
    model_dir = _save_config_only_dir(tmp_path)

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        built = _build_from_config_only(
            Eagle3DraftModel,
            str(model_dir),
            None,
            None,
            draft_attn_impl="sdpa",
        )

    assert built.config.transformer_layer_config._attn_implementation == "sdpa"


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
        draft_mrope_full_head_hack=True,
    )

    built = build_draft_model(args, _FakeMTP, None, None, None)  # type: ignore[arg-type]

    assert built == "MTP_MODEL"
    assert captured["verifier_config"] is verifier_cfg
    assert captured["num_speculative_steps"] == 3
    # mask token stays unset for MTP (resolve_mask_token_id was never called)
    assert args.mask_token_id is None


# ---------------------------------------------------------------------------
# build_draft_model: sliding-window default routing by speculator type
# ---------------------------------------------------------------------------


def _capture_full_attention_indices(
    monkeypatch, speculator_type: str, requested_indices: list[int]
):
    """Run build_draft_model for a synthesized draft (no --from-pretrained /
    --draft-config) and return the full_attention_indices it forwards to
    create_transformer_layer_config."""
    captured = {}

    def _fake_create(*, full_attention_indices, **_kwargs):
        captured["full_attention_indices"] = full_attention_indices
        return SimpleNamespace(vocab_size=128)

    monkeypatch.setattr("scripts.train.create_transformer_layer_config", _fake_create)
    monkeypatch.setattr("scripts.train.resolve_mask_token_id", lambda *_a, **_k: 0)

    class _FakeModel:
        @classmethod
        def from_training_args(cls, **_kwargs):
            return "MODEL"

    args = SimpleNamespace(
        speculator_type=speculator_type,
        from_pretrained="",
        draft_config="",
        verifier_name_or_path="some-verifier",
        num_layers=3,
        draft_arch="qwen3",
        draft_hidden_act=None,
        sliding_window=2048,
        full_attention_indices=requested_indices,
        mask_token_id=None,
        trust_remote_code=False,
        draft_mrope_full_head_hack=True,
    )
    build_draft_model(args, _FakeModel, None, None, 128)  # type: ignore[arg-type]
    return captured["full_attention_indices"]


@pytest.mark.parametrize(
    ("speculator_type", "requested_indices", "expected_indices"),
    [
        ("dflash", [], []),
        ("dspark", [], []),
        ("dflash", [1], [1]),
        ("eagle3", [], []),
        ("peagle", [], []),
        ("eagle3", [0, 2], [0, 2]),
    ],
)
def test_build_draft_model_routing(
    monkeypatch, speculator_type, requested_indices, expected_indices
):
    """All speculator types (except mtp) default every layer to sliding window
    (empty opt-out list) and forward an explicit non-empty list unchanged."""
    assert (
        _capture_full_attention_indices(monkeypatch, speculator_type, requested_indices)
        == expected_indices
    )


# ---------------------------------------------------------------------------
# intermediate_size resolution (dense + MoE verifiers)
# ---------------------------------------------------------------------------


def _create_layer_config_for(verifier: SimpleNamespace):
    with patch("scripts.train.AutoConfig.from_pretrained", return_value=verifier):
        return create_transformer_layer_config(
            "target",
            num_layers=2,
            draft_arch="llama",
            hidden_act=None,
            sliding_window=2048,
            full_attention_indices=[],
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


# ---------------------------------------------------------------------------
# _warm_start_from_checkpoint (cross-type warm start)
# ---------------------------------------------------------------------------

TINY_QWEN3_CONFIG = Qwen3Config(
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
)


def _make_dflash_model() -> DFlashDraftModel:
    """Create a tiny DFlash model with real initialized weights."""
    transformer_config = copy.deepcopy(TINY_QWEN3_CONFIG)
    transformer_config._attn_implementation = "eager"
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
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
    model = DFlashDraftModel(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buf in model.buffers():
            if buf.is_floating_point() and buf.isnan().any():
                buf.zero_()
    return model


def _save_dflash_checkpoint(tmp_path: Path) -> Path:
    """Save a DFlash checkpoint with known trainable weights and return the
    directory path."""
    model = _make_dflash_model()
    with torch.no_grad():
        model.fc.weight.fill_(42.0)
        for layer in model.layers:
            for p in layer.parameters():
                p.fill_(7.0)
    model_dir = tmp_path / "dflash_ckpt"
    model.save_pretrained(str(model_dir))
    return model_dir


def _make_warm_start_args(checkpoint_dir: str) -> Namespace:
    """Minimal args namespace for _warm_start_from_checkpoint."""
    return Namespace(
        from_pretrained=checkpoint_dir,
        speculator_type="dspark",
        verifier_name_or_path="dummy-verifier",
        target_layer_ids=[0, 1],
        block_size=4,
        mask_token_id=0,
        sample_from_anchor=None,
        sliding_window_non_causal=False,
        draft_attn_impl="eager",
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )


def _patch_verifier_config():
    """Patch VerifierConfig.from_pretrained to avoid network calls."""
    return patch.object(
        VerifierConfig,
        "from_pretrained",
        return_value=VerifierConfig(
            name_or_path="dummy-verifier",
            architectures=["Qwen3ForCausalLM"],
        ),
    )


def test_warm_start_produces_dspark_model(tmp_path):
    """Cross-type warm start from DFlash checkpoint creates a DSpark model."""
    ckpt_dir = _save_dflash_checkpoint(tmp_path)
    args = _make_warm_start_args(str(ckpt_dir))

    with (
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        _patch_verifier_config(),
    ):
        model = _warm_start_from_checkpoint(
            DSparkDraftModel, args, None, None, 64
        )

    assert isinstance(model, DSparkDraftModel)
    assert isinstance(model.config, DSparkSpeculatorConfig)


def test_warm_start_transfers_backbone_weights(tmp_path):
    """Backbone weights (fc, layers) are transferred from the DFlash checkpoint."""
    ckpt_dir = _save_dflash_checkpoint(tmp_path)
    args = _make_warm_start_args(str(ckpt_dir))

    with (
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        _patch_verifier_config(),
    ):
        model = _warm_start_from_checkpoint(
            DSparkDraftModel, args, None, None, 64
        )

    assert torch.allclose(model.fc.weight, torch.tensor(42.0))
    for layer in model.layers:
        for p in layer.parameters():
            assert torch.allclose(p, torch.tensor(7.0), atol=1e-4)


def test_warm_start_dspark_heads_are_new(tmp_path):
    """DSpark-only heads (MarkovHead, ConfidenceHead) are NOT 42.0 or 7.0
    — they should have their own random initialization, not DFlash weights."""
    ckpt_dir = _save_dflash_checkpoint(tmp_path)
    args = _make_warm_start_args(str(ckpt_dir))

    with (
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        _patch_verifier_config(),
    ):
        model = _warm_start_from_checkpoint(
            DSparkDraftModel, args, None, None, 64
        )

    assert model.markov_head is not None
    assert model.confidence_head is not None
    markov_w1 = model.markov_head.markov_w1.weight
    assert not torch.allclose(markov_w1, torch.tensor(42.0))
    assert not torch.allclose(markov_w1, torch.tensor(7.0))


def test_warm_start_preserves_config_fields(tmp_path):
    """Source architecture (block_size, aux layers) is inherited;
    target-specific fields (markov_rank) come from CLI args."""
    ckpt_dir = _save_dflash_checkpoint(tmp_path)
    args = _make_warm_start_args(str(ckpt_dir))
    args.markov_rank = 128

    with (
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        _patch_verifier_config(),
    ):
        model = _warm_start_from_checkpoint(
            DSparkDraftModel, args, None, None, 64
        )

    assert model.config.block_size == 4
    assert model.config.aux_hidden_state_layer_ids == [0, 1]
    assert model.config.markov_rank == 128


def test_build_draft_model_routes_cross_type(tmp_path):
    """build_draft_model detects cross-type and routes to warm start."""
    ckpt_dir = _save_dflash_checkpoint(tmp_path)
    args = _make_warm_start_args(str(ckpt_dir))

    with (
        patch.object(DSparkDraftModel, "load_verifier_weights"),
        _patch_verifier_config(),
    ):
        model = build_draft_model(args, DSparkDraftModel, None, None, 64)

    assert isinstance(model, DSparkDraftModel)
    assert torch.allclose(model.fc.weight, torch.tensor(42.0))
