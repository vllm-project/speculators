"""Verifier final-norm class resolution.

Some verifier families store the final RMSNorm weight in the Gemma
convention — the applied gain is ``1 + w`` — while a plain ``Qwen3RMSNorm``
applies gain ``w``. This family includes Gemma itself, the Qwen3.5/Qwen3.8
models, Qwen3-Next and Step3.5, whose final norms vLLM builds from
``GemmaRMSNorm``.
The frozen ``verifier_norm`` must therefore be constructed from the class
matching the verifier's convention, or the reconstructed verifier targets
are silently mis-scaled.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash.core import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.models.utils import (
    _verifier_model_type,
    resolve_verifier_norm_class,
    uses_gemma_style_final_norm,
)
from speculators.proposals import GreedyTokenProposalConfig

from .test_checkpoint_key_ownership import (
    _fake_verifier,
    _make_fake_loader,
    _make_model,
)


@pytest.fixture(autouse=True)
def _isolate_verifier_model_type_cache():
    """Keep _verifier_model_type's @cache from leaking results across tests."""
    _verifier_model_type.cache_clear()
    yield
    _verifier_model_type.cache_clear()


def _point_at_fake_verifier(
    model, tmp_path, model_type: str, text_model_type: str = ""
):
    """Point the model's verifier at a fake checkpoint dir with this model_type."""
    verifier_dir = tmp_path / model_type / text_model_type
    verifier_dir.mkdir(parents=True)
    raw: dict = {"model_type": model_type}
    if text_model_type:  # multimodal wrapper carrying the family in text_config
        raw["text_config"] = {"model_type": text_model_type}
    (verifier_dir / "config.json").write_text(json.dumps(raw))
    model.config.speculators_config.verifier.name_or_path = str(verifier_dir)


@pytest.mark.parametrize(
    ("model_type", "text_model_type", "expected"),
    [
        ("qwen3_5", "", True),
        ("qwen3_5", "qwen3_5_text", True),
        ("qwen3_5_moe", "", True),
        ("qwen3_5_moe", "qwen3_5_moe_text", True),
        ("qwen3_next", "", True),
        ("step3p7", "step3p5", True),
        ("gemma3_text", "", True),
        ("gemma2", "", True),
        ("gemma3n", "", False),  # dropped the (1 + w) convention
        ("gemma4", "gemma4_text", False),  # dropped the (1 + w) convention
        ("qwen3", "", False),
        ("llama", "", False),
    ],
    ids=[
        "qwen3_5",
        "qwen3_5_text_nested",
        "qwen3_5_moe",
        "qwen3_5_moe_text_nested",
        "qwen3_next",
        "step3p5_text_nested",
        "gemma3_text",
        "gemma2",
        "gemma3n_negative",
        "gemma4_negative",
        "qwen3_negative",
        "llama_negative",
    ],
)
def test_detection_by_verifier_model_type(
    tmp_path, model_type: str, text_model_type: str, expected: bool
):
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    _point_at_fake_verifier(model, tmp_path, model_type, text_model_type)
    assert uses_gemma_style_final_norm(model.config) is expected
    assert (resolve_verifier_norm_class(model.config) is Gemma3RMSNorm) is expected


def test_unresolvable_verifier_defaults_to_plain(monkeypatch: pytest.MonkeyPatch):
    """A verifier whose config.json cannot be read keeps the plain convention."""

    calls: list[str] = []

    def _raise(name_or_path, *args, **kwargs):
        calls.append(name_or_path)
        raise OSError("cannot resolve")

    monkeypatch.setattr(
        "speculators.models.utils.PretrainedConfig.get_config_dict",
        _raise,
    )
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    assert uses_gemma_style_final_norm(model.config) is False
    assert isinstance(model.verifier_norm, Qwen3RMSNorm)
    assert calls, "the patched config.json read was never reached"


def test_missing_model_type_is_unresolved(tmp_path, caplog: pytest.LogCaptureFixture):
    """A config.json without model_type is unresolved and warns, not a silent ''."""
    verifier_dir = tmp_path / "gemma3_no_model_type"
    verifier_dir.mkdir()
    (verifier_dir / "config.json").write_text(json.dumps({"hidden_size": 16}))

    with caplog.at_level("WARNING", logger="speculators.models.utils"):
        assert _verifier_model_type(str(verifier_dir)) is None
    assert "declares no model_type" in caplog.text


def test_plain_construction_unchanged():
    """A plain-convention verifier (the default dummy) keeps Qwen3RMSNorm."""
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    assert isinstance(model.verifier_norm, Qwen3RMSNorm)


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_5", "qwen3_5_moe", "qwen3_next", "gemma3_text"],
    ids=["qwen3_5", "qwen3_5_moe", "qwen3_next", "gemma3"],
)
def test_gemma_style_construction_swaps_class(
    tmp_path, monkeypatch: pytest.MonkeyPatch, model_type: str
):
    """Gemma-convention verifiers construct verifier_norm as Gemma3RMSNorm
    and load the raw checkpoint weight unchanged (the +1 lives in forward)."""
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    _point_at_fake_verifier(model, tmp_path, model_type)
    rebuilt = DFlashDraftModel(cast("DFlashSpeculatorConfig", model.config))
    assert isinstance(rebuilt.verifier_norm, Gemma3RMSNorm)

    # The weight must load verbatim: the convention is applied in forward,
    # not baked into the parameter.
    fake = _fake_verifier()
    fake["model.norm.weight"] = torch.randn(16)
    rebuilt.save_pretrained(tmp_path / "draft")
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        _make_fake_loader(fake),
    )
    loaded = cast(
        "DFlashDraftModel",
        DFlashDraftModel.from_pretrained(tmp_path / "draft", local_files_only=True),
    )
    assert isinstance(loaded.verifier_norm, Gemma3RMSNorm)
    assert torch.equal(loaded.verifier_norm.weight, fake["model.norm.weight"])


def test_gemma3_rmsnorm_matches_folded_qwen3_rmsnorm():
    """Reference equivalence: Gemma3RMSNorm(w) == Qwen3RMSNorm(w + 1).

    The two fix mechanisms for the convention mismatch (class swap vs
    folding +1 into the loaded weight) must produce the same function.
    """
    torch.manual_seed(0)
    w = torch.randn(16)
    x = torch.randn(8, 16)
    gemma = Gemma3RMSNorm(16, eps=1e-6)
    gemma.weight.data = w.clone()
    qwen_folded = Qwen3RMSNorm(16, eps=1e-6)
    qwen_folded.weight.data = w + 1.0
    torch.testing.assert_close(gemma(x), qwen_folded(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("qwen3_5", Gemma3RMSNorm),
        ("gemma3_text", Gemma3RMSNorm),
        ("llama", Qwen3RMSNorm),
    ],
    ids=["qwen3_5", "gemma3", "llama"],
)
def test_eagle3_resolves_verifier_norm_class(
    tmp_path,
    model_type: str,
    expected_cls: type,
):
    """Eagle3 picks the same verifier norm class as DFlash for the same verifier."""
    verifier_dir = tmp_path / model_type
    verifier_dir.mkdir()
    (verifier_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        draft_vocab_size=64,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=str(verifier_dir), architectures=[]),
        ),
    )
    model = Eagle3DraftModel(config)
    assert type(model.verifier_norm) is expected_cls


@pytest.mark.parametrize(
    ("module", "norm_class"),
    [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNorm"),
        ("transformers.models.step3p7.modeling_step3p7", "Step3p7RMSNorm"),
    ],
    ids=["qwen3_5", "qwen3_next", "step3p5"],
)
def test_gemma3_rmsnorm_matches_verifier_final_norm(module: str, norm_class: str):
    """Gemma3RMSNorm reproduces the listed verifiers' own final-norm class."""
    verifier_norm_cls = getattr(pytest.importorskip(module), norm_class)
    torch.manual_seed(0)
    w = torch.randn(16)
    x = torch.randn(8, 16)
    gemma = Gemma3RMSNorm(16, eps=1e-6)
    gemma.weight.data = w.clone()
    verifier_norm = verifier_norm_cls(16, eps=1e-6)
    verifier_norm.weight.data = w.clone()
    torch.testing.assert_close(gemma(x), verifier_norm(x), rtol=1e-5, atol=1e-5)


def test_custom_code_verifier_resolves_without_trust_remote_code(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A verifier declaring an auto_map still resolves its final-norm convention."""
    verifier_dir = tmp_path / "custom_code_verifier"
    verifier_dir.mkdir()
    (verifier_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3",
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
            }
        )
    )

    def _refuse(*args, **kwargs):
        raise ValueError(
            "contains custom code which must be executed to correctly load the model"
        )

    # Reaching for AutoConfig would make this verifier unresolvable again.
    monkeypatch.setattr("speculators.models.utils.AutoConfig.from_pretrained", _refuse)

    assert _verifier_model_type(str(verifier_dir)) == "gemma3"
