"""Verifier final-norm class resolution.

Some verifier families store the final RMSNorm weight in the Gemma
convention — the applied gain is ``1 + w`` — while a plain ``Qwen3RMSNorm``
applies gain ``w``. This family includes Gemma itself and the Qwen3.5/Qwen3.8
models, whose ``Qwen3_5RMSNorm`` is an alias of vLLM's ``GemmaRMSNorm``.
The frozen ``verifier_norm`` must therefore be constructed from the class
matching the verifier's convention, or the reconstructed verifier targets
are silently mis-scaled.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.utils import (
    resolve_verifier_norm_class,
    uses_gemma_style_final_norm,
)

from .test_checkpoint_key_ownership import (
    _fake_verifier,
    _make_fake_loader,
    _make_model,
)


def _point_at_fake_verifier(
    model, tmp_path, model_type: str, text_model_type: str = ""
):
    """Point the model's verifier at a fake checkpoint dir with this model_type."""
    verifier_dir = tmp_path / model_type / text_model_type
    verifier_dir.mkdir(parents=True)
    raw = {"model_type": model_type}
    if text_model_type:  # multimodal wrapper carrying the family in text_config
        raw["text_config"] = {"model_type": text_model_type}
    (verifier_dir / "config.json").write_text(json.dumps(raw))
    model.config.speculators_config.verifier.name_or_path = str(verifier_dir)


@pytest.mark.parametrize(
    ("model_type", "text_model_type", "expected"),
    [
        ("qwen3_5", "", True),
        ("qwen3_5", "qwen3_5_text", True),
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
    """A verifier config AutoConfig cannot resolve keeps the plain convention."""

    def _raise(*args, **kwargs):
        raise OSError("cannot resolve")

    monkeypatch.setattr(
        "speculators.models.utils.AutoConfig.from_pretrained",
        _raise,
    )
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    assert uses_gemma_style_final_norm(model.config) is False
    assert isinstance(model.verifier_norm, Qwen3RMSNorm)


def test_plain_construction_unchanged():
    """A plain-convention verifier (the default dummy) keeps Qwen3RMSNorm."""
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    assert isinstance(model.verifier_norm, Qwen3RMSNorm)


@pytest.mark.parametrize(
    "model_type", ["qwen3_5", "gemma3_text"], ids=["qwen3_5", "gemma3"]
)
def test_gemma_style_construction_swaps_class(
    tmp_path, monkeypatch: pytest.MonkeyPatch, model_type: str
):
    """Gemma-convention verifiers construct verifier_norm as Gemma3RMSNorm
    and load the raw checkpoint weight unchanged (the +1 lives in forward)."""
    model = _make_model(DFlashDraftModel, draft_vocab_size=64)
    _point_at_fake_verifier(model, tmp_path, model_type)
    rebuilt = DFlashDraftModel(model.config)
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
