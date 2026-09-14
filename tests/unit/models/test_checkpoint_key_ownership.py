"""Checkpoint ownership for verifier-derived weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

if TYPE_CHECKING:
    from pathlib import Path

VERIFIER_VOCAB = 64
VERIFIER_OWNED_OMITTED = {
    "embed_tokens.weight",
    "verifier_lm_head.weight",
    "verifier_norm.weight",
}


def _make_model(model_cls: type, draft_vocab_size: int) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VERIFIER_VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    speculators_config = SpeculatorsConfig(
        algorithm="dflash",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
    )
    config_kwargs: dict[str, Any] = {
        "transformer_layer_config": tl_config,
        "draft_vocab_size": draft_vocab_size,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "sample_from_anchor": True,
        "speculators_config": speculators_config,
    }
    config = (
        DSparkSpeculatorConfig(**config_kwargs)
        if model_cls is DSparkDraftModel
        else DFlashSpeculatorConfig(**config_kwargs)
    )
    model = model_cls(config)
    for param in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(param)
    torch.nn.init.ones_(model.verifier_norm.weight)
    if model.use_draft_vocab:
        t2d = torch.zeros(model.verifier_vocab_size, dtype=torch.bool)
        t2d[:draft_vocab_size] = True
        model.load_vocab_mappings(t2d, torch.arange(draft_vocab_size, dtype=torch.long))
    return model.eval()


def _fake_verifier() -> dict[str, torch.Tensor]:
    return {
        "embed_tokens.weight": torch.randn(VERIFIER_VOCAB, 16),
        "lm_head.weight": torch.randn(VERIFIER_VOCAB, 16),
        "model.norm.weight": torch.ones(16),
    }


def _make_fake_loader(fake):
    """Fake load_model_layers returning the requested verifier tensors."""

    def loader(weights_to_load, name_or_path):
        return {name: fake[name] for name in weights_to_load if name in fake}

    return loader


def _saved_keys(model: DFlashDraftModel, tmp_path: Path) -> set[str]:
    model.save_pretrained(tmp_path)
    return set(load_file(tmp_path / "model.safetensors"))


@pytest.mark.parametrize(
    "model_cls",
    [DFlashDraftModel, DSparkDraftModel],
    ids=["dflash", "dspark"],
)
def test_full_vocab_save_omits_verifier_owned(model_cls, tmp_path: Path):
    full = _make_model(model_cls, draft_vocab_size=VERIFIER_VOCAB)
    assert not full.use_draft_vocab
    keys = _saved_keys(full, tmp_path / "full")
    assert VERIFIER_OWNED_OMITTED.isdisjoint(keys)
    assert "lm_head.weight" not in keys

    # Class-level ignore lists are instance-copied: a reduced-vocab sibling
    # created after the full-vocab model must not inherit lm_head omission.
    reduced = _make_model(model_cls, draft_vocab_size=32)
    reduced_keys = _saved_keys(reduced, tmp_path / "reduced")
    assert VERIFIER_OWNED_OMITTED.isdisjoint(reduced_keys)
    assert "lm_head.weight" in reduced_keys
    assert {"t2d", "d2t"}.issubset(reduced_keys)


def test_full_vocab_slim_roundtrip_reconstructs_from_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = _make_model(DFlashDraftModel, draft_vocab_size=VERIFIER_VOCAB)
    model.save_pretrained(tmp_path)
    fake = _fake_verifier()
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        _make_fake_loader(fake),
    )
    loaded = cast(
        "DFlashDraftModel",
        DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True),
    )
    assert torch.equal(loaded.embed_tokens.weight, fake["embed_tokens.weight"])
    assert torch.equal(loaded.lm_head.weight, fake["lm_head.weight"])
    assert torch.equal(loaded.verifier_lm_head.weight, fake["lm_head.weight"])
    assert torch.equal(loaded.verifier_norm.weight, fake["model.norm.weight"])


def test_checkpoint_weights_take_precedence(monkeypatch: pytest.MonkeyPatch):
    model = DraftVocabMixin()
    model.config = cast(
        "Any",
        SimpleNamespace(
            speculators_config=SimpleNamespace(
                verifier=SimpleNamespace(name_or_path="dummy")
            )
        ),
    )
    model.embed_tokens = nn.Embedding(VERIFIER_VOCAB, 16)
    model.lm_head = nn.Linear(16, VERIFIER_VOCAB, bias=False)
    model.verifier_lm_head = nn.Linear(16, VERIFIER_VOCAB, bias=False)
    model.use_draft_vocab = False
    model.t2d = None
    model.d2t = None
    with torch.no_grad():
        model.embed_tokens.weight.fill_(0.25)
        model.lm_head.weight.fill_(0.5)
    expected_embed = model.embed_tokens.weight.detach().clone()
    expected_head = model.lm_head.weight.detach().clone()
    fake = _fake_verifier()
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        _make_fake_loader(fake),
    )

    model.load_verifier_weights()

    assert torch.equal(model.embed_tokens.weight, expected_embed)
    assert torch.equal(model.lm_head.weight, expected_head)
