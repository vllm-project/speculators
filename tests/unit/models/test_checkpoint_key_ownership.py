"""Verifier-owned checkpoint key ownership for DFlash/DSpark.

embed_tokens is always an exact frozen verifier copy, and a full-vocab
lm_head is the exact frozen verifier projection; both are reconstructed on
load from the verifier and omitted from saved checkpoints. A reduced-vocab
lm_head is a runtime-required verifier-derived head: current vLLM cannot
reconstruct an arbitrary reduced head from the full verifier head, so it
must be serialized together with the t2d/d2t vocab mappings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
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
    config_kwargs = {
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


# ── 1. save-key ownership ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_cls",
    [DFlashDraftModel, DSparkDraftModel],
    ids=["dflash", "dspark"],
)
def test_full_vocab_save_omits_verifier_owned(model_cls, tmp_path: Path):
    full = _make_model(model_cls, draft_vocab_size=VERIFIER_VOCAB)
    assert not full.use_draft_vocab
    keys = _saved_keys(full, tmp_path / "full")
    assert VERIFIER_OWNED_OMITTED - keys == VERIFIER_OWNED_OMITTED
    assert "lm_head.weight" not in keys
    assert "layers.0.self_attn.q_proj.weight" in keys

    # Class-level ignore lists are instance-copied: a reduced-vocab sibling
    # created after the full-vocab model must not inherit lm_head omission.
    reduced = _make_model(model_cls, draft_vocab_size=32)
    reduced_keys = _saved_keys(reduced, tmp_path / "reduced")
    assert "lm_head.weight" in reduced_keys


@pytest.mark.parametrize(
    "model_cls",
    [DFlashDraftModel, DSparkDraftModel],
    ids=["dflash", "dspark"],
)
def test_reduced_vocab_save_keeps_runtime_required_head(model_cls, tmp_path: Path):
    model = _make_model(model_cls, draft_vocab_size=32)
    assert model.use_draft_vocab
    keys = _saved_keys(model, tmp_path)
    assert "embed_tokens.weight" not in keys
    assert "lm_head.weight" in keys
    assert "verifier_lm_head.weight" not in keys
    assert "verifier_norm.weight" not in keys
    assert "t2d" in keys
    assert "d2t" in keys


# ── 2. slim roundtrip ────────────────────────────────────────────────────────


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
    loaded = DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True)
    assert torch.equal(loaded.embed_tokens.weight, fake["embed_tokens.weight"])
    assert torch.equal(loaded.lm_head.weight, fake["lm_head.weight"])
    assert torch.equal(loaded.verifier_lm_head.weight, fake["lm_head.weight"])
    assert torch.equal(loaded.verifier_norm.weight, fake["model.norm.weight"])


def test_reduced_vocab_slim_roundtrip_reconstructs_head_through_t2d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = _make_model(DFlashDraftModel, draft_vocab_size=32)
    model.save_pretrained(tmp_path)
    fake = _fake_verifier()
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        _make_fake_loader(fake),
    )
    # t2d/d2t reload from the checkpoint itself; do not pass them explicitly.
    loaded = DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.use_draft_vocab
    assert loaded.lm_head.weight.shape == (32, 16)
    expected = fake["lm_head.weight"][model.t2d.bool(), :]
    assert torch.equal(loaded.lm_head.weight, expected)
    assert torch.equal(loaded.t2d, model.t2d)
    assert torch.equal(loaded.d2t, model.d2t)


# ── 3. legacy self-contained checkpoint compatibility ────────────────────────


def test_self_contained_checkpoint_still_loads(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """Old self-contained checkpoints (with verifier-owned weights present) must
    still load; verifier-owned tensors now use the verifier as source of truth."""
    model = _make_model(DFlashDraftModel, draft_vocab_size=VERIFIER_VOCAB)
    model.save_pretrained(tmp_path)
    # Overwrite the safetensors payload with the full state dict to simulate a
    # checkpoint written before slim saving.
    tensors = {
        key: value
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    save_file(tensors, tmp_path / "model.safetensors")
    keys = set(load_file(tmp_path / "model.safetensors"))
    assert "embed_tokens.weight" in keys
    assert "lm_head.weight" in keys

    fake = _fake_verifier()
    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        _make_fake_loader(fake),
    )
    loaded = DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True)
    assert torch.equal(loaded.embed_tokens.weight, fake["embed_tokens.weight"])
    assert torch.equal(loaded.lm_head.weight, fake["lm_head.weight"])
