"""Offline verifier resolution through both real draft initialization paths."""

import argparse
from typing import Any
from unittest.mock import patch

import pytest
import torch
from huggingface_hub import constants as hub_constants
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HFValidationError
from safetensors.torch import save_file
from transformers import LlamaConfig, PretrainedConfig, Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train import cli
from speculators.utils.loading import is_config_only_dir


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    return cache


@pytest.fixture
def local_verifier(tmp_path):
    directory = tmp_path / "teacher"
    directory.mkdir()
    Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
    ).save_pretrained(directory)
    weights = {
        "model.embed_tokens.weight": torch.full((64, 32), 0.25),
        "lm_head.weight": torch.full((64, 32), 0.5),
        "model.norm.weight": torch.ones(32),
    }
    save_file(weights, directory / "model.safetensors")
    return directory


def _config(algorithm, saved):
    decoder_class = Qwen3Config if algorithm == "dspark" else LlamaConfig
    decoder_kwargs: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "_attn_implementation": "eager",
    }
    decoder = decoder_class(**decoder_kwargs)
    config_class = (
        DSparkSpeculatorConfig if algorithm == "dspark" else Eagle3SpeculatorConfig
    )
    extra: dict[str, Any] = (
        {"aux_hidden_state_layer_ids": [0]}
        if algorithm == "dspark"
        else {"eagle_aux_hidden_state_layer_ids": [0]}
    )
    return config_class(
        transformer_layer_config=decoder,
        draft_vocab_size=64,
        speculators_config=SpeculatorsConfig(
            algorithm=algorithm,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=saved,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
        **extra,
    )


@pytest.mark.parametrize("algorithm", ["eagle3", "dspark"])
@pytest.mark.parametrize("config_only", [True, False])
@pytest.mark.parametrize("saved", ["Qwen/model", "", None])
def test_offline_verifier_loads_local_weights(
    tmp_path, offline_cache, local_verifier, algorithm, config_only, saved, caplog
):
    model_class = DSparkDraftModel if algorithm == "dspark" else Eagle3DraftModel
    config = _config(algorithm, saved)
    draft = tmp_path / "draft"
    if config_only:
        config.save_pretrained(draft)
    else:
        original = model_class(config)
        # Finite draft weights allow checkpoint preservation to be checked.
        with torch.no_grad():
            original.fc.weight.fill_(0.125)
            original.embed_tokens.weight.fill_(0.75)
        original.save_pretrained(draft)
    assert is_config_only_dir(draft) is config_only
    args = argparse.Namespace(
        from_pretrained=str(draft),
        verifier_name_or_path=str(local_verifier),
        draft_attn_impl="eager",
        speculator_type=algorithm,
    )
    with patch(
        "speculators.utils.loading.hf_hub_download", wraps=hf_hub_download
    ) as hub_download:
        built = cli.build_draft_model(args, model_class, None, None, 64)
    hub_download.assert_not_called()
    assert built.config.speculators_config.verifier.name_or_path == str(local_verifier)
    torch.testing.assert_close(
        built.get_parameter("verifier_lm_head.weight"), torch.full((64, 32), 0.5)
    )
    if config_only or algorithm == "dspark":
        # DSpark deliberately reconstructs frozen embeddings from its verifier.
        torch.testing.assert_close(
            built.get_parameter("embed_tokens.weight"), torch.full((64, 32), 0.25)
        )
    else:
        torch.testing.assert_close(
            built.get_parameter("embed_tokens.weight"), torch.full((64, 32), 0.75)
        )
    if not config_only:
        torch.testing.assert_close(
            built.get_parameter("fc.weight"),
            torch.full_like(built.get_parameter("fc.weight"), 0.125),
        )
    assert built.config.transformer_layer_config._attn_implementation == "eager"
    if saved:
        assert "must contain the same verifier weights" in caplog.text


@pytest.mark.parametrize(
    "saved",
    [
        "/missing/local/model",
        "/missing-model",
        "missing/local/model",
        "./missing/model",
        "../missing/model",
        "~/missing/model",
        "namespace/invalid..model",
        "namespace/model/",
        "bare-name",
    ],
)
def test_preserves_missing_local_paths_and_invalid_ids(
    offline_cache, local_verifier, saved
):
    config = _config("eagle3", saved)
    cli._resolve_config_verifier(config, str(local_verifier))
    assert config.speculators_config.verifier.name_or_path == saved


def test_preserves_existing_saved_path(offline_cache, local_verifier, tmp_path):
    saved = tmp_path / "saved"
    saved.mkdir()
    config = _config("eagle3", str(saved))
    cli._resolve_config_verifier(config, str(local_verifier))
    assert config.speculators_config.verifier.name_or_path == str(saved)


def test_preserves_cached_hub_id(offline_cache, local_verifier):
    repo = offline_cache / "models--Qwen--model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("a" * 40)
    (repo / "snapshots" / ("a" * 40)).mkdir(parents=True)
    config = _config("eagle3", "Qwen/model")
    cli._resolve_config_verifier(config, str(local_verifier))
    assert config.speculators_config.verifier.name_or_path == "Qwen/model"


def test_preserves_online_hub_id(offline_cache, local_verifier, monkeypatch):
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    config = _config("eagle3", "Qwen/model")
    cli._resolve_config_verifier(config, str(local_verifier))
    assert config.speculators_config.verifier.name_or_path == "Qwen/model"


@pytest.mark.parametrize("fallback", [None, "", "missing/teacher", "another/model"])
def test_requires_existing_local_fallback(offline_cache, fallback):
    config = _config("eagle3", "Qwen/model")
    cli._resolve_config_verifier(config, fallback)
    assert config.speculators_config.verifier.name_or_path == "Qwen/model"


def test_ignores_config_without_verifier(offline_cache, local_verifier):
    cli._resolve_config_verifier(PretrainedConfig(), str(local_verifier))


def test_preserves_mtp_verifier(offline_cache, local_verifier):
    config = _config("eagle3", "Qwen/model")
    config.speculators_model_type = "mtp"
    cli._resolve_config_verifier(config, str(local_verifier))
    assert config.speculators_config.verifier.name_or_path == "Qwen/model"


@pytest.mark.parametrize("config_only", [True, False])
def test_missing_saved_local_path_still_fails(
    tmp_path, offline_cache, local_verifier, config_only
):
    config = _config("eagle3", "/missing/local/model")
    draft = tmp_path / "draft"
    if config_only:
        config.save_pretrained(draft)
    else:
        Eagle3DraftModel(config).save_pretrained(draft)
    args = argparse.Namespace(
        from_pretrained=str(draft),
        verifier_name_or_path=str(local_verifier),
        draft_attn_impl="eager",
        speculator_type="eagle3",
    )
    with pytest.raises(HFValidationError):
        cli.build_draft_model(args, Eagle3DraftModel, None, None, 64)
