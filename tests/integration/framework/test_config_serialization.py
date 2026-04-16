"""Integration tests for config serialization round-trips in the Speculators library.

Tests that speculator configurations can survive full save -> load -> verify
cycles through multiple serialization formats (Pydantic, HF-style dicts,
and local filesystem persistence).
"""

import json
import tempfile
from pathlib import Path
from typing import Literal

import pytest

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_schemas,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig

# ===== Test-Specific Config Subclass =====


@SpeculatorModelConfig.register("serial_test_algo")
class SerialTestAlgoConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["serial_test_algo"] = "serial_test_algo"
    num_draft_layers: int = 2
    draft_hidden_size: int = 256
    use_shared_weights: bool = True


reload_schemas()


# ===== Fixtures =====


@pytest.fixture
def full_config():
    proposal = GreedyTokenProposalConfig(speculative_tokens=7)
    verifier = VerifierConfig(
        name_or_path="test/serial-verifier",
        architectures=["SerialTestModel"],
    )
    spec_config = SpeculatorsConfig(
        algorithm="serial_test_algo",
        proposal_methods=[proposal],
        default_proposal_method="greedy",
        verifier=verifier,
    )
    return SerialTestAlgoConfig(
        speculators_config=spec_config,
        num_draft_layers=4,
        draft_hidden_size=512,
        use_shared_weights=False,
    )


# ===== Pydantic Round-Trip Tests =====


@pytest.mark.smoke
def test_pydantic_model_dump_round_trip(full_config):
    """model_dump -> model_validate should preserve all fields."""
    dumped = full_config.model_dump()
    restored = SpeculatorModelConfig.model_validate(dumped)

    assert isinstance(restored, SerialTestAlgoConfig)
    assert restored.speculators_model_type == "serial_test_algo"
    assert restored.num_draft_layers == 4
    assert restored.draft_hidden_size == 512
    assert restored.use_shared_weights is False
    assert restored.speculators_config.algorithm == "serial_test_algo"
    assert restored.speculators_config.verifier.name_or_path == "test/serial-verifier"
    assert restored.speculators_config.proposal_methods[0].speculative_tokens == 7


@pytest.mark.sanity
def test_pydantic_round_trip_preserves_defaults():
    """Default field values should survive serialization round-trip."""
    proposal = GreedyTokenProposalConfig()
    verifier = VerifierConfig(
        name_or_path="test/default-verifier",
        architectures=["DefaultModel"],
    )
    spec_config = SpeculatorsConfig(
        algorithm="serial_test_algo",
        proposal_methods=[proposal],
        default_proposal_method="greedy",
        verifier=verifier,
    )
    config = SerialTestAlgoConfig(speculators_config=spec_config)

    dumped = config.model_dump()
    restored = SpeculatorModelConfig.model_validate(dumped)

    assert isinstance(restored, SerialTestAlgoConfig)
    assert restored.num_draft_layers == 2
    assert restored.draft_hidden_size == 256
    assert restored.use_shared_weights is True


# ===== HF-Style Dict Round-Trip Tests =====


@pytest.mark.smoke
def test_to_dict_from_dict_round_trip(full_config):
    """to_dict -> from_dict should preserve all fields."""
    config_dict = full_config.to_dict()
    restored = SpeculatorModelConfig.from_dict(config_dict)

    assert isinstance(restored, SerialTestAlgoConfig)
    assert restored.speculators_model_type == "serial_test_algo"
    assert restored.num_draft_layers == 4
    assert restored.draft_hidden_size == 512


@pytest.mark.sanity
def test_to_diff_dict_from_dict_round_trip(full_config):
    """to_diff_dict -> from_dict should preserve non-default fields."""
    diff_dict = full_config.to_diff_dict()
    restored = SpeculatorModelConfig.from_dict(diff_dict)

    assert isinstance(restored, SerialTestAlgoConfig)
    assert restored.speculators_model_type == "serial_test_algo"
    assert restored.num_draft_layers == 4


# ===== Filesystem Persistence Tests =====


@pytest.mark.smoke
def test_save_load_pretrained_local(full_config):
    """save_pretrained -> from_pretrained with local directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "config.json"
        full_config.save_pretrained(save_path)
        assert save_path.exists()

        restored = SpeculatorModelConfig.from_pretrained(save_path)

        assert isinstance(restored, SerialTestAlgoConfig)
        assert restored.speculators_model_type == "serial_test_algo"
        assert restored.num_draft_layers == 4
        assert restored.draft_hidden_size == 512
        assert restored.use_shared_weights is False
        assert (
            restored.speculators_config.verifier.name_or_path == "test/serial-verifier"
        )


@pytest.mark.sanity
def test_saved_json_is_valid(full_config):
    """Saved config should be valid JSON with expected structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "config.json"
        full_config.save_pretrained(save_path)

        # save_pretrained may create a directory; find the actual JSON file
        if save_path.is_dir():
            json_file = save_path / "config.json"
        else:
            json_file = save_path

        with json_file.open() as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert data["speculators_model_type"] == "serial_test_algo"
        assert "speculators_config" in data
        assert data["speculators_config"]["algorithm"] == "serial_test_algo"
        assert (
            data["speculators_config"]["verifier"]["name_or_path"]
            == "test/serial-verifier"
        )


@pytest.mark.regression
def test_save_load_preserves_nested_proposal_config(full_config):
    """Nested proposal config should survive filesystem round-trip."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "config.json"
        full_config.save_pretrained(save_path)
        restored = SpeculatorModelConfig.from_pretrained(save_path)

        original_proposal = full_config.speculators_config.proposal_methods[0]
        restored_proposal = restored.speculators_config.proposal_methods[0]

        assert isinstance(restored_proposal, GreedyTokenProposalConfig)
        assert (
            restored_proposal.speculative_tokens == original_proposal.speculative_tokens
        )


# ===== Cross-Format Consistency Tests =====


@pytest.mark.regression
def test_pydantic_and_hf_dict_produce_same_result(full_config):
    """Pydantic dump and HF to_dict reload identically."""
    pydantic_dict = full_config.model_dump()
    hf_dict = full_config.to_dict()

    restored_pydantic = SpeculatorModelConfig.model_validate(pydantic_dict)
    restored_hf = SpeculatorModelConfig.from_dict(hf_dict)

    assert (
        restored_pydantic.speculators_model_type == restored_hf.speculators_model_type
    )
    assert restored_pydantic.num_draft_layers == restored_hf.num_draft_layers
    assert restored_pydantic.draft_hidden_size == restored_hf.draft_hidden_size
    assert restored_pydantic.use_shared_weights == restored_hf.use_shared_weights
    assert (
        restored_pydantic.speculators_config.algorithm
        == restored_hf.speculators_config.algorithm
    )


@pytest.mark.regression
def test_json_string_round_trip(full_config):
    """to_json_string -> json.loads -> model_validate should work."""
    json_str = full_config.to_json_string()
    data = json.loads(json_str)
    restored = SpeculatorModelConfig.model_validate(data)

    assert isinstance(restored, SerialTestAlgoConfig)
    assert restored.num_draft_layers == 4
    assert restored.draft_hidden_size == 512
