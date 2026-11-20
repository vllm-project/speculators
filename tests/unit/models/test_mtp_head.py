"""Unit tests for MultiLevelLMHead in Gemma4 MTP."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.core import MTPDraftModel
from speculators.models.mtp.head import MultiLevelLMHead
from speculators.proposals.greedy import GreedyTokenProposalConfig


@pytest.fixture
def head_setup():
    hidden_size = 16
    vocab_size = 256
    num_centroids = 4
    return MultiLevelLMHead(hidden_size, vocab_size, num_centroids)


def test_initialization(head_setup):
    """Test head initializes and tokens_per_centroid is computed correctly."""
    head = head_setup
    assert head.tokens_per_centroid == 256 // 4
    assert head.centroids.in_features == 16
    assert head.centroids.out_features == 4
    assert head.token_ordering.shape == (256,)
    assert torch.equal(head.token_ordering, torch.arange(256))


def test_compute_centroid_loss_basic(head_setup):
    """Test basic loss computation without ignore_index."""
    head = head_setup
    batch_size, seq_len = 2, 3
    hidden_states = torch.randn(batch_size, seq_len, 16)

    targets = torch.tensor(
        [
            [0, 64, 128],  # belongs to centroids 0, 1, 2
            [192, 10, 70],  # belongs to centroids 3, 0, 1
        ]
    )
    expected_centroids = torch.tensor([[0, 1, 2], [3, 0, 1]])

    loss = head.compute_centroid_loss(hidden_states, targets)

    assert loss.shape == (batch_size, seq_len)

    flat_hidden = hidden_states.view(-1, 16)
    expected_logits = head.centroids(flat_hidden)
    expected_loss = F.cross_entropy(
        expected_logits, expected_centroids.view(-1), reduction="none"
    ).view(batch_size, seq_len)

    assert torch.allclose(loss, expected_loss)


def test_compute_centroid_loss_with_ignore_index(head_setup):
    """Test that ignore_index is properly masked out."""
    head = head_setup
    batch_size, seq_len = 2, 3
    hidden_states = torch.randn(batch_size, seq_len, 16)

    targets = torch.tensor([[0, -100, 128], [-100, -100, 70]])
    expected_centroids = torch.tensor([[0, -100, 2], [-100, -100, 1]])

    loss = head.compute_centroid_loss(hidden_states, targets, ignore_index=-100)

    assert loss[0, 1].item() == 0.0
    assert loss[1, 0].item() == 0.0
    assert loss[1, 1].item() == 0.0

    valid_mask = targets != -100
    valid_hidden = hidden_states[valid_mask]
    valid_centroids = expected_centroids[valid_mask]

    expected_logits = head.centroids(valid_hidden)
    expected_loss = F.cross_entropy(expected_logits, valid_centroids, reduction="none")

    assert torch.allclose(loss[valid_mask], expected_loss)


def test_compute_centroid_loss_all_ignored(head_setup):
    """Test behavior when all targets are ignore_index."""
    head = head_setup
    batch_size, seq_len = 2, 3
    hidden_states = torch.randn(batch_size, seq_len, 16)

    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    loss = head.compute_centroid_loss(hidden_states, targets, ignore_index=-100)

    assert loss.shape == (batch_size, seq_len)
    assert torch.all(loss == 0.0)


def test_compute_centroid_loss_with_custom_ordering(head_setup):
    """Test that the token_ordering buffer scrambles the target mapping."""
    head = head_setup

    head.token_ordering.copy_(torch.arange(255, -1, -1))
    head._rebuild_inverse()

    hidden_states = torch.randn(1, 1, 16)
    targets = torch.tensor([[0]])

    loss = head.compute_centroid_loss(hidden_states, targets)

    expected_logits = head.centroids(hidden_states.view(-1, 16))
    expected_loss = F.cross_entropy(
        expected_logits, torch.tensor([3]), reduction="none"
    ).view(1, 1)

    assert torch.allclose(loss, expected_loss)


def test_initialization_validation():
    """Test constructor argument validation."""
    with pytest.raises(ValueError, match="num_centroids must be positive"):
        MultiLevelLMHead(16, 256, 0)

    with pytest.raises(ValueError, match="must be divisible by num_centroids"):
        MultiLevelLMHead(16, 256, 3)


def test_mtp_draft_model_integration_centroid_loss():
    """Test MTPDraftModel includes centroid loss when num_centroids is set."""
    tc = Qwen3Config(
        hidden_size=16,
        vocab_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    tc._attn_implementation = "eager"
    config = MTPSpeculatorConfig(
        transformer_layer_config=tc,
        num_centroids=4,
        centroid_intermediate_top_k=2,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                architectures=["Qwen3ForCausalLM"], name_or_path="dummy"
            ),
        ),
    )

    model = MTPDraftModel(config)
    assert model.masked_embedding is not None

    batch_sz, seq_len = 1, 4
    input_ids = torch.randint(0, 256, (batch_sz, seq_len))
    hidden_states = torch.randn(batch_sz, seq_len, 16)

    original_compute = model.masked_embedding.compute_centroid_loss
    called = False

    def mock_compute(*args, **kwargs):
        nonlocal called
        called = True
        return original_compute(*args, **kwargs)

    model.masked_embedding.compute_centroid_loss = mock_compute  # type: ignore[method-assign]
    model(input_ids, hidden_states)

    assert called, "centroid loss was not computed during forward pass"
