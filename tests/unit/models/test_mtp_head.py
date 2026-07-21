"""Unit tests for MultiLevelLMHead in Gemma4 MTP."""

import pytest
import torch
import torch.nn.functional as F

from speculators.models.mtp.head import MultiLevelLMHead


@pytest.fixture
def head_setup():
    hidden_size = 16
    vocab_size = 256
    num_centroids = 4
    head = MultiLevelLMHead(hidden_size, vocab_size, num_centroids)
    return head


def test_initialization(head_setup):
    """Test that the head initializes and tokens_per_centroid is computed correctly."""
    head = head_setup
    assert head.tokens_per_centroid == 256 // 4
    assert head.centroids.in_features == 16
    assert head.centroids.out_features == 4
    assert head.token_ordering.shape == (256,)
    
    # Check that sequential token mapping works
    assert torch.equal(head.token_ordering, torch.arange(256))


def test_compute_centroid_loss_basic(head_setup):
    """Test basic loss computation without ignore_index."""
    head = head_setup
    batch_size, seq_len = 2, 3
    hidden_states = torch.randn(batch_size, seq_len, 16)
    
    # Targets for the first 3 centroids
    targets = torch.tensor([
        [0, 64, 128], # belongs to centroids 0, 1, 2
        [192, 10, 70] # belongs to centroids 3, 0, 1
    ])
    expected_centroids = torch.tensor([
        [0, 1, 2],
        [3, 0, 1]
    ])
    
    loss = head.compute_centroid_loss(hidden_states, targets)
    
    # Loss should have the same shape as targets
    assert loss.shape == (batch_size, seq_len)
    
    # Manually compute the expected loss
    flat_hidden = hidden_states.view(-1, 16)
    expected_logits = head.centroids(flat_hidden)
    expected_loss = F.cross_entropy(
        expected_logits, 
        expected_centroids.view(-1), 
        reduction="none"
    ).view(batch_size, seq_len)
    
    assert torch.allclose(loss, expected_loss)


def test_compute_centroid_loss_with_ignore_index(head_setup):
    """Test that ignore_index is properly masked out."""
    head = head_setup
    batch_size, seq_len = 2, 3
    hidden_states = torch.randn(batch_size, seq_len, 16)
    
    targets = torch.tensor([
        [0, -100, 128], 
        [-100, -100, 70]
    ])
    expected_centroids = torch.tensor([
        [0, -100, 2],
        [-100, -100, 1]
    ])
    
    loss = head.compute_centroid_loss(hidden_states, targets, ignore_index=-100)
    
    # Loss should be 0.0 where ignore_index was used
    assert loss[0, 1].item() == 0.0
    assert loss[1, 0].item() == 0.0
    assert loss[1, 1].item() == 0.0
    
    # Check valid positions
    valid_mask = targets != -100
    valid_hidden = hidden_states[valid_mask]
    valid_centroids = expected_centroids[valid_mask]
    
    expected_logits = head.centroids(valid_hidden)
    expected_loss = F.cross_entropy(
        expected_logits, 
        valid_centroids, 
        reduction="none"
    )
    
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
    """Test that the token_ordering buffer correctly scrambles the target mapping."""
    head = head_setup
    
    # Reverse the token ordering
    head.token_ordering.copy_(torch.arange(255, -1, -1))
    head._rebuild_inverse()
    
    hidden_states = torch.randn(1, 1, 16)
    
    # The token '0' is now at the very end of the array (position 255)
    # Position 255 belongs to centroid 255 // 64 = 3
    targets = torch.tensor([[0]])
    
    loss = head.compute_centroid_loss(hidden_states, targets)
    
    # Manually check that target centroid is 3
    expected_logits = head.centroids(hidden_states.view(-1, 16))
    expected_loss = F.cross_entropy(
        expected_logits, 
        torch.tensor([3]), 
        reduction="none"
    ).view(1, 1)
    
    assert torch.allclose(loss, expected_loss)


def test_initialization_validation():
    """Test constructor argument validation."""
    with pytest.raises(ValueError, match="num_centroids must be positive"):
        MultiLevelLMHead(16, 256, 0)
        
    with pytest.raises(ValueError, match="must be divisible by num_centroids"):
        MultiLevelLMHead(16, 256, 3)


def test_mtp_draft_model_integration_centroid_loss():
    """Test that MTPDraftModel includes centroid loss when num_centroids is configured."""
    from speculators.models.mtp.core import MTPDraftModel
    from speculators.models.mtp.config import MTPSpeculatorConfig
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
    
    tc = Gemma2Config(
        hidden_size=16, 
        vocab_size=256, 
        num_hidden_layers=1, 
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8
    )
    config = MTPSpeculatorConfig(
        transformer_layer_config=tc,
        num_centroids=4,
        centroid_intermediate_top_k=2,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(architectures=["Gemma2ForCausalLM"], name_or_path="dummy")
        )
    )
    
    model = MTPDraftModel(config)
    assert model.masked_embedding is not None
    
    batch_sz, seq_len = 1, 3
    input_ids = torch.randint(0, 256, (batch_sz, seq_len))
    hidden_states = torch.randn(batch_sz, seq_len, 16)
    
    _, loss, metrics = model(input_ids, hidden_states)
    
    # The total loss should include the masked_embedding centroid loss.
    # We can verify the masked_embedding was called by replacing its compute_centroid_loss with a mock.
    original_compute = model.masked_embedding.compute_centroid_loss
    called = False
    
    def mock_compute(*args, **kwargs):
        nonlocal called
        called = True
        return original_compute(*args, **kwargs)
        
    model.masked_embedding.compute_centroid_loss = mock_compute
    model(input_ids, hidden_states)
    
    assert called, "centroid loss was not computed during forward pass"
