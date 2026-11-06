import pytest
import torch

from speculators.models.eagle3.config import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel


def test_eagle3_gradient_checkpointing_ttt_steps():
    """Test that gradient checkpointing throws an error when ttt_steps > 1 in Eagle3."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    
    tc = Qwen3Config(hidden_size=16, vocab_size=256, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=tc,
        speculators_config={
            "algorithm": "eagle3",
            "verifier": {"name_or_path": "HuggingFaceH4/zephyr-7b-beta", "architectures": ["LlamaForCausalLM"]},
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 3}],
            "default_proposal_method": "greedy",
        }
    )
    from unittest.mock import patch
    from transformers import PretrainedConfig

    with patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = PretrainedConfig(hidden_size=16)
        model = Eagle3DraftModel(config)

    dummy_hidden = torch.randn(1, 10, 3 * model.hidden_size)
    dummy_input = torch.randint(0, 100, (1, 10))

    # Enable gradient checkpointing and set to train mode
    model.gradient_checkpointing_enable()
    model.train()

    # ttt_steps > 1 should raise ValueError
    with pytest.raises(ValueError, match="Eagle3 gradient checkpointing is incompatible with ttt_steps > 1"):
        model(
            hidden_states=dummy_hidden,
            input_ids=dummy_input,
            document_ids=dummy_input,
            ttt_steps=2,
        )

    # ttt_steps = 1 should not raise the ValueError (might fail later due to dummy inputs/mock, but not this ValueError)
    try:
        model(
            hidden_states=dummy_hidden,
            input_ids=dummy_input,
            document_ids=dummy_input,
            ttt_steps=1,
        )
    except ValueError as e:
        if "Eagle3 gradient checkpointing is incompatible" in str(e):
            pytest.fail("Should not have raised gradient checkpointing ValueError for ttt_steps=1")
    except Exception:
        # We just care that it doesn't raise our specific ValueError
        pass
