from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.eagle3.config import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel


@pytest.mark.smoke
def test_eagle3_gradient_checkpointing_ttt_steps():
    """Test gradient checkpointing throws an error when ttt_steps > 1 in Eagle3."""
    tc = Qwen3Config(
        hidden_size=16,
        vocab_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    tc._attn_implementation = "eager"
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=tc,
        speculators_config={
            "algorithm": "eagle3",
            "verifier": {
                "name_or_path": "HuggingFaceH4/zephyr-7b-beta",
                "architectures": ["LlamaForCausalLM"],
            },
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 3}],
            "default_proposal_method": "greedy",
        },
    )

    with patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = PretrainedConfig(hidden_size=16)
        model = Eagle3DraftModel(config)

    with torch.no_grad():
        model.embed_tokens.weight.normal_(0, 0.02)
        model.lm_head.weight.normal_(0, 0.02)
        model.verifier_lm_head.weight.normal_(0, 0.02)

    dummy_hidden = torch.randn(1, 10, 3 * model.hidden_size)
    dummy_verifier_hs = torch.randn(1, 10, model.hidden_size)
    dummy_input = torch.randint(0, 100, (1, 10))
    dummy_doc_ids = torch.zeros((1, 10), dtype=torch.long)

    # Enable gradient checkpointing and set to train mode
    model.gradient_checkpointing_enable()
    model.train()

    # ttt_steps > 1 should raise ValueError
    with pytest.raises(
        ValueError,
        match="Eagle3 gradient checkpointing is incompatible with ttt_steps > 1",
    ):
        model(
            hidden_states=dummy_hidden,
            input_ids=dummy_input,
            document_ids=dummy_doc_ids,
            ttt_steps=2,
        )

    # ttt_steps = 1 should succeed and propagate gradients on backward()
    with patch(
        "speculators.models.eagle3.core.compute_metrics",
        side_effect=lambda logits, targets, *_args, **_kwargs: (
            torch.nn.functional.mse_loss(logits, targets),
            {},
        ),
    ):
        _, loss, _ = model(
            hidden_states=dummy_hidden,
            input_ids=dummy_input,
            document_ids=dummy_doc_ids,
            verifier_last_hidden_states=dummy_verifier_hs,
            ttt_steps=1,
        )
    loss.backward()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()
