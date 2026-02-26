import pytest
import torch
from torch import nn

from speculators import SpeculatorModel
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator

_CHECKPOINTS = [
    "inference-optimization/test_tencentbac_fastmtp",
    "inference-optimization/test_qwen3_next_mtp",
]


@pytest.mark.sanity
@pytest.mark.parametrize("model_id", _CHECKPOINTS)
def test_fastmtp_checkpoint_forward_pass(model_id):
    model = SpeculatorModel.from_pretrained(model_id)
    assert isinstance(model, FastMTPSpeculator)
    config = model.config
    assert isinstance(config, FastMTPConfig)

    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        output = model(input_ids=input_ids, hidden_states=hidden_states)

    assert "logits_list" in output
    assert len(output["logits_list"]) == config.num_speculative_steps
    for logits in output["logits_list"]:
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
