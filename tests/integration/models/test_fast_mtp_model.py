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

    # embed_tokens and lm_head are populated by _setup_embeddings_and_lm_head when
    # speculators_config.verifier.name_or_path is set. Here we attach random weights
    # to test forward-pass correctness in isolation.
    model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    model.eval()

    num_steps = config.num_speculative_steps
    batch_size, seq_len = 2, num_steps + 4  # seq_len must be > num_steps
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        output = model(input_ids=input_ids, hidden_states=hidden_states)

    assert "logits_list" in output
    assert len(output["logits_list"]) == num_steps
    for step, logits in enumerate(output["logits_list"]):
        valid_len = seq_len - step - 1
        assert logits.shape == (batch_size, valid_len, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
