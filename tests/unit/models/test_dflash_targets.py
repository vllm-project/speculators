import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RMSNorm

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _tiny_model(
    sample_from_anchor: bool,
    verifier_architectures: list[str] | None = None,
) -> DFlashDraftModel:
    tl_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    verifier_kwargs = {}
    if verifier_architectures is not None:
        verifier_kwargs["speculators_config"] = SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="dummy",
                architectures=verifier_architectures,
            ),
        )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        **verifier_kwargs,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


@pytest.mark.parametrize(
    ("verifier_architectures", "expected_norm_cls"),
    [
        (["Gemma3ForCausalLM"], Gemma3RMSNorm),
        (["Qwen2ForCausalLM"], Qwen3RMSNorm),
    ],
)
def test_verifier_norm_matches_verifier_architecture(
    verifier_architectures,
    expected_norm_cls,
):
    model = _tiny_model(False, verifier_architectures)
    assert isinstance(model.verifier_norm, expected_norm_cls)


@pytest.mark.parametrize("max_anchors", [5, 16])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_targets_match_full_sequence_roll(sample_from_anchor, max_anchors):
    torch.manual_seed(0)
    model = _tiny_model(sample_from_anchor)
    seq_len = 32
    hidden_states = torch.randn(1, seq_len, 2 * 16)
    verifier_last_hidden_states = torch.randn(1, seq_len, 16)
    input_ids = torch.randint(0, 64, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        _, _, targets, _, anchored_block_indices = model._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            max_anchors=max_anchors,
        )

        full_logits = model.verifier_lm_head(
            model.verifier_norm(verifier_last_hidden_states)
        )
        if not sample_from_anchor:
            full_logits = torch.roll(full_logits, 1, dims=1)
        expected = full_logits[:, anchored_block_indices]

    torch.testing.assert_close(targets, expected, atol=1e-5, rtol=0)
