"""Unit tests for MTPDraftModel training compatibility."""

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPConfig, MTPDraftModel, compute_step_weights
from speculators.models.mtp.model_definitions import (
    _last_full_attention_idx,
    mtp_model_classes,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from tests.conftest import requires_transformers_version

try:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
except ImportError:
    Qwen3NextConfig = None  # type: ignore[assignment, misc]

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
except ImportError:
    Qwen3_5TextConfig = None  # type: ignore[assignment, misc]

try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
except ImportError:
    Qwen3_5MoeTextConfig = None  # type: ignore[assignment, misc]

SMALL_QWEN3_CONFIG = Qwen3Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_act="silu",
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
)

SMALL_QWEN3NEXT_CONFIG = (
    Qwen3NextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    if Qwen3NextConfig is not None
    else None
)

SMALL_QWEN35_CONFIG = (
    Qwen3_5TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    if Qwen3_5TextConfig is not None
    else None
)

SMALL_QWEN35_MOE_CONFIG = (
    Qwen3_5MoeTextConfig(
        vocab_size=1000,
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    if Qwen3_5MoeTextConfig is not None
    else None
)


def _make_mtp_config():
    return MTPConfig(
        transformer_layer_config=SMALL_QWEN3_CONFIG,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )


def _make_mtp_config_for(transformer_config):
    return MTPConfig(
        transformer_layer_config=transformer_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )


class TestMTPConfigValidation:
    def test_single_layer_enforced(self):
        with pytest.raises(ValueError, match="MTP currently supports 1 layer"):
            MTPConfig(
                transformer_layer_config=SMALL_QWEN3_CONFIG,
                num_nextn_predict_layers=2,
            )

    def test_single_layer_accepted(self):
        config = MTPConfig(
            transformer_layer_config=SMALL_QWEN3_CONFIG,
            num_nextn_predict_layers=1,
        )
        assert config.num_nextn_predict_layers == 1

    def test_serialization_round_trip(self):
        config = _make_mtp_config()
        config_dict = config.to_dict()
        restored = MTPConfig.from_dict(config_dict)
        assert restored.speculators_model_type == "mtp"
        assert restored.hidden_size == 64
        assert restored.vocab_size == 1000
        assert restored.num_nextn_predict_layers == 1

    def test_transformer_layer_config_dict_conversion(self):
        config = MTPConfig(
            transformer_layer_config={"model_type": "qwen3", "hidden_size": 128},  # type: ignore[arg-type]
        )
        assert isinstance(config.transformer_layer_config, Qwen3Config)
        assert config.transformer_layer_config.hidden_size == 128


class TestComputeStepWeights:
    def test_weights_sum_to_one(self):
        weights = compute_step_weights(beta=0.6, num_steps=3)
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_earlier_steps_weighted_higher(self):
        weights = compute_step_weights(beta=0.6, num_steps=3)
        assert weights[0] > weights[1] > weights[2]

    def test_beta_one_gives_uniform(self):
        weights = compute_step_weights(beta=1.0, num_steps=3)
        for w in weights:
            assert abs(w - 1 / 3) < 1e-6

    def test_single_step_gets_full_weight(self):
        weights = compute_step_weights(beta=0.6, num_steps=1)
        assert weights == pytest.approx([1.0])


class TestGetTrainerKwargs:
    def test_computes_step_weights_from_beta(self):
        train_kwargs, val_kwargs = MTPDraftModel.get_trainer_kwargs(
            step_weight_beta=0.6,
            num_speculative_steps=3,
        )
        assert len(train_kwargs["step_weights"]) == 3
        assert abs(sum(train_kwargs["step_weights"]) - 1.0) < 1e-6
        assert train_kwargs["step_weights"] == val_kwargs["step_weights"]

    def test_explicit_step_weights_override_beta(self):
        custom = [0.5, 0.3, 0.2]
        train_kwargs, val_kwargs = MTPDraftModel.get_trainer_kwargs(
            step_weights=custom,
            step_weight_beta=0.9,
        )
        assert train_kwargs["step_weights"] == custom
        assert val_kwargs["step_weights"] == custom


class TestFromTrainingArgs:
    """from_training_args must absorb extra kwargs from train.py without raising."""

    def test_extra_kwargs_do_not_raise(self):
        model = MTPDraftModel.from_training_args(
            SMALL_QWEN3_CONFIG,
            num_speculative_steps=3,
            verifier_name_or_path=None,
            epochs=20,
            lr=1e-4,
            t2d=None,
            d2t=None,
            draft_vocab_size=1000,
            save_path="./output",
            total_seq_len=8192,
        )
        assert isinstance(model, MTPDraftModel)
        assert model.config.num_speculative_steps == 3


class TestForwardPass:
    @pytest.fixture
    def model(self):
        config = _make_mtp_config()
        m = MTPDraftModel(config)
        m.eval()
        return m

    def test_returns_logits_loss_metrics(self, model):
        input_ids = torch.randint(0, 1000, (1, 16))
        hidden_states = torch.randn(1, 16, 64)

        with torch.no_grad():
            logits_list, loss, metrics = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
            )

        assert len(logits_list) == 3
        assert loss.ndim == 0
        assert all(f"loss_step_{i}" in metrics for i in range(3))

    def test_logit_shapes_decrease_per_step(self, model):
        seq_len = 16
        input_ids = torch.randint(0, 1000, (1, seq_len))
        hidden_states = torch.randn(1, seq_len, 64)

        with torch.no_grad():
            logits_list, _, _ = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
            )

        for step, logits in enumerate(logits_list):
            expected_len = seq_len - step - 2
            assert logits.shape == (1, expected_len, 1000)

    def test_loss_with_step_weights(self, model):
        input_ids = torch.randint(0, 1000, (1, 16))
        hidden_states = torch.randn(1, 16, 64)
        weights = compute_step_weights(beta=0.6, num_steps=3)

        with torch.no_grad():
            _, loss, _ = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                step_weights=weights,
            )

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_embed_tokens_frozen(self, model):
        assert not model.embed_tokens.weight.requires_grad

    def test_lm_head_frozen(self, model):
        assert not model.lm_head.weight.requires_grad

    def test_layers_exposes_mtp_layers_for_fsdp(self, model):
        assert model.layers is model.mtp_layers


class TestArchitectureRegistration:
    @pytest.mark.sanity
    @requires_transformers_version("4.57.0")
    def test_qwen3_next_registered(self):
        assert "qwen3_next" in mtp_model_classes

    @pytest.mark.sanity
    @requires_transformers_version("5.2.0")
    def test_qwen3_5_text_registered(self):
        assert "qwen3_5_text" in mtp_model_classes

    @pytest.mark.sanity
    @requires_transformers_version("5.2.0")
    def test_qwen3_5_moe_text_registered(self):
        assert "qwen3_5_moe_text" in mtp_model_classes


class _FakeConfig:
    """Minimal stand-in for _last_full_attention_idx tests."""

    def __init__(self, layer_types=None):
        self.layer_types = layer_types or []


class TestHybridAttentionIndex:
    def test_picks_last_full_attention(self):
        config = _FakeConfig(
            ["linear_attention", "full_attention", "linear_attention", "full_attention"]
        )
        assert _last_full_attention_idx(config) == 3  # type: ignore[arg-type]

    def test_single_full_attention(self):
        config = _FakeConfig(["linear_attention", "full_attention"])
        assert _last_full_attention_idx(config) == 1  # type: ignore[arg-type]

    def test_no_full_attention_raises(self):
        config = _FakeConfig(["linear_attention", "linear_attention"])
        with pytest.raises(ValueError, match="full_attention"):
            _last_full_attention_idx(config)  # type: ignore[arg-type]

    def test_empty_layer_types_returns_zero(self):
        config = _FakeConfig()
        assert _last_full_attention_idx(config) == 0  # type: ignore[arg-type]


@requires_transformers_version("4.57.0")
class TestQwen3NextForward:
    @pytest.fixture
    def model(self):
        m = MTPDraftModel(_make_mtp_config_for(SMALL_QWEN3NEXT_CONFIG))
        m.eval()
        return m

    @pytest.mark.sanity
    def test_returns_three_step_logits(self, model):
        with torch.no_grad():
            logits_list, loss, metrics = model(
                input_ids=torch.randint(0, 1000, (1, 16)),
                hidden_states=torch.randn(1, 16, 64),
            )
        assert len(logits_list) == 3
        assert loss.ndim == 0

    def test_logit_shapes(self, model):
        seq_len = 16
        with torch.no_grad():
            logits_list, _, _ = model(
                input_ids=torch.randint(0, 1000, (1, seq_len)),
                hidden_states=torch.randn(1, seq_len, 64),
            )
        for step, logits in enumerate(logits_list):
            assert logits.shape == (1, seq_len - step - 2, 1000)


@requires_transformers_version("5.2.0")
class TestQwen35Forward:
    @pytest.fixture
    def model(self):
        m = MTPDraftModel(_make_mtp_config_for(SMALL_QWEN35_CONFIG))
        m.eval()
        return m

    @pytest.mark.sanity
    def test_returns_three_step_logits(self, model):
        with torch.no_grad():
            logits_list, loss, metrics = model(
                input_ids=torch.randint(0, 1000, (1, 16)),
                hidden_states=torch.randn(1, 16, 64),
            )
        assert len(logits_list) == 3
        assert loss.ndim == 0

    def test_logit_shapes(self, model):
        seq_len = 16
        with torch.no_grad():
            logits_list, _, _ = model(
                input_ids=torch.randint(0, 1000, (1, seq_len)),
                hidden_states=torch.randn(1, seq_len, 64),
            )
        for step, logits in enumerate(logits_list):
            assert logits.shape == (1, seq_len - step - 2, 1000)

    def test_frozen_weights(self, model):
        assert not model.embed_tokens.weight.requires_grad
        assert not model.lm_head.weight.requires_grad


@requires_transformers_version("5.2.0")
class TestQwen35MoeForward:
    @pytest.fixture
    def model(self):
        m = MTPDraftModel(_make_mtp_config_for(SMALL_QWEN35_MOE_CONFIG))
        m.eval()
        return m

    @pytest.mark.sanity
    def test_returns_three_step_logits(self, model):
        with torch.no_grad():
            logits_list, loss, metrics = model(
                input_ids=torch.randint(0, 1000, (1, 16)),
                hidden_states=torch.randn(1, 16, 64),
            )
        assert len(logits_list) == 3
        assert loss.ndim == 0

    def test_logit_shapes(self, model):
        seq_len = 16
        with torch.no_grad():
            logits_list, _, _ = model(
                input_ids=torch.randint(0, 1000, (1, seq_len)),
                hidden_states=torch.randn(1, seq_len, 64),
            )
        for step, logits in enumerate(logits_list):
            assert logits.shape == (1, seq_len - step - 2, 1000)

    def test_frozen_weights(self, model):
        assert not model.embed_tokens.weight.requires_grad
        assert not model.lm_head.weight.requires_grad

    def test_layers_exposes_mtp_layers_for_fsdp(self, model):
        assert model.layers is model.mtp_layers
