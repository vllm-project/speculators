"""Unit tests for DSpine transfer, injection, and transition scoring."""

from copy import deepcopy
from typing import Any, cast

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.dspine import DSpineDraftModel, DSpineSpeculatorConfig
from speculators.models.dspine.metrics import compute_dspine_loss
from speculators.models.dspine.model_definitions import (
    AdjacentCausalInjection,
    build_output_embedding_transfer_space,
)
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.config.schema import DSpineArgs


def _tiny_config(**overrides) -> DSpineSpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    transformer_config.is_causal = True
    values: dict[str, Any] = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": 16,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "transfer_dim": 4,
        "message_dim": 3,
        "transition_top_k": 2,
        "curriculum_layers": 2,
        "known_anchor_injection": True,
    }
    values.update(overrides)
    return DSpineSpeculatorConfig(**values)


def test_transfer_space_has_unit_transfer_embeddings_and_pca_readout():
    torch.manual_seed(0)
    embeddings = torch.randn(32, 8)

    transfer_embeddings, readout = build_output_embedding_transfer_space(embeddings, 4)

    assert transfer_embeddings.shape == (32, 4)
    assert readout.shape == (4, 8)
    torch.testing.assert_close(transfer_embeddings.norm(dim=-1), torch.ones(32))
    torch.testing.assert_close(
        readout @ readout.transpose(0, 1), torch.eye(4), atol=1e-5, rtol=1e-5
    )


def test_transfer_space_rejects_oversized_dimension():
    with pytest.raises(ValueError, match="cannot exceed hidden_size"):
        build_output_embedding_transfer_space(torch.randn(8, 4), 5)


def test_injection_starts_as_identity_and_keeps_anchor_unchanged():
    torch.manual_seed(1)
    injection = AdjacentCausalInjection(
        hidden_size=8,
        transfer_dim=4,
        message_dim=3,
        num_layers=2,
        block_size=4,
    )
    hidden = torch.randn(1, 8, 8)

    output, features = injection.inject_layer(0, hidden)

    torch.testing.assert_close(output, hidden)
    assert features.shape == (1, 8, 4)
    torch.testing.assert_close(output[:, ::4], hidden[:, ::4])


def test_injection_never_crosses_block_boundaries():
    injection = AdjacentCausalInjection(
        hidden_size=2,
        transfer_dim=2,
        message_dim=2,
        num_layers=1,
        block_size=2,
    )
    with torch.no_grad():
        cast("torch.nn.Linear", injection.readouts[0]).weight.copy_(torch.eye(2))
        injection.message.weight.copy_(torch.eye(2))
        injection.receiver_gate.weight.zero_()
        injection.predecessor_gate.weight.zero_()
        injection.gate_bias.fill_(20)
        cast("torch.nn.Linear", injection.writes[0]).weight.copy_(torch.eye(2))
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0]]])

    output, _ = injection.inject_layer(0, hidden)

    torch.testing.assert_close(output[:, 0], hidden[:, 0])
    torch.testing.assert_close(output[:, 2], hidden[:, 2])
    assert not torch.equal(output[:, 1], hidden[:, 1])
    assert not torch.equal(output[:, 3], hidden[:, 3])


def test_curriculum_replaces_predecessors_per_block():
    injection = AdjacentCausalInjection(
        hidden_size=4,
        transfer_dim=2,
        message_dim=2,
        num_layers=1,
        block_size=3,
    )
    hidden = torch.randn(1, 6, 4)
    transfer_embeddings = torch.nn.functional.normalize(torch.randn(7, 2), dim=-1)
    predecessor_ids = torch.tensor([[1, 2], [3, 4]])

    output, _ = injection.inject_layer(
        0,
        hidden,
        transfer_embeddings=transfer_embeddings,
        correct_predecessor_ids=predecessor_ids,
        curriculum_blocks=torch.tensor([True, False]),
    )

    torch.testing.assert_close(output, hidden)  # zero writes preserve warm start


@pytest.mark.parametrize("curriculum", [False, True])
def test_known_anchor_uses_token_feature_independently_of_curriculum(curriculum):
    """Eq. 4/5 use the decided anchor, not its unsupervised predicted feature."""
    torch.manual_seed(19)
    injection = AdjacentCausalInjection(
        hidden_size=4, transfer_dim=2, message_dim=3, num_layers=1, block_size=3
    )
    with torch.no_grad():
        cast("torch.nn.Linear", injection.writes[0]).weight.normal_()
        injection.predecessor_gate.weight.normal_()
    hidden = torch.randn(2, 3, 4)
    transfer_embeddings = torch.nn.functional.normalize(torch.randn(7, 2), dim=-1)
    predecessor_ids = torch.tensor([[1, 2], [3, 4]])
    features = (
        torch.nn.functional.normalize(
            hidden @ cast("torch.nn.Linear", injection.readouts[0]).weight.T, dim=-1
        )
        * 2**0.5
    )
    predecessor = features[:, :-1].clone()
    predecessor[:, 0] = transfer_embeddings[predecessor_ids[:, 0]] * 2**0.5
    if curriculum:
        predecessor = transfer_embeddings[predecessor_ids] * 2**0.5
    gate = torch.sigmoid(
        features[:, 1:] @ injection.receiver_gate.weight.T
        + predecessor @ injection.predecessor_gate.weight.T
        + injection.gate_bias
    )
    update = gate * (predecessor @ injection.message.weight.T)
    update = update @ cast("torch.nn.Linear", injection.writes[0]).weight.T
    rms = (hidden[:, 1:].square().mean(-1, keepdim=True) + injection.eps).sqrt()
    expected = torch.cat((hidden[:, :1], hidden[:, 1:] + rms * update), dim=1)

    actual, post_features = injection.inject_layer(
        0,
        hidden,
        transfer_embeddings=transfer_embeddings,
        anchor_token_ids=predecessor_ids[:, 0],
        correct_predecessor_ids=predecessor_ids,
        curriculum_blocks=torch.full((2,), curriculum),
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 0], hidden[:, 0])
    torch.testing.assert_close(post_features, injection.project(0, expected))


def test_config_round_trip_preserves_dspine_contract(tmp_path):
    config = _tiny_config(
        speculators_config=SpeculatorsConfig(
            algorithm="dspine",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B",
                architectures=["Qwen3ForCausalLM"],
            ),
        )
    )

    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DSpineSpeculatorConfig)
    assert loaded.speculators_model_type == "dspine"
    assert loaded.speculators_config.algorithm == "dspine"
    assert loaded.architectures == ["DSpineDraftModel"]
    assert loaded.transformer_layer_config.is_causal is True
    assert loaded.known_anchor_injection is True
    assert loaded.transfer_dim == 4
    assert loaded.message_dim == 3
    assert loaded.transition_top_k == 2


def test_legacy_checkpoint_keeps_predicted_anchor_but_new_training_uses_known_token():
    legacy = _tiny_config().model_dump(exclude_none=True)
    legacy.pop("known_anchor_injection")
    assert DSpineSpeculatorConfig(**legacy).known_anchor_injection is False
    assert DSpineArgs().known_anchor_injection is True


def test_checkpoint_round_trip_preserves_transfer_and_injection(tmp_path):
    torch.manual_seed(2)
    config = _tiny_config(
        speculators_config=SpeculatorsConfig(
            algorithm="dspine",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        )
    )
    model = DSpineDraftModel(config)
    with torch.no_grad():
        model.transfer_embeddings.copy_(
            torch.nn.functional.normalize(
                torch.randn_like(model.transfer_embeddings), dim=-1
            )
        )
        cast("torch.nn.Linear", model.injection.writes[0]).weight.normal_(std=0.1)
    expected_transfer_embeddings = model.transfer_embeddings.clone()
    expected_write = cast("torch.nn.Linear", model.injection.writes[0]).weight.clone()

    model.save_pretrained(tmp_path)
    loaded = cast(
        "DSpineDraftModel",
        DSpineDraftModel.from_pretrained(tmp_path, local_files_only=True),
    )

    torch.testing.assert_close(loaded.transfer_embeddings, expected_transfer_embeddings)
    torch.testing.assert_close(
        cast("torch.nn.Linear", loaded.injection.writes[0]).weight, expected_write
    )


def test_model_rejects_pruned_draft_vocabulary():
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        DSpineDraftModel(_tiny_config(draft_vocab_size=8))


def test_model_rejects_sampling_from_anchor():
    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        DSpineDraftModel(_tiny_config(sample_from_anchor=True))


def test_candidate_logits_preserve_bfloat16_autocast():
    torch.manual_seed(25)
    model = DSpineDraftModel(_tiny_config()).float()
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.1)
    hidden = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    candidates = torch.randint(0, 16, (2, 3, 4))[..., ::2]

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model._compute_candidate_logits(hidden, candidates)
        expected = model.lm_head(hidden).gather(-1, candidates)

    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, hidden, upstream, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-3, rtol=2e-2)


def test_candidate_refinement_preserves_forward_loss_and_gradients(monkeypatch):
    torch.manual_seed(26)
    model = DSpineDraftModel(_tiny_config())
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.1)
        model.lm_head.weight.normal_(std=0.1)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
        model.transfer_embeddings.copy_(
            torch.nn.functional.normalize(
                torch.randn_like(model.transfer_embeddings), dim=-1
            )
        )
        for write in model.injection.writes:
            cast("torch.nn.Linear", write).weight.normal_(std=0.1)
    reference = deepcopy(model)
    monkeypatch.setattr(
        reference,
        "_compute_candidate_logits",
        lambda hidden, candidates: reference.lm_head(hidden).gather(-1, candidates),
    )
    inputs = {
        "hidden_states": torch.randn(1, 16, 16),
        "input_ids": torch.randint(0, 16, (1, 16)),
        "loss_mask": torch.ones(1, 16),
        "verifier_last_hidden_states": torch.randn(1, 16, 8),
        "document_ids": torch.zeros(1, 16, dtype=torch.long),
        "max_anchors": 2,
        "global_step": 2,
        "total_training_steps": 12,
    }

    # Match sampled anchors and curriculum masks in the two forward passes.
    torch.manual_seed(27)
    _, actual_loss, actual_metrics = model(**inputs)
    torch.manual_seed(27)
    _, expected_loss, expected_metrics = reference(**inputs)

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_metrics, expected_metrics)
    actual_loss.backward()
    expected_loss.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)


def test_tiny_forward_and_backward_is_finite(monkeypatch):
    torch.manual_seed(4)
    model = DSpineDraftModel(_tiny_config())
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.1)
        model.lm_head.weight.normal_(std=0.1)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
        model.transfer_embeddings.copy_(
            torch.nn.functional.normalize(
                torch.randn_like(model.transfer_embeddings), dim=-1
            )
        )
    hidden_states = torch.randn(1, 8, 16)
    input_ids = torch.randint(0, 16, (1, 8))
    loss_mask = torch.ones(1, 8)
    verifier_last_hidden_states = torch.randn(1, 8, 8)
    document_ids = torch.zeros(1, 8, dtype=torch.long)
    anchored_ids = []
    model.embed_tokens.register_forward_pre_hook(
        lambda module, args: anchored_ids.append(args[0][:, :: model.block_size])
    )
    original_inject = model.injection.inject_layer
    seen_layers = []

    def checked_inject(layer_idx, hidden, **kwargs):
        torch.testing.assert_close(
            kwargs["anchor_token_ids"], anchored_ids[-1].reshape(-1)
        )
        seen_layers.append(layer_idx)
        return original_inject(layer_idx, hidden, **kwargs)

    monkeypatch.setattr(model.injection, "inject_layer", checked_inject)
    full_projection_shapes = []
    model.lm_head.register_forward_hook(
        lambda module, args, output: full_projection_shapes.append(output.shape)
    )

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hidden_states,
        document_ids=document_ids,
        max_anchors=1,
        global_step=2,
        total_training_steps=12,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["eal_sum"])
    assert cast("torch.nn.Linear", model.injection.writes[0]).weight.grad is not None
    assert seen_layers == list(range(len(model.layers)))
    # Only unary candidate selection needs a full-vocabulary projection.
    assert full_projection_shapes == [torch.Size([1, 4, 16])]


def test_dspine_loss_reaches_unary_refinement_and_layer_features():
    torch.manual_seed(5)
    unary = torch.randn(1, 4, 7, requires_grad=True)
    targets = torch.randn(1, 4, 7)
    candidate_ids = unary.detach().topk(3, dim=-1).indices
    refined = torch.randn(1, 4, 3, requires_grad=True)
    layer_features = [torch.randn(1, 4, 2, requires_grad=True) for _ in range(2)]
    target_features = torch.randn(1, 4, 2)
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

    loss, metrics = compute_dspine_loss(
        unary_logits=unary,
        refined_candidate_logits=refined,
        candidate_ids=candidate_ids,
        targets=targets,
        target_ids=targets.argmax(dim=-1),
        layer_features=layer_features,
        target_features=target_features,
        loss_mask=loss_mask,
        block_size=4,
        gamma=7.0,
        alpha=0.1,
        embedding_beta=0.5,
        embedding_warmup_steps=10,
        global_step=10,
        transition_min_target_mass=1e-4,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert unary.grad is not None
    assert refined.grad is not None
    assert all(feature.grad is not None for feature in layer_features)
    assert metrics["embedding_lambda_sum"] > 0


def test_hard_sequence_labels_are_not_replaced_by_teacher_argmax():
    """A sampled sequence label controls CE and top-k membership, not soft targets."""
    unary = torch.tensor([[[2.0, 1.0, -1.0], [2.0, 1.0, -1.0]]])
    targets = torch.tensor([[[4.0, 0.0, -2.0], [4.0, 0.0, -2.0]]])
    target_ids = torch.tensor([[0, 2]])
    loss, metrics = compute_dspine_loss(
        unary_logits=unary,
        refined_candidate_logits=unary[..., :2],
        candidate_ids=torch.tensor([[[0, 1], [0, 1]]]),
        targets=targets,
        target_ids=target_ids,
        layer_features=[],
        target_features=torch.zeros(1, 2, 2),
        loss_mask=torch.tensor([[0.0, 1.0]]),
        block_size=2,
        gamma=7.0,
        alpha=0.1,
        embedding_beta=0.0,
        embedding_warmup_steps=0,
        global_step=0,
        transition_min_target_mass=1e-4,
    )
    torch.testing.assert_close(metrics["unary_ce_sum"], -unary[0, 1].log_softmax(-1)[2])
    torch.testing.assert_close(
        metrics["unary_l1_sum"],
        (unary[0, 1].softmax(-1) - targets[0, 1].softmax(-1)).abs().sum(),
    )
    assert metrics["refined_ce_sum"].item() == 0
    assert metrics["unary_candidate_recall_at_2_sum"].item() == 0
    assert torch.isfinite(loss)


def test_training_eal_uses_sequence_labels_and_stops_at_first_mismatch():
    """The training proxy must use sampled tokens and contiguous acceptance."""
    candidate_ids = torch.tensor([[[0, 1], [2, 0], [1, 0], [0, 1], [2, 0], [2, 0]]])
    target_ids = torch.tensor([[0, 2, 1, 0, 1, 2]])
    teacher_logits = torch.tensor([[[4.0, 0.0, 0.0]]]).expand(1, 6, 3)
    assert not torch.equal(teacher_logits.argmax(-1), target_ids)

    _, metrics = compute_dspine_loss(
        unary_logits=torch.zeros(1, 6, 3),
        refined_candidate_logits=torch.tensor([[[2.0, 0.0]]]).expand(1, 6, 2),
        candidate_ids=candidate_ids,
        targets=teacher_logits,
        target_ids=target_ids,
        layer_features=[],
        target_features=torch.zeros(1, 6, 2),
        loss_mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]),
        block_size=3,
        gamma=7.0,
        alpha=0.1,
        embedding_beta=0.0,
        embedding_warmup_steps=0,
        global_step=0,
        transition_min_target_mass=1e-4,
    )

    # First block accepts both drafts; second rejects its first draft.
    torch.testing.assert_close(metrics["eal_sum"], torch.tensor(4.0))
    torch.testing.assert_close(metrics["eal_total"], torch.tensor(2.0))
