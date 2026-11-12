import math
from typing import Any, ClassVar

import torch
from transformers import PretrainedConfig

from speculators.dspine_settings import DSpineSettings
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspine.config import DSpineSpeculatorConfig
from speculators.models.dspine.metrics import compute_dspine_loss
from speculators.models.dspine.model_definitions import (
    AdjacentCausalInjection,
    build_output_embedding_transfer_space,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DSpineDraftModel",
]


@SpeculatorModel.register("dspine")
class DSpineDraftModel(DFlashDraftModel):
    """DFlash backbone with causal conditioning propagated through depth."""

    config_class: ClassVar[type[DSpineSpeculatorConfig]] = DSpineSpeculatorConfig  # type: ignore[misc,assignment]
    injection: AdjacentCausalInjection
    transfer_embeddings: torch.Tensor
    requires_global_step: ClassVar[bool] = True
    disable_ddp_buffer_broadcast: ClassVar[bool] = True

    def __init__(self, config: DSpineSpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        hidden_size = config.transformer_layer_config.hidden_size
        assert config.draft_vocab_size == target_vocab_size, (  # noqa: S101
            "DSpine requires the full verifier vocabulary."
        )
        assert config.curriculum_end_ratio >= config.curriculum_hold_ratio, (  # noqa: S101
            "curriculum_end_ratio must be >= curriculum_hold_ratio."
        )
        assert not config.sample_from_anchor, (  # noqa: S101
            "DSpine requires sample_from_anchor=False."
        )
        super().__init__(config=config)

        self.injection = AdjacentCausalInjection(
            hidden_size=hidden_size,
            transfer_dim=config.transfer_dim,
            message_dim=config.message_dim,
            num_layers=config.transformer_layer_config.num_hidden_layers,
            block_size=config.block_size,
            eps=config.transfer_space_eps,
        )
        # Register C(V * r)
        self.register_buffer(
            "transfer_embeddings",
            torch.full(
                (target_vocab_size, config.transfer_dim),
                torch.nan,
                dtype=self.lm_head.weight.dtype,
            ),
        )

    def load_verifier_weights(self) -> None:
        """Load verifier weights and derive the unified transfer space C."""
        super().load_verifier_weights()
        if self.transfer_embeddings.isnan().any():
            # Initialize the transfer embeddings and per layer readouts.
            transfer_embeddings, readout = build_output_embedding_transfer_space(
                self.lm_head.weight,
                self.config.transfer_dim,
                eps=self.config.transfer_space_eps,
            )
            self.transfer_embeddings.copy_(transfer_embeddings)
            self.injection.initialize_readouts(readout)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,  # noqa: ARG003
        d2t: torch.Tensor | None = None,  # noqa: ARG003
        **kwargs,
    ) -> "DSpineDraftModel":
        """Create DSpine from the shared DFlash-family training arguments."""
        config = DSpineSpeculatorConfig(
            **cls._build_base_config_kwargs("dspine", verifier_config, **kwargs),
            **{
                name: kwargs[name]
                for name in DSpineSettings.model_fields
                if name in kwargs
            },
        )
        model = cls(config=config)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Return paper-specific loss and curriculum settings."""
        shared = {
            "gamma": kwargs.get("dflash_decay_gamma", 7.0),
            "max_anchors": kwargs.get("max_anchors", 512),
        }
        return dict(shared), dict(shared)

    def _curriculum_probability(
        self, global_step: int, total_training_steps: int
    ) -> float:
        """Compute the curriculum probability from training progress."""
        if not self.training or total_training_steps <= 0:
            return 0.0

        config = self.config
        progress = global_step / total_training_steps
        span = config.curriculum_end_ratio - config.curriculum_hold_ratio
        if span == 0:
            return config.curriculum_probability * (
                progress <= config.curriculum_hold_ratio
            )

        remaining = (config.curriculum_end_ratio - progress) / span
        return config.curriculum_probability * min(1.0, max(0.0, remaining))

    def _forward_backbone_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        layer_context: dict[str, Any] | None,
        **kwargs,
    ) -> torch.Tensor:
        assert layer_context is not None  # noqa: S101
        if layer_idx == 0:
            if torch.is_autocast_enabled(hidden_states.device.type):
                hidden_states = hidden_states.to(
                    torch.get_autocast_dtype(hidden_states.device.type)
                )
            block_tokens = input_ids[0, anchored_block_indices].view(
                -1, self.block_size
            )
            layer_context["block_tokens"] = block_tokens
            probability = layer_context["curriculum_probability"]
            layer_context["curriculum_blocks"] = (
                torch.rand(block_tokens.shape[0], device=hidden_states.device)
                < probability
                if probability > 0
                else None
            )

        pre_injection = super()._forward_backbone_layer(
            layer_idx,
            hidden_states,
            input_ids,
            anchored_block_indices,
            layer_context,
            **kwargs,
        )
        # Keep the final raw decoder states for last-write refinement.
        layer_context["final_pre_injection"] = pre_injection
        block_tokens = layer_context["block_tokens"]
        hidden_states, features = self.injection.inject_layer(
            layer_idx,
            pre_injection,
            transfer_embeddings=self.transfer_embeddings,
            block_tokens=block_tokens,
            curriculum_blocks=(
                layer_context["curriculum_blocks"]
                if layer_idx < self.config.curriculum_layers
                else None
            ),
        )
        layer_context["layer_features"].append(features)
        return hidden_states

    def _compute_candidate_logits(
        self, hidden_states: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score selected vocabulary rows without materializing full logits."""
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            candidate_weights = self.lm_head.weight[candidate_ids].float()
            return torch.einsum(
                "...h,...kh->...k", hidden_states.float(), candidate_weights
            )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        gamma: float = 7.0,
        max_anchors: int = 512,
        global_step: int = 0,
        total_training_steps: int = 1,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        layer_context: dict[str, Any] = {
            "curriculum_probability": self._curriculum_probability(
                global_step, total_training_steps
            ),
            "layer_features": [],
        }
        _, unary_logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            layer_context=layer_context,
            **kwargs,
        )
        block_tokens = layer_context["block_tokens"]

        candidate_ids = unary_logits.topk(self.config.transition_top_k, dim=-1).indices

        correct_predecessor_features = self.transfer_embeddings[
            block_tokens[:, :-1]
        ].float() * math.sqrt(self.config.transfer_dim)
        refined_successors = self.injection.last_write(
            layer_context["final_pre_injection"], correct_predecessor_features
        )
        with torch.autocast(device_type=refined_successors.device.type, enabled=False):
            refined_hidden = (
                refined_successors
                / (
                    refined_successors.square().mean(-1, keepdim=True)
                    + self.norm.variance_epsilon
                ).sqrt()
                * self.norm.weight.float()
            )

        candidate_blocks = candidate_ids.view(
            max_anchors, self.block_size, self.config.transition_top_k
        )
        refined_logits = self._compute_candidate_logits(
            refined_hidden, candidate_blocks[:, 1:]
        )
        refined_candidate_logits = unary_logits.new_zeros(
            candidate_ids.shape, dtype=torch.float32
        )
        refined_candidate_logits[:, :: self.block_size] = unary_logits[
            :, :: self.block_size
        ].gather(-1, candidate_ids[:, :: self.block_size])
        refined_candidate_logits.view(
            max_anchors, self.block_size, self.config.transition_top_k
        )[:, 1:] = refined_logits

        target_ids = block_tokens.reshape(1, -1)
        target_features = self.injection.token_features(
            self.transfer_embeddings, target_ids
        )
        loss, metrics = compute_dspine_loss(
            unary_logits=unary_logits,
            refined_candidate_logits=refined_candidate_logits,
            candidate_ids=candidate_ids,
            targets=targets,
            target_ids=target_ids,
            layer_features=layer_context["layer_features"],
            target_features=target_features,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            gamma=gamma,
            alpha=self.config.distribution_loss_alpha,
            embedding_beta=self.config.embedding_loss_beta,
            embedding_warmup_steps=self.config.embedding_loss_warmup_steps,
            global_step=global_step,
            transition_min_target_mass=self.config.transition_min_target_mass,
        )
        return None, loss, metrics
