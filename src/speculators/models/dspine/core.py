from typing import Any, ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
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
    requires_global_step: ClassVar[bool] = True
    disable_ddp_buffer_broadcast: ClassVar[bool] = True

    def __init__(self, config: DSpineSpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        hidden_size = config.transformer_layer_config.hidden_size
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "DSpine requires the full verifier vocabulary: "
                f"draft_vocab_size={config.draft_vocab_size}, "
                f"verifier_vocab_size={target_vocab_size}."
            )
        if config.curriculum_end_ratio < config.curriculum_hold_ratio:
            raise ValueError(
                "curriculum_end_ratio must be greater than or equal to "
                "curriculum_hold_ratio."
            )
        if config.sample_from_anchor:
            raise ValueError("DSpine requires sample_from_anchor=False.")
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
        """Load frozen verifier weights and derive the unified transfer space. (C(V * r))"""
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
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSpineDraftModel":
        """Create DSpine from the shared DFlash-family training arguments."""
        config = DSpineSpeculatorConfig(
            **cls._build_base_config_kwargs("dspine", verifier_config, **kwargs),
            known_anchor_injection=kwargs.get("known_anchor_injection", True),
            transfer_dim=kwargs.get("transfer_dim", 1024),
            message_dim=kwargs.get("message_dim", 512),
            transition_top_k=kwargs.get("transition_top_k", 16),
            distribution_loss_alpha=kwargs.get("distribution_loss_alpha", 0.1),
            embedding_loss_beta=kwargs.get("embedding_loss_beta", 0.5),
            embedding_loss_warmup_steps=kwargs.get("embedding_loss_warmup_steps", 146),
            curriculum_probability=kwargs.get("curriculum_probability", 0.5),
            curriculum_hold_ratio=kwargs.get("curriculum_hold_ratio", 1.0 / 6.0),
            curriculum_end_ratio=kwargs.get("curriculum_end_ratio", 1.0 / 3.0),
            curriculum_layers=kwargs.get("curriculum_layers", 3),
            transition_min_target_mass=kwargs.get("transition_min_target_mass", 1e-4),
            transfer_space_eps=kwargs.get("transfer_space_eps", 1e-6),
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
        """Compute the probability of using curriculum blocks based on training progress."""
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

    def _dspine_backbone_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        *,
        max_anchors: int,
        global_step: int,
        total_training_steps: int,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
    ]:
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, max_anchors, document_ids, device)
        )
        mask_tokens_size = max_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        draft_hidden = self.embed_tokens(mask_token_ids)

        target_hidden = self.hidden_norm(self.fc(hidden_states))
        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        all_position_ids = torch.cat(
            [position_ids, mask_position_ids.unsqueeze(0)], dim=1
        )
        position_embeddings = self.rotary_emb(hidden_states, all_position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )
        with torch.no_grad():
            if anchored_block_indices.numel() < total_seq_len:
                target_indices = (anchored_block_indices - 1) % total_seq_len
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )
            else:
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                targets = torch.roll(verifier_logits, 1, dims=1)[
                    :, anchored_block_indices
                ]

        # Compute the correct predecessor and probability for curriculum learning and injection.
        block_tokens = input_ids[0, anchored_block_indices].view(
            max_anchors, self.block_size
        )
        correct_predecessor_ids = block_tokens[:, :-1]
        curriculum_probability = self._curriculum_probability(
            global_step, total_training_steps
        )
        curriculum_blocks = None
        if curriculum_probability > 0:
            curriculum_blocks = (
                torch.rand(max_anchors, device=device) < curriculum_probability
            )

        layer_features: list[torch.Tensor] = []
        final_pre_injection: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            pre_injection = layer(
                hidden_states=draft_hidden,
                target_hidden=target_hidden,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=all_position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_idx == len(self.layers) - 1:
                # Cache u_{L,t} so last-write refinement can replace the final
                # injection with one conditioned on a chosen predecessor token,
                # without rerunning the Transformer (paper §4.3, p. 7).
                final_pre_injection = pre_injection
            use_curriculum = (
                curriculum_blocks if layer_idx < self.config.curriculum_layers else None
            )
            # [1, max_anchors * block_size, hidden_size], [1, max_anchors * block_size, transfer_dim]
            draft_hidden, post_features = self.injection.inject_layer(
                layer_idx,
                pre_injection,
                transfer_embeddings=self.transfer_embeddings,
                anchor_token_ids=(
                    block_tokens[:, 0] if self.config.known_anchor_injection else None
                ),
                correct_predecessor_ids=correct_predecessor_ids,
                curriculum_blocks=use_curriculum,
            )
            layer_features.append(post_features)

        assert final_pre_injection is not None  # noqa: S101
        hidden = self.norm(draft_hidden)
        # shape: [1, max_anchors * block_size, vocab_size]
        unary_logits = self.lm_head(hidden)
        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0
        return (
            hidden,
            unary_logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            layer_features,
            final_pre_injection,
        )

    def _compute_candidate_logits(
        self, hidden_states: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score selected vocabulary rows without materializing full logits."""
        candidate_weights = self.lm_head.weight[candidate_ids]
        return torch.einsum("...h,...kh->...k", hidden_states, candidate_weights)

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
        (
            _hidden,
            unary_logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            layer_features,
            final_pre_injection,
        ) = self._dspine_backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            global_step=global_step,
            total_training_steps=total_training_steps,
            **kwargs,
        )

        candidate_ids = unary_logits.topk(self.config.transition_top_k, dim=-1).indices
        block_tokens = input_ids[0, anchored_block_indices].view(
            max_anchors, self.block_size
        )

        correct_predecessor_features = self.injection.token_features(
            self.transfer_embeddings, block_tokens[:, :-1]
        )
        refined_successors = self.injection.last_write(
            final_pre_injection, correct_predecessor_features
        )
        refined_hidden = self.norm(refined_successors)

        candidate_blocks = candidate_ids.view(
            max_anchors, self.block_size, self.config.transition_top_k
        )
        refined_logits = self._compute_candidate_logits(
            refined_hidden, candidate_blocks[:, 1:]
        )
        refined_candidate_logits = unary_logits.new_zeros(candidate_ids.shape)
        refined_candidate_logits[:, :: self.block_size] = unary_logits[
            :, :: self.block_size
        ].gather(-1, candidate_ids[:, :: self.block_size])
        refined_candidate_logits.view(
            max_anchors, self.block_size, self.config.transition_top_k
        )[:, 1:] = refined_logits

        target_ids = input_ids[:, anchored_block_indices]
        target_features = self.injection.token_features(
            self.transfer_embeddings, target_ids
        )
        loss, metrics = compute_dspine_loss(
            unary_logits=unary_logits,
            refined_candidate_logits=refined_candidate_logits,
            candidate_ids=candidate_ids,
            targets=targets,
            target_ids=target_ids,
            layer_features=layer_features,
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
