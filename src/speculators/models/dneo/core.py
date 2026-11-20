import logging
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dneo.config import DNeoSpeculatorConfig
from speculators.models.dneo.model_definitions import (
    DNeoCausalHead,
    FeatureEncoder,
    Qwen3DNeoDecoderLayer,
    create_prefix_only_mask_mod,
    dneo_target_layer_ids,
)
from speculators.models.utils import conditional_torch_compile, get_verifier_config

logger = logging.getLogger(__name__)

__all__ = [
    "DNeoDraftModel",
]


@SpeculatorModel.register("dneo")
class DNeoDraftModel(DFlashDraftModel):
    """DNeo parallel drafter (arXiv:2611.21925) on the DFlash backbone.

    Differences from DFlash, each switchable in the config:

    - ``fc`` is a :class:`FeatureEncoder` (MoFA or diagonal) that fuses target
      layers coordinate-wise instead of projecting their concatenation;
    - prefix-only attention: draft positions attend to the context features only;
    - DFlash-2's 2-tap dynamic convolutions around attention and MLP;
    - the fused DNeo causal head adds a previous-token-conditioned logit bias,
      teacher-forced on the block's ground-truth tokens during training.

    Rejection memory (cross-attention to the verifier's features at the previous
    round's rejected positions) is not implemented: it needs on-policy rollouts
    with those features, which the offline data pipeline does not produce.
    """

    config_class: ClassVar[type[DNeoSpeculatorConfig]] = DNeoSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DNeoDecoderLayer"]

    def _make_decoder_layer(
        self, config: DNeoSpeculatorConfig, layer_idx: int
    ) -> nn.Module:
        return Qwen3DNeoDecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
            prefix_only_attention=config.prefix_only_attention,
        )

    def __init__(self, config: DNeoSpeculatorConfig) -> None:
        hidden_size = config.transformer_layer_config.hidden_size
        if config.target_hidden_size not in (None, hidden_size):
            raise ValueError(
                "DNeo's feature encoder needs the drafter's hidden size to equal the "
                f"target's ({config.target_hidden_size}), got {hidden_size}"
            )
        super().__init__(config=config)
        # post_init re-randomized the convolutions' kernel projections: start them,
        # as DFlash2 does, as identity transforms.
        for layer in self.layers:
            layer.reset_convolutions()
        # Replace DFlash's dense encoder; hidden_norm still follows it.
        self.fc = FeatureEncoder(  # type: ignore[assignment]
            len(self.target_layer_ids), hidden_size, mode=config.feature_encoder
        )
        self.causal_head: DNeoCausalHead | None = None
        if config.causal_head_rank > 0:
            self.causal_head = DNeoCausalHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                hidden_size=hidden_size,
                rank=config.causal_head_rank,
                block_size=config.block_size,
            )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DNeoDraftModel":
        """Create a DNeo model from training arguments.

        Without ``target_layer_ids`` the DNeo protocol picks them: the first
        target layer plus ``num_target_layers - 1`` from its second half.
        """
        if kwargs.get("target_layer_ids") is None:
            num_layers = get_verifier_config(
                kwargs["verifier_name_or_path"],
                trust_remote_code=kwargs.get("trust_remote_code", False),
            ).num_hidden_layers
            kwargs["target_layer_ids"] = dneo_target_layer_ids(
                num_layers, kwargs.get("num_target_layers") or 6
            )
            logger.warning(
                "DNeo protocol target layers %s: capture the training data with "
                "--target-layer-ids %s",
                kwargs["target_layer_ids"],
                " ".join(map(str, kwargs["target_layer_ids"])),
            )
        config = DNeoSpeculatorConfig(
            **cls._build_base_config_kwargs("dneo", verifier_config, **kwargs),
            feature_encoder=kwargs.get("feature_encoder", "mofa"),
            prefix_only_attention=kwargs.get("prefix_only_attention", True),
            conv_kernel_size=kwargs.get("dneo_conv_kernel_size", 2),
            conv_group_size=kwargs.get("dneo_conv_group_size", 16),
            causal_head_rank=kwargs.get("causal_head_rank", 256),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        """Anchor only where the drafter has context: with prefix-only attention a
        document's first token has no earlier position to attend to."""
        if self.config.prefix_only_attention:
            docs = document_ids.to(loss_mask.device)
            has_context = torch.zeros_like(loss_mask, dtype=torch.bool)
            has_context[:, 1:] = (docs[:, 1:] == docs[:, :-1]) & (docs[:, 1:] != -1)
            loss_mask = loss_mask * has_context.to(loss_mask.dtype)
        return super()._build_attention_mask(
            loss_mask, max_anchors, document_ids, device
        )

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
    ):
        if not self.config.prefix_only_attention:
            return super()._create_attention_mask(
                document_ids, total_seq_len, anchor_positions, device, sliding_window
            )
        mask_mod, q_len, kv_len = create_prefix_only_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
        )
        return self._create_mask_fn(
            mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
        )

    def _previous_token_ids(
        self,
        input_ids: torch.Tensor,  # [1, total_seq_len]
        anchored_block_indices: torch.Tensor,  # [num_anchors*block_size]
    ) -> torch.Tensor:
        """Each draft slot's previous token in the ground-truth block (teacher
        forcing), as the causal head sees the previously sampled token at
        inference."""
        block_tokens = input_ids[0, anchored_block_indices].view(-1, self.block_size)
        if self.config.sample_from_anchor:
            # slot k predicts token p+k+1; its predecessor is token p+k
            return block_tokens
        # slot k predicts token p+k; its predecessor is token p+k-1
        return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        if self.causal_head is not None:
            num_blocks = anchored_block_indices.numel() // self.block_size
            bias = self.causal_head(
                self._previous_token_ids(input_ids, anchored_block_indices),
                hidden.view(num_blocks, self.block_size, -1),
            )
            logits = logits + bias.view_as(logits).to(logits.dtype)
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
