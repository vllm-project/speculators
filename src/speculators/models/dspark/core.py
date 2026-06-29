"""DSpark draft model core implementation.

Anchor-based training with cross-attention, Markov head, and confidence head.
"""

from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_dspark_metrics
from speculators.models.dspark.model_definitions import DSparkDecoderLayer
from speculators.models.dspark.utils import (
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DraftVocabMixin, SpeculatorModel):
    """DSpark draft model with anchor-based training.

    Uses cross-attention where draft queries attend to target hidden states
    and their own block. Supports Markov head for token sequence modeling
    and confidence head for early-stop during inference.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig
    _no_split_modules = ["DSparkDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        if config.transformer_layer_config._attn_implementation is None:
            config.transformer_layer_config._attn_implementation = "flex_attention"
        self._attn_impl = config.transformer_layer_config._attn_implementation
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                DSparkDecoderLayer(config.transformer_layer_config, layer_idx)
                for layer_idx in range(num_draft_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,
        )
        self.verifier_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,
        )
        self.verifier_norm.weight.requires_grad = False

        self.block_size = config.block_size
        self.num_anchors = config.num_anchors

        # Markov head
        self.markov_head = self._build_markov_head(config)

        # Confidence head
        self.enable_confidence_head = config.enable_confidence_head
        self.confidence_head_with_markov = config.confidence_head_with_markov
        self.confidence_head = self._build_confidence_head(config)

        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.aux_hidden_state_layer_ids

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training."
            )
        return self.config.mask_token_id

    def _build_markov_head(self, config: DSparkSpeculatorConfig) -> nn.Module | None:
        """Build Markov head based on config."""
        from speculators.models.dspark.markov_head import build_markov_head

        return build_markov_head(config)

    def _build_confidence_head(self, config: DSparkSpeculatorConfig) -> nn.Module | None:
        """Build confidence head based on config."""
        from speculators.models.dspark.confidence_head import build_confidence_head

        return build_confidence_head(config)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create DSpark model from training arguments.

        Args:
            verifier_config: Verifier model configuration with num_hidden_layers
                set to the number of DRAFT layers.
            t2d: Target-to-draft vocabulary mapping tensor.
            d2t: Draft-to-target vocabulary mapping tensor.
            **kwargs: Training arguments with DSpark-specific params.
        """
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.proposals.greedy import GreedyTokenProposalConfig

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        verifier_config._attn_implementation = kwargs.get(
            "draft_attn_impl", "flex_attention"
        )

        config = DSparkSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            num_anchors=kwargs.get("num_anchors", 256),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            markov_rank=kwargs.get("markov_rank", 0),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", False),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", False),
            ce_loss_alpha=kwargs.get("ce_loss_alpha", 1.0),
            l1_loss_alpha=kwargs.get("l1_loss_alpha", 0.0),
            confidence_head_alpha=kwargs.get("confidence_head_alpha", 0.0),
            loss_decay_gamma=kwargs.get("loss_decay_gamma", None),
            speculators_config=SpeculatorsConfig(
                algorithm="dspark",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("block_size", 8),
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DSpark."""
        train_kwargs = {
            "ce_loss_alpha": kwargs.get("ce_loss_alpha", 1.0),
            "l1_loss_alpha": kwargs.get("l1_loss_alpha", 0.0),
            "confidence_head_alpha": kwargs.get("confidence_head_alpha", 0.0),
            "loss_decay_gamma": kwargs.get("loss_decay_gamma", None),
        }
        return train_kwargs, train_kwargs

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        confidence_head_alpha: float = 0.0,
        loss_decay_gamma: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass for DSpark training.

        Args:
            hidden_states: [1, T, num_layers*D] target hidden states.
            input_ids: [1, T] source token IDs.
            loss_mask: [1, T] supervised token mask.
            verifier_last_hidden_states: [1, T, D] last layer target hidden states.
            document_ids: [1, T] document IDs for packed sequences.
            position_ids: [1, T] position IDs (optional).
            ce_loss_alpha: CE loss weight.
            l1_loss_alpha: L1 loss weight.
            confidence_head_alpha: Confidence loss weight.
            loss_decay_gamma: Position-wise loss decay rate.

        Returns:
            Tuple of (draft_tokens, loss, metrics).
        """
        device = hidden_states.device
        bsz, seq_len = hidden_states.shape[:2]

        # 1. Sample anchor positions
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )

        # 2. Create noise embedding
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )

        # 3. Project target hidden states
        target_hidden = self.hidden_norm(self.fc(hidden_states))

        # 4. Build attention mask
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )

        # 5. Build position IDs
        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        position_embeddings = self.rotary_emb(noise_embedding, full_position_ids)

        # 6. Transformer forward
        hidden = noise_embedding
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                target_hidden_states=target_hidden,
                attention_mask=dspark_attn_mask,
                position_ids=full_position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden = self.norm(hidden)

        # 7. Compute logits
        num_blocks = anchor_positions.size(1)
        logits = self.lm_head(hidden).reshape(bsz, num_blocks, self.block_size, -1)
        hidden_4d = hidden.reshape(bsz, num_blocks, self.block_size, -1)

        # 8. Build target IDs and eval mask
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )

        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )

        # 9. Compute aligned target logits for L1 loss and confidence supervision
        aligned_target_logits = None
        if verifier_last_hidden_states is not None and (l1_loss_alpha > 0 or confidence_head_alpha > 0):
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                verifier_last_hidden_states.unsqueeze(1).expand(
                    -1, num_blocks, -1, -1
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, verifier_last_hidden_states.size(-1)
                ),
            )
            aligned_target_logits = self.verifier_lm_head(
                self.verifier_norm(aligned_target_hidden)
            )

        # 10. Apply Markov head
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        if self.markov_head is not None:
            logits = self.markov_head.apply_block_logits(
                logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_4d,
            )

        # 11. Confidence head prediction
        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(dtype=hidden_4d.dtype)
                confidence_features = torch.cat(
                    [hidden_4d, prev_embeddings], dim=-1
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(hidden_4d).float()

        # 12. Compute loss and metrics
        loss, metrics = compute_dspark_metrics(
            logits=logits,
            targets=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
            ce_loss_alpha=ce_loss_alpha,
            l1_loss_alpha=l1_loss_alpha,
            confidence_head_alpha=confidence_head_alpha,
            loss_decay_gamma=loss_decay_gamma,
        )

        draft_tokens = torch.argmax(logits, dim=-1).reshape(bsz, -1)
        return draft_tokens, loss, metrics
