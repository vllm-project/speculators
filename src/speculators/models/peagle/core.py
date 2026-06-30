"""P-EAGLE draft model implementation with parallel multi-token prediction."""

from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.peagle.attention import create_peagle_mask_mod
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.data import generate_cod_sample_indices
from speculators.models.peagle.metrics import compute_metrics
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig


@SpeculatorModel.register("peagle")
class PEagleDraftModel(Eagle3DraftModel):
    """
    P-EAGLE (Parallel EAGLE) draft model for speculative decoding.

    P-EAGLE extends EAGLE-3 with parallel multi-token prediction using
    Conditional-On-Distribution (COD) sampling for memory-efficient training.
    """

    config_class: ClassVar[type[PEagleSpeculatorConfig]] = PEagleSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        *Eagle3DraftModel._keys_to_ignore_on_load_missing,  # noqa: SLF001
        "mask_hidden",
    ]

    def __init__(
        self,
        config: PEagleSpeculatorConfig,
    ):
        super().__init__(config=config)

        self.num_depths = config.num_depths
        self.down_sample_ratio = config.down_sample_ratio
        self.down_sample_ratio_min = config.down_sample_ratio_min
        self.mask_token_id = config.mask_token_id

        # Learnable mask_hidden parameter for padding unsampled positions
        self.mask_hidden = torch.nn.Parameter(torch.randn(1, 1, 3 * self.hidden_size))

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        **kwargs,
    ):
        """
        Forward pass for P-EAGLE model training with parallel group prediction.

        Args:
            hidden_states: Verifier hidden states [1, seq_len, 3*hidden_size]
            input_ids: Input token IDs [1, seq_len]
            document_ids: Document IDs [1, seq_len], maps positions to doc index, pad -1
            position_ids: Position IDs [1, seq_len] (optional)
            loss_mask: Loss mask for which tokens to compute loss on
                [1, seq_len]
            verifier_last_hidden_states: Verifier final hidden states for
                targets [1, seq_len, hidden_size]

        Returns:
            Tuple of (draft_tokens, loss, metrics)
        """
        if verifier_last_hidden_states is None:
            raise ValueError("verifier_last_hidden_states required for training")

        device = hidden_states.device
        seq_length = input_ids.shape[1]

        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

        # Generate COD sampling indices
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            num_depths=self.num_depths,
            down_sample_ratio=self.down_sample_ratio,
            down_sample_ratio_min=self.down_sample_ratio_min,
        )
        total_sampled = anchor_pos.shape[0]

        orig_positions = anchor_pos + depth
        is_depth_0 = depth == 0  # [total_sampled]

        # Build sampled input_ids: real tokens for depth 0, mask for others
        sampled_ids = torch.where(
            is_depth_0,
            input_ids[0, orig_positions],
            torch.tensor(self.mask_token_id, dtype=input_ids.dtype, device=device),
        ).unsqueeze(0)  # [1, total_sampled]
        inputs_embeds = self.embed_tokens(sampled_ids).to(
            hidden_states.dtype
        )  # [1, total_sampled, hidden_size]

        # Build sampled hidden states: real for depth 0, mask_hidden for others
        mask_hidden = self.mask_hidden.to(device=device, dtype=hidden_states.dtype)
        sampled_hidden = torch.where(
            is_depth_0.unsqueeze(-1),
            hidden_states[0, orig_positions],
            mask_hidden.squeeze(0).expand(orig_positions.shape[0], -1),
        ).unsqueeze(0)  # [1, total_sampled, 3*hidden_size]

        # Project concatenated hidden states (3*hidden_size) -> hidden_size
        if self.input_norm is not None:
            sampled_hidden = self.input_norm(sampled_hidden)
        sampled_hidden = self.fc(sampled_hidden)  # [1, total_sampled, hidden_size]

        layer_input = torch.cat(
            [inputs_embeds, sampled_hidden], dim=-1
        )  # [1, total_sampled, 2*hidden_size]

        position_ids = orig_positions.unsqueeze(0)  # [1, total_sampled]

        position_embeddings = self.rotary_emb(layer_input, position_ids)

        mask_mod = create_peagle_mask_mod(
            anchor_pos=anchor_pos,
            depth=depth,
            document_ids=document_ids.squeeze(0).to(device),
        )

        attention_mask = self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total_sampled,
            KV_LEN=total_sampled,
            device=device,
        )

        hidden_states = layer_input
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        logits = self.lm_head(
            self.norm(hidden_states)
        )  # [1, total_sampled, vocab_size]

        with torch.no_grad():
            targets = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )

        targets = targets[:, orig_positions, :]  # [1, total_sampled, vocab_size]

        loss, metrics = compute_metrics(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
            anchor_pos=anchor_pos,
            depth=depth,
            num_depths=self.num_depths,
            loss_config=loss_config,
        )

        return None, loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "PEagleDraftModel":
        """
        Create P-EAGLE model from training arguments.

        Args:
            verifier_config: Verifier model configuration
            **kwargs: Training arguments with P-EAGLE-specific params
                - draft_vocab_size: Size of draft vocabulary
                - norm_before_residual: Whether to normalize before residual
                - num_depths: Number of parallel groups (default 8)
                - down_sample_ratio: COD sampling ratio (default 0.7)
                - down_sample_ratio_min: Minimum sampling ratio (default 0.2)
                - mask_token_id: Mask token ID
                - t2d: Target-to-draft vocabulary mapping
                - d2t: Draft-to-target vocabulary mapping
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized PEagleDraftModel
        """
        # Resolve target layer IDs if not provided
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )

        config = PEagleSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs.get("norm_before_residual", False),
            norm_before_fc=kwargs.get("norm_before_fc", False),
            norm_output=kwargs.get("norm_output", False),
            eagle_aux_hidden_state_layer_ids=target_layer_ids,
            num_depths=kwargs.get("num_depths", 8),
            down_sample_ratio=kwargs.get("down_sample_ratio", 0.7),
            down_sample_ratio_min=kwargs.get("down_sample_ratio_min", 0.2),
            mask_token_id=kwargs.get("mask_token_id"),
            speculators_config=SpeculatorsConfig(
                algorithm="peagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("num_depths", 8),
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
        """
        Get training and validation kwargs for P-EAGLE.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        return {"loss_config": loss_config}, {"loss_config": loss_config}
