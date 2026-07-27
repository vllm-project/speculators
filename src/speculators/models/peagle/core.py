"""P-EAGLE draft model implementation with parallel multi-token prediction."""

from typing import ClassVar

import torch
import torch.distributed as dist
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
    _sp_splits_batch: ClassVar[bool] = False

    def __init__(
        self,
        config: PEagleSpeculatorConfig,
    ):
        super().__init__(config=config)

        self.mask_token_id = config.mask_token_id

        # Learnable mask_hidden parameter for padding unsampled positions
        num_aux = (
            len(self.config.eagle_aux_hidden_state_layer_ids)
            if self.config.eagle_aux_hidden_state_layer_ids
            else 3
        )
        self.mask_hidden = torch.nn.Parameter(
            torch.randn(1, 1, num_aux * self.hidden_size)
        )

    @conditional_torch_compile
    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        max_anchors: int | None = None,
        num_depths: int = 8,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
        **kwargs,
    ):
        from speculators.train.distributed import (  # noqa: PLC0415
            get_sp_group,
            get_sp_rank,
            get_sp_size,
        )

        if verifier_last_hidden_states is None:
            raise ValueError("verifier_last_hidden_states required for training")

        device = hidden_states.device
        seq_length = input_ids.shape[1]
        sp_size = get_sp_size()

        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            num_depths=num_depths,
            down_sample_ratio=down_sample_ratio,
            down_sample_ratio_min=down_sample_ratio_min,
            max_anchors=max_anchors,
        )

        if sp_size > 1:
            sp_group = get_sp_group()
            sp_rank = get_sp_rank()
            src = dist.get_process_group_ranks(sp_group)[0]
            size_t = torch.tensor(anchor_pos.shape[0], device=device)
            dist.broadcast(size_t, src=src, group=sp_group)
            total_sampled = size_t.item()
            if anchor_pos.shape[0] != total_sampled:
                anchor_pos = torch.empty(total_sampled, device=device, dtype=torch.long)
                depth = torch.empty(total_sampled, device=device, dtype=torch.long)
            dist.broadcast(anchor_pos, src=src, group=sp_group)
            dist.broadcast(depth, src=src, group=sp_group)

            remainder = total_sampled % sp_size
            if remainder != 0:
                pad_len = sp_size - remainder
                anchor_pos = torch.nn.functional.pad(anchor_pos, (0, pad_len))
                depth = torch.nn.functional.pad(depth, (0, pad_len))

        total_sampled = anchor_pos.shape[0]
        full_anchor_pos = anchor_pos
        full_depth = depth
        full_orig_positions = full_anchor_pos + full_depth

        if sp_size > 1:
            n_per_rank = total_sampled // sp_size
            local_start = sp_rank * n_per_rank
            local_anchor_pos = anchor_pos[local_start : local_start + n_per_rank]
            local_depth = depth[local_start : local_start + n_per_rank]
            local_orig_positions = local_anchor_pos + local_depth
            local_sampled = n_per_rank
        else:
            local_anchor_pos = anchor_pos
            local_depth = depth
            local_orig_positions = full_orig_positions
            local_sampled = total_sampled

        is_depth_0 = local_depth == 0

        sampled_ids = torch.where(
            is_depth_0,
            input_ids[0, local_orig_positions],
            torch.tensor(self.mask_token_id, dtype=input_ids.dtype, device=device),
        ).unsqueeze(0)
        inputs_embeds = self.embed_tokens(sampled_ids).to(hidden_states.dtype)

        mask_hidden = self.mask_hidden.to(device=device, dtype=hidden_states.dtype)
        sampled_hidden = torch.where(
            is_depth_0.unsqueeze(-1),
            hidden_states[0, local_orig_positions],
            mask_hidden.squeeze(0).expand(local_sampled, -1),
        ).unsqueeze(0)

        if self.input_norm is not None:
            sampled_hidden = self.input_norm(sampled_hidden)
        if self.fc_norm is not None:
            chunks = sampled_hidden.chunk(len(self.fc_norm), dim=-1)
            sampled_hidden = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)],
                dim=-1,
            )
        sampled_hidden = self.fc(sampled_hidden)

        layer_input = torch.cat([inputs_embeds, sampled_hidden], dim=-1)

        position_ids = local_orig_positions.unsqueeze(0)

        position_embeddings = self.rotary_emb(layer_input, position_ids)

        doc_ids_1d = document_ids.squeeze(0).to(device)

        # Build masks at FULL total_sampled scale (matches post-all-to-all layout)
        def _build_attn_mask(sliding_window=None):
            mask_mod = create_peagle_mask_mod(
                anchor_pos=full_anchor_pos,
                depth=full_depth,
                document_ids=doc_ids_1d,
                sliding_window=sliding_window,
            )
            return self._create_mask_fn(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=total_sampled,
                KV_LEN=total_sampled,
                device=device,
            )

        full_attn_mask = _build_attn_mask() if self.uses_full_attn else None
        sliding_window_attn_mask = (
            _build_attn_mask(self.sliding_window)
            if self.uses_sliding_window_attn
            else None
        )

        hidden_states = layer_input
        for layer_idx, layer in enumerate(self.layers):
            layer_mask = (
                sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        logits = self.lm_head(self.norm(hidden_states))

        with torch.no_grad():
            targets = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )

        targets = targets[:, local_orig_positions, :]

        loss, metrics = compute_metrics(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
            anchor_pos=local_anchor_pos,
            depth=local_depth,
            num_depths=num_depths,
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
            fc_norm=kwargs.get("fc_norm", False),
            norm_output=kwargs.get("norm_output", False),
            eagle_aux_hidden_state_layer_ids=target_layer_ids,
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
        max_anchors = kwargs.get("max_anchors")
        num_depths = kwargs.get("num_depths", 8)
        down_sample_ratio = kwargs.get("down_sample_ratio", 0.7)
        down_sample_ratio_min = kwargs.get("down_sample_ratio_min", 0.2)
        shared = {
            "loss_config": loss_config,
            "max_anchors": max_anchors,
            "num_depths": num_depths,
            "down_sample_ratio": down_sample_ratio,
            "down_sample_ratio_min": down_sample_ratio_min,
        }
        return dict(shared), dict(shared)
