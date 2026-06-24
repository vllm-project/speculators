"""P-EAGLE draft model implementation with parallel multi-token prediction."""

from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.peagle.attention import create_peagle_mask_mod
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.data import generate_cod_sample_indices
from speculators.models.peagle.metrics import compute_metrics
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.distributed import get_sp_rank, get_sp_size


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

    @torch.compiler.disable
    def _sync_and_partition_cod(
        self, anchor_pos, depth, document_ids, device
    ):
        """Broadcast COD samples from rank 0, pad for SP divisibility, create mask, partition."""
        import torch.distributed as dist  # noqa: PLC0415

        from speculators.train.distributed import get_sp_group  # noqa: PLC0415

        sp_size = get_sp_size()
        sp_rank = get_sp_rank()
        sp_group = get_sp_group()
        src = dist.get_process_group_ranks(sp_group)[0]

        actual_total = torch.tensor(
            [anchor_pos.shape[0]], dtype=torch.long, device=device
        )
        dist.broadcast(actual_total, src=src, group=sp_group)
        actual_total = int(actual_total.item())

        if anchor_pos.shape[0] != actual_total:
            anchor_pos = torch.zeros(actual_total, dtype=torch.long, device=device)
            depth = torch.zeros(actual_total, dtype=torch.long, device=device)
        dist.broadcast(anchor_pos, src=src, group=sp_group)
        dist.broadcast(depth, src=src, group=sp_group)

        padded_total = ((actual_total + sp_size - 1) // sp_size) * sp_size
        pad_amount = padded_total - actual_total

        valid_mask = torch.ones(padded_total, dtype=torch.bool, device=device)
        if pad_amount > 0:
            valid_mask[actual_total:] = False
            anchor_pos = torch.cat(
                [
                    anchor_pos,
                    torch.zeros(pad_amount, dtype=torch.long, device=device),
                ]
            )
            depth = torch.cat(
                [
                    depth,
                    torch.zeros(pad_amount, dtype=torch.long, device=device),
                ]
            )

        mask_mod = create_peagle_mask_mod(
            anchor_pos=anchor_pos,
            depth=depth,
            document_ids=document_ids.squeeze(0).to(device),
            valid_mask=valid_mask,
        )
        attention_mask = self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=padded_total,
            KV_LEN=padded_total,
            device=device,
        )

        n_per_rank = padded_total // sp_size
        local_start = sp_rank * n_per_rank
        anchor_pos = anchor_pos[local_start : local_start + n_per_rank]
        depth = depth[local_start : local_start + n_per_rank]
        valid_mask = valid_mask[local_start : local_start + n_per_rank]

        return anchor_pos, depth, valid_mask, attention_mask, n_per_rank

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_fn=kl_div_loss,
        **kwargs,
    ):
        """
        Forward pass for P-EAGLE model training with parallel group prediction.

        Args:
            hidden_states: Verifier hidden states [batch, seq_len, 3*hidden_size]
            input_ids: Input token IDs [batch, seq_len]
            document_ids: Document IDs [1, seq_len], maps positions to doc index, pad -1
            position_ids: Position IDs [batch, seq_len] (optional)
            loss_mask: Loss mask for which tokens to compute loss on
                [batch, seq_len]
            verifier_last_hidden_states: Verifier final hidden states for
                targets [batch, seq_len, hidden_size]

        Returns:
            Tuple of (draft_tokens, loss, metrics)
        """
        if verifier_last_hidden_states is None:
            raise ValueError("verifier_last_hidden_states required for training")

        device = hidden_states.device
        sp_size = get_sp_size()
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

        # SP: sync COD samples across ranks, pad, create global mask, partition
        if sp_size > 1:
            anchor_pos, depth, valid_mask, attention_mask, total_sampled = (
                self._sync_and_partition_cod(
                    anchor_pos, depth, document_ids, device
                )
            )
        else:
            total_sampled = anchor_pos.shape[0]
            valid_mask = None

        orig_positions = anchor_pos + depth
        is_depth_0 = depth == 0  # [total_sampled]

        # Clamp for safe indexing (SP padding positions may exceed seq_length)
        clamped_orig = torch.clamp(orig_positions, 0, seq_length - 1)

        # Build sampled input_ids: real tokens for depth 0, mask for others
        sampled_ids = torch.where(
            is_depth_0,
            input_ids[0, clamped_orig],
            torch.tensor(self.mask_token_id, dtype=input_ids.dtype, device=device),
        ).unsqueeze(0)  # [1, total_sampled]
        inputs_embeds = self.embed_tokens(sampled_ids).to(
            hidden_states.dtype
        )  # [1, total_sampled, hidden_size]

        # Build sampled hidden states: real for depth 0, mask_hidden for others
        mask_hidden = self.mask_hidden.to(device=device, dtype=hidden_states.dtype)
        sampled_hidden = torch.where(
            is_depth_0.unsqueeze(-1),
            hidden_states[0, clamped_orig],
            mask_hidden.squeeze(0).expand(total_sampled, -1),
        ).unsqueeze(0)  # [1, total_sampled, 3*hidden_size]

        # Project concatenated hidden states (3*hidden_size) -> hidden_size
        sampled_hidden = self.fc(sampled_hidden)  # [1, total_sampled, hidden_size]

        layer_input = torch.cat(
            [inputs_embeds, sampled_hidden], dim=-1
        )  # [1, total_sampled, 2*hidden_size]

        position_ids = orig_positions.unsqueeze(0)  # [1, total_sampled]

        position_embeddings = self.rotary_emb(layer_input, position_ids)

        if sp_size <= 1:
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
                self.verifier_norm(verifier_last_hidden_states[:, clamped_orig])
            )
            # [1, total_sampled, vocab_size]

        loss, metrics = compute_metrics(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
            anchor_pos=anchor_pos,
            depth=depth,
            num_depths=self.num_depths,
            loss_fn=loss_fn,
            valid_mask=valid_mask,
        )

        if sp_size > 1:
            for k in list(metrics):
                if k.endswith("_total") and metrics[k].numel() == 1:
                    metrics[k] = metrics[k] / sp_size

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
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        return {"loss_fn": loss_fn}, {"loss_fn": loss_fn}
