# ruff: noqa: ERA001
"""
P-EAGLE (Parallel EAGLE) draft model implementation.

P-EAGLE extends EAGLE-3 with parallel multi-token prediction through
Conditional-On-Distribution (COD) sampling. Instead of sequential TTT steps,
P-EAGLE generates predictions for multiple future positions in a single
forward pass using parallel prediction groups.

Key differences from Eagle3DraftModel:
- mask_hidden: Learnable hidden state parameter for MTP padding positions
- mask_token_embedding: Learnable embedding for unknown predicted tokens
- Parallel forward pass via COD-sampled position groups
- Cross-entropy loss by default (instead of KL divergence)
"""

from __future__ import annotations

import copy
from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig, DynamicCache, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
)
from speculators.models.eagle3.model_definitions import model_classes
from speculators.models.peagle.cod_sampling import cod_sample
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.loss import (
    cross_entropy_loss,
    kl_div_loss,
    per_depth_accuracy,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers


def build_parallel_group_mask_mod(
    ntp_lengths: torch.Tensor,
    depth_indices: list[torch.Tensor],
    total_len: int,
):
    """Build a mask modification function for parallel group attention.

    Creates an attention mask that allows:
    1. Causal attention within NTP (depth-0) positions
    2. Each MTP position attends to NTP positions before its original
       sequence position AND to itself (diagonal)
    3. MTP positions do NOT attend to other MTP positions (independence)

    The combined sequence layout is:
        [NTP positions (depth 0)] [MTP depth 1] [MTP depth 2] ... [MTP depth K-1]

    Args:
        ntp_lengths: Tensor of per-document NTP sequence lengths for
            document-boundary masking.
        depth_indices: COD-sampled indices per depth from cod_sample().
        total_len: Total combined sequence length across all depths.

    Returns:
        A mask modification function compatible with flex_attention's
        create_block_mask.
    """
    ntp_len = depth_indices[0].shape[0] if len(depth_indices) > 0 else 0

    # Build a mapping from global position to original sequence position
    # for all depths beyond 0
    # Also track boundary offsets for each depth
    depth_offsets = [0]
    offset = ntp_len
    for k in range(1, len(depth_indices)):
        depth_offsets.append(offset)
        offset += depth_indices[k].shape[0]

    # Build document IDs for NTP positions
    document_ids = torch.repeat_interleave(
        torch.arange(ntp_lengths.shape[0], device=ntp_lengths.device, dtype=torch.long),
        ntp_lengths,
    )
    # Pad to ntp_len
    if document_ids.shape[0] < ntp_len:
        document_ids = torch.cat([
            document_ids,
            -1 * torch.ones(
                ntp_len - document_ids.shape[0],
                device=ntp_lengths.device,
                dtype=torch.long,
            ),
        ])

    # Build original position mapping for MTP positions
    mtp_original_positions = []
    for k in range(1, len(depth_indices)):
        if depth_indices[k].numel() > 0:
            mtp_original_positions.append(depth_indices[k])
        else:
            mtp_original_positions.append(
                torch.tensor([], dtype=torch.long, device=ntp_lengths.device)
            )

    if mtp_original_positions:
        mtp_positions_cat = torch.cat(mtp_original_positions).contiguous()
    else:
        mtp_positions_cat = torch.tensor(
            [], dtype=torch.long, device=ntp_lengths.device
        )

    def mask_mod(_b, _h, q_idx, kv_idx):
        q_is_ntp = q_idx < ntp_len
        kv_is_ntp = kv_idx < ntp_len

        # Case 1: Both NTP - standard causal + document mask
        both_ntp = torch.logical_and(q_is_ntp, kv_is_ntp)
        ntp_causal = q_idx >= kv_idx
        ntp_same_doc = document_ids[torch.clamp(q_idx, max=ntp_len - 1)] == \
            document_ids[torch.clamp(kv_idx, max=ntp_len - 1)]
        ntp_valid = torch.logical_and(
            document_ids[torch.clamp(q_idx, max=ntp_len - 1)] != -1, ntp_same_doc
        )
        ntp_mask = torch.logical_and(ntp_causal, ntp_valid)

        # Case 2: Query is MTP, KV is NTP - MTP attends to NTP positions
        # before or at its original position
        q_is_mtp = ~q_is_ntp
        mtp_to_ntp = torch.logical_and(q_is_mtp, kv_is_ntp)
        mtp_q_offset = q_idx - ntp_len
        mtp_q_orig_pos = mtp_positions_cat[
            torch.clamp(mtp_q_offset, min=0, max=max(mtp_positions_cat.shape[0] - 1, 0))
        ]
        mtp_attends_ntp = kv_idx <= mtp_q_orig_pos

        # Case 3: Diagonal self-attention for MTP positions
        diagonal = q_idx == kv_idx

        # Combine: NTP-NTP causal OR MTP-to-NTP OR diagonal
        return torch.logical_or(
            torch.logical_and(both_ntp, ntp_mask),
            torch.logical_or(
                torch.logical_and(mtp_to_ntp, mtp_attends_ntp),
                torch.logical_and(q_is_mtp, diagonal),
            ),
        )

    return mask_mod


@SpeculatorModel.register("peagle")
class PEagleDraftModel(SpeculatorModel):
    """P-EAGLE draft model with parallel multi-token prediction.

    Extends the EAGLE-3 architecture with:
    - mask_hidden: Learnable padding hidden state for MTP positions
    - mask_token_embedding: Learnable embedding for unknown tokens at MTP positions
    - Parallel prediction groups via COD sampling
    - Single forward pass for all prediction depths
    """

    config_class: ClassVar[type[PEagleSpeculatorConfig]] = PEagleSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    def __init__(
        self,
        config: PEagleSpeculatorConfig,
        t2d: torch.Tensor | None,
        d2t: torch.Tensor | None,
    ):
        # Initialize via SpeculatorModel.__init__ (not Eagle3DraftModel)
        # to avoid complex MRO issues - we replicate needed setup
        SpeculatorModel.__init__(
            self,
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.para_num = config.para_num
        self.down_sample_ratio = config.down_sample_ratio
        self.down_sample_ratio_min = config.down_sample_ratio_min
        self.ptd_token_id = config.ptd_token_id
        self.loss_type = config.loss_type

        # Verify mapping tensors
        if (t2d is None) != (d2t is None):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be None. "
                f"Got t2d={'provided' if t2d is not None else 'None'}, "
                f"d2t={'provided' if d2t is not None else 'None'}"
            )

        if t2d is not None:
            self.register_buffer("t2d", t2d)
            if int(t2d.sum(dtype=torch.long).item()) != self.draft_vocab_size:
                raise ValueError(
                    f"t2d has {int(t2d.sum(dtype=torch.long).item())} non-zero values, "
                    f"expected {self.draft_vocab_size}."
                )
        else:
            self.register_buffer("t2d", None)

        if d2t is not None:
            self.register_buffer("d2t", d2t)
            if d2t.shape[0] != self.draft_vocab_size:
                raise ValueError(
                    f"d2t.shape[0] ({d2t.shape[0]}) must match"
                    f" draft_vocab_size ({self.draft_vocab_size})."
                )
        else:
            self.register_buffer("d2t", None)

        # === P-EAGLE specific: Learnable padding parameters ===
        # mask_hidden substitutes for missing hidden vectors at MTP positions
        # Shape: [1, 1, 3 * hidden_size]
        self.mask_hidden = torch.nn.Parameter(
            torch.zeros(1, 1, 3 * self.hidden_size)
        )
        torch.nn.init.normal_(self.mask_hidden, mean=0.0, std=0.02)

        # mask_token_embedding substitutes for unknown predicted tokens at MTP positions
        # Shape: [1, 1, hidden_size]
        self.mask_token_embedding = torch.nn.Parameter(
            torch.zeros(1, 1, self.hidden_size)
        )
        torch.nn.init.normal_(self.mask_token_embedding, mean=0.0, std=0.02)

        # === Standard EAGLE-3 architecture components ===
        self.fc = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self._model_definitions = model_classes[
            config.transformer_layer_config.model_type
        ]
        self._setup_decoder_layers(
            config.transformer_layer_config, config.norm_before_residual
        )
        self.norm = self._model_definitions.norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self._setup_rotary_embedding(config.transformer_layer_config)
        self._setup_embeddings_and_lm_heads(
            config.speculators_config.verifier, t2d, config.embed_requires_grad
        )

    def _setup_decoder_layers(
        self, transformer_layer_config: PretrainedConfig, norm_before_residual: bool
    ):
        """Setup decoder layers (same as EAGLE-3)."""
        num_hidden_layers = transformer_layer_config.num_hidden_layers
        layers = [
            self._model_definitions.first_layer_class(
                transformer_layer_config,
                layer_idx=0,
                norm_before_residual=norm_before_residual,
            )
        ]
        layers.extend([
            self._model_definitions.decoder_layer_class(
                transformer_layer_config, layer_idx
            )
            for layer_idx in range(1, num_hidden_layers)
        ])
        self.layers = torch.nn.ModuleList(layers)

    def _setup_rotary_embedding(self, transformer_layer_config: PretrainedConfig):
        """Setup rotary embedding with 2x hidden size (same as EAGLE-3)."""
        modified_config = copy.copy(transformer_layer_config)
        modified_config.hidden_size = modified_config.hidden_size * 2
        self.rotary_emb = self._model_definitions.rotary_emb_class(modified_config)

    def _setup_embeddings_and_lm_heads(
        self,
        config: VerifierConfig,
        t2d: torch.Tensor | None,
        embed_requires_grad: bool,
    ):
        """Setup embeddings and LM heads (same as EAGLE-3 but embed unfrozen by default)."""
        if config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")
        verifier_model_config = AutoConfig.from_pretrained(config.name_or_path)

        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )

        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            config.name_or_path,
        )

        if "embed_tokens.weight" not in verifier_weights:
            raise KeyError(
                f"Could not find embedding weights in {config.name_or_path}. "
                "Expected a key ending with 'embed_tokens.weight'."
            )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)

        self.verifier_norm = self._model_definitions.norm_class(
            self.hidden_size, eps=verifier_model_config.rms_norm_eps,
        )

        # EMBEDDINGS
        self.embed_tokens = torch.nn.Embedding(
            verifier_model_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_model_config.pad_token_id,
        )
        default_dtype = self.embed_tokens.weight.dtype
        self.embed_tokens.load_state_dict(
            {"weight": embed_tokens_weight.to(default_dtype)}
        )
        # P-EAGLE: embeddings trainable by default
        self.embed_tokens.weight.requires_grad = embed_requires_grad

        # LM HEADS
        self.lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        if t2d is not None:
            lm_head_weight = lm_head_weight.to(
                device=t2d.device, dtype=default_dtype
            )[t2d.to(torch.bool), :]
        else:
            lm_head_weight = lm_head_weight.to(dtype=default_dtype)

        if lm_head_weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Verifier lm head data shape {lm_head_weight.shape} does not match "
                f"draft lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

        if "model.norm.weight" in verifier_weights:
            verifier_norm_weight = verifier_weights["model.norm.weight"]
            self.verifier_norm.load_state_dict(
                {"weight": verifier_norm_weight.to(default_dtype)}
            )
        self.verifier_norm.weight.requires_grad = False

    def _build_parallel_hidden_states(
        self,
        hidden_states: torch.Tensor,
        depth_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Build combined hidden states for parallel group processing.

        Concatenates NTP hidden states with mask_hidden for MTP positions.

        Args:
            hidden_states: NTP hidden states [1, ntp_len, 3 * hidden_size].
            depth_indices: COD-sampled indices per depth.

        Returns:
            Combined hidden states [1, total_positions, 3 * hidden_size]
            where MTP positions use mask_hidden.
        """
        parts = [hidden_states]  # Start with depth 0 (NTP)

        for k in range(1, len(depth_indices)):
            n_positions = depth_indices[k].shape[0]
            if n_positions > 0:
                # Use learnable mask_hidden for MTP positions
                mtp_hidden = self.mask_hidden.expand(1, n_positions, -1)
                parts.append(mtp_hidden)

        return torch.cat(parts, dim=1)

    def _build_parallel_input_ids(
        self,
        input_ids: torch.Tensor,
        depth_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Build input IDs for parallel group processing.

        NTP positions use actual input IDs. MTP positions use ptd_token_id
        as placeholder.

        Args:
            input_ids: NTP input IDs [1, ntp_len].
            depth_indices: COD-sampled indices per depth.

        Returns:
            Combined input IDs [1, total_positions].
        """
        device = input_ids.device
        parts = [input_ids]  # Start with depth 0 (NTP)

        for k in range(1, len(depth_indices)):
            n_positions = depth_indices[k].shape[0]
            if n_positions > 0:
                # MTP positions get ptd_token_id
                mtp_ids = torch.full(
                    (1, n_positions), self.ptd_token_id,
                    dtype=input_ids.dtype, device=device,
                )
                parts.append(mtp_ids)

        return torch.cat(parts, dim=1)

    def _build_parallel_position_ids(
        self,
        position_ids: torch.Tensor,
        depth_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Build position IDs for parallel group processing.

        MTP positions use their original sequence positions (shifted by depth k)
        to maintain proper positional encoding.

        Args:
            position_ids: NTP position IDs [1, ntp_len].
            depth_indices: COD-sampled indices per depth.

        Returns:
            Combined position IDs [1, total_positions].
        """
        device = position_ids.device
        parts = [position_ids]  # Start with depth 0 (NTP)

        for k in range(1, len(depth_indices)):
            if depth_indices[k].numel() > 0:
                # MTP position IDs: original positions shifted by depth
                # depth k predicts token k+1 positions ahead
                mtp_pos = depth_indices[k].unsqueeze(0) + k + 1
                parts.append(mtp_pos.to(device))

        return torch.cat(parts, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # [1, seq_len]
        lengths: torch.Tensor | None = None,  # [batch_size]
        loss_mask: torch.Tensor | None = None,  # [1, seq_len]
        position_ids: torch.Tensor | None = None,  # [1, seq_len]
        verifier_last_hidden_states: torch.Tensor | None = None,
        para_num: int | None = None,
        down_sample_ratio: float | None = None,
        **kwargs,
    ):
        """Forward pass with parallel multi-token prediction.

        Performs COD sampling to generate parallel prediction groups, then
        processes all groups in a single forward pass through the decoder.

        Args:
            hidden_states: Concatenated verifier hidden states [1, seq_len, 3*H].
            input_ids: Input token IDs [1, seq_len].
            lengths: Per-document sequence lengths for attention masking.
            loss_mask: Boolean mask for valid training positions [1, seq_len].
            position_ids: Token position IDs [1, seq_len].
            verifier_last_hidden_states: Verifier output for loss computation.
            para_num: Override for number of parallel depths.
            down_sample_ratio: Override for COD retention rate.
            **kwargs: Additional keyword arguments.

        Returns:
            If verifier_last_hidden_states is provided (training):
                Tuple of (draft_tokens_list, loss, metrics_dict)
            Otherwise (inference):
                List of draft token predictions per depth.
        """
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        _para_num = para_num if para_num is not None else self.para_num
        _down_sample_ratio = (
            down_sample_ratio if down_sample_ratio is not None else
            self.down_sample_ratio
        )

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        # === COD Sampling ===
        sample_mask = (
            loss_mask.squeeze(0).bool()
            if loss_mask is not None
            else torch.ones(total_seq_len, dtype=torch.bool, device=device)
        )
        depth_indices = cod_sample(
            sample_mask,
            down_sample_ratio=_down_sample_ratio,
            num_depths=_para_num,
            down_sample_ratio_min=self.down_sample_ratio_min,
        )

        # === Build parallel inputs ===
        parallel_hidden = self._build_parallel_hidden_states(
            hidden_states, depth_indices
        )
        parallel_ids = self._build_parallel_input_ids(input_ids, depth_indices)
        parallel_pos = self._build_parallel_position_ids(position_ids, depth_indices)

        combined_len = parallel_hidden.shape[1]

        # === Attention mask ===
        mask_mod = build_parallel_group_mask_mod(
            lengths.to(device), depth_indices, combined_len,
        )
        attention_mask = create_block_mask(
            mask_mod,
            B=None, H=None,
            Q_LEN=combined_len,
            KV_LEN=combined_len,
            device=device,
        )

        # === Forward through decoder ===
        past_key_values = DynamicCache(
            config=self.config.transformer_layer_config
        )

        combined_hidden = self.fc(parallel_hidden)
        # [1, combined_len, hidden_size]

        with torch.no_grad():
            # For MTP positions, use mask_token_embedding instead of actual embeddings
            ntp_len = depth_indices[0].shape[0]
            input_embeds_ntp = self.embed_tokens(parallel_ids[:, :ntp_len])
            mtp_len = combined_len - ntp_len
            if mtp_len > 0:
                input_embeds_mtp = self.mask_token_embedding.expand(1, mtp_len, -1)
                input_embeds = torch.cat([input_embeds_ntp, input_embeds_mtp], dim=1)
            else:
                input_embeds = input_embeds_ntp

        combined_hidden = torch.cat([input_embeds, combined_hidden], dim=-1)
        # [1, combined_len, 2 * hidden_size]

        position_embeddings = self.rotary_emb(combined_hidden, parallel_pos)

        cache_position = torch.arange(
            combined_len, dtype=torch.long, device=device
        )

        for decoder_layer in self.layers:
            combined_hidden = decoder_layer(
                combined_hidden,
                attention_mask=attention_mask,
                position_ids=parallel_pos,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        logits = self.lm_head(self.norm(combined_hidden))
        # [1, combined_len, draft_vocab_size]

        # === Split logits by depth and compute loss ===
        return_loss = verifier_last_hidden_states is not None
        draft_tokens = []
        offset = 0

        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )

            loss = torch.tensor(0.0, device=device)
            metrics: dict[str, torch.Tensor] = {}

        for k in range(len(depth_indices)):
            n_k = depth_indices[k].shape[0]
            depth_logits = logits[:, offset:offset + n_k, :]
            draft_tokens.append(torch.argmax(depth_logits, dim=-1).detach().clone())

            if return_loss and n_k > 0:
                # Get targets at appropriate positions for this depth
                # Depth k predicts token at position i + k + 1
                idx = depth_indices[k]
                if k == 0:
                    # NTP: predict next token (shift by 1)
                    valid = idx < total_seq_len - 1
                    shifted_idx = torch.clamp(idx + 1, max=total_seq_len - 1)
                else:
                    # MTP depth k: predict token k+1 ahead
                    valid = idx + k + 1 < total_seq_len
                    shifted_idx = torch.clamp(idx + k + 1, max=total_seq_len - 1)

                if self.loss_type == "cross_entropy":
                    # Cross-entropy with target token IDs
                    depth_targets = torch.argmax(
                        targets[:, shifted_idx, :], dim=-1
                    )
                    depth_mask = valid.unsqueeze(0).float()
                    depth_loss = cross_entropy_loss(
                        depth_logits, depth_targets, depth_mask
                    )
                else:
                    # KL divergence with target distributions
                    depth_target_dist = targets[:, shifted_idx, :]
                    depth_mask = valid.unsqueeze(0).float()
                    depth_loss = kl_div_loss(
                        depth_logits, depth_target_dist, depth_mask
                    )

                loss = loss + depth_loss
                metrics[f"loss_depth_{k}"] = depth_loss.detach().clone()

                # Accuracy for this depth
                pred_ids = torch.argmax(depth_logits, dim=-1)
                target_ids = torch.argmax(targets[:, shifted_idx, :], dim=-1)
                correct = (pred_ids == target_ids).float()
                if valid.any():
                    acc = (correct * valid.unsqueeze(0).float()).sum() / (
                        valid.float().sum() + 1e-5
                    )
                else:
                    acc = torch.tensor(0.0, device=device)
                metrics[f"acc_depth_{k}"] = acc

            offset += n_k

        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "PEagleDraftModel":
        """Create P-EAGLE model from training arguments.

        Args:
            verifier_config: Verifier model configuration.
            **kwargs: Training arguments including P-EAGLE-specific params:
                - num_layers: Number of decoder layers
                - norm_before_residual: Whether to normalize before residual
                - t2d/d2t: Vocabulary mapping tensors
                - para_num: Number of parallel prediction depths
                - down_sample_ratio: COD retention rate
                - down_sample_ratio_min: Minimum retention rate
                - ptd_token_id: Padding token ID for MTP
                - loss_type: Loss function type
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized PEagleDraftModel.
        """
        config = PEagleSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            embed_requires_grad=kwargs.get("embed_requires_grad", True),
            para_num=kwargs.get("para_num", 8),
            down_sample_ratio=kwargs.get("down_sample_ratio", 0.5),
            down_sample_ratio_min=kwargs.get("down_sample_ratio_min", 0.0),
            ptd_token_id=kwargs.get("ptd_token_id", 0),
            loss_type=kwargs.get("loss_type", "cross_entropy"),
            speculators_config=SpeculatorsConfig(
                algorithm="peagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("para_num", 8),
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        return cls(config=config, t2d=kwargs.get("t2d"), d2t=kwargs.get("d2t"))

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for P-EAGLE.

        Args:
            **kwargs: Training arguments.

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs).
        """
        train_kwargs = {
            "para_num": kwargs.get("para_num", 8),
            "down_sample_ratio": kwargs.get("down_sample_ratio", 0.5),
        }
        val_kwargs = {
            "para_num": kwargs.get("para_num", 8),
            "down_sample_ratio": kwargs.get("down_sample_ratio", 0.5),
        }
        return train_kwargs, val_kwargs
