from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.domino import DominoHead
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
        # Domino head weights are absent when loading a plain DFlash checkpoint
        # (no Domino head trained). Suppress the spurious missing-key warning.
        "domino_head.prefix_gru.weight_ih_l0",
        "domino_head.prefix_gru.weight_hh_l0",
        "domino_head.embed_proj.0.weight",
        "domino_head.embed_proj.2.weight",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size
        self.projector_type = config.projector_type
        if self.projector_type == "domino":
            self.domino_head = DominoHead(
                hidden_size=self.hidden_size,
                gru_hidden_dim=config.gru_hidden_dim,
                emb_dim=config.emb_dim,
                draft_vocab_size=self.draft_vocab_size,
            )
        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)
        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": kwargs.get("sliding_window_non_causal", False),
            "projector_type": kwargs.get("projector_type", "dflash"),
            "pure_draft_prefix_len": kwargs.get("domino_pure_draft_prefix_len", 1),
            "emb_dim": kwargs.get("domino_emb_dim", 256),
            "gru_hidden_dim": kwargs.get("domino_gru_hidden_dim", 1024),
            "lambda_base_start": kwargs.get("domino_lambda_start", 1.0),
            "lambda_base_decay_ratio": kwargs.get("domino_lambda_decay_ratio", 1.0),
            "auf": kwargs.get("auf", False),
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    # First block position is the anchor, not emitted during gen.
                    GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        normalize_by_decay = kwargs.get("normalize_loss_by_decay", False)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "normalize_by_decay": normalize_by_decay,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @staticmethod
    def _auf_mask(
        logits: torch.Tensor,
        targets: torch.Tensor,
        base_mask: torch.Tensor,
        num_anchors: int,
        block_size: int,
        return_j_star: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the Accept-Until-Fail (AUF) loss mask for the base branch.

        Truncates the cross-entropy support at the first greedy prediction error
        within each block. Positions up to and including the first error (the
        "breaker" token j*) retain their gradient; positions strictly after j*
        are zeroed. This aligns training supervision with the prefix-acceptance
        semantics of the verifier and eliminates wasted capacity on unreachable
        suffix positions.

        Reference: Yang & Li, "Spec-AUF: Accept-Until-Fail Training under
        Train-Inference Misalignment for Masked Block Drafters" (arXiv 2607.01893).

        Args:
            logits: Base logits [1, T, vocab]. Detached internally — j* is not
                a gradient path.
            targets: Verifier logit distributions [1, T, vocab]. Detached
                internally.
            base_mask: Boolean validity mask [1, T] (e.g. domino_loss_mask).
            num_anchors: Number of anchor blocks.
            block_size: Tokens per block.
            return_j_star: If True, also return j* (first error position) per block.

        Returns:
            If return_j_star is False:
                Boolean mask [1, T] with the same shape as base_mask, zeroed out
                for all positions strictly after the first error in each block.
            If return_j_star is True:
                Tuple of (mask, j_star) where j_star is [num_anchors] tensor
                containing the first error position (0-indexed) per block.
                j* = block_size means no error (all accepted).
        """
        base_preds_4d = (
            logits.detach().argmax(dim=-1).reshape(1, num_anchors, block_size)
        )
        target_ids_4d = (
            targets.detach().argmax(dim=-1).reshape(1, num_anchors, block_size)
        )
        mask_4d = base_mask.bool().reshape(1, num_anchors, block_size)
        errors = (base_preds_4d != target_ids_4d) & mask_4d
        error_floats = errors.float()
        running_errors = error_floats.cumsum(dim=-1)
        # errors_before == 0: keep accepted prefix + breaker token j*
        # errors_before >= 1: zero the unreachable suffix
        errors_before = running_errors - error_floats
        auf_mask = (mask_4d & (errors_before == 0)).reshape_as(base_mask)

        if not return_j_star:
            return auf_mask

        # j* = first error position per block (0-indexed within block)
        # If no error, j* = block_size (all positions accepted)
        first_error = errors.float().argmax(dim=-1)  # [1, num_anchors]
        has_error = errors.any(dim=-1)  # [1, num_anchors]
        j_star = torch.where(
            has_error, first_error, torch.full_like(first_error, block_size)
        )
        return auf_mask, j_star.squeeze(0)  # [num_anchors]

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        **kwargs,
    ):
        """Run the anchored-block draft transformer up to the draft logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. DSpark reuses this and adds its Markov and
        confidence heads before computing its own loss.
        """
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            # Shift right by 1 so verifier_logits[i] predicts token at position i
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            target_indices = anchored_block_indices + (
                1 if self.config.shift_label else 0
            )
            target_indices = target_indices.clamp(max=verifier_logits.shape[1] - 1)
            targets = verifier_logits[:, target_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        # shape: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        aligned_loss_mask[:, :: self.block_size] = 0

        if self.config.projector_type == "domino":
            oob = (anchored_block_indices + 1) >= verifier_logits.shape[1]
            aligned_loss_mask[:, oob] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @torch.compiler.disable
    def _compute_domino_metrics(
        self,
        *,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        loss_config: "LossConfig | None",
        gamma: float,
        normalize_by_decay: bool,
        global_step: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, dict]:
        """Compute Domino dual-branch loss and metrics."""
        decay_ratio = self.config.lambda_base_decay_ratio
        if decay_ratio > 0 and total_steps > 0:
            decay_steps = int(total_steps * decay_ratio)
            progress = min(global_step / decay_steps, 1.0)
            lambda_base = self.config.lambda_base_start * (1.0 - progress)
            lambda_base = max(0.0, min(1.0, lambda_base))
        else:
            lambda_base = self.config.lambda_base_start

        num_anchors = hidden.shape[1] // self.block_size
        hidden_4d = hidden.reshape(1, num_anchors, self.block_size, -1)
        base_logits_4d = logits.reshape(1, num_anchors, self.block_size, -1)

        prev_token_ids_4d = input_ids[:, anchored_block_indices].reshape(
            1, num_anchors, self.block_size
        )
        refined_logits = self.domino_head(
            hidden_states_4d=hidden_4d,
            base_logits_4d=base_logits_4d,
            prev_token_ids=prev_token_ids_4d,
            suffix_start=self.config.pure_draft_prefix_len,
            embed_tokens=self.embed_tokens,
        ).reshape(1, num_anchors * self.block_size, -1)

        domino_loss_mask = aligned_loss_mask.clone()
        anchor_pos_in_mask = anchored_block_indices[:: self.block_size]
        domino_loss_mask[:, :: self.block_size] = loss_mask[:, anchor_pos_in_mask]

        # B-AUF+D (arXiv 2607.01893): L_final keeps full mask;
        # L_base is optionally truncated.
        j_star = None
        if self.config.auf:
            base_mask, j_star = self._auf_mask(
                logits,
                targets,
                domino_loss_mask,
                num_anchors,
                self.block_size,
                return_j_star=True,
            )
        else:
            base_mask = domino_loss_mask

        base_loss, base_metrics = compute_metrics(
            logits,
            targets,
            base_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            normalize_by_decay=normalize_by_decay,
            decay_mode="domino",
        )
        final_loss, final_metrics = compute_metrics(
            refined_logits,
            targets,
            domino_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            normalize_by_decay=normalize_by_decay,
            decay_mode="domino",
        )
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        ones = torch.tensor(1.0, device=logits.device)
        metrics = {
            "loss_sum": loss.detach().clone(),
            "loss_total": ones,
            "base_loss_sum": base_loss.detach().clone(),
            "base_loss_total": ones,
            "full_acc_sum": base_metrics["full_acc_sum"],
            "full_acc_total": base_metrics["full_acc_total"],
            "final_loss_sum": final_loss.detach().clone(),
            "final_loss_total": ones,
            "final_full_acc_sum": final_metrics["full_acc_sum"],
            "final_full_acc_total": final_metrics["full_acc_total"],
            "lambda_base_sum": torch.tensor(lambda_base, device=logits.device),
            "lambda_base_total": ones,
            **{
                k: v
                for k, v in final_metrics.items()
                if k.startswith(("position_", "eal"))
            },
        }

        # AUF observability: mean j* (first error position) - proxy for acceptance
        if j_star is not None:
            metrics["j_star_sum"] = j_star.float().sum()
            metrics["j_star_total"] = torch.tensor(
                float(j_star.numel()), device=logits.device
            )

        return loss, metrics

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        normalize_by_decay: bool = False,
        global_step: int = 0,
        total_steps: int = 0,
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

        draft_tokens = torch.argmax(logits, dim=-1)

        if self.projector_type == "domino":
            loss, metrics = self._compute_domino_metrics(
                hidden=hidden,
                logits=logits,
                targets=targets,
                input_ids=input_ids,
                loss_mask=loss_mask,
                aligned_loss_mask=aligned_loss_mask,
                anchored_block_indices=anchored_block_indices,
                loss_config=loss_config,
                gamma=gamma,
                normalize_by_decay=normalize_by_decay,
                global_step=global_step,
                total_steps=total_steps,
            )
        else:
            j_star = None
            if self.config.auf:
                num_anchors = aligned_loss_mask.shape[1] // self.block_size
                auf_loss_mask, j_star = self._auf_mask(
                    logits,
                    targets,
                    aligned_loss_mask,
                    num_anchors,
                    self.block_size,
                    return_j_star=True,
                )
            else:
                auf_loss_mask = aligned_loss_mask

            loss, metrics = compute_metrics(
                logits,
                targets,
                auf_loss_mask,
                self.block_size,
                gamma=gamma,
                loss_config=loss_config,
                normalize_by_decay=normalize_by_decay,
                per_position_loss_weight=per_position_loss_weight,
                dpace_alpha=dpace_alpha,
            )

            if j_star is not None:
                metrics["j_star_sum"] = j_star.float().sum()
                metrics["j_star_total"] = torch.tensor(
                    float(j_star.numel()), device=logits.device
                )

        return draft_tokens, loss, metrics
