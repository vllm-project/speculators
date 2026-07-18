import logging
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
from speculators.models.dflash.metrics import compute_fused_ce_metrics, compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids
from speculators.ops.fused_linear_cross_entropy import (
    frozen_linear_cross_entropy,
    validate_liger_installation,
)

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


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
            _compiled_create_block_mask
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

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
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

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1

        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": kwargs.get("sliding_window_non_causal", False),
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
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
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        linear_ce_backend = kwargs.get("dflash_linear_cross_entropy_backend", "torch")
        compact_ce_rows = kwargs.get("dflash_compact_zero_weight_ce_rows", False)
        label_source = kwargs.get("dflash_label_source", "verifier_argmax")
        verifier_argmax_chunk_size = kwargs.get("dflash_verifier_argmax_chunk_size", 0)
        if linear_ce_backend not in {"liger", "torch"}:
            raise ValueError(
                "dflash_linear_cross_entropy_backend must be 'torch' or 'liger'"
            )
        if label_source not in {"input_ids", "verifier_argmax"}:
            raise ValueError(
                "dflash_label_source must be 'verifier_argmax' or 'input_ids'"
            )
        if verifier_argmax_chunk_size < 0:
            raise ValueError("dflash_verifier_argmax_chunk_size must be non-negative")
        if compact_ce_rows and linear_ce_backend != "liger":
            raise ValueError(
                "dflash_compact_zero_weight_ce_rows requires "
                "dflash_linear_cross_entropy_backend='liger'"
            )
        if (
            label_source != "verifier_argmax" or verifier_argmax_chunk_size
        ) and linear_ce_backend != "liger":
            raise ValueError(
                "DFlash label-source and verifier-argmax chunking options require "
                "dflash_linear_cross_entropy_backend='liger'"
            )
        if linear_ce_backend == "liger":
            if set(loss_config) != {"ce"}:
                raise ValueError(
                    "dflash_linear_cross_entropy_backend='liger' requires an "
                    "exactly-CE --loss-fn configuration"
                )
            validate_liger_installation()
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "linear_cross_entropy_backend": linear_ce_backend,
            "compact_zero_weight_ce_rows": compact_ce_rows,
            "label_source": label_source,
            "verifier_argmax_chunk_size": verifier_argmax_chunk_size,
        }
        return dict(shared), dict(shared)

    def prepare_fused_linear_cross_entropy(self, compute_dtype: torch.dtype) -> None:
        """Prepare the ignored verifier head as the fused CE compute weight."""

        if not torch.equal(
            self.lm_head.weight.detach(), self.verifier_lm_head.weight.detach()
        ):
            raise RuntimeError(
                "Liger CE requires identical frozen draft and verifier LM heads; "
                "the loaded checkpoint and verifier weights differ"
            )
        # This head is excluded from checkpoints. Casting it once avoids a persistent
        # extra weight copy while matching autocast's compute precision.
        self.verifier_lm_head.to(dtype=compute_dtype)

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
        target_ids_only: bool = False,
        materialize_draft_logits: bool = True,
        label_source: str = "verifier_argmax",
        verifier_argmax_chunk_size: int = 0,
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

        if target_ids_only:
            targets = self._ce_target_ids(
                input_ids,
                verifier_last_hidden_states,
                anchored_block_indices,
                label_source=label_source,
                verifier_argmax_chunk_size=verifier_argmax_chunk_size,
            )
        else:
            with torch.no_grad():
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                if not self.config.sample_from_anchor:
                    # False: shift right so slot j predicts token at position j.
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                targets = verifier_logits[:, anchored_block_indices]

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
        logits = self.lm_head(hidden) if materialize_draft_logits else None

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @torch.no_grad()
    def _ce_target_ids(
        self,
        input_ids: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        *,
        label_source: str,
        verifier_argmax_chunk_size: int,
    ) -> torch.Tensor:
        if label_source == "input_ids":
            if self.use_draft_vocab:
                raise ValueError(
                    "dflash_label_source='input_ids' requires the full verifier "
                    "vocabulary"
                )
            label_indices = anchored_block_indices
            if self.config.sample_from_anchor:
                label_indices = label_indices + 1
            return input_ids[:, label_indices]
        if label_source != "verifier_argmax":
            raise ValueError(f"unsupported DFlash label source: {label_source!r}")

        normalized = self.verifier_norm(verifier_last_hidden_states)
        sequence_length = normalized.shape[1]
        if (
            verifier_argmax_chunk_size == 0
            or verifier_argmax_chunk_size >= sequence_length
        ):
            verifier_ids = self.verifier_lm_head(normalized).argmax(dim=-1)
        else:
            verifier_ids = torch.cat(
                [
                    self.verifier_lm_head(
                        normalized[:, start : start + verifier_argmax_chunk_size]
                    ).argmax(dim=-1)
                    for start in range(0, sequence_length, verifier_argmax_chunk_size)
                ],
                dim=1,
            )
        if not self.config.sample_from_anchor:
            verifier_ids = torch.roll(verifier_ids, 1, dims=1)
        return verifier_ids[:, anchored_block_indices]

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
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        linear_cross_entropy_backend: str = "torch",
        compact_zero_weight_ce_rows: bool = False,
        label_source: str = "verifier_argmax",
        verifier_argmax_chunk_size: int = 0,
        **kwargs,
    ):
        use_fused_ce = linear_cross_entropy_backend == "liger"
        hidden, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            target_ids_only=use_fused_ce,
            materialize_draft_logits=not use_fused_ce,
            label_source=label_source,
            verifier_argmax_chunk_size=verifier_argmax_chunk_size,
            **kwargs,
        )
        if use_fused_ce:
            if loss_config is None or set(loss_config) != {"ce"}:
                raise ValueError("Liger DFlash CE requires an exactly-CE loss config")
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            flat_targets = targets.reshape(-1)
            flat_mask = aligned_loss_mask.reshape(-1)
            active_indices = None
            if compact_zero_weight_ce_rows:
                active_indices = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
                if active_indices.numel() > 0:
                    flat_hidden = flat_hidden.index_select(0, active_indices)
                    flat_targets = flat_targets.index_select(0, active_indices)

            if active_indices is not None and active_indices.numel() == 0:
                loss_per_token = hidden.reshape(-1, hidden.shape[-1]).sum(dim=-1) * 0
                token_accuracy = torch.zeros_like(loss_per_token)
            else:
                compute_weight = self.verifier_lm_head.weight.detach()
                flat_hidden = flat_hidden.to(compute_weight.dtype)
                loss_per_token, token_accuracy = frozen_linear_cross_entropy(
                    flat_hidden,
                    compute_weight,
                    flat_targets,
                )
                if active_indices is not None:
                    loss_per_token = loss_per_token.new_zeros(
                        flat_mask.shape
                    ).index_copy(0, active_indices, loss_per_token)
                    token_accuracy = token_accuracy.new_zeros(
                        flat_mask.shape
                    ).index_copy(0, active_indices, token_accuracy)
            return (
                None,
                *compute_fused_ce_metrics(
                    loss_per_token,
                    token_accuracy,
                    aligned_loss_mask,
                    self.block_size,
                    loss_weight=loss_config["ce"][1],
                    gamma=gamma,
                    per_position_loss_weight=per_position_loss_weight,
                    dpace_alpha=dpace_alpha,
                    sample_from_anchor=self.config.sample_from_anchor,
                ),
            )

        if linear_cross_entropy_backend != "torch":
            raise ValueError(
                "linear_cross_entropy_backend must be either 'torch' or 'liger'"
            )
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
