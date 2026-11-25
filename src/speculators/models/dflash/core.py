import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.components import BackboneInputs, BlockLayout
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.backbone import DFlashBackbone
from speculators.models.dflash.metrics import (
    compute_metrics,
    compute_metrics_from_hidden,
)
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    rename_checkpoint_key,
    select_anchors,
)
from speculators.models.utils import (
    conditional_torch_compile,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)

logger = logging.getLogger(__name__)

_BACKBONE_CHECKPOINT_PREFIXES = {
    name: f"backbone.{name}"
    for name in ("layers", "fc", "hidden_norm", "norm", "rotary_emb")
}

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    supports_gradient_checkpointing = True  # noqa: D003  # Qwen3DFlashDecoderLayer inherits GradientCheckpointingLayer
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        # lm_head is handled per-instance in __init__: omitted only for full-vocab
        # drafts, where it is the exact frozen verifier projection.
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

    def _make_backbone(self, config: DFlashSpeculatorConfig) -> DFlashBackbone:
        return DFlashBackbone(
            config.transformer_layer_config, len(self.target_layer_ids)
        )

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
        self.backbone = self._make_backbone(config)
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        sliding_layers = self.backbone.sliding_window_indices
        self.uses_sliding_window_attn = bool(sliding_layers)
        self.uses_full_attn = len(sliding_layers) < len(self.layers)

        self.verifier_norm = resolve_verifier_norm_class(config)(
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
        self.backbone.post_init()
        # Verifier-owned weights are reconstructed on load. Keep a reduced-vocab
        # lm_head serialized because current runtimes cannot derive it from the
        # full verifier head.
        #
        # Shadow the ClassVar lists with per-instance copies so full- and
        # reduced-vocabulary siblings cannot mutate each other's save rules.
        keys_to_ignore_on_save = list(type(self)._keys_to_ignore_on_save)  # noqa: SLF001
        keys_to_ignore_on_load_missing = list(
            type(self)._keys_to_ignore_on_load_missing  # noqa: SLF001
        )
        keys_to_ignore_on_save.append("embed_tokens.weight")
        if not self.use_draft_vocab:
            keys_to_ignore_on_save.append("lm_head.weight")
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        mapping = {
            rf"^{source}\.": f"{target}."
            for source, target in _BACKBONE_CHECKPOINT_PREFIXES.items()
        }
        mapping.update(kwargs.pop("key_mapping", {}) or {})
        return super().from_pretrained(*args, key_mapping=mapping, **kwargs)

    def remap_checkpoint_keys_for_load(self, state_dict):
        """Translate flat checkpoint names for direct training-state loads."""
        return {
            rename_checkpoint_key(name, _BACKBONE_CHECKPOINT_PREFIXES): tensor
            for name, tensor in state_dict.items()
        }

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        push_to_hub=False,
        max_shard_size="50GB",
        variant=None,
        token=None,
        save_peft_format=True,
        save_original_format=True,  # noqa: ARG002 - model type owns conversion
        distributed_checkpoint=False,
        **kwargs,
    ):
        """Save backbone weights under the existing flat checkpoint names."""
        state_dict = self.state_dict() if state_dict is None else state_dict
        save_prefixes = {
            target: source for source, target in _BACKBONE_CHECKPOINT_PREFIXES.items()
        }
        exported = {
            rename_checkpoint_key(name, save_prefixes): tensor
            for name, tensor in state_dict.items()
        }
        return super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=exported,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            save_original_format=False,
            distributed_checkpoint=distributed_checkpoint,
            **kwargs,
        )

    @property
    def layers(self) -> nn.ModuleList:
        """Expose decoder layers to the training infrastructure."""
        return self.backbone.layers

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def load_verifier_weights(self):
        """Reconstruct weights intentionally omitted from DFlash checkpoints."""
        self._load_verifier_weights(
            overwrite_embed_tokens=True,
            overwrite_lm_head=not self.use_draft_vocab,
        )

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
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        if kwargs.get("causal_attention") is not None:
            verifier_config.is_causal = kwargs["causal_attention"]
        block_size = kwargs.get("block_size", 8)

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )
        default_non_causal = algorithm == "dflash2"
        non_causal_arg = kwargs.get("sliding_window_non_causal")
        sliding_window_non_causal = (
            default_non_causal if non_causal_arg is None else non_causal_arg
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
            "sliding_window_non_causal": sliding_window_non_causal,
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
        implementation = kwargs.get("loss_implementation", "fused")
        # fused_linear computes the same fused losses from hidden states; the
        # config is resolved as "fused" and forward takes the hidden-state path.
        fused_linear = implementation == "fused_linear"
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], "fused" if fused_linear else implementation
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 512)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "fused_linear": fused_linear,
        }
        return dict(shared), dict(shared)

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
    ):
        """Honor explicit causality, preserving legacy per-layer defaults otherwise."""
        is_causal = getattr(self.config.transformer_layer_config, "is_causal", None)
        if is_causal is None:
            is_causal = (
                sliding_window is not None and not self.config.sliding_window_non_causal
            )
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            is_causal=is_causal,
            sliding_window=sliding_window,
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
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _build_targets(
        self,
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        block_positions: torch.Tensor,  # [num_blocks * block_size]
    ) -> torch.Tensor:  # [1, num_blocks * block_size, draft_vocab_size]
        """Build verifier-logit targets for the selected draft block positions."""
        total_seq_len = verifier_last_hidden_states.shape[1]

        with torch.no_grad():
            # If there are fewer block positions than base positions, compute
            # the verifier head only for the positions used as targets.
            if block_positions.numel() < total_seq_len:
                target_indices = (
                    block_positions
                    if self.config.sample_from_anchor
                    else (block_positions - 1) % total_seq_len
                )
                return self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )

            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            if not self.config.sample_from_anchor:
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            return verifier_logits[:, block_positions]

    def prepare_backbone_inputs(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        max_anchors: int = 512,
    ) -> tuple[BackboneInputs, BlockLayout]:
        """Prepare embeddings, positions, and masks for the backbone."""
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
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

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

        layout = BlockLayout(
            anchor_positions,
            anchored_block_indices,
            aligned_loss_mask,
            self.block_size,
            self.config.sample_from_anchor,
        )
        inputs = BackboneInputs(
            noise_embedding,
            hidden_states,
            position_ids,
            sliding_window_attn_mask,
            full_attn_mask,
        )
        return inputs, layout

    @conditional_torch_compile
    def forward(
        self,
        # [1, total_seq_len, num_target_layers * hidden_size]
        hidden_states: torch.Tensor,
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
        materialize_logits: bool = True,
        *,
        fused_linear: bool = False,
        correction_kwargs=None,
        loss_kwargs=None,
        **backbone_kwargs,
    ):
        inputs, layout = self.prepare_backbone_inputs(
            hidden_states, input_ids, loss_mask, document_ids, position_ids, max_anchors
        )
        hidden = self.backbone(*inputs, **backbone_kwargs)
        logits = (
            self.lm_head(hidden) if materialize_logits and not fused_linear else hidden
        )
        targets = self._build_targets(
            verifier_last_hidden_states, layout.block_positions
        )
        correction = self.apply_training_correction(
            hidden,
            logits,
            targets,
            layout,
            input_ids,
            document_ids,
            **(correction_kwargs or {}),
        )
        loss, metrics = self.compute_loss_and_metrics(
            logits,
            targets,
            layout,
            correction,
            fused_linear=fused_linear,
            loss_config=loss_config,
            gamma=gamma,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            **(loss_kwargs or {}),
        )
        return None, loss, metrics

    def apply_training_correction(self, *_args, **_kwargs):
        """Prepare algorithm-specific outputs for the training loss."""
        return None  # noqa: RET501 - plain DFlash has no correction output

    def compute_loss_and_metrics(
        self,
        logits,
        targets,
        layout: BlockLayout,
        correction,  # noqa: ARG002 - plain DFlash has no correction output
        *,
        fused_linear: bool = False,
        loss_config: LossConfig | None,
        gamma: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
        **_kwargs,
    ):
        metric_kwargs = {
            "gamma": gamma,
            "loss_config": loss_config,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "sample_from_anchor": self.config.sample_from_anchor,
        }
        if fused_linear:
            return compute_metrics_from_hidden(
                logits,  # backbone hidden states in fused-linear mode
                self.lm_head.weight,
                targets,
                layout.loss_mask,
                self.block_size,
                **metric_kwargs,
            )

        return compute_metrics(
            logits, targets, layout.loss_mask, self.block_size, **metric_kwargs
        )
