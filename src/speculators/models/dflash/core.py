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
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.utils import resolve_target_layer_ids


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
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
            create_block_mask
            if self._attn_impl == "simple_flex_attention"
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
        self.post_init()

        # Re-initialize lm_head weights to NaN after post_init() resets them to zeros.
        # This ensures load_verifier_weights() will load them from the verifier model.
        # We can't check isnan() on meta tensors, so we always re-initialize.
        if hasattr(self, "embed_tokens"):
            torch.nn.init.constant_(self.embed_tokens.weight, torch.nan)
        if hasattr(self, "lm_head"):
            torch.nn.init.constant_(self.lm_head.weight, torch.nan)
        if hasattr(self, "verifier_lm_head"):
            torch.nn.init.constant_(self.verifier_lm_head.weight, torch.nan)

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def load_verifier_weights(self) -> None:
        """Load verifier model weights, always loading lm_head from verifier.

        Overrides base class to always load lm_head from verifier, even if it's
        not NaN (since checkpoint may have zeros from default initialization).
        """
        import warnings  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        speculators_config = getattr(
            getattr(self, "config", None), "speculators_config", None
        )
        if speculators_config is None:
            return
        verifier_config = speculators_config.verifier
        if verifier_config.name_or_path is None:
            return

        # Determine which weights to load based on model attributes
        weights_to_load = ["embed_tokens.weight", "lm_head.weight"]
        if hasattr(self, "verifier_norm"):
            weights_to_load.append("model.norm.weight")

        verifier_weights = load_model_layers(
            weights_to_load,
            verifier_config.name_or_path,
        )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)

        # Load embed_tokens if not already loaded (NaN or all zeros means uninitialized)
        if self.embed_tokens.weight.isnan().any() or (self.embed_tokens.weight == 0).all():
            self.embed_tokens.load_state_dict({"weight": embed_tokens_weight})

        if self.use_draft_vocab:
            if self.t2d is None or not torch.any(self.t2d).item():  # type: ignore[arg-type]
                raise ValueError(
                    "t2d tensor hasn't been set. Please call "
                    "`.load_vocab_mappings(t2d, d2t)` before `.load_verifier_weights()`"
                )
            lm_head_weight = lm_head_weight[
                self.t2d.to(device=lm_head_weight.device, dtype=torch.bool), :  # type: ignore[union-attr,index]
            ]

        # Always load lm_head from verifier (don't check for NaN)
        # This ensures we get the correct weights even if checkpoint has zeros
        self.lm_head.load_state_dict(
            {"weight": lm_head_weight.detach().clone()}, strict=False
        )
        self.verifier_lm_head.load_state_dict(
            {"weight": lm_head_weight.detach().clone()}, strict=False
        )

        # Load verifier norm weights if the model has verifier_norm
        if hasattr(self, "verifier_norm"):
            if "model.norm.weight" not in verifier_weights:
                warnings.warn(
                    f"Could not find final norm weights in "
                    f"{verifier_config.name_or_path}. "
                    "Using default initialization (weight=1.0).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                verifier_norm_sd = {"weight": verifier_weights["model.norm.weight"]}
                self.verifier_norm.load_state_dict(verifier_norm_sd)  # type: ignore[attr-defined]

        # HF's from_pretrained resets requires_grad=True on all parameters.
        # Re-freeze verifier weights that should never be trained.
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        if hasattr(self, "verifier_norm"):
            self.verifier_norm.weight.requires_grad_(False)  # type: ignore[attr-defined]

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
                - max_anchors: Max anchor positions during training (default: 256)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        # Resolve target layer IDs if not provided
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )

        config = DFlashSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            max_anchors=kwargs.get("max_anchors", 3072),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            speculators_config=SpeculatorsConfig(
                algorithm="dflash",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        # DFlash first position is anchor position, not used during gen
                        speculative_tokens=kwargs.get("block_size", 8) - 1,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        return {"loss_fn": loss_fn}, {"loss_fn": loss_fn}

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    def _create_attention_mask(
        self,
        lengths: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            lengths=lengths.to(device),
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
    def _build_attention_mask(self, loss_mask, lengths, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, self.config.max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                lengths=lengths,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                lengths=lengths,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    @torch.compiler.disable
    def _build_inference_aligned_mask(self, total_seq_len, device, sliding_window=None):
        """Build attention masks for inference-aligned single-block forward.

        Produces masks where query tokens (one block of block_size tokens) may
        attend to ALL context positions, with CAUSAL attention within the block.
        This matches vLLM's DFlash inference behavior where causal=True is set
        in the CommonAttentionMetadata (see llm_base_proposer.py:1067).
        """
        block_size = self.block_size
        q_len = block_size
        kv_len = total_seq_len + q_len

        def base_context_mod(_b, _h, q_idx, kv_idx):
            kv_is_context = kv_idx < total_seq_len
            if sliding_window is not None:
                q_pos = total_seq_len + q_idx
                kv_pos = kv_idx
                in_window = kv_pos >= q_pos - sliding_window
                return kv_is_context & in_window
            return kv_is_context

        def block_mod(_b, _h, q_idx, kv_idx):
            # CAUSAL within block: attend to block positions up to q_idx
            kv_is_block = kv_idx >= total_seq_len
            kv_block_idx = kv_idx - total_seq_len
            return kv_is_block & (kv_block_idx <= q_idx)

        from torch.nn.attention.flex_attention import or_masks
        mask_mod = or_masks(base_context_mod, block_mod)

        mask = self._create_mask_fn(
            mask_mod,
            B=1,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        return mask

    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_fn=kl_div_loss,
        return_intermediates: bool = False,  # whether to return intermediate tensors
        inference_aligned: bool = False,  # match vLLM inference single-block behavior
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        # Initialize intermediates dict if requested
        intermediates = {} if return_intermediates else None

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        if inference_aligned:
            return self._forward_inference_aligned(
                hidden_states=hidden_states,
                input_ids=input_ids,
                total_seq_len=total_seq_len,
                position_ids=position_ids,
                return_intermediates=return_intermediates,
                **kwargs,
            )

        num_anchors = self.config.max_anchors

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, lengths, device)
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
        if return_intermediates:
            intermediates["embed_output"] = noise_embedding

        # FC + hidden_norm on context (same as training)
        fc_output_raw = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output_raw)
        # shape: [1, total_seq_len, hidden_size]
        if return_intermediates:
            intermediates["fc_output"] = fc_output_raw
            intermediates["hidden_norm_output"] = fc_output

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            # Shift right by 1 so verifier_logits[i] predicts token at position i
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        # Store layer outputs if requested
        if return_intermediates:
            intermediates["layer_outputs"] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                return_intermediates=return_intermediates,
                **kwargs,
            )
            
            if return_intermediates:
                noise_embedding, attn_intermediates = layer_output
                intermediates["layer_outputs"].append(noise_embedding)
                intermediates[f"layer_{layer_idx}_attn"] = attn_intermediates
            else:
                noise_embedding = layer_output

        final_normed = self.norm(noise_embedding)
        logits = self.lm_head(final_normed)
        # shape: [1, num_anchors*block_size, vocab_size]
        if return_intermediates:
            intermediates["final_norm_output"] = final_normed
            intermediates["logits"] = logits

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        aligned_loss_mask[:, :: self.block_size] = 0
        loss, metrics = compute_metrics(
            logits, targets, aligned_loss_mask, self.block_size, loss_fn=loss_fn
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        if return_intermediates:
            return {
                "draft_tokens": draft_tokens,
                "loss": loss,
                "metrics": metrics,
                **intermediates
            }

        return draft_tokens, loss, metrics

    def _forward_inference_aligned(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        total_seq_len: int,
        position_ids: torch.Tensor,
        return_intermediates: bool,
        **kwargs,
    ):
        """Single-block forward matching vLLM inference behavior.

        Processes only ``block_size`` query tokens (1 bonus + block_size-1
        mask) attending to ALL context positions. This produces outputs at
        the same positions as vLLM's speculative decoding inference step.
        """
        device = hidden_states.device
        block_size = self.block_size
        intermediates = {} if return_intermediates else None

        # FC + hidden_norm on context (same as training)
        fc_output_raw = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output_raw)
        if return_intermediates:
            intermediates["fc_output"] = fc_output_raw
            intermediates["hidden_norm_output"] = fc_output

        # Embed block_size query tokens: first token = bonus (last input),
        # rest = mask tokens
        mask_token_ids = torch.full(
            (1, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        mask_token_ids[:, 0] = input_ids[:, -1]  # bonus token
        noise_embedding = self.embed_tokens(mask_token_ids)
        if return_intermediates:
            intermediates["embed_output"] = noise_embedding
            intermediates["mask_token_ids"] = mask_token_ids
            intermediates["input_ids"] = input_ids  # Save for debugging

        # Position IDs: context positions + sequential block positions
        # Use 0-indexed positions to match vLLM's inference engine
        # (vLLM uses target_positions which are 0-indexed)
        ctx_positions = torch.arange(
            total_seq_len, device=device
        ).unsqueeze(0)  # [0, 1, ..., total_seq_len-1]
        last_ctx_pos = ctx_positions[:, -1]  # total_seq_len - 1
        block_positions = last_ctx_pos + 1 + torch.arange(
            block_size, device=device
        ).unsqueeze(0)  # [total_seq_len, total_seq_len+1, ..., total_seq_len+block_size-1]
        combined_position_ids = torch.cat([ctx_positions, block_positions], dim=1)

        position_embeddings = self.rotary_emb(hidden_states, combined_position_ids)

        # Build inference-aligned masks
        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._build_inference_aligned_mask(
                total_seq_len, device, sliding_window=None
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._build_inference_aligned_mask(
                total_seq_len, device, sliding_window=self.sliding_window
            )

        if return_intermediates:
            intermediates["layer_outputs"] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=combined_position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                return_intermediates=return_intermediates,
                **kwargs,
            )
            
            if return_intermediates:
                noise_embedding, attn_intermediates = layer_output
                intermediates["layer_outputs"].append(noise_embedding)
                intermediates[f"layer_{layer_idx}_attn"] = attn_intermediates
            else:
                noise_embedding = layer_output

        final_normed = self.norm(noise_embedding)
        logits = self.lm_head(final_normed)
        if return_intermediates:
            intermediates["final_norm_output"] = final_normed
            intermediates["logits"] = logits

        draft_tokens = torch.argmax(logits, dim=-1)

        if return_intermediates:
            return {
                "draft_tokens": draft_tokens,
                **intermediates,
            }

        return draft_tokens
