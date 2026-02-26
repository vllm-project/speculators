"""FastMTP speculator model implementation."""

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp.config import FastMTPConfig
from speculators.models.fast_mtp.model_definitions import get_fast_mtp_components
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["FastMTPLayer", "FastMTPSpeculator"]


class FastMTPLayer(nn.Module):
    """Single FastMTP layer combining verifier hidden states with token embeddings.

    Architecture flow: normalize inputs separately (hidden states and embeddings),
    concatenate and project to hidden_size, apply self-attention with residual,
    apply MLP with residual, and final normalization.

    Attribute names must match checkpoint key structure exactly. The component
    registry ensures correct module types for different architectures.

    :param config: FastMTP configuration
    :param layer_idx: Layer index (0-indexed)
    """

    def __init__(self, config: FastMTPConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        model_type = config.transformer_config.model_type
        components = get_fast_mtp_components(model_type)
        arch_config = self._create_architecture_config(config, components)

        norm_class = components.norm_class
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.token_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        self.input_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.self_attn = components.attention_class(arch_config, layer_idx=layer_idx)
        self.mlp = components.mlp_class(arch_config)

    def _create_architecture_config(self, config: FastMTPConfig, components):
        """Convert generic FastMTPConfig to architecture-specific config.

        Different architectures expect different config classes. This builds
        the appropriate config object (e.g., Qwen2Config) from FastMTPConfig.

        :param config: Generic FastMTP configuration
        :param components: FastMTP component registry for the architecture
        :return: Architecture-specific configuration object
        """
        config_class = config.transformer_config.__class__

        arch_config = config_class(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            attention_bias=components.attention_bias,
        )

        if components.use_moe:
            if hasattr(config.transformer_config, "num_experts"):
                arch_config.num_experts = config.transformer_config.num_experts
            if hasattr(config.transformer_config, "num_experts_per_tok"):
                arch_config.num_experts_per_tok = (
                    config.transformer_config.num_experts_per_tok
                )
            if hasattr(config.transformer_config, "moe_intermediate_size"):
                arch_config.moe_intermediate_size = (
                    config.transformer_config.moe_intermediate_size
                )

        return arch_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through FastMTP layer.

        :param hidden_states: Hidden states from verifier [batch, seq_len, hidden_size]
        :param token_embeddings: Token embeddings [batch, seq_len, hidden_size]
        :param attention_mask: Optional attention mask
        :param position_ids: Optional position IDs
        :param kwargs: Additional arguments
        :return: Output hidden states [batch, seq_len, hidden_size]
        """
        hidden_states_norm = self.hidden_layernorm(hidden_states)
        token_embeddings_norm = self.token_layernorm(token_embeddings)

        combined = torch.cat([hidden_states_norm, token_embeddings_norm], dim=-1)
        proj_hidden = self.input_proj(combined)
        proj_hidden = self.input_layernorm(proj_hidden)

        residual = proj_hidden
        attn_output, _ = self.self_attn(
            hidden_states=proj_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return self.final_layernorm(hidden_states)


@SpeculatorModel.register("mtp")
class FastMTPSpeculator(SpeculatorModel):
    """FastMTP speculator model for multi-token prediction.

    FastMTP predicts multiple future tokens (default: 3) per forward pass using
    a single layer with weighted multi-step loss for training.

    Architecture components:
    - Single FastMTP layer (mtp_layers.0)
    - Shared embed_tokens and lm_head with verifier
    - Multi-step prediction where each step uses previous step's output

    :param config: FastMTP configuration
    :param verifier: Optional verifier model (path or PreTrainedModel)
    :param verifier_attachment_mode: How to attach verifier (detached/full/train_only)
    """

    config_class: ClassVar[type[FastMTPConfig]] = FastMTPConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "lm_head.weight",
        "embed_tokens.weight",
    ]

    def __init__(
        self,
        config: FastMTPConfig,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        verifier_attachment_mode: Literal["detached", "full", "train_only"]
        | None = None,
    ):
        """Initialize FastMTP speculator.

        :param config: FastMTP configuration
        :param verifier: Optional verifier model (path or PreTrainedModel)
        :param verifier_attachment_mode: How to attach verifier
            (detached/full/train_only)
        """
        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode or "detached",
        )

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_speculative_steps = config.num_speculative_steps

        # Use ModuleList even with single layer for checkpoint compatibility
        # Keys will be: model.mtp_layers.0.*
        self.mtp_layers = nn.ModuleList([FastMTPLayer(config, layer_idx=0)])

        self.embed_tokens: nn.Embedding | None = None
        self.lm_head: nn.Linear | None = None

        if verifier is not None and verifier_attachment_mode != "detached":
            self.attach_verifier(verifier, mode=verifier_attachment_mode)

    def attach_verifier(
        self,
        verifier: str | os.PathLike | PreTrainedModel,
        mode: Literal["full", "train_only"] | None = None,
    ):
        """Attach verifier model and extract embed_tokens and lm_head.

        :param verifier: Verifier model (path or PreTrainedModel)
        :param mode: Attachment mode (full or train_only)
        """
        super().attach_verifier(verifier, mode)

        if isinstance(verifier, (str, os.PathLike)):
            verifier_model = AutoModelForCausalLM.from_pretrained(verifier)
        else:
            verifier_model = verifier  # type: ignore[assignment]

        if hasattr(verifier_model, "model"):
            base_model = verifier_model.model
        else:
            base_model = verifier_model

        if not hasattr(base_model, "embed_tokens"):
            raise AttributeError(
                f"Verifier model does not have 'embed_tokens' attribute. "
                f"Model type: {type(verifier_model)}"
            )

        self.embed_tokens = base_model.embed_tokens
        self.lm_head = verifier_model.lm_head

        self.embed_tokens.weight.requires_grad = False  # type: ignore[union-attr]
        self.lm_head.weight.requires_grad = False  # type: ignore[union-attr]

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> dict[str, Any] | tuple:
        """Forward pass for FastMTP multi-token prediction.

        :param input_ids: Token IDs [batch, seq_len]
        :param hidden_states: Hidden states from verifier [batch, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [batch, seq_len]
        :param position_ids: Optional position IDs [batch, seq_len]
        :param labels: Optional ground truth labels for loss computation
        :param return_dict: Whether to return dict or tuple
        :param kwargs: Additional arguments
        :return: Dictionary with logits_list, loss (if labels), and metrics (if labels)
        :raises RuntimeError: If embed_tokens or lm_head not attached
        """
        if self.embed_tokens is None or self.lm_head is None:
            raise RuntimeError(
                "Must attach verifier before forward pass. "
                "Call model.attach_verifier(verifier, mode='full') first."
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        all_logits = []
        total_loss = torch.tensor(0.0, device=device) if labels is not None else None
        metrics = {}

        current_hidden = hidden_states
        current_input_ids = input_ids

        for step in range(self.num_speculative_steps):
            current_embeddings = self.embed_tokens(current_input_ids)

            mtp_output = self.mtp_layers[0](
                hidden_states=current_hidden,
                token_embeddings=current_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

            logits = self.lm_head(mtp_output)
            all_logits.append(logits)

            if labels is not None:
                step_loss = self._compute_step_loss(
                    logits=logits,
                    labels=labels,
                    step=step,
                    weight=self.config.mtp_loss_step_weights[step],
                )
                total_loss += step_loss  # type: ignore[operator]
                metrics[f"loss_step_{step}"] = step_loss.item()

            current_input_ids = torch.argmax(logits, dim=-1)
            current_hidden = mtp_output

        if return_dict:
            return {
                "logits_list": all_logits,
                "loss": total_loss,
                "metrics": metrics,
            }
        return (all_logits, total_loss, metrics)

    def _compute_step_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        weight: float,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for a single prediction step.

        :param logits: Predicted logits [batch, seq_len, vocab_size]
        :param labels: Ground truth labels [batch, seq_len]
        :param step: Prediction step index
        :param weight: Loss weight for this step
        :return: Weighted loss tensor
        """
        # Shift labels for next-token prediction
        # step 0: predict token at position i+1, step 1: predict i+2, etc.
        shift = step + 1

        shifted_logits = logits[:, :-shift] if shift > 0 else logits
        shifted_labels = labels[:, shift:]

        loss = nn.functional.cross_entropy(
            shifted_logits.reshape(-1, self.vocab_size),
            shifted_labels.reshape(-1),
            reduction="mean",
        )

        return weight * loss

    def load_state_dict(self, state_dict: dict, *args, **kwargs):  # type: ignore[override]
        """Load checkpoint with automatic format detection and key remapping."""
        sample_keys = list(state_dict.keys())[:20]
        detected_format = self._detect_checkpoint_format(sample_keys)

        if detected_format != "tencentbac":
            state_dict = self._remap_checkpoint_keys(
                state_dict, from_format=detected_format, to_format="tencentbac"
            )

        return super().load_state_dict(state_dict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return state dict with keys in tencentbac format convention."""
        # Always use tencentbac format (internal format)
        return super().state_dict(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):  # type: ignore[override]
        """Save model with config verification."""
        super().save_pretrained(save_directory, **kwargs)

        config_path = Path(save_directory) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)

            if "speculators_model_type" not in config:
                raise ValueError("Config missing 'speculators_model_type' key")
            if config["speculators_model_type"] != "mtp":
                raise ValueError(
                    f"Expected speculators_model_type 'mtp', got "
                    f"'{config['speculators_model_type']}'"
                )
            if "num_speculative_steps" not in config:
                raise ValueError("Config missing 'num_speculative_steps' key")

    def _detect_checkpoint_format(self, keys: list[str]) -> str:
        """Infer checkpoint format from weight key prefixes.

        :param keys: List of weight keys from checkpoint
        :return: Detected format (tencentbac, qwen3_next, or deepseek)
        """
        if any(k.startswith("model.mtp_layers.0.") for k in keys):
            return "tencentbac"

        if any(k.startswith("mtp.layers.0.") for k in keys):
            return "qwen3_next"

        if any(k.startswith("mtp.") and not k.startswith("mtp.layers") for k in keys):
            return "deepseek"

        return "tencentbac"

    def _remap_checkpoint_keys(
        self, state_dict: dict, from_format: str, to_format: str
    ) -> dict:
        """Translate weight keys between checkpoint formats.

        :param state_dict: State dictionary with original keys
        :param from_format: Source checkpoint format
        :param to_format: Target checkpoint format
        :return: State dictionary with remapped keys
        """
        remapped = {}

        for key, value in state_dict.items():
            new_key = key

            if from_format == "qwen3_next" and to_format == "tencentbac":
                if key.startswith("mtp.layers.0."):
                    new_key = key.replace("mtp.layers.0.", "model.mtp_layers.0.")
                elif key.startswith("mtp.fc."):
                    new_key = key.replace("mtp.fc.", "model.mtp_layers.0.input_proj.")

            elif from_format == "tencentbac" and to_format == "qwen3_next":
                if key.startswith("model.mtp_layers.0."):
                    new_key = key.replace("model.mtp_layers.0.", "mtp.layers.0.")

            elif from_format == "deepseek" and to_format == "tencentbac":
                if key.startswith("mtp.") and not key.startswith("mtp.layers"):
                    new_key = key.replace("mtp.", "model.mtp_layers.0.")

            elif (
                from_format == "tencentbac"
                and to_format == "deepseek"
                and key.startswith("model.mtp_layers.0.")
            ):
                new_key = key.replace("model.mtp_layers.0.", "mtp.")

            remapped[new_key] = value

        return remapped

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: Qwen2Config,
        **kwargs,
    ) -> "FastMTPSpeculator":
        """Create FastMTP model from training arguments.

        :param verifier_config: Verifier model configuration
        :param kwargs: Additional arguments (num_speculative_steps,
            verifier_name_or_path)
        :return: FastMTP model instance
        """
        num_speculative_steps = kwargs.get("num_speculative_steps", 3)

        config = FastMTPConfig(
            transformer_config=verifier_config,
            num_speculative_steps=num_speculative_steps,
            hidden_size=verifier_config.hidden_size,
            intermediate_size=verifier_config.intermediate_size,
            num_attention_heads=verifier_config.num_attention_heads,
            num_key_value_heads=verifier_config.num_key_value_heads,
            vocab_size=verifier_config.vocab_size,
            speculators_config=SpeculatorsConfig(
                algorithm="mtp",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=num_speculative_steps,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=kwargs.get("verifier_name_or_path"),
                ),
            ),
        )

        return cls(config=config)

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for FastMTP.

        :param kwargs: Training arguments
        :return: Tuple of (train_kwargs, val_kwargs)
        """
        train_kwargs = {
            "num_speculative_steps": kwargs.get("num_speculative_steps", 3),
        }
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
