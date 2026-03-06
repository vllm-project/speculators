"""FastMTP speculator model implementation."""

from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp.config import FastMTPConfig
from speculators.models.fast_mtp.model_definitions import get_fast_mtp_components
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

__all__ = ["FastMTPLayer", "FastMTPSpeculator"]


class FastMTPLayer(nn.Module):
    """Single MTP layer combining verifier hidden states with token embeddings.

    Differs from Eagle3's approach in two key ways:

    1. **Projection strategy**: Eagle3 patches q/k/v projections to 2x width, then
       concatenates [embed, hidden]. FastMTPLayer uses an explicit
       ``input_proj: Linear(2*hidden → hidden)`` that collapses both inputs to
       hidden_size *before* attention, so attention runs at standard width.

    2. **Implementation pattern**: Eagle3 is a mixin on an existing decoder layer
       (Llama, Qwen3, etc.) inheriting its MLP/attention. FastMTPLayer is a standalone
       nn.Module that instantiates attention and MLP from the component registry.

    Both normalize the two inputs separately before combining; the difference is where
    dimension-reduction happens relative to attention.

    Attribute names must match checkpoint key structure exactly (e.g. `mtp_layers.0.*`).

    :param config: FastMTP configuration
    :param layer_idx: Layer index (0-indexed)
    """

    def __init__(self, config: FastMTPConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        tc = config.transformer_layer_config
        components = get_fast_mtp_components(tc.model_type)
        arch_config = self._create_architecture_config(config, components)

        norm_class = components.norm_class
        eps = tc.rms_norm_eps
        self.input_layernorm = norm_class(hidden_size, eps=eps)
        self.post_attention_layernorm = norm_class(hidden_size, eps=eps)
        self.hidden_layernorm = norm_class(hidden_size, eps=eps)
        self.token_layernorm = norm_class(hidden_size, eps=eps)
        self.final_layernorm = norm_class(hidden_size, eps=eps)

        self.use_moe = components.use_moe
        self.input_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.self_attn = components.attention_class(arch_config, layer_idx=layer_idx)
        self.mlp = components.mlp_class(arch_config)
        self.rotary_emb = components.rotary_emb_class(arch_config)

    def _create_architecture_config(self, config: FastMTPConfig, components):
        """Build an architecture-specific config from transformer_layer_config.

        Passes all dimension fields through from the transformer config so that the
        attention/MLP modules are initialized with the correct sizes.

        :param config: FastMTP configuration
        :param components: FastMTP component registry for the architecture
        :return: Architecture-specific configuration object
        """
        tc = config.transformer_layer_config
        config_class = tc.__class__

        arch_config = config_class(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=tc.num_key_value_heads,
            max_position_embeddings=tc.max_position_embeddings,
            rms_norm_eps=tc.rms_norm_eps,
            vocab_size=tc.vocab_size,
            attention_bias=components.attention_bias,
        )
        arch_config._attn_implementation = "eager"  # noqa: SLF001

        if components.use_moe:
            if hasattr(tc, "num_experts"):
                arch_config.num_experts = tc.num_experts
            if hasattr(tc, "num_experts_per_tok"):
                arch_config.num_experts_per_tok = tc.num_experts_per_tok
            if hasattr(tc, "moe_intermediate_size"):
                arch_config.moe_intermediate_size = tc.moe_intermediate_size

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

        if position_ids is None:
            batch_size, seq_len, _ = proj_hidden.shape
            position_ids = (
                torch.arange(seq_len, device=proj_hidden.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        position_embeddings = self.rotary_emb(proj_hidden, position_ids)

        residual = proj_hidden
        attn_output, _ = self.self_attn(
            hidden_states=proj_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        if self.use_moe:
            mlp_output = mlp_output[0]
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
    - Teacher-forced multi-step prediction (ground-truth tokens, not sampled)

    :param config: FastMTP configuration
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

    def __init__(self, config: FastMTPConfig):
        super().__init__(config=config)

        self.mtp_layers = nn.ModuleList([FastMTPLayer(config, layer_idx=0)])

        self.embed_tokens: nn.Embedding | None = None
        self.lm_head: nn.Linear | None = None

        self._setup_embeddings_and_lm_head()

    def _setup_embeddings_and_lm_head(self):
        """Load embed_tokens and lm_head weights from verifier checkpoint.

        Only runs if speculators_config.verifier.name_or_path is set.
        Loads only the needed tensors without loading the full verifier model.
        """
        if (
            self.config.speculators_config is None
            or self.config.speculators_config.verifier is None
            or self.config.speculators_config.verifier.name_or_path is None
        ):
            return

        path = self.config.speculators_config.verifier.name_or_path
        weights = load_model_layers(["embed_tokens.weight", "lm_head.weight"], path)

        embed_weight = weights["embed_tokens.weight"]
        lm_head_weight = weights.get("lm_head.weight", embed_weight)

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.embed_tokens.weight = nn.Parameter(
            embed_weight.detach().clone(), requires_grad=False
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = nn.Parameter(
            lm_head_weight.detach().clone(), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> dict[str, Any] | tuple:
        """Forward pass for FastMTP multi-token prediction (teacher-forced).

        At step k, uses ground-truth input_ids[t+k] as the embedding input and
        hidden_states[t] from the verifier. No autoregressive rollout — each step
        is independent, enabling parallel computation during training.

        :param input_ids: Token IDs [batch, seq_len]
        :param hidden_states: Hidden states from verifier [batch, seq_len, hidden_size]
        :param attention_mask: Optional attention mask [batch, seq_len]
        :param position_ids: Optional position IDs [batch, seq_len]
        :param labels: Optional ground truth labels [batch, seq_len]
        :param step_weights: Per-step loss weights (None = uniform). Training only.
        :param return_dict: Whether to return dict or tuple
        :param kwargs: Additional arguments passed to FastMTPLayer
        :return: Dictionary with logits_list, loss (if labels), and metrics (if labels)
        :raises RuntimeError: If embed_tokens or lm_head not attached
        """
        if self.embed_tokens is None or self.lm_head is None:
            raise RuntimeError(
                "embed_tokens and lm_head must be set before calling forward. "
                "Ensure speculators_config.verifier.name_or_path is configured."
            )

        device = input_ids.device
        seq_len = input_ids.shape[1]
        num_steps = self.config.num_speculative_steps

        all_logits = []
        total_loss = torch.tensor(0.0, device=device) if labels is not None else None
        metrics = {}

        for step in range(num_steps):
            valid_len = seq_len - step - 1
            step_hidden = hidden_states[:, :valid_len]
            step_embeds = self.embed_tokens(input_ids[:, step : step + valid_len])
            step_pos_ids = (
                position_ids[:, :valid_len] if position_ids is not None else None
            )

            mtp_output = self.mtp_layers[0](
                hidden_states=step_hidden,
                token_embeddings=step_embeds,
                attention_mask=attention_mask,
                position_ids=step_pos_ids,
                **kwargs,
            )

            logits = self.lm_head(mtp_output)
            all_logits.append(logits)

            if labels is not None:
                step_labels = labels[:, step + 1 : step + 1 + valid_len]
                weight = step_weights[step] if step_weights is not None else 1.0
                step_loss = weight * nn.functional.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    step_labels.reshape(-1),
                )
                total_loss = total_loss + step_loss  # type: ignore[operator]
                metrics[f"loss_step_{step}"] = step_loss.item()

        if return_dict:
            return {
                "logits_list": all_logits,
                "loss": total_loss,
                "metrics": metrics,
            }
        return (all_logits, total_loss, metrics)

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:  # noqa: PLR0911
        """Remap checkpoint keys to model parameter names for HF weight loading.

        HF calls this per-key before resolving missing/unexpected keys. The model's
        base_model_prefix is "model", so speculators keys (model.mtp_layers.0.*) are
        handled by HF's built-in prefix-stripping — only external formats need
        remapping.

        Applies the same rules as _remap_qwen3_next_key and _remap_key, but without
        the "model." prefix that HF strips before invoking this hook. The deepseek
        catch-all (mtp.<rest> → mtp_layers.0.<rest>) must remain last.
        """
        if key.startswith("mtp.layers.0."):
            return key.replace("mtp.layers.0.", "mtp_layers.0.", 1), True
        if key.startswith("mtp.fc."):
            return key.replace("mtp.fc.", "mtp_layers.0.input_proj.", 1), True
        if key == "mtp.norm.weight":
            return "mtp_layers.0.final_layernorm.weight", True
        if key.startswith("mtp.pre_fc_norm_hidden."):
            return key.replace(
                "mtp.pre_fc_norm_hidden.", "mtp_layers.0.hidden_layernorm."
            ), True
        if key.startswith("mtp.pre_fc_norm_embedding."):
            return key.replace(
                "mtp.pre_fc_norm_embedding.", "mtp_layers.0.token_layernorm."
            ), True
        if key.startswith("mtp.") and not key.startswith("mtp.layers"):
            return key.replace("mtp.", "mtp_layers.0.", 1), True
        return key, False

    @classmethod
    def from_training_args(  # type: ignore[override]
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "FastMTPSpeculator":
        """Create FastMTP model from training arguments.

        :param verifier_config: Verifier model configuration
        :param kwargs: Additional keyword arguments (num_speculative_steps,
            verifier_name_or_path)
        :return: FastMTP model instance
        """
        num_speculative_steps = kwargs.get("num_speculative_steps", 3)

        config = FastMTPConfig(
            transformer_layer_config=verifier_config,
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
            "step_weights": kwargs.get("step_weights", [0.51, 0.31, 0.18]),
        }
        val_kwargs = train_kwargs.copy()

        return train_kwargs, val_kwargs
