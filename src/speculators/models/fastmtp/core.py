import importlib
import inspect
import re
import warnings
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from speculators.config import VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.core import (
    compute_metrics,
    conditional_torch_compile,
)
from speculators.models.fastmtp.config import FastMTPSpeculatorConfig
from speculators.utils.loading import load_model_layers

__all__ = [
    "FastMTPDraftModel",
]


@SpeculatorModel.register("fastmtp")
class FastMTPDraftModel(SpeculatorModel):
    """FastMTP speculator model (arXiv:2509.18362).

    A single MTP head with position-shared weights is applied recursively
    K times to predict K future tokens. RoPE embeddings are computed once
    and reused across all steps.

    MTP head architecture::

        token_layernorm(embeds) + hidden_layernorm(hidden)
        -> input_proj(cat[hidden, embeds])
        -> decoder_layer (self_attn + MLP with residuals)
        -> final_layernorm

    :param config: FastMTP speculator configuration
    :param t2d: Target-to-draft vocabulary mapping (bool mask)
    :param d2t: Draft-to-target vocabulary mapping (index tensor)
    """

    config_class: ClassVar[type[FastMTPSpeculatorConfig]] = FastMTPSpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = ["verifier_lm_head.weight"]  # type: ignore[misc,assignment]

    def __init__(
        self,
        config: FastMTPSpeculatorConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            verifier=kwargs.get("verifier"),
            verifier_attachment_mode=kwargs.get(
                "verifier_attachment_mode", "train_only"
            ),
        )
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.num_speculative_steps = config.num_speculative_steps

        # _attn_implementation is not serialized; default to sdpa if missing
        if (
            getattr(config.transformer_layer_config, "_attn_implementation", None)
            is None
        ):
            config.transformer_layer_config._attn_implementation = "sdpa"  # noqa: SLF001

        self._setup_vocab_mapping(t2d, d2t)

        decoder_class, norm_class, rotary_emb_class = self._import_model_classes(
            config.transformer_layer_config
        )

        self.token_layernorm = norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self.hidden_layernorm = norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self.input_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.decoder_layer = decoder_class(config.transformer_layer_config, layer_idx=0)
        self.final_layernorm = norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )

        self.rotary_emb = rotary_emb_class(config.transformer_layer_config)
        self._setup_embeddings_and_lm_heads(config.speculators_config.verifier, t2d)

    def _setup_vocab_mapping(self, t2d: torch.Tensor | None, d2t: torch.Tensor | None):
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

    def _import_model_classes(
        self, transformer_layer_config: PretrainedConfig
    ) -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
        """Dynamically import decoder layer, norm, and rotary embedding classes
        from the base model's modeling module."""
        config_class = type(transformer_layer_config)
        if config_class not in MODEL_FOR_CAUSAL_LM_MAPPING:
            raise TypeError(
                f"Config class {config_class} is not a valid causal language model "
                f"config class. Please use a valid config, e.g., LlamaConfig."
            )

        causal_lm_model_class = MODEL_FOR_CAUSAL_LM_MAPPING[config_class]
        modeling_module = importlib.import_module(causal_lm_model_class.__module__)

        model_prefix = config_class.__name__.removesuffix("Config")
        decoder_name = f"{model_prefix}DecoderLayer"
        try:
            decoder_class = getattr(modeling_module, decoder_name)
        except AttributeError as e:
            raise ValueError(
                f"Decoder layer class {decoder_name} not found in "
                f"{causal_lm_model_class.__module__}."
            ) from e

        norm_class = self._find_norm_class(modeling_module)

        rotary_name = f"{model_prefix}RotaryEmbedding"
        try:
            rotary_class = getattr(modeling_module, rotary_name)
        except AttributeError:
            classes = dict(inspect.getmembers(modeling_module, inspect.isclass))
            rotary_class = next(
                (
                    cls
                    for name, cls in classes.items()
                    if re.match(r".*RotaryEmbedding$", name)
                ),
                None,
            )
            if rotary_class is None:
                raise ValueError(
                    f"Could not find RotaryEmbedding class in "
                    f"{causal_lm_model_class.__module__}."
                ) from None

        return decoder_class, norm_class, rotary_class

    @staticmethod
    def _find_norm_class(modeling_module) -> type[nn.Module]:
        classes = dict(inspect.getmembers(modeling_module, inspect.isclass))
        for pat in [r".*RMSNorm$", r".*Norm$"]:
            for name, cls in classes.items():
                if re.match(pat, name):
                    return cls

        warnings.warn(
            "Unable to find layer normalization class. "
            "Falling back to torch.nn.LayerNorm.",
            stacklevel=2,
        )
        return nn.LayerNorm

    def _setup_embeddings_and_lm_heads(
        self, config: VerifierConfig, t2d: torch.Tensor | None
    ):
        """Create embed_tokens, lm_head, verifier_lm_head and load verifier weights.

        When called from from_pretrained (meta device), only creates layers;
        weights are loaded from the checkpoint by HF's loading machinery.
        """
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
        if t2d is not None and t2d.shape[0] != verifier_model_config.vocab_size:
            raise ValueError(
                f"t2d.shape[0] ({t2d.shape[0]}) must match"
                f" verifier_vocab_size ({verifier_model_config.vocab_size})."
            )

        self.embed_tokens = nn.Embedding(
            verifier_model_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_model_config.pad_token_id,
        )
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.verifier_lm_head = nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        # Skip weight loading on meta device (from_pretrained loads from checkpoint)
        if self.embed_tokens.weight.is_meta:
            return

        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight"],
            config.name_or_path,
        )

        if "embed_tokens.weight" not in verifier_weights:
            raise KeyError(
                f"Could not find embedding weights in {config.name_or_path}. "
                "Expected a key ending with 'embed_tokens.weight'."
            )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)

        default_dtype = self.embed_tokens.weight.dtype
        self.embed_tokens.load_state_dict(
            {"weight": embed_tokens_weight.to(default_dtype)}
        )
        self.embed_tokens.weight.requires_grad = False

        if t2d is not None:
            lm_head_weight = lm_head_weight.to(device=t2d.device, dtype=default_dtype)[
                t2d.to(torch.bool), :
            ]
        else:
            lm_head_weight = lm_head_weight.to(dtype=default_dtype)

        if lm_head_weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Verifier lm head data shape "
                f"{lm_head_weight.shape} does not match draft "
                f"lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

    def _mtp_head_forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single MTP head pass: norm -> cat+proj -> decoder_layer -> final_norm."""
        normed_embeds = self.token_layernorm(input_embeds)
        normed_hidden = self.hidden_layernorm(hidden_states)
        projected = self.input_proj(torch.cat([normed_hidden, normed_embeds], dim=-1))
        layer_output = self.decoder_layer(
            projected,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=False,
            **kwargs,
        )
        return self.final_layernorm(layer_output)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        current_hidden = (
            verifier_last_hidden_states
            if verifier_last_hidden_states is not None
            else hidden_states[:, :, : self.hidden_size]
        )

        # Position-shared: compute RoPE once, reuse across all K steps
        position_embeddings = self.rotary_emb(current_hidden, position_ids)

        batch_size = hidden_states.shape[0]
        causal_mask_2d = torch.ones(
            batch_size, total_seq_len, dtype=torch.long, device=device
        )
        attention_mask = _prepare_4d_causal_attention_mask(
            causal_mask_2d,
            (batch_size, total_seq_len),
            current_hidden,
            past_key_values_length=0,
        )

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(verifier_last_hidden_states)
            loss = torch.tensor(0.0, device=device)
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        cache_position = torch.arange(total_seq_len, dtype=torch.long, device=device)

        for ttt_step in range(ttt_steps):
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)

            current_hidden = self._mtp_head_forward(
                input_embeds=input_embeds,
                hidden_states=current_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                **kwargs,
            )

            logits = self.lm_head(current_hidden)

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                )
                loss += s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())

            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )

        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens
