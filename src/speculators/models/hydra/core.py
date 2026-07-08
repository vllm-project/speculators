import copy
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.hydra.config import HydraSpeculatorConfig
from speculators.models.hydra.metrics import compute_metrics
from speculators.models.hydra.model_definitions import ResBlock
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig


@SpeculatorModel.register("hydra")
class HydraDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[HydraSpeculatorConfig]] = HydraSpeculatorConfig  # type: ignore[misc]
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

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: HydraSpeculatorConfig):
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config
        num_heads = config.num_hydra_heads
        num_layers = config.num_hydra_layers
        dropout_rate = config.dropout_rate

        # Norm class detection from the transformer config
        norm_eps = getattr(tl_config, "rms_norm_eps", 1e-6)
        self.verifier_norm = nn.RMSNorm(self.hidden_size, eps=norm_eps)
        self.verifier_norm.weight.requires_grad = False

        # Optional prefix attention layer (Hydra++)
        if config.use_prefix_mlp:
            prefix_config = copy.deepcopy(tl_config)
            prefix_config.num_hidden_layers = 1
            from transformers import AutoModel  # noqa: PLC0415

            self.prefix_layer = AutoModel.from_config(prefix_config)
            # Re-initialize (we'll train from scratch)
            self.prefix_layer.apply(self._init_weights_fn)
        else:
            self.prefix_layer = None

        # Hydra MLP heads
        # Each head i takes input of dim (i+2)*hidden_size when grounded:
        #   hidden_state + embed(x_t) + embed(x_{t+1}) + ... + embed(x_{t+i})
        # First ResBlock reduces to hidden_size; rest are hidden_size -> hidden_size
        head_modules = []
        for head_idx in range(num_heads):
            layers: list[nn.Module] = []
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(ResBlock(self.hidden_size, num_condition=head_idx + 1))
            for _ in range(num_layers - 1):
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(ResBlock(self.hidden_size))
            head_modules.append(nn.Sequential(*layers))
        self.hydra_mlp = nn.ModuleList(head_modules)

        # Per-head LM heads
        hydra_lm_heads = []
        for _ in range(num_heads):
            head_layers: list[nn.Module] = []
            if dropout_rate > 0:
                head_layers.append(nn.Dropout(dropout_rate))
            head_layers.append(
                nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
            )
            hydra_lm_heads.append(nn.Sequential(*head_layers))
        self.hydra_lm_heads = nn.ModuleList(hydra_lm_heads)

        # Initialize hydra lm_head weights from the verifier lm_head
        # (done later in load_verifier_weights)

        self.post_init()

    @property
    def layers(self) -> nn.ModuleList:
        """Alias for FSDP layer wrapping which expects a `layers` attribute."""
        return self.hydra_mlp

    @staticmethod
    def _init_weights_fn(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def target_layer_ids(self) -> list[int]:
        """Hydra uses only the final verifier hidden state."""
        verifier_path = getattr(
            getattr(getattr(self, "config", None), "speculators_config", None),
            "verifier",
            None,
        )
        if verifier_path is not None and hasattr(verifier_path, "name_or_path"):
            path = verifier_path.name_or_path
            if path:
                v_config = AutoConfig.from_pretrained(path)
                if hasattr(v_config, "text_config"):
                    v_config = v_config.text_config
                return [v_config.num_hidden_layers]
        return [32]  # fallback

    def load_verifier_weights(self):
        super().load_verifier_weights()
        # Initialize hydra lm_head weights from the shared lm_head
        with torch.no_grad():
            lm_weight = self.lm_head.weight.data
            for head in self.hydra_lm_heads:
                linear = head[-1] if isinstance(head, nn.Sequential) else head
                if (
                    isinstance(linear, nn.Linear)
                    and linear.weight.shape == lm_weight.shape
                ):
                    linear.weight.data.copy_(lm_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,  # noqa: ARG002
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_config: LossConfig | None = None,
        **kwargs,  # noqa: ARG002
    ):
        """Forward pass with teacher forcing.

        Args:
            hidden_states: Verifier hidden states
                [1, seq_len, hidden_size]
            input_ids: Ground truth token ids [1, seq_len]
            document_ids: Document boundary ids [1, seq_len]
            loss_mask: Valid positions [1, seq_len]
            position_ids: Position ids [1, seq_len]
            verifier_last_hidden_states: Last layer hidden
                states [1, seq_len, hidden_size]
            loss_config: Loss function configuration
        """
        device = hidden_states.device
        num_heads = self.config.num_hydra_heads

        # Apply prefix attention layer if using Hydra++
        if self.prefix_layer is not None:
            prefix_out = self.prefix_layer(
                inputs_embeds=hidden_states,
                position_ids=position_ids,
            )
            base_repr = prefix_out.last_hidden_state
        else:
            base_repr = hidden_states

        # Get token embeddings for grounding (teacher-forced)
        with torch.no_grad():
            input_embeds = self.embed_tokens(input_ids)

        # Build grounding inputs: for head i, concatenate
        # [base_repr, embed(x_{t+1}), ..., embed(x_{t+i+1})]
        # Using roll to shift embeddings to the right positions
        grounding_embeds = [base_repr]
        for i in range(num_heads):
            grounding_embeds.append(torch.roll(input_embeds, shifts=-(i + 1), dims=1))

        # Compute targets from verifier if training
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
            loss = torch.tensor(0.0, device=device)
            metrics: dict = {}

        # Run each head
        draft_logits = []
        for head_idx in range(num_heads):
            head_input = torch.cat(grounding_embeds[: head_idx + 2], dim=-1)
            head_hidden = self.hydra_mlp[head_idx](head_input)
            head_logits = self.hydra_lm_heads[head_idx](head_hidden)
            draft_logits.append(head_logits)

            if return_loss:
                h_loss, h_metrics = compute_metrics(
                    head_logits,
                    targets,
                    loss_mask,
                    head_idx,
                    loss_config=loss_config,
                )
                loss += h_loss
                metrics.update(h_metrics)

        # Return draft token predictions (argmax per head)
        draft_tokens = [torch.argmax(logits, dim=-1) for logits in draft_logits]

        if return_loss:
            metrics["loss_sum"] = loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return draft_tokens, loss, metrics
        return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "HydraDraftModel":
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        # For hydra, we only use the last target layer
        if target_layer_ids and len(target_layer_ids) > 1:
            target_layer_ids = [target_layer_ids[-1]]

        # Override num_hidden_layers for the prefix layer
        prefix_config = copy.deepcopy(verifier_config)
        prefix_config.num_hidden_layers = 1

        config = HydraSpeculatorConfig(
            transformer_layer_config=prefix_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            num_hydra_heads=kwargs.get("num_hydra_heads", 4),
            num_hydra_layers=kwargs.get("num_hydra_layers", 4),
            use_prefix_mlp=kwargs.get("use_prefix_mlp", True),
            dropout_rate=kwargs.get("dropout_rate", 0.0),
            speculators_config=SpeculatorsConfig(
                algorithm="hydra",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("num_hydra_heads", 4),
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
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        train_kwargs = {
            "loss_config": loss_config,
        }
        val_kwargs = {
            "loss_config": loss_config,
        }
        return train_kwargs, val_kwargs
