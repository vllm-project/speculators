import os
from typing import ClassVar, Literal, Optional, Union

import torch
from pydantic.fields import Field
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from speculators import SpeculatorModelConfig, SpeculatorsConfig
from speculators.model import SpeculatorModel

__all__ = ["IndependentSpeculatorConfig"]


@SpeculatorModelConfig.register("independent")
class IndependentSpeculatorConfig(SpeculatorModelConfig):
    @classmethod
    def from_pretrained_config(
        cls, pretrained_config: PretrainedConfig, speculators_config: SpeculatorsConfig
    ) -> "IndependentSpeculatorConfig":
        pretrained_dict = pretrained_config.to_dict()
        pretrained_dict["model_type"] = pretrained_config.model_type

        return cls(**pretrained_dict, speculators_config=speculators_config)

    speculators_model_type: Literal["independent"] = "independent"
    architectures: list[str] = Field(
        default_factory=lambda: ["LlamaForCausalLM"],
        description=("List of model architectures that can be used with the model "),
    )
    draft_model: str = Field(
        default="",
        description=(
            "The name or path to the draft model to use for the speculator. "
            "Must be a model that is compatible with the speculator."
        ),
    )


@SpeculatorModel.register("independent")
class IndependentSpeculator(SpeculatorModel):
    config_class: ClassVar[type[IndependentSpeculatorConfig]] = (  # type: ignore[misc]
        IndependentSpeculatorConfig
    )
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "verifier*",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[assignment,misc]
        "verifier*",
    ]

    def __init__(
        self,
        config: IndependentSpeculatorConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
    ):
        if not isinstance(config, IndependentSpeculatorConfig):
            raise ValueError(
                "config must be an instance of IndependentSpeculatorConfig, "
                f"got {type(config)} instead."
            )

        super().__init__(
            config=config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
        )

        self.draft_model: PreTrainedModel = self.resolve_verifier(config.draft_model)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # noqa: ARG002
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        return self.draft_model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
