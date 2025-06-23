from transformers import PretrainedConfig

from speculators import SpeculatorModelConfig, SpeculatorsConfig

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

    speculators_model_type: str = "independent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ensure we set the model_type to the one from the original config
        self._model_type = kwargs.get("model_type")

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict["model_type"] = self._model_type
        del config_dict["_model_type"]
        return config_dict
