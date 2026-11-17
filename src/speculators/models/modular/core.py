"""Composition of block backbones, correction modules, and training strategies."""

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml

from speculators.model import SpeculatorModel
from speculators.models.components import TrainingObjective
from speculators.models.dflash.backbone import DFlashBackbone
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.objective import dflash_objective
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.dflash2.objective import dflash2_objective
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.dspark.objective import dspark_objective
from speculators.models.modular.config import BlockRecipe, ModularSpeculatorConfig
from speculators.models.xpress import XPressDraftModel, XPressSpeculatorConfig
from speculators.models.xpress.model_definitions import XPressRefinerHead
from speculators.models.xpress.objective import xpress_objective

_OBJECTIVES: dict[str, TrainingObjective] = {
    "dflash": dflash_objective,
    "dspark": dspark_objective,
    "dflash2": dflash2_objective,
    "xpress": xpress_objective,
}
_TRAINER_RECIPES: dict[str, type[DFlashDraftModel]] = {
    "dflash": DFlashDraftModel,
    "dspark": DSparkDraftModel,
    "dflash2": DFlash2DraftModel,
    "xpress": XPressDraftModel,
}


@SpeculatorModel.register("modular")
class ModularDraftModel(DFlashDraftModel):
    """Train supported mixtures without a subclass for each pairing.

    Native mixed checkpoints require a matching serving implementation. Legacy
    export is available only when the whole recipe matches an existing model.
    """

    config_class: ClassVar[type[ModularSpeculatorConfig]] = ModularSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "Qwen3DFlash2DecoderLayer"]

    def __init__(self, config: ModularSpeculatorConfig):
        super().__init__(config)
        recipe = config.recipe
        assert recipe.objective is not None  # noqa: S101
        self.objective = _OBJECTIVES[recipe.objective]
        if recipe.backbone.kind == "dflash2":
            for layer in self.get_decoder_layers():
                assert isinstance(layer, Qwen3DFlash2DecoderLayer)  # noqa: S101
                layer.reset_convolutions()
        correction = recipe.correction
        if correction.kind == "markov":
            self.correction = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=correction.rank,
                hidden_size=self.hidden_size,
                head_type=correction.markov_head_type,
            )
        elif correction.kind == "selector":
            self.correction = CandidateSelector(
                vocab_size=self.draft_vocab_size,
                hidden_size=self.hidden_size,
                rank=correction.rank,
                top_k=correction.selector_top_k,
                initializer_range=getattr(
                    config.transformer_layer_config, "initializer_range", 0.02
                ),
            )
        elif correction.kind == "xpress":
            self.correction = XPressRefinerHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                hidden_size=self.hidden_size,
                block_size=self.block_size,
                rank=correction.rank,
                mlp_ratio=correction.xpress_mlp_ratio,
            )
            self.post_init()
        self.confidence_head: ConfidenceHead | None = None
        self.confidence_with_correction = recipe.confidence.with_correction_features
        if recipe.confidence.enabled:
            size = self.hidden_size + (
                correction.rank if self.confidence_with_correction else 0
            )
            self.confidence_head = ConfidenceHead(size)

    def _create_backbone(self, config: DFlashSpeculatorConfig) -> DFlashBackbone:
        assert isinstance(config, ModularSpeculatorConfig)  # noqa: S101
        backbone = config.recipe.backbone
        if backbone.kind == "dflash":
            return super()._create_backbone(config)
        return DFlashBackbone(
            config.transformer_layer_config,
            len(self.target_layer_ids),
            decoder_layer_factory=partial(
                Qwen3DFlash2DecoderLayer,
                block_size=config.block_size,
                conv_kernel_size=backbone.conv_kernel_size,
                conv_group_size=backbone.conv_group_size,
            ),
        )

    @classmethod
    def from_training_args(cls, verifier_config, t2d=None, d2t=None, **kwargs):
        path = kwargs.get("composition_config")
        if not path:
            raise ValueError(
                "Fresh modular training requires --composition-config "
                "or a saved full configuration"
            )
        recipe = BlockRecipe.model_validate(yaml.safe_load(Path(path).read_text()))
        config = ModularSpeculatorConfig(
            **cls._build_base_config_kwargs("modular", verifier_config, **kwargs),
            recipe=recipe,
        )
        model = cls(config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs):
        config = kwargs.get("model_config")
        if not isinstance(config, ModularSpeculatorConfig):
            raise ValueError(
                "Modular objective resolution requires the constructed model_config"
            )
        assert config.recipe.objective is not None  # noqa: S101
        return _TRAINER_RECIPES[config.recipe.objective].get_trainer_kwargs(**kwargs)

    def save_pretrained(self, save_directory, *args, checkpoint_format=None, **kwargs):
        if checkpoint_format not in (None, "legacy", "native"):
            raise ValueError(f"Unknown checkpoint format: {checkpoint_format!r}")
        if checkpoint_format != "legacy":
            return super().save_pretrained(
                save_directory, *args, checkpoint_format="native", **kwargs
            )
        model_class, config = self._legacy_recipe()
        # A meta-device serialization facade gives the serving export the actual
        # legacy class/config without allocating another model's tensor storage.
        with torch.device("meta"):
            facade = model_class(config).to(dtype=self.dtype)
        state = kwargs.pop("state_dict", None)
        kwargs["state_dict"] = self.state_dict() if state is None else state
        kwargs["checkpoint_format"] = "legacy"
        return facade.save_pretrained(save_directory, *args, **kwargs)

    def _legacy_recipe(self):
        recipe = self.config.recipe
        backbone, correction, confidence = (
            recipe.backbone,
            recipe.correction,
            recipe.confidence,
        )
        common = {
            key: deepcopy(getattr(self.config, key))
            for key in (
                "transformer_layer_config",
                "draft_vocab_size",
                "block_size",
                "aux_hidden_state_layer_ids",
                "mask_token_id",
                "target_hidden_size",
                "sliding_window_non_causal",
                "sample_from_anchor",
            )
        }
        extra: dict[str, Any] = {}
        cls: type[DFlashDraftModel]
        config_cls: type[DFlashSpeculatorConfig]
        if backbone.kind == "dflash2" and correction.kind == "selector":
            name, cls, config_cls = (
                "dflash2",
                DFlash2DraftModel,
                DFlash2SpeculatorConfig,
            )
            extra = {
                "conv_kernel_size": backbone.conv_kernel_size,
                "conv_group_size": backbone.conv_group_size,
                "selector_rank": correction.rank,
                "selector_top_k": correction.selector_top_k,
            }
        elif backbone.kind == "dflash" and correction.kind == "xpress":
            name, cls, config_cls = "xpress", XPressDraftModel, XPressSpeculatorConfig
            extra = {
                "xpress_rank": correction.rank,
                "xpress_mlp_ratio": correction.xpress_mlp_ratio,
                "num_jacobi_passes": correction.num_jacobi_passes,
            }
        elif backbone.kind == "dflash" and recipe.objective == "dspark":
            name, cls, config_cls = "dspark", DSparkDraftModel, DSparkSpeculatorConfig
            extra = {
                "markov_rank": correction.rank if correction.kind == "markov" else 0,
                "markov_head_type": correction.markov_head_type,
                "enable_confidence_head": confidence.enabled,
                "confidence_head_with_markov": confidence.with_correction_features,
            }
        elif backbone.kind == "dflash" and recipe.objective == "dflash":
            name, cls, config_cls = "dflash", DFlashDraftModel, DFlashSpeculatorConfig
        else:
            raise ValueError(
                "This mixed recipe has no legacy serving export; save the native format"
            )
        serving = deepcopy(self.config.speculators_config)
        if serving is None:
            raise ValueError(
                "Legacy serving export requires verifier/proposal metadata"
            )
        serving.algorithm = name
        return cls, config_cls(**common, **extra, speculators_config=serving)
