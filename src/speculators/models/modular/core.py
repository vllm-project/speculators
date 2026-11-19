"""Composition of block backbones, correction modules, and training strategies."""

from functools import partial
from pathlib import Path
from typing import ClassVar

import yaml

from speculators.model import SpeculatorModel
from speculators.models.components import TrainingObjective
from speculators.models.dflash.backbone import DFlashBackbone
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.objective import dflash_objective
from speculators.models.dflash2 import DFlash2DraftModel
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.dflash2.objective import dflash2_objective
from speculators.models.dspark import DSparkDraftModel
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.dspark.objective import dspark_objective
from speculators.models.modular.config import BlockRecipe, ModularSpeculatorConfig
from speculators.models.xpress import XPressDraftModel
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
    """Train supported mixtures and save hierarchical checkpoints."""

    hierarchical_checkpoint = True
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
