"""DFlash2 transformer body with local convolution in each decoder layer."""

from typing import cast

from speculators.models.dflash.backbone import DFlashBackbone
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer


class DFlash2Backbone(DFlashBackbone):
    decoder_layer_class = Qwen3DFlash2DecoderLayer

    def post_init(self) -> None:
        super().post_init()
        for layer in self.layers:
            decoder_layer = cast("Qwen3DFlash2DecoderLayer", layer)
            decoder_layer.reset_convolutions()
