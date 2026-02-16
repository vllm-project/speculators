from typing import NamedTuple


class ModelComponents(NamedTuple):
    first_layer_class: type
    decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type

