from speculators.models.eagle3.model_components import ModelComponents
from speculators.models.eagle3.model_definitions_llama import LLAMA_MODEL_COMPONENTS
from speculators.models.eagle3.model_definitions_qwen3 import QWEN3_MODEL_COMPONENTS

model_classes: dict[str, ModelComponents] = {
    "llama": LLAMA_MODEL_COMPONENTS,
    "qwen3": QWEN3_MODEL_COMPONENTS,
    "qwen3_vl": QWEN3_MODEL_COMPONENTS,
    "qwen3_vl_text": QWEN3_MODEL_COMPONENTS,
    "qwen3vl": QWEN3_MODEL_COMPONENTS,
}

