import warnings

from transformers import AutoConfig, PretrainedConfig


def get_verifier_config(verifier_name_or_path: str) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
) -> list[int]:
    if target_layer_ids is not None:
        return target_layer_ids

    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    target_layer_ids = [2, num_layers // 2, num_layers - 3]
    warnings.warn(
        DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
        stacklevel=3,
    )
    return target_layer_ids
