import warnings
from functools import partial

import torch
from transformers import AutoConfig, PretrainedConfig


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(verifier_name_or_path: str) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
    text_config = getattr(verifier_config, "text_config", None)
    if text_config is not None:
        verifier_config = text_config
    return verifier_config


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def default_auxiliary_target_layer_ids(num_layers: int) -> list[int]:
    """Choose up to three unique, valid auxiliary verifier layers."""
    if num_layers < 2:
        raise ValueError(
            "A verifier must expose at least two hidden layers so extraction can "
            "include one auxiliary layer and the final layer."
        )

    target_count = min(3, num_layers - 1)
    selected = {
        layer_id
        for layer_id in (2, num_layers // 2, num_layers - 3)
        if 1 <= layer_id < num_layers
    }
    for layer_id in (1, num_layers - 1, *range(1, num_layers)):
        if len(selected) >= target_count:
            break
        selected.add(layer_id)
    return sorted(selected)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
) -> list[int]:
    if target_layer_ids is not None:
        return target_layer_ids

    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    target_layer_ids = default_auxiliary_target_layer_ids(num_layers)
    warnings.warn(
        DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
        stacklevel=3,
    )
    return target_layer_ids


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers have no dense ``intermediate_size`` (their FFN is a routed set of
      small experts), so the draft falls back to the widely used ``3 * hidden_size``
      gated-MLP ratio -- the Qwen3 dense convention that the dflash draft decoder
      follows. Pass ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier config has no dense intermediate_size (likely MoE); using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size


def strip_verifier_final_layer_id(
    target_layer_ids: list[int],
    verifier_name_or_path: str,
) -> list[int]:
    """Remove the verifier final layer from training auxiliary layer IDs."""
    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError(
            "--target-layer-ids must not contain duplicate layer IDs: "
            f"{target_layer_ids}"
        )
    out_of_range = [
        layer_id
        for layer_id in target_layer_ids
        if layer_id < 1 or layer_id > num_layers
    ]
    if out_of_range:
        raise ValueError(
            "--target-layer-ids must be in the inclusive range "
            f"[1, {num_layers}]; got {out_of_range}"
        )
    aux_target_layer_ids = [
        layer_id for layer_id in target_layer_ids if layer_id != num_layers
    ]
    if not aux_target_layer_ids:
        raise ValueError(
            "--target-layer-ids must include at least one auxiliary verifier layer "
            f"other than the final layer ({num_layers})."
        )
    if len(aux_target_layer_ids) != len(target_layer_ids):
        warnings.warn(
            "Stripping the verifier's final layer "
            f"({num_layers}) from --target-layer-ids for training. "
            "The last extracted layer is consumed separately as "
            "`verifier_last_hidden_states`.",
            stacklevel=2,
        )
    return aux_target_layer_ids
