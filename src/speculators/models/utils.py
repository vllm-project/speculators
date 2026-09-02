import logging
import warnings
from copy import deepcopy
from functools import partial

import torch
from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
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
    trust_remote_code: bool = False,
) -> list[int]:
    num_layers = get_verifier_config(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    ).num_hidden_layers

    if target_layer_ids is None:
        explicit = False
        target_layer_ids = [2, num_layers // 2, num_layers - 3]
    else:
        explicit = True

    # Layer id ``num_layers`` (the final hidden state) is valid, matching the
    # ids scripts/launch_vllm.py emits with --include-last-layer.
    invalid = (
        not target_layer_ids
        or min(target_layer_ids) < 0
        or max(target_layer_ids) > num_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    )
    if invalid:
        if explicit:
            raise ValueError(
                f"target_layer_ids must be distinct and within [0, {num_layers}] "
                f"for a verifier with {num_layers} hidden layers, "
                f"got {target_layer_ids}"
            )
        raise ValueError(
            f"Default target layer ids {target_layer_ids} are invalid for a verifier "
            f"with {num_layers} hidden layers; pass --target-layer-ids explicitly."
        )

    if not explicit:
        warnings.warn(
            DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
            stacklevel=3,
        )
    return target_layer_ids


def flatten_rope_parameters(config: PretrainedConfig) -> PretrainedConfig:
    """Flatten nested per-layer-type ``rope_parameters`` for rotary embedding init.

    Models like Laguna store separate rope configs per layer type
    (``sliding_attention``, ``full_attention``). Rotary embedding classes expect
    a flat dict with ``rope_type``/``rope_theta`` at the top level. This helper
    selects the ``sliding_attention`` variant when nested parameters are detected
    and returns a deep-copied config; otherwise returns the original unchanged.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or "sliding_attention" not in rope_params:
        return config
    config = deepcopy(config)
    config.rope_parameters = rope_params["sliding_attention"]
    return config


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
