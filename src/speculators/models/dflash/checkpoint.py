"""DFlash-family checkpoint names, independent of physical module ownership."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping

import torch

BACKBONE_NAMES = frozenset({"layers", "fc", "hidden_norm", "norm", "rotary_emb"})
LEGACY_KEY_MAPPING = {
    r"^(layers|fc|hidden_norm|norm|rotary_emb)\.": r"backbone.\1.",
}


def native_key(key: str) -> str:
    return f"backbone.{key}" if key.split(".", 1)[0] in BACKBONE_NAMES else key


def legacy_key(key: str) -> str:
    if key.startswith("backbone."):
        candidate = key.removeprefix("backbone.")
        if candidate.split(".", 1)[0] in BACKBONE_NAMES:
            return candidate
    return key


def checkpoint_key_map(keys: Iterable[str]) -> dict[str, str]:
    """Validate source names before loading any tensors, including shard headers."""
    result = {}
    destinations = set()
    layouts = set()
    for key in keys:
        destination = native_key(key)
        if destination in destinations:
            raise ValueError(f"Checkpoint keys collide at {destination!r}")
        destinations.add(destination)
        result[key] = destination
        if key != destination:
            layouts.add("legacy")
        elif legacy_key(key) != key:
            layouts.add("native")
    if len(layouts) > 1:
        raise ValueError("Checkpoint mixes legacy and native backbone keys")
    return result


def import_state_dict(state_dict: Mapping[str, torch.Tensor]):
    """Rename keys without copying tensor storage or changing caller dictionaries."""
    mapping = checkpoint_key_map(state_dict)
    renamed = OrderedDict((mapping[key], value) for key, value in state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        renamed.__dict__["_metadata"] = {
            native_key(key): value for key, value in metadata.items()
        }
    return renamed


def export_state_dict(state_dict: Mapping[str, torch.Tensor], *, native: bool):
    renamed = import_state_dict(state_dict)
    if native:
        return renamed
    return OrderedDict((legacy_key(key), value) for key, value in renamed.items())
