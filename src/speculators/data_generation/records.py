"""Training records shared by response regeneration and data preparation."""

from typing import Any, TypedDict

from typing_extensions import NotRequired


class PreparedSample(TypedDict):
    """Token IDs and their supervision mask, with optional serving messages.

    Messages carry media that token IDs alone cannot represent. IDs, provenance,
    and readable conversations belong to the enclosing regeneration row.
    Preparation validates and finalizes these fields before saving the dataset.
    """

    input_ids: list[int]
    loss_mask: list[int]
    messages: NotRequired[list[dict[str, Any]]]


def build_boundary_sample(
    input_ids: list[int],
    boundary: int,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> PreparedSample:
    """Mark tokens before the boundary as context and the rest as supervised."""
    if not 0 <= boundary <= len(input_ids):
        raise ValueError(
            f"Boundary {boundary} is outside a sequence of length {len(input_ids)}"
        )
    sample: PreparedSample = {
        "input_ids": input_ids,
        "loss_mask": [0] * boundary + [1] * (len(input_ids) - boundary),
    }
    if messages is not None:
        sample["messages"] = messages
    return sample
