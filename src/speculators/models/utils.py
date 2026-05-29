import warnings

import torch
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


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Shared utility used by DFlash and PEagle for anchor-based attention.

    Args:
        loss_mask: Binary mask indicating valid positions [1, total_seq_len]
        num_anchors: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded).
            For DFlash, this is the draft block size. For PEagle, use 1.

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [num_anchors]
            - anchor_valid: Boolean mask for valid anchors [num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size:] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())
    if k > 0:
        perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
        anchors[:k] = valid_indices[perm[:k]]
        anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]
