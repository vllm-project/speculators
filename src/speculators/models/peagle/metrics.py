import torch


def compute_metrics(
    loss: torch.Tensor,
    prediction_loss: float | torch.Tensor,
    accuracy: torch.Tensor,
    num_tokens: torch.Tensor | float,
    all_indices: torch.Tensor,
    seq_length: int,
    para_depth: int,
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    device: torch.device,
    epsilon: float = 1e-5,
) -> dict[str, torch.Tensor]:
    per_position_accuracy = {}
    depths = all_indices // seq_length
    pred_tokens_flat = pred_tokens.squeeze(0)
    target_tokens_flat = target_tokens.squeeze(0)

    for depth in range(min(para_depth, 10)):
        depth_mask = depths == depth
        has_depth = depth_mask.sum() > 0
        if has_depth:
            depth_correct = (
                (pred_tokens_flat == target_tokens_flat).float()
                * depth_mask.float()
            ).sum()
            depth_total = depth_mask.sum().float()
            depth_accuracy = depth_correct / (depth_total + epsilon)
        else:
            depth_accuracy = torch.tensor(0.0, device=device)
        per_position_accuracy[f"position {depth} acc"] = depth_accuracy.detach()
        per_position_accuracy[f"position {depth} count"] = torch.tensor(
            1.0 if has_depth else 0.0, device=device
        )

    if not isinstance(num_tokens, torch.Tensor):
        num_tokens = torch.tensor(num_tokens, device=device)

    return {
        "loss": loss.detach(),
        "full_acc": accuracy.detach(),
        "num_tokens": num_tokens.detach(),
        **per_position_accuracy,
    }
