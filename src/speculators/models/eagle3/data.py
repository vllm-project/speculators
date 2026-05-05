from speculators.train.data import BatchType

__all__ = ["shift_batch"]


def shift_batch(batch: BatchType):
    input_ids = batch["input_ids"]  # shape: [seq_len]
    # [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    hidden_states = batch["hidden_states"]  # shape: [seq_len, hidden_size]
    # [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
    verifier_last_hidden_states = batch[
        "verifier_last_hidden_states"
    ]  # shape: [seq_len, hidden_size]
    # [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]
    loss_mask = batch["loss_mask"]  # shape: [seq_len]
    # [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9]
    lengths = batch["lengths"]  # shape: [1]
    # [10]
    position_ids = batch["position_ids"]  # shape: [seq_len]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Need to align (x1, g0, y1, l1)

    # Drop x0, g(-1), y0, l0, reduce seq_len by 1

    input_ids = input_ids[1:]
    hidden_states = hidden_states[:-1]
    verifier_last_hidden_states = verifier_last_hidden_states[1:]
    loss_mask = loss_mask[1:]
    lengths = lengths - 1
    position_ids = position_ids[1:]  # Note: position_ids now start at 1

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "verifier_last_hidden_states": verifier_last_hidden_states,
        "loss_mask": loss_mask,
        "lengths": lengths,
        "position_ids": position_ids,
    }
