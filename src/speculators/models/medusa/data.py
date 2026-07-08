from speculators.train.data import BatchType

__all__ = ["shift_batch_medusa"]


def shift_batch_medusa(batch: BatchType) -> BatchType:
    """Rename verifier_last_hidden_states to hidden_states for Medusa.

    No token-level shifting — the Medusa forward pass handles alignment
    internally via per-head offset slicing of input_ids.
    """
    return {
        "input_ids": batch["input_ids"],
        "hidden_states": batch["verifier_last_hidden_states"],
        "loss_mask": batch["loss_mask"],
        "lengths": batch["lengths"],
        "position_ids": batch["position_ids"],
    }
