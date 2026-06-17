import logging
from pathlib import Path

from safetensors import safe_open

logger = logging.getLogger(__name__)


def check_safetensors_file(path: Path, tokens: list[int]):
    with safe_open(path, "pt") as f:
        t_ids = f.get_tensor("token_ids").tolist()
        if t_ids != tokens:
            raise ValueError(
                f"Token ids in {path} don't match expected token ids {tokens}"
            )

        hs = f.get_tensor("hidden_states")
        if hs.isnan().any():
            raise ValueError(
                f"Hidden states in {path} contain NaN values"
            )
        hs_shape = list(hs.shape)
        if len(tokens) != hs_shape[0]:
            raise ValueError(
                f"Sequence length of hidden states {hs_shape[0]} in {path}"
                f" doesn't match num tokens {len(tokens)}"
            )


def get_existing_hidden_state_indices(output_path: Path) -> list[int]:
    """Find existing `hs_i.safetensors` files (where i is the file index)"""

    existing_file_indices_set: set[int] = set()

    if not output_path.exists():
        return []

    for file_path in output_path.iterdir():
        if file_path.name.startswith("hs_") and file_path.name.endswith(".safetensors"):
            index_str = file_path.stem[3:]  # Remove "hs_" prefix
            try:
                file_index = int(index_str)
                existing_file_indices_set.add(file_index)
            except ValueError:
                continue

    return sorted(existing_file_indices_set)


def get_indices_to_process(
    num_samples: int,
    max_samples: int | None,
    existing: list[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed
        world_size: Number of nodes to generate on
        rank: The rank of the local node

    Returns:
        list of dataset indices to process
    """

    target = min(max_samples, num_samples) if max_samples is not None else num_samples

    if target <= 0:
        return []

    chunk_size = target // world_size
    remainder = target % world_size
    # Distribute remainder across the first `remainder` ranks so chunks differ
    # by at most 1.
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)

    existing_s = set(existing)
    to_process = [i for i in range(start, end) if i not in existing_s]

    if not to_process:
        logger.info("All samples for this rank already processed!")
        return []

    if len(existing_s & set(range(start, end))) > 0:
        logger.info(
            f"Found {len(existing_s & set(range(start, end)))} existing samples"
            f" for rank {rank}."
        )

    return to_process
