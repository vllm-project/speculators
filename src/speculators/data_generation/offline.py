import logging
from pathlib import Path

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def check_hidden_states(data: dict, tokens: list[int]):
    required = {"token_ids", "hidden_states"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Hidden-state payload is missing keys: {missing}")

    t_ids = data["token_ids"].tolist()
    if t_ids != tokens:
        raise ValueError(f"Token ids don't match expected token ids {tokens}")

    hs = data["hidden_states"]
    if not isinstance(hs, torch.Tensor):
        raise ValueError(f"Hidden states must be a tensor, got {type(hs).__name__}")
    if len(tokens) != hs.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hs.shape[0]}"
            f" doesn't match num tokens {len(tokens)}"
        )

    lo, hi = torch.aminmax(hs)
    if bool(torch.isfinite(torch.stack((lo, hi))).all()):
        return

    raise ValueError(f"Hidden states contain non-finite values (min={lo}, max={hi})")


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


def find_corrupt_hidden_state_indices(
    output_path: Path, indices: list[int]
) -> dict[int, str]:
    """Find cached `hs_i.safetensors` files that cannot be opened.

    Returns a mapping of file index to the reason that file is unreadable.

    A file can be present and still be unusable. When the directory vLLM stages
    hidden states in and ``output_path`` are on different filesystems, the
    ``shutil.move`` in ``generate_offline_data`` hits ``EXDEV`` and falls back to
    a non-atomic byte copy, so a run interrupted mid-copy leaves a truncated
    file at its final name. :func:`get_existing_hidden_state_indices` only
    inspects filenames and reports such a file as done, so every later run skips
    it and training fails when it reads it.

    ``safe_open`` performs safetensors' own header validation in its constructor
    -- header length, header JSON, offset contiguity, dtype/shape/offset
    consistency, and exact buffer coverage -- without reading any tensor data, so
    the cost scales with the file count rather than the tensor size. Indices
    whose file is absent are skipped; absence is already handled as "not generated".

    Note that this cannot detect corruption that preserves the file's size: the
    format carries no checksum, so bit-rot inside the tensor data still requires
    loading it (see :func:`check_hidden_states`).
    """
    corrupt: dict[int, str] = {}
    for idx in indices:
        path = output_path / f"hs_{idx}.safetensors"
        try:
            with safe_open(path, framework="pt"):
                pass
        except FileNotFoundError:
            continue
        except Exception as e:
            corrupt[idx] = str(e) or type(e).__name__
    return corrupt


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
