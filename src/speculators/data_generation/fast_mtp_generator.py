"""FastMTP data generation — captures the last verifier hidden layer and saves samples.

FastMTP needs a single [seq_len, H] hidden state tensor (the verifier's last
layer output) rather than the multiple intermediate layers Eagle3 requires.
:func:`generate_and_save_fast_mtp` wraps :class:`VllmHiddenStatesGenerator`,
selects the last layer automatically, and writes one ``.pt`` file per sequence
together with a ``sample_lengths.json`` index.
"""

import json
from collections.abc import Callable
from pathlib import Path

import torch

from speculators.data_generation._model_utils import num_hidden_layers
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)

__all__ = ["generate_and_save_fast_mtp"]

log = PipelineLogger(__name__)


def _last_hidden_layer(model_path: str) -> int:
    """Return the index of the last hidden layer for *model_path*."""
    return num_hidden_layers(model_path) - 1


def _resolve_loss_mask(
    input_ids: torch.Tensor,
    raw_loss_mask: torch.Tensor | None,
    precomputed: list[int] | None,
    loss_mask_fn: Callable[[list[int]], list[int]] | None,
) -> torch.Tensor:
    """Return a ``long`` loss mask tensor for one sequence.

    Priority (highest to lowest):

    1. *loss_mask_fn* — called with the token IDs; highest flexibility.
    2. *precomputed* — pre-built mask from ``loss_masks[i]``; zero overhead.
    3. *raw_loss_mask* — mask returned by :class:`VllmHiddenStatesGenerator`.
    4. All-ones fallback — train on every position.
    """
    if loss_mask_fn is not None:
        return torch.tensor(loss_mask_fn(input_ids.tolist()), dtype=torch.long)
    if precomputed is not None:
        return torch.tensor(precomputed, dtype=torch.long)
    if raw_loss_mask is not None:
        return raw_loss_mask
    return torch.ones(len(input_ids), dtype=torch.long)


def generate_and_save_fast_mtp(
    model_path: str,
    token_ids: list[list[int]],
    output_dir: Path | str,
    *,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    tensor_parallel_size: int = 1,
    max_num_batched_tokens: int | None = None,
    generate_batch_size: int = 500,
    loss_masks: list[list[int]] | None = None,
    loss_mask_fn: Callable[[list[int]], list[int]] | None = None,
) -> None:
    """Generate FastMTP training data and save one ``.pt`` file per sequence.

    Runs prefill-only inference through *model_path* via vLLM, captures the
    last verifier hidden state (``[seq_len, H]``) for each sequence, and writes::

        {
            "input_ids":     Tensor[seq_len],       # long
            "hidden_states": Tensor[seq_len, H],    # float32, last verifier layer
            "loss_mask":     Tensor[seq_len],       # long, 1 = train here
        }

    A ``sample_lengths.json`` index mapping ``str(index) → seq_len`` is also
    written for fast length look-ups during training.

    :param model_path: HuggingFace model ID or local path for the verifier.
    :param token_ids: Pre-tokenized sequences to process.
    :param output_dir: Destination directory for ``.pt`` files and the index.
    :param max_model_len: Maximum sequence length passed to vLLM.
    :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
    :param tensor_parallel_size: Number of GPUs for tensor parallelism.
    :param max_num_batched_tokens: Maximum tokens processed in one forward pass.
        Defaults to ``max(8192, max_model_len)``. Reducing this value (e.g. to
        2048) lowers peak activation memory, which is useful for large models
        that use memory-intensive attention kernels (e.g. Qwen3-Next's GDN
        linear attention). Has no effect on output quality.
    :param generate_batch_size: Number of sequences passed to the generator per
        call. Hidden states are collected, saved to disk, and freed between
        batches, keeping CPU memory bounded. At mean seq_len 1184 and hidden_dim
        7168, each batch of 500 sequences uses ~8.5 GB of CPU RAM. Lower this
        value if you see CPU OOM; raise it to reduce per-batch overhead.
    :param loss_masks: Pre-computed 0/1 masks, one list per sequence (parallel
        to *token_ids*). Preferred when masks are already available (e.g. from
        :func:`~speculators.data_generation.preprocessing.tokenize_conversations`).
        Mutually exclusive with *loss_mask_fn*.
    :param loss_mask_fn: ``(token_ids: list[int]) -> list[int]`` called per
        sequence when masks must be computed on-the-fly. Mutually exclusive
        with *loss_masks*.  If neither is provided, all positions are trainable.

    :raises ValueError: if both *loss_masks* and *loss_mask_fn* are provided,
        or if *loss_masks* length does not match *token_ids*.

    Example — using pre-computed masks (most common)::

        dataset = tokenize_conversations(raw, tokenizer, max_length=4096)
        token_ids  = [s["input_ids"].tolist() for s in dataset]
        loss_masks = [s["loss_mask"].tolist()  for s in dataset]

        generate_and_save_fast_mtp(
            model_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
            token_ids=token_ids,
            loss_masks=loss_masks,
            output_dir="/path/to/output",
            tensor_parallel_size=8,
        )
    """
    if loss_masks is not None and loss_mask_fn is not None:
        raise ValueError("Provide loss_masks or loss_mask_fn, not both.")
    if loss_masks is not None and len(loss_masks) != len(token_ids):
        raise ValueError(
            f"loss_masks length ({len(loss_masks)}) must match "
            f"token_ids length ({len(token_ids)})."
        )

    last_layer = _last_hidden_layer(model_path)
    log.info(f"FastMTP: capturing layer {last_layer} (last) from {model_path}")

    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=[last_layer],
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(token_ids)
    sample_lengths: dict[str, int] = {}
    global_idx = 0

    for chunk_start in range(0, total, generate_batch_size):
        chunk_end = min(chunk_start + generate_batch_size, total)
        chunk_tids = token_ids[chunk_start:chunk_end]
        chunk_masks = (
            loss_masks[chunk_start:chunk_end] if loss_masks is not None else None
        )

        log.info(
            f"Processing sequences {chunk_start + 1}–{chunk_end} of {total} "
            f"(chunk size {len(chunk_tids)})"
        )

        for i, item in enumerate(generator.generate(chunk_tids)):
            input_ids: torch.Tensor = item["input_ids"]
            # VllmHiddenStatesGenerator returns hidden_states as list[Tensor], one
            # per requested layer. We requested exactly one (the last layer).
            hs = item["hidden_states"]
            hidden_states: torch.Tensor = hs[0] if isinstance(hs, list) else hs
            precomputed = chunk_masks[i] if chunk_masks is not None else None
            loss_mask = _resolve_loss_mask(
                input_ids, item["loss_mask"], precomputed, loss_mask_fn
            )
            torch.save(
                {
                    "input_ids": input_ids,
                    "hidden_states": hidden_states,
                    "loss_mask": loss_mask,
                },
                str(output_dir / f"data_{global_idx}.pt"),
            )
            sample_lengths[str(global_idx)] = len(input_ids)
            global_idx += 1

    with (output_dir / "sample_lengths.json").open("w") as fh:
        json.dump(sample_lengths, fh)

    log.info(f"Saved {total} samples to {output_dir}")
