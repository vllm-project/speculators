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
from transformers import AutoConfig

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)

__all__ = ["generate_and_save_fast_mtp"]

log = PipelineLogger(__name__)


def _last_hidden_layer(model_path: str) -> int:
    """Return the index of the last hidden layer for *model_path*."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers - 1  # type: ignore[no-any-return]
    if hasattr(config, "text_config"):
        return config.text_config.num_hidden_layers - 1  # type: ignore[no-any-return]
    raise ValueError(f"Cannot determine num_hidden_layers from config for {model_path}")


def _resolve_loss_mask(
    input_ids: torch.Tensor,
    raw_loss_mask: torch.Tensor | None,
    loss_mask_fn: Callable[[list[int]], list[int]] | None,
) -> torch.Tensor:
    """Return a ``long`` loss mask tensor for one sequence.

    Priority: explicit *loss_mask_fn* > pre-computed *raw_loss_mask* from the
    generator > all-ones fallback (train on every position).
    """
    if loss_mask_fn is not None:
        return torch.tensor(loss_mask_fn(input_ids.tolist()), dtype=torch.long)
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
    :param loss_mask_fn: ``(token_ids: list[int]) -> list[int]`` returning a
        0/1 mask (1 = compute loss).  If ``None``, all positions are trainable.

    Example::

        generate_and_save_fast_mtp(
            model_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
            token_ids=tokenized_conversations,
            output_dir="/path/to/output",
            tensor_parallel_size=8,
            loss_mask_fn=my_mask_fn,
        )
    """
    last_layer = _last_hidden_layer(model_path)
    log.info(f"FastMTP: capturing layer {last_layer} (last) from {model_path}")

    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=[last_layer],
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_lengths: dict[str, int] = {}
    for i, item in enumerate(generator.generate(token_ids)):
        input_ids: torch.Tensor = item["input_ids"]
        # VllmHiddenStatesGenerator returns hidden_states as list[Tensor], one
        # per requested layer. We requested exactly one (the last layer).
        hs = item["hidden_states"]
        hidden_states: torch.Tensor = hs[0] if isinstance(hs, list) else hs
        loss_mask = _resolve_loss_mask(input_ids, item["loss_mask"], loss_mask_fn)
        torch.save(
            {
                "input_ids": input_ids,
                "hidden_states": hidden_states,
                "loss_mask": loss_mask,
            },
            str(output_dir / f"data_{i}.pt"),
        )
        sample_lengths[str(i)] = len(input_ids)

    with (output_dir / "sample_lengths.json").open("w") as fh:
        json.dump(sample_lengths, fh)

    log.info(f"Saved {len(token_ids)} samples to {output_dir}")
