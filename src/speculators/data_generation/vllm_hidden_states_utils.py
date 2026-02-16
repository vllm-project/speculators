"""Helper utilities for vLLM hidden state generation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

_PROMPT_OVERFLOW_PATTERN = re.compile(
    r"length\s+(\d+)\)\s+is longer than the maximum model length of\s+(\d+)"
)


def ensure_tokenizer_max_token_id(tokenizer: Any) -> None:
    """Backfill tokenizer.max_token_id for tokenizers that do not expose it."""
    if hasattr(tokenizer, "max_token_id"):
        return
    try:
        max_token_id = tokenizer.vocab_size - 1
    except Exception:
        max_token_id = len(tokenizer) - 1
    tokenizer.max_token_id = max_token_id


def infer_num_hidden_layers(config: Any) -> int:
    """Infer model hidden layer count from HF config."""
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(
        config.text_config, "num_hidden_layers"
    ):
        return config.text_config.num_hidden_layers
    raise ValueError("Cannot determine num_layers from config")


def resolve_layer_ids(layer_ids: list[int] | None, num_layers: int) -> list[int]:
    """Resolve default/specified layer IDs and validate index range."""
    resolved = [2, num_layers // 2, num_layers - 3, num_layers - 1]
    if layer_ids is not None:
        resolved = layer_ids
    for layer_id in resolved:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"Layer index {layer_id} out of bounds [0, {num_layers - 1}]"
            )
    return resolved


def normalize_token_ids_batch(token_ids: list[list[int]] | torch.Tensor) -> list:
    """Normalize batch token IDs input to a Python list."""
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if not token_ids:
        raise ValueError("token_ids cannot be empty")
    return token_ids


def validate_multimodal_batch_alignment(
    input_ids_list: list,
    prompt_texts: list[str] | None,
    multimodal_inputs: list[dict] | None,
) -> None:
    """Ensure multimodal-side inputs are present and batch aligned."""
    if multimodal_inputs is None:
        return
    if prompt_texts is None:
        raise ValueError("prompt_texts is required for multimodal inputs")
    if len(prompt_texts) != len(input_ids_list):
        raise ValueError(
            "prompt_texts length must match token_ids length for multimodal"
        )
    if len(multimodal_inputs) != len(input_ids_list):
        raise ValueError("multimodal_inputs length must match token_ids length")


def truncate_text_only_batch(input_ids_list: list, max_len: int) -> list:
    """Truncate text-only token IDs to prefill-safe length."""
    return [ids[:max_len] for ids in input_ids_list]


def to_token_id_list(token_ids: list[int] | torch.Tensor) -> list[int]:
    """Normalize one sequence into a list of token IDs."""
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    return token_ids


def get_request_id(
    request_ids: list[str] | None, request_counter: int, batch_index: int
) -> str:
    """Get caller-provided request ID or a deterministic fallback."""
    if request_ids is not None:
        return request_ids[batch_index]
    return f"req_{request_counter}_{batch_index}"


def parse_prompt_overflow(error_msg: str) -> int | None:
    """Parse prompt overflow count from vLLM length validation errors."""
    match = _PROMPT_OVERFLOW_PATTERN.search(error_msg)
    if not match:
        return None
    prompt_len = int(match.group(1))
    max_len = int(match.group(2))
    overflow = prompt_len - max_len
    return overflow if overflow > 0 else None


def validate_multimodal_engine_features(engine_req: Any) -> None:
    """Validate vLLM multimodal feature payload before Request conversion."""
    if engine_req.mm_features is None:
        raise ValueError(
            "Multimodal request missing mm_features; "
            "prompt may lack multimodal placeholders."
        )

    for mm_index, mm_feature in enumerate(engine_req.mm_features):
        mm_data = mm_feature.data
        if mm_data is None:
            raise ValueError(
                "Multimodal feature data is None; "
                f"feature_index={mm_index} modality={mm_feature.modality}"
            )
        if mm_feature.modality == "image":
            if not hasattr(mm_data, "get") or mm_data.get("image_grid_thw") is None:
                keys = (
                    list(mm_data.keys()) if hasattr(mm_data, "keys") else type(mm_data)
                )
                raise ValueError(
                    "Multimodal image feature missing image_grid_thw; "
                    f"feature_index={mm_index} keys={keys}"
                )


def truncate_engine_prompt_token_ids(
    engine_req: Any,
    *,
    max_len: int,
    request_id: str,
    log_warning: Callable[[str], None],
) -> list[int]:
    """Ensure multimodal prompt token IDs fit max_len and stay synchronized."""
    prompt_token_ids = engine_req.prompt_token_ids
    if prompt_token_ids is None:
        raise ValueError("Multimodal request missing prompt_token_ids")
    if len(prompt_token_ids) > max_len:
        truncated_ids = prompt_token_ids[:max_len]
        engine_req.prompt_token_ids = truncated_ids
        if (
            hasattr(engine_req, "decoder_inputs")
            and engine_req.decoder_inputs is not None
            and hasattr(engine_req.decoder_inputs, "prompt_token_ids")
        ):
            engine_req.decoder_inputs.prompt_token_ids = truncated_ids
        prompt_token_ids = truncated_ids
        log_warning(
            "Multimodal prompt token ids exceeded limit and were "
            f"truncated to {max_len} tokens for request_id={request_id}"
        )
    return prompt_token_ids


def sequence_lengths(input_ids_list: list) -> list[int]:
    """Get per-sample sequence lengths."""
    return [len(ids) for ids in input_ids_list]


def get_embed_length_mismatch(
    captured_input_embeds: torch.Tensor | None, seq_lens: list[int]
) -> tuple[bool, int, int]:
    """Check whether captured embeddings length matches flattened input tokens."""
    if captured_input_embeds is None:
        return False, 0, 0
    expected_num_tokens = sum(seq_lens)
    actual_num_tokens = int(captured_input_embeds.shape[0])
    return (
        actual_num_tokens != expected_num_tokens,
        expected_num_tokens,
        actual_num_tokens,
    )


def build_generation_results(
    *,
    aux_hidden_states: list[torch.Tensor],
    input_ids_list: list,
    captured_input_embeds: torch.Tensor | None,
) -> list[dict[str, Any]]:
    """Slice flattened capture buffers back to sample-level outputs."""
    results: list[dict[str, Any]] = []
    offset = 0
    for i, seq_len in enumerate(sequence_lengths(input_ids_list)):
        layer_states = [
            h[offset : offset + seq_len].clone().cpu() for h in aux_hidden_states
        ]
        input_ids_tensor = torch.as_tensor(input_ids_list[i], dtype=torch.long)
        input_embeds_slice = None
        if captured_input_embeds is not None:
            input_embeds_slice = (
                captured_input_embeds[offset : offset + seq_len].clone().cpu()
            )
        results.append(
            {
                "input_ids": input_ids_tensor,
                "hidden_states": layer_states,
                "loss_mask": None,
                "input_embeds": input_embeds_slice,
            }
        )
        offset += seq_len
    return results
