"""Convert on-policy conversations or prepared rows into training datasets."""

import logging
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset

from speculators.data_generation.records import (
    PreparedSample,
    adapt_conversation_for_vllm,
    normalize_conversation,
    parse_tools,
    prepared_sample,
)
from speculators.data_generation.render_client import render_conversation
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_speculator_training_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

logger = logging.getLogger(__name__)


class BoundaryUnstableError(ValueError):
    """The chat template is not prefix-stable at an assistant boundary."""


def _encode_render(
    conversation: list[dict],
    endpoint: str,
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None,
) -> list[int]:
    return render_conversation(
        endpoint,
        adapt_conversation_for_vllm(conversation),
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _render_boundary_samples(
    conversation: list[dict],
    endpoint: str,
    max_length: int,
    *,
    tools: list[dict] | None = None,
) -> list[PreparedSample]:
    """Render one prepared sample for every non-leading assistant turn."""
    samples: list[PreparedSample] = []
    for index, turn in enumerate(conversation):
        if index == 0 or turn["role"] != "assistant":
            continue

        history = conversation[:index]
        prompt_ids = _encode_render(
            history,
            endpoint,
            add_generation_prompt=True,
            tools=tools,
        )
        if len(prompt_ids) >= max_length:
            continue

        prefix = conversation[: index + 1]
        full_ids = _encode_render(
            prefix,
            endpoint,
            add_generation_prompt=False,
            tools=tools,
        )
        if full_ids[: len(prompt_ids)] == prompt_ids:
            boundary = len(prompt_ids)
        else:
            boundary = _common_prefix_len(prompt_ids, full_ids)
            history_ids = _encode_render(
                history,
                endpoint,
                add_generation_prompt=False,
                tools=tools,
            )
            if full_ids[: len(history_ids)] != history_ids or boundary < len(
                history_ids
            ):
                raise BoundaryUnstableError(
                    "prompt and full renders diverge inside history at "
                    f"assistant turn {index}"
                )

        samples.append(
            prepared_sample(
                full_ids,
                boundary,
                messages=adapt_conversation_for_vllm(prefix),
            )
        )
    return samples


def _warn_truncation(unsupervised: int, clipped: int) -> None:
    if unsupervised:
        logger.warning(
            "Dropped %d rows with no supervised tokens after truncation",
            unsupervised,
        )
    if clipped:
        logger.warning(
            "Clipped %d assistant responses at --seq-length; increase it to "
            "supervise those responses completely",
            clipped,
        )


def _finalize_samples(
    samples: list[PreparedSample],
    max_length: int,
    minimum_valid_tokens: int | None,
) -> dict[str, list]:
    """Validate, truncate, filter, and tensorize canonical prepared samples."""
    results: dict[str, list] = {
        "input_ids": [],
        "loss_mask": [],
        "seq_len": [],
        "messages": [],
    }
    unsupervised = 0
    clipped = 0

    for sample in samples:
        input_ids = list(sample["input_ids"])
        loss_mask = list(sample["loss_mask"])
        if len(input_ids) != len(loss_mask):
            raise ValueError(
                "Prepared row shape mismatch: "
                f"input_ids={len(input_ids)}, loss_mask={len(loss_mask)}"
            )
        if any(value not in (0, 1) for value in loss_mask):
            raise ValueError("Prepared row loss_mask must contain only 0 and 1")

        was_clipped = len(input_ids) > max_length
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        valid_tokens = sum(loss_mask)
        if valid_tokens == 0:
            unsupervised += 1
            continue
        if minimum_valid_tokens is not None and valid_tokens < minimum_valid_tokens:
            continue

        results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
        results["loss_mask"].append(torch.tensor(loss_mask, dtype=torch.long))
        results["seq_len"].append(len(input_ids))
        results["messages"].append(sample.get("messages", []))
        clipped += was_clipped

    _warn_truncation(unsupervised, clipped)
    return results


def _finalize_prepared_batch(
    examples: dict,
    max_length: int,
    minimum_valid_tokens: int | None,
) -> dict[str, list]:
    samples: list[PreparedSample] = [
        {"input_ids": input_ids, "loss_mask": loss_mask}
        for input_ids, loss_mask in zip(
            examples["input_ids"], examples["loss_mask"], strict=True
        )
    ]
    return _finalize_samples(samples, max_length, minimum_valid_tokens)


def _render_conversation_batch(
    examples: dict,
    endpoint: str,
    max_length: int,
    minimum_valid_tokens: int | None,
) -> dict[str, list]:
    conversations = examples.get("messages")
    if conversations is None:
        conversations = examples.get("conversations")
    if conversations is None:
        raise ValueError("Natural-language input requires messages or conversations")

    tools_column = examples.get("tools")
    if tools_column is not None and len(tools_column) != len(conversations):
        raise ValueError("tools and conversation columns have different lengths")

    samples: list[PreparedSample] = []
    accepted_conversations = 0
    failed_conversations = 0
    for index, raw_conversation in enumerate(conversations):
        conversation = normalize_conversation(raw_conversation)
        if not conversation:
            continue
        accepted_conversations += 1
        try:
            tools = parse_tools(tools_column[index]) if tools_column else None
            samples.extend(
                _render_boundary_samples(
                    conversation,
                    endpoint,
                    max_length,
                    tools=tools,
                )
            )
        except Exception as exc:
            logger.error(
                "Skipping conversation %d after %s: %s",
                index,
                type(exc).__name__,
                exc,
            )
            failed_conversations += 1

    if failed_conversations:
        logger.warning(
            "%d/%d conversations could not be rendered",
            failed_conversations,
            accepted_conversations,
        )
    if len(samples) > accepted_conversations:
        logger.info(
            "Per-turn fan-out: %d conversations -> %d rows",
            accepted_conversations,
            len(samples),
        )
    return _finalize_samples(samples, max_length, minimum_valid_tokens)


def build_speculator_training_dataset(
    dataset: HFDataset,
    max_length: int = 2048,
    num_proc: int = 8,
    *,
    render_endpoint: str | None = None,
    minimum_valid_tokens: int | None = None,
) -> HFDataset:
    """Convert one on-policy dataset to the canonical training representation.

    Natural-language rows must contain complete target-model conversations in a
    ``messages`` or ``conversations`` column. The vLLM render endpoint converts
    them to token IDs; it never generates responses. Rows already containing
    ``input_ids`` and ``loss_mask`` skip rendering.
    """
    columns = set(dataset.column_names)
    prepared = {"input_ids", "loss_mask"} <= columns
    conversational = bool({"messages", "conversations"} & columns)

    if prepared:
        logger.info("Prepared rows: validating without rendering")

        def map_batch(examples):
            return _finalize_prepared_batch(examples, max_length, minimum_valid_tokens)

    elif conversational and render_endpoint is not None:
        logger.info("Rendering on-policy conversations via %s", render_endpoint)

        def map_batch(examples):
            return _render_conversation_batch(
                examples,
                render_endpoint,
                max_length,
                minimum_valid_tokens,
            )

    elif conversational:
        raise ValueError(
            "render_endpoint is required for natural-language on-policy data"
        )
    else:
        raise ValueError(
            "Input must contain messages/conversations or input_ids/loss_mask"
        )

    original_columns = dataset.column_names
    dataset = dataset.map(
        map_batch,
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_columns,
        keep_in_memory=True,
    )
    dataset.set_format(type="torch")
    return dataset


def _load_hf_dataset(spec: str) -> HFDataset:
    """Load ``hf:<dataset>[:<subset>:<split>]`` on-policy data."""
    subset: str | None
    match spec.removeprefix("hf:").split(":"):
        case [dataset_id]:
            subset, split = None, "train"
        case [dataset_id, split]:
            subset = None
        case [dataset_id, subset, split]:
            pass
        case _:
            raise ValueError(
                f"Invalid hf: spec {spec!r}; expected hf:<dataset>[:<subset>:<split>]"
            )

    if not dataset_id or not split or subset == "":
        raise ValueError(f"Invalid hf: spec {spec!r}")
    return load_dataset(dataset_id, name=subset, split=split)


def load_raw_dataset(source: str) -> HFDataset:
    """Load user-provided on-policy JSON/JSONL data or an ``hf:`` dataset."""
    if source.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=source, split="train")

    path = Path(source)
    if path.is_dir():
        files = sorted(
            str(file) for file in (*path.rglob("*.json"), *path.rglob("*.jsonl"))
        )
        if not files:
            raise ValueError(f"No .json/.jsonl files found in directory: {source}")
        return load_dataset("json", data_files=files, split="train")

    if source.startswith("hf:"):
        return _load_hf_dataset(source)

    raise ValueError(
        f"Unsupported input {source!r}. Use a local JSON/JSONL path or an "
        "hf:<dataset> spec. Raw source presets belong to response regeneration."
    )


def load_and_preprocess_dataset(
    train_data_paths: list[str],
    *,
    seq_length: int,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: int | None = None,
    token_freq_path: Path | str = "./token_freq.pt",  # noqa: S107
    render_endpoint: str | None = None,
    minimum_valid_tokens: int | None = None,
    allow_empty_output: bool = False,
) -> HFDataset:
    """Load and combine on-policy conversations or canonical prepared rows."""
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")

    processed_datasets = []
    for source in train_data_paths:
        logger.info("Loading %s", source)
        dataset = load_raw_dataset(source).shuffle(seed=seed)
        if max_samples is not None and len(dataset) > 3 * max_samples:
            dataset = dataset.select(range(3 * max_samples))

        processed_datasets.append(
            build_speculator_training_dataset(
                dataset,
                max_length=seq_length,
                num_proc=build_dataset_num_proc,
                render_endpoint=render_endpoint,
                minimum_valid_tokens=minimum_valid_tokens,
            )
        )

    combined = concatenate_datasets(processed_datasets).shuffle(seed=seed)
    if max_samples is not None and len(combined) > max_samples:
        combined = combined.select(range(max_samples))
    if len(combined) == 0 and not allow_empty_output:
        raise ValueError(
            "No samples remain after preparation; check the input schema and "
            "sequence-length filters"
        )

    save_token_frequency_distribution(combined, token_freq_path)
    return combined
