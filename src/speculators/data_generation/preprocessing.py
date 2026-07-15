import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.render_client import render_conversation
from speculators.data_generation.torch_utils import set_default_torch_num_threads
from speculators.train.vocab_mapping import save_token_frequency_distribution

__all__ = [
    "build_eagle3_dataset",
    "load_and_preprocess_dataset",
    "load_raw_dataset",
]

log = PipelineLogger(__name__)


ProcessorLike = PreTrainedTokenizerBase | ProcessorMixin


def _visualize_sample(preprocessed: HFDataset, processor: ProcessorLike, idx: int = 0):
    """Visualize a single sample with color-coded trainable regions."""
    # Get preprocessed sample
    prep_sample = preprocessed[idx]
    input_ids = prep_sample["input_ids"].tolist()
    loss_mask = prep_sample["loss_mask"].tolist()

    log.info(f"SAMPLE #{idx}")
    log.info("HIGHLIGHTED TEXT (BLUE = trainable, GREY = masked)")

    # Create color-highlighted text
    blue = "\033[38;5;153m"  # Very light blue text for trainable tokens
    grey = "\033[90m"  # Grey text for masked tokens
    reset = "\033[0m"  # Reset color

    output = []
    prev_state = None

    for i in range(len(input_ids)):
        is_train = loss_mask[i] == 1
        token = processor.decode([input_ids[i]])
        assert isinstance(token, str)

        # Switch colors when state changes
        if is_train != prev_state:
            output.append(blue if is_train else grey)
            prev_state = is_train

        output.append(token)

    output.append(reset)
    highlighted = "".join(output)

    log.info(highlighted)


def _normalize_conversation(
    conv: list[dict],
) -> list[dict]:
    """Normalize conversation to standard format with role/content keys.

    Args:
        conv: Raw conversation turns

    Returns:
        Normalized conversation
    """
    normalized = []
    for turn in conv:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value") or turn.get("content") or ""

        # Map various role names to standard user/assistant
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "system":
            role = "system"
        elif role == "tool":
            role = "tool"
        else:
            log.warning(f"Unknown role '{role}', skipping turn")
            continue

        # Build normalized turn with role and content
        normalized_turn = {"role": role, "content": content}

        # Preserve tool_calls and tool_call_id if present
        if turn.get("tool_calls"):
            normalized_turn["tool_calls"] = turn["tool_calls"]
        if turn.get("tool_call_id"):
            normalized_turn["tool_call_id"] = turn["tool_call_id"]

        thinking = turn.get("thinking") or turn.get("reasoning_content")
        if thinking:
            normalized_turn["thinking"] = thinking
            normalized_turn["reasoning_content"] = thinking

        normalized.append(normalized_turn)

    return normalized


def _adapt_part_for_vllm(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    if part_type == "text":
        return {"type": "text", "text": part["text"]}

    for modality in ("image", "video", "audio"):
        if part_type == modality:
            if local_path := part.get("path"):
                file_url = f"file://{Path(local_path).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            if part.get("base64"):
                expr = {"type": modality, "base64": "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )
            if part.get(modality):
                expr = {"type": modality, modality: "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )

            expr = {"type": modality} | {k: "..." for k in part if k != "type"}
            raise NotImplementedError(f"Unknown content part: {expr}")

    expr = dict.fromkeys(part.keys(), "...")
    raise NotImplementedError(f"Unknown content part: {expr}")


def _adapt_turn_for_vllm(turn: dict):
    if isinstance(turn["content"], str):
        return turn

    return turn | {"content": [_adapt_part_for_vllm(part) for part in turn["content"]]}


def _adapt_conv_for_vllm(normalized_conv: list[dict]):
    return [_adapt_turn_for_vllm(turn) for turn in normalized_conv]


class BoundaryUnstableError(ValueError):
    """The chat template is not prefix-stable at an assistant turn boundary."""


def _encode_render(
    conv_prefix: list[dict],
    render_endpoint: str,
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
) -> list[int]:
    """Render a conversation prefix via the vLLM ``/render`` endpoint.

    Returns the token ids only; the loss mask is derived from the boundary
    between two renders, not from the server's tag-gated assistant mask.
    """
    messages = _adapt_conv_for_vllm(conv_prefix)
    return render_conversation(
        render_endpoint,
        messages,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


class _Turn(NamedTuple):
    """One assistant turn's renders: context ends at ``boundary`` in ``full_ids``."""

    prompt_ids: list[int]
    boundary: int
    full_ids: list[int]
    idx: int


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    length = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        length += 1
    return length


def _render_boundary_rows(
    normalized_conv: list[dict],
    render_endpoint: str,
    max_length: int,
    *,
    tools: list[dict] | None = None,
) -> list[dict]:
    """Build training rows whose loss mask is the render boundary of each
    assistant turn.

    For assistant turn ``j``, the boundary is the longest common token prefix
    of the ``conv[:j]`` generation-prompt render and the ``conv[:j+1]`` full
    render: everything before it is context (mask 0), the tokens past it are
    supervised (mask 1) -- the same prompt/completion boundary the serving
    engine reports for on-policy data, reconstructed from the model's own
    chat template. No markers, no regex. Usually the whole prompt render is
    that prefix; when the generation prompt itself diverges (e.g. a
    pre-filled empty ``<think>`` scaffold while the data records real
    reasoning), the boundary sits where the renders part ways -- which is
    where generation starts on auto-opened thinking templates -- guarded by
    the history render staying a token-prefix of the full render.

    Emission is packed into a single row when the template renders append-only
    (each turn's full render is a token-prefix of the next turn's prompt
    render). Templates that rewrite history -- e.g. reasoning templates that
    strip past ``<think>`` blocks -- break that chain and fan out to one row
    per assistant turn, which supervises each turn's completion against the
    same context inference would see.

    Turns whose context alone reaches ``max_length`` are not rendered: their
    supervised span would start past the window, so they could only produce
    dropped rows.

    Raises:
        BoundaryUnstableError: The renders diverge inside history, so no
            boundary mask can be derived.
    """
    # An assistant turn with no preceding context (i == 0) has no prompt to
    # render a boundary against; keep it as context for later turns only.
    assistant_indices = [
        i
        for i, turn in enumerate(normalized_conv)
        if turn["role"] == "assistant" and i > 0
    ]

    turns: list[_Turn] = []
    for j in assistant_indices:
        prompt_ids = _encode_render(
            normalized_conv[:j],
            render_endpoint,
            add_generation_prompt=True,
            tools=tools,
        )
        if len(prompt_ids) >= max_length:
            # Context already fills the window; this and every later turn
            # could only yield rows with no supervised tokens in-window.
            break
        full_ids = _encode_render(
            normalized_conv[: j + 1],
            render_endpoint,
            add_generation_prompt=False,
            tools=tools,
        )
        if full_ids[: len(prompt_ids)] == prompt_ids:
            boundary = len(prompt_ids)
        else:
            # The generation prompt diverges from the completed turn (e.g.
            # Qwen3.5's no-think scaffold vs recorded reasoning). The boundary
            # is the common prefix, valid only if the divergence is confined
            # to the generation-prompt tail: history itself must agree.
            boundary = _common_prefix_len(prompt_ids, full_ids)
            hist_ids = _encode_render(
                normalized_conv[:j],
                render_endpoint,
                add_generation_prompt=False,
                tools=tools,
            )
            if full_ids[: len(hist_ids)] != hist_ids or boundary < len(hist_ids):
                raise BoundaryUnstableError(
                    f"prompt and full renders diverge inside history at "
                    f"assistant turn {j}; cannot derive a boundary loss mask"
                )
        turns.append(_Turn(prompt_ids, boundary, full_ids, j))

    if not turns:
        return []

    # Packed: append-only chain holds, all spans valid in the final render.
    append_only = all(
        nxt.prompt_ids[: len(cur.full_ids)] == cur.full_ids
        for cur, nxt in zip(turns, turns[1:], strict=False)
    )
    if append_only:
        input_ids = turns[-1].full_ids
        loss_mask = [0] * len(input_ids)
        for turn in turns:
            loss_mask[turn.boundary : len(turn.full_ids)] = [1] * (
                len(turn.full_ids) - turn.boundary
            )
        # Trailing non-assistant messages are dropped: nothing after the last
        # assistant turn is supervised or conditioned on.
        return [
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "conv": normalized_conv[: turns[-1].idx + 1],
            }
        ]

    # Fan-out: history is rewritten between turns (e.g. stripped thinking).
    return [
        {
            "input_ids": turn.full_ids,
            "loss_mask": [0] * turn.boundary
            + [1] * (len(turn.full_ids) - turn.boundary),
            "conv": normalized_conv[: turn.idx + 1],
        }
        for turn in turns
    ]


def _parse_conv_tools(conv_tools: object, idx: int) -> list | None:
    """Parse the tools JSON string for one conversation; warn and return None
    on invalid JSON or unexpected types."""
    if not conv_tools:
        return None
    if isinstance(conv_tools, list):
        return conv_tools
    if not isinstance(conv_tools, str):
        log.warning(
            f"Non-string value in tools column for conversation {idx}: "
            f"{type(conv_tools).__name__}, proceeding without tools"
        )
        return None
    try:
        return json.loads(conv_tools)
    except json.JSONDecodeError as e:
        log.warning(
            f"Invalid JSON in tools column for conversation {idx}: {e}, "
            "proceeding without tools"
        )
        return None


def _append_row(
    results: dict[str, list],
    input_ids: list[int],
    loss_mask: list[int],
    max_length: int,
    minimum_valid_tokens: int | None,
) -> Literal["kept", "unsupervised", "filtered"]:
    """Clip a row to the window, filter it, and tensorize it into ``results``.

    Rows with no supervised tokens in-window contribute zero gradient and are
    dropped ("unsupervised"); rows below ``minimum_valid_tokens`` are
    "filtered".
    """
    input_ids = input_ids[:max_length]
    loss_mask = loss_mask[:max_length]
    num_valid_tokens = sum(loss_mask)
    if num_valid_tokens == 0:
        return "unsupervised"
    if minimum_valid_tokens is not None and num_valid_tokens < minimum_valid_tokens:
        return "filtered"
    results["input_ids"].append(torch.tensor(input_ids, dtype=torch.long))
    results["loss_mask"].append(torch.tensor(loss_mask, dtype=torch.long))
    results["seq_len"].append(len(input_ids))
    return "kept"


def _warn_unsupervised(num_dropped: int) -> None:
    if num_dropped:
        log.warning(
            f"Dropped {num_dropped} rows with no supervised tokens. "
            f"If unexpected, consider increasing --seq-length to avoid "
            f"truncating assistant responses."
        )


def _passthrough_pretokenized(
    examples: dict, max_length: int, minimum_valid_tokens: int | None = None
) -> dict[str, list]:
    """Carry pre-tokenized ``(input_ids, loss_mask)`` rows through, truncated only.

    On-policy regeneration already applied the boundary as the mask, so these rows
    need no rendering.
    """
    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    num_unsupervised = 0
    for ids, mask in zip(examples["input_ids"], examples["loss_mask"], strict=True):
        # `strict=True` only pairs the columns; a per-row skew would survive it and
        # the collator packs each key independently, silently shifting the mask
        # against the ids for every sample packed after this one.
        if len(ids) != len(mask):
            raise ValueError(
                f"Pre-tokenized row shape mismatch: "
                f"input_ids={len(ids)}, loss_mask={len(mask)}"
            )
        status = _append_row(results, ids, mask, max_length, minimum_valid_tokens)
        num_unsupervised += status == "unsupervised"
    _warn_unsupervised(num_unsupervised)
    return results


def _preprocess_batch(
    examples: dict,
    is_multimodal: bool,
    render_endpoint: str | None,
    max_length: int,
    minimum_valid_tokens: int | None = None,
) -> dict[str, list]:
    """Process a batch of conversations into tokenized rows with boundary masks."""

    # On-policy regeneration rows are already masked (boundary); pass them through
    # instead of re-rendering.
    if "input_ids" in examples and "loss_mask" in examples:
        return _passthrough_pretokenized(examples, max_length, minimum_valid_tokens)

    if render_endpoint is None:
        raise ValueError(
            "render_endpoint is required to derive loss masks for off-policy "
            "conversations"
        )

    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    conversations: list[dict] = examples.get("conversations", [])

    # MM inputs are extracted via the Chat Completions API, which needs the
    # original messages -- token ids alone cannot carry the images.
    if is_multimodal:
        results["messages"] = []

    if not conversations:
        log.warning(f"No conversations key found. Keys: {list(examples.keys())}")
        return results

    tools_col = examples.get("tools")
    if tools_col is not None and len(tools_col) != len(conversations):
        log.warning(
            f"Tools column length ({len(tools_col)}) does not match "
            f"conversations length ({len(conversations)}), proceeding without tools"
        )
        tools_col = None

    num_unsupervised = 0
    num_convs_in = 0
    num_convs_empty = 0

    for idx, conv in enumerate(conversations):
        conv_tools = tools_col[idx] if tools_col is not None else None

        if not conv or not isinstance(conv, list):
            continue

        normalized_conv = _normalize_conversation(conv)
        if not normalized_conv:
            continue

        parsed_tools = _parse_conv_tools(conv_tools, idx)
        num_convs_in += 1

        try:
            rows = _render_boundary_rows(
                normalized_conv,
                render_endpoint,
                max_length,
                tools=parsed_tools,
            )
        # One row the render endpoint or boundary derivation can't handle must
        # not kill the run. The failure modes can't be enumerated -- templates
        # are swappable and raise arbitrary types -- so catch broadly and skip.
        except Exception as e:
            log.error(f"Failed to process conversation {idx}: {type(e).__name__}: {e}")
            num_convs_empty += 1
            continue

        num_kept = 0
        for row in rows:
            status = _append_row(
                results,
                row["input_ids"],
                row["loss_mask"],
                max_length,
                minimum_valid_tokens,
            )
            num_unsupervised += status == "unsupervised"
            if status == "kept":
                num_kept += 1
                if "messages" in results:
                    results["messages"].append(_adapt_conv_for_vllm(row["conv"]))
        num_convs_empty += num_kept == 0

    _warn_unsupervised(num_unsupervised)
    if num_convs_empty:
        log.warning(
            f"{num_convs_empty}/{num_convs_in} conversations produced no training "
            f"rows (no assistant turn with context, unstable template, or fully "
            f"truncated)"
        )
    num_rows = len(results["input_ids"])
    if num_rows > num_convs_in:
        log.info(f"Per-turn fan-out: {num_convs_in} conversations -> {num_rows} rows")

    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    processor: ProcessorLike,
    max_length: int = 2048,
    num_proc: int = 8,
    *,
    render_endpoint: str | None = None,
    minimum_valid_tokens: int | None = None,
) -> HFDataset:
    """Build an EAGLE3 dataset with render-boundary loss masks.

    Off-policy conversations are tokenized by the vLLM ``/render`` endpoint and
    masked at the render boundary of each assistant turn (see
    ``_render_boundary_rows``); append-only templates keep one row per
    conversation, history-rewriting templates (e.g. reasoning models) fan out
    to one row per assistant turn. Pre-tokenized rows (on-policy regeneration)
    carry their own boundary mask and pass straight through.

    Args:
        dataset: Raw dataset with conversations, or pre-tokenized rows.
        processor: Processor, used to detect multimodal inputs and to decode.
        max_length: Maximum sequence length.
        num_proc: Number of worker processes; each renders concurrently.
        render_endpoint: Base URL of a vLLM server. Required unless the dataset
            is already pre-tokenized.
        minimum_valid_tokens: Minimum supervised tokens for a row to be kept.
    """
    original_cols = dataset.column_names
    # These rows carry the generation boundary as their mask, so _preprocess_batch
    # passes them through: no rendering, no boundary derivation.
    pretokenized = {"input_ids", "loss_mask"} <= set(original_cols)
    # Multimodal rows keep their `messages` so the images survive to hidden-state
    # extraction. Compute once here rather than pickling the heavyweight processor
    # into every map worker just to recheck it.
    is_multimodal = isinstance(processor, ProcessorMixin)

    if pretokenized:
        log.info("Pre-tokenized rows: using their loss mask, skipping render")
    elif render_endpoint is None:
        raise ValueError(
            "render_endpoint is required to derive loss masks for off-policy "
            "conversations. Pass --render-endpoint pointing at a vLLM server."
        )
    else:
        log.info("Deriving loss masks from vLLM render boundaries")

    # Avoid CPU contention for MM processing:
    # https://github.com/vllm-project/vllm/pull/31879
    with set_default_torch_num_threads() if is_multimodal else nullcontext():
        dataset = dataset.map(
            lambda examples: _preprocess_batch(
                examples,
                is_multimodal,
                render_endpoint,
                max_length,
                minimum_valid_tokens,
            ),
            batched=True,
            num_proc=num_proc,
            batch_size=1000,
            remove_columns=original_cols,
            keep_in_memory=True,  # skip caching
        )

    dataset.set_format(type="torch")
    return dataset


def _load_hf_dataset(spec: str) -> tuple[HFDataset, None]:
    """Load an arbitrary HuggingFace dataset from an ``hf:`` spec.

    Args:
        spec: ``hf:<dataset_id>[:<subset>:<split>]``. The split defaults to
            ``train``. A single suffix (``hf:<id>:<split>``) selects a split
            without a subset; both can be given as ``hf:<id>:<subset>:<split>``.

    Returns:
        Tuple of (raw_dataset, None). No normalize_fn is applied: the dataset
        must already be in conversations format.

    Raises:
        ValueError: If the spec is malformed or the loaded dataset has no
            ``conversations`` column.
    """
    subset: str | None
    match spec.removeprefix("hf:").split(":"):
        case [hf_id]:
            subset, split = None, "train"
        case [hf_id, split]:
            subset = None
        case [hf_id, subset, split]:
            pass
        case _:
            raise ValueError(
                f"Invalid hf: spec '{spec}'. "
                f"Expected hf:<dataset_id>[:<subset>:<split>]."
            )

    if not hf_id:
        raise ValueError(f"Invalid hf: spec '{spec}': missing dataset id.")
    if subset == "":
        raise ValueError(f"Invalid hf: spec '{spec}': empty subset.")
    if not split:
        raise ValueError(f"Invalid hf: spec '{spec}': empty split.")

    raw_dataset = load_dataset(hf_id, name=subset, split=split)

    if "conversations" not in raw_dataset.column_names:
        raise ValueError(
            f"HuggingFace dataset '{hf_id}' (split '{split}') is not in "
            f"conversations format: expected a 'conversations' column but found "
            f"{raw_dataset.column_names}. Pass a dataset already in conversations "
            f"format, or add a preset to DATASET_CONFIGS with a normalize_fn."
        )

    return raw_dataset, None


def load_raw_dataset(
    train_data_path: str,
) -> tuple[HFDataset, Callable[[dict], dict] | None]:
    """Load a raw dataset from one of several source types.

    Resolution order:
        1. Local ``.json``/``.jsonl`` file.
        2. Local directory: recursively load all ``*.json``/``*.jsonl`` files
           as a single dataset.
        3. Named preset from ``DATASET_CONFIGS``.
        4. ``hf:<id>[:<subset>:<split>]`` for an arbitrary HuggingFace dataset.

    Args:
        train_data_path: File path, directory path, preset name, or ``hf:`` spec.

    Returns:
        Tuple of (raw_dataset, normalize_fn). normalize_fn is None for sources
        already in conversations format.

    Raises:
        ValueError: If the source cannot be resolved or a local directory
            contains no ``.json``/``.jsonl`` files.
    """
    # 1. Local file
    if train_data_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=train_data_path, split="train"), None

    # 2. Local directory
    path = Path(train_data_path)
    if path.is_dir():
        data_files = sorted(
            str(p) for p in (*path.rglob("*.json"), *path.rglob("*.jsonl"))
        )
        if not data_files:
            raise ValueError(
                f"No .json/.jsonl files found in directory: {train_data_path}"
            )
        return load_dataset("json", data_files=data_files, split="train"), None

    # 3. Named preset
    if train_data_path in DATASET_CONFIGS:
        config = DATASET_CONFIGS[train_data_path]
        raw_dataset = load_dataset(
            config.hf_path, name=config.subset, split=config.split
        )
        if config.filter_fn is not None:
            raw_dataset = raw_dataset.filter(config.filter_fn)
        return raw_dataset, config.normalize_fn

    # 4. Arbitrary HuggingFace dataset
    if train_data_path.startswith("hf:"):
        return _load_hf_dataset(train_data_path)

    raise ValueError(
        f"Unsupported dataset: {train_data_path}. Supported: local .json/.jsonl "
        f"file, local directory of .json/.jsonl files, hf:<id>[:<subset>:<split>], "
        f"or a preset {list(DATASET_CONFIGS.keys())}."
    )


def get_tokenizer(processor: ProcessorLike):
    if isinstance(processor, ProcessorMixin):
        return processor.tokenizer  # type: ignore[attr-defined]

    return processor


def _resolve_pad_token(processor: ProcessorLike):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_processor(target_model_path: str, *, trust_remote_code: bool = False):
    processor = AutoProcessor.from_pretrained(
        target_model_path,
        trust_remote_code=trust_remote_code,
    )
    _resolve_pad_token(processor)

    return processor


def load_and_preprocess_dataset(
    target_model_path: str,
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
    trust_remote_code: bool = False,
) -> tuple[HFDataset, ProcessorLike]:
    """Load, tokenize, and preprocess a dataset for EAGLE3 training.

    Off-policy conversations are tokenized by a vLLM ``/render`` endpoint and
    masked at the render boundary; pre-tokenized rows pass straight through.
    Caching is handled automatically by HuggingFace datasets.

    Args:
        target_model_path: HuggingFace model ID or local path
        train_data_path: Dataset name or path to JSON/JSONL file
        seq_length: Maximum sequence length
        build_dataset_num_proc: Number of processes for dataset building
        seed: Random seed for shuffling
        max_samples: Optional limit on number of samples
        token_freq_path: Path to save token frequency distribution
        cache_dir: Directory to cache HuggingFace datasets (optional)
        render_endpoint: Base URL of a running vLLM server (e.g.
            ``http://localhost:8000``) used to render conversations. Required
            unless every dataset is already pre-tokenized.
        minimum_valid_tokens: Number of tokens to consider for a valid sample
        allow_empty_output: If True, allow returning an empty dataset instead of
                          raising when no samples survive preprocessing.
        trust_remote_code: If True, allows executing code from HF Hub.

    Returns:
        Tuple of (preprocessed_dataset, processor)
    """
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")
    log.section("Starting dataset preprocessing")
    if minimum_valid_tokens is not None:
        log.info(
            f"Filtering samples with fewer than {minimum_valid_tokens} valid tokens"
        )

    log.subsection("Loading processor")
    processor = load_processor(target_model_path, trust_remote_code=trust_remote_code)

    if render_endpoint is not None:
        log.info(f"Rendering conversations via vLLM endpoint: {render_endpoint}")

    processed_datasets = []
    for train_data_path in train_data_paths:
        log.subsection(f"Processing {train_data_path}")
        raw_dataset, normalize_fn = load_raw_dataset(train_data_path)
        raw_dataset = raw_dataset.shuffle(seed=seed)

        if max_samples is not None and len(raw_dataset) > 3 * max_samples:
            # Reduce size to 3 * max_samples to reduce processing
            # This will then be reduced further to max_samples
            # after combining datasets and shuffling
            raw_dataset = raw_dataset.select(range(3 * max_samples))

        if normalize_fn is not None:
            raw_dataset = raw_dataset.map(
                normalize_fn,
                num_proc=build_dataset_num_proc,
                keep_in_memory=True,  # skip caching
            )

        log.info(f"Loaded {len(raw_dataset)} samples")

        preprocessed_dataset = build_eagle3_dataset(
            dataset=raw_dataset,
            processor=processor,
            max_length=seq_length,
            num_proc=build_dataset_num_proc,
            render_endpoint=render_endpoint,
            minimum_valid_tokens=minimum_valid_tokens,
        )
        if minimum_valid_tokens is not None:
            log.info(f"Kept {len(preprocessed_dataset)} samples after filtering")
        processed_datasets.append(preprocessed_dataset)

    combined_dataset = concatenate_datasets(processed_datasets)
    combined_dataset = combined_dataset.shuffle(seed=seed)
    if max_samples is not None and len(combined_dataset) > max_samples:
        combined_dataset = combined_dataset.select(range(max_samples))

    if len(combined_dataset) == 0 and not allow_empty_output:
        raise ValueError(
            "No samples remain after preprocessing. Check the dataset schema, "
            "assistant masking, and --minimum-valid-tokens. Pass "
            "--allow-empty-output if an empty dataset is intentional."
        )

    log.subsection("Computing token frequency distribution")
    save_token_frequency_distribution(
        dataset=combined_dataset,
        output_path=token_freq_path,
    )

    if len(combined_dataset) == 0:
        log.warning("No samples remain after preprocessing; skipping visualization")
    else:
        log.subsection("Visualizing sample")
        _visualize_sample(combined_dataset, processor, idx=0)

    log.section("Dataset preprocessing complete")

    return combined_dataset, processor
