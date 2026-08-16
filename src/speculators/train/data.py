import json
import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.offline import align_hidden_states_to_tokens
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    multimodal_types = {
        "audio",
        "audio_url",
        "image",
        "image_url",
        "input_audio",
        "input_image",
        "input_video",
        "video",
        "video_url",
    }
    text_types = {None, "input_text", "text"}

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                return True
            part_type = part.get("type")
            if part_type in multimodal_types:
                return True
            if part_type not in text_types:
                return True

    return False


def build_client_item(dataset_item: dict) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    input_ids = dataset_item["input_ids"]
    out_dict: dict = {
        "input_ids": input_ids.tolist()
        if hasattr(input_ids, "tolist")
        else list(input_ids)
    }

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]
        tools = dataset_item.get("tools")
        if tools:
            out_dict["tools"] = json.loads(tools) if isinstance(tools, str) else tools

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)

        if data is None:
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data

def _check_hidden_states_self_consistent(data: dict[str, torch.Tensor]) -> None:
    token_ids = data["token_ids"]
    hidden_states = data["hidden_states"]
    if hidden_states.isnan().any():
        raise ValueError("Hidden states contain NaN values")
    if token_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hidden_states.shape[0]}"
            f" doesn't match num tokens {token_ids.shape[0]}"
        )


def _is_multimodal_dataset_item(dataset_item: dict[str, Any]) -> bool:
    return "messages" in dataset_item and _has_multimodal_content(
        dataset_item["messages"]
    )


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _align_loss_mask_to_token_ids(  # noqa: C901
    source_token_ids: list[int],
    source_loss_mask: list[int],
    target_token_ids: list[int],
    *,
    min_match_run: int = 4,
    max_resync_scan: int = 2048,
) -> torch.Tensor:
    """Map a preprocessed loss mask onto vLLM's actual multimodal token IDs.

    vLLM Chat Completions re-tokenizes multimodal messages and can expand image
    placeholders to a different number of image tokens than the HF preprocessing
    path.  Hidden states are valid for vLLM's token IDs, so training must use the
    same token IDs and a loss mask projected from the preprocessed sample.

    The projection only permits skipping source spans whose loss mask is all 0;
    target-only spans are assigned 0.  This keeps assistant text supervision
    intact while tolerating differences in user-side vision-token blocks and
    Chat Completions continuation suffixes.
    """
    if len(source_token_ids) != len(source_loss_mask):
        raise ValueError(
            "Cannot align loss mask with mismatched source lengths: "
            f"input_ids={len(source_token_ids)}, loss_mask={len(source_loss_mask)}"
        )

    def source_span_is_masked(start: int, end: int) -> bool:
        return all(int(value) == 0 for value in source_loss_mask[start:end])

    def following_run_matches(src_pos: int, tgt_pos: int) -> bool:
        remaining = min(
            min_match_run,
            len(source_token_ids) - src_pos,
            len(target_token_ids) - tgt_pos,
        )
        if remaining <= 0:
            return True
        return (
            source_token_ids[src_pos : src_pos + remaining]
            == target_token_ids[tgt_pos : tgt_pos + remaining]
        )

    def find_resync(src_pos: int, tgt_pos: int) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_cost: int | None = None
        max_src = min(len(source_token_ids), src_pos + max_resync_scan)
        max_tgt = min(len(target_token_ids), tgt_pos + max_resync_scan)
        for next_src in range(src_pos, max_src):
            if not source_span_is_masked(src_pos, next_src):
                break
            for next_tgt in range(tgt_pos, max_tgt):
                if next_tgt > tgt_pos and int(source_loss_mask[src_pos]) != 0:
                    continue
                if source_token_ids[next_src] != target_token_ids[next_tgt]:
                    continue
                if not following_run_matches(next_src, next_tgt):
                    continue
                cost = (next_src - src_pos) + (next_tgt - tgt_pos)
                if best_cost is None or cost < best_cost:
                    best = (next_src, next_tgt)
                    best_cost = cost
                    break
            if best is not None and best_cost == next_src - src_pos:
                break
        return best

    aligned: list[int] = []
    src_idx = 0
    tgt_idx = 0

    while src_idx < len(source_token_ids) and tgt_idx < len(target_token_ids):
        if source_token_ids[src_idx] == target_token_ids[tgt_idx]:
            aligned.append(int(source_loss_mask[src_idx]))
            src_idx += 1
            tgt_idx += 1
            continue

        resync = find_resync(src_idx, tgt_idx)
        if resync is None:
            break
        next_src, next_tgt = resync
        aligned.extend([0] * (next_tgt - tgt_idx))
        src_idx = next_src
        tgt_idx = next_tgt

    if tgt_idx < len(target_token_ids):
        if not source_span_is_masked(src_idx, len(source_token_ids)):
            raise ValueError(
                "Unable to align vLLM multimodal token IDs without dropping "
                f"trainable source tokens at source={src_idx}, target={tgt_idx}"
            )
        aligned.extend([0] * (len(target_token_ids) - tgt_idx))
        tgt_idx = len(target_token_ids)

    if src_idx < len(source_token_ids) and not source_span_is_masked(
        src_idx, len(source_token_ids)
    ):
        raise ValueError(
            "Unable to align vLLM multimodal token IDs; remaining source suffix "
            f"contains trainable tokens at source={src_idx}"
        )

    if len(aligned) != len(target_token_ids):
        raise ValueError(
            "Aligned loss mask length mismatch: "
            f"mask={len(aligned)}, target_ids={len(target_token_ids)}"
        )
    return torch.tensor(aligned, dtype=torch.long)


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        train_ratio: float = 1.0,
        split: Literal["train", "val"] = "train",
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.data = load_from_disk(datapath)
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0.0, 1.0], got {train_ratio}")
        if split == "val" and train_ratio == 1.0:
            raise ValueError("train_ratio=1.0 leaves no validation split")

        # Both splits derive their boundary from this one expression,
        # so they are exactly complementary.
        split_idx = int(len(self.data) * train_ratio)
        start, stop = (
            (0, split_idx) if split == "train" else (split_idx, len(self.data))
        )
        if start >= stop:
            raise ValueError(
                f"{split} split is empty (dataset has {len(self.data)} rows, "
                f"train_ratio={train_ratio} gives split_idx={split_idx})"
            )
        self.start_file_idx = start
        self.data = self.data.select(range(start, stop))

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_client(self):
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:  # noqa: C901
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        client_item = build_client_item(dataset_item)

        try:
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            if "messages" in client_item:
                _check_hidden_states_self_consistent(loaded_hs)
            else:
                loaded_hs, _ = align_hidden_states_to_tokens(
                    loaded_hs,
                    dataset_item["input_ids"].tolist(),
                    allow_prefix_truncation=False,
                )

            file_idx = self._map_to_file_idx(index)
            match self.on_generate:
                case "cache":
                    self.transfer.cache(handle, file_idx)
                case "delete":
                    self.transfer.delete(handle)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Failed to load/cache hidden states for sample {index}: {e}",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _get_raw_data(self, index):
        file_idx = self._map_to_file_idx(index)
        loaded_hs = self.transfer.get_cached(file_idx)

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
                case "skip":
                    return None
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )

        if loaded_hs is None:
            return loaded_hs

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        input_ids = self.data[index]["input_ids"]
        loss_mask = self.data[index]["loss_mask"]
        if torch.equal(loaded_hs["token_ids"], input_ids):
            aligned_loss_mask = loss_mask
        elif _is_multimodal_dataset_item(self.data[index]):
            try:
                aligned_loss_mask = _align_loss_mask_to_token_ids(
                    _as_list(input_ids),
                    _as_list(loss_mask),
                    _as_list(loaded_hs["token_ids"]),
                )
            except ValueError as e:
                warnings.warn(
                    f"Failed to align multimodal token ids for index {index}: {e}",
                    stacklevel=1,
                )
                return None
        else:
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {input_ids}",
                stacklevel=1,
            )
            return None

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": aligned_loss_mask,  # [seq_len]
        }


class CollateFn:
    """Picklable collate function for use with ``multiprocessing_context='spawn'``."""

    def __init__(
        self,
        max_len: int,
        hidden_size: int,
        num_target_layers: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        preprocess: Callable[[BatchType], BatchType] | None = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers
        self.dtype = dtype
        self.preprocess = preprocess

    def __call__(self, batch: Sequence[BatchType | None]) -> BatchType:
        max_len = self.max_len
        dtype = self.dtype
        preprocess = self.preprocess

        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(
                self.hidden_size, self.num_target_layers, dtype=dtype
            )
            if preprocess:
                empty = preprocess(empty)
            batch = [empty]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
            # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids

        return collated_data
