# ruff: noqa: ERA001
import functools
import json
import math
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from openai import OpenAI
from safetensors import safe_open
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    _normalize_conversation,
    create_loss_mask_from_token_ids,
    detect_assistant_token_markers,
    load_raw_dataset,
)
from speculators.train.noise_transforms import TransformTensors

BatchType = dict[str, Any]


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def slice_and_pad_to_length(tensor, length):
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding)


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
    # todo: verify loss mask shift is correct

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


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def standardize_data_v0(data: dict[str, Any]) -> dict[str, Any]:
    # v0 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_state": [seq_len, 3 * hidden_size],
    #  "target": [seq_len, hidden_size],
    # }

    return {
        "hidden_states": data["hidden_state"],
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["target"],
        "loss_mask": data["loss_mask"],
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def standardize_data_v2(
    data: dict[str, Any],
    start_marker_ids: list[int] | None = None,
    end_marker_ids: list[int] | None = None,
) -> dict[str, Any]:
    # v2 data format:
    # {
    #  "token_ids": [seq_len],
    #  "hidden_states": [seq_len, num_hidden_layers, hidden_size],
    # }
    # where num_hidden_layers = 4

    token_ids = data["token_ids"]

    if start_marker_ids is not None and end_marker_ids is not None:
        loss_mask = create_loss_mask_from_token_ids(
            token_ids, start_marker_ids, end_marker_ids
        )
    else:
        loss_mask = torch.ones_like(token_ids, dtype=torch.bool)

    return {
        "hidden_states": data["hidden_states"][:, :-1].flatten(1),
        "input_ids": token_ids,
        "verifier_last_hidden_states": data["hidden_states"][:, -1],
        "loss_mask": loss_mask,
    }


def _create_empty_sample():
    """Creates an empty sample which the collator can ignore. This is useful as a
    replacement batch if data loading fails for a sample."""
    return {
        "hidden_states": torch.empty(0, dtype=torch.float),
        "input_ids": torch.zeros(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, dtype=torch.float),
        "loss_mask": torch.zeros(0, dtype=torch.bool),
        "lengths": torch.zeros(1, dtype=torch.long),
        "position_ids": torch.zeros(0, dtype=torch.long),
    }


def vllm_generate_hidden_states(
    vllm_url: str, model: str, conversation: list[dict[str, str]]
) -> dict[str, torch.Tensor] | None:
    """Generates hidden states using VLLM."""

    client = OpenAI(base_url=vllm_url, api_key="EMPTY")

    completion = client.chat.completions.create(
        model=model, messages=conversation, max_tokens=1
    )
    hidden_states_path = completion.kv_transfer_params["hidden_states_path"]

    f = safe_open(hidden_states_path, "pt")

    data = {k: f.get_tensor(k) for k in f.keys()}

    # Cleanup file path
    os.remove(hidden_states_path)
    return data


class Eagle3OnlineVLLMDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        dataset: str,
        vllm_url: str,
        model: str,
        lengths_file: str,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.float,
        standardize_fn: StandardizeFnSig = standardize_data_v2,
    ):
        """Initialize the Eagle3OnlineVLLMDataset.

        Args:
            max_len: The maximum length of the sequence.
            dataset: The name of the dataset.
            vllm_url: The URL of the VLLM server to generate hidden states.
            model: The name of the model to generate hidden states.
            lengths_file: Path to sample_lengths.json produced by
                preprocess_data.py. Required for accurate batch packing.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.
        """
        self.max_len = max_len
        self.text_dataset = load_raw_dataset(dataset, num_proc=8, cache_dir=None)

        self.vllm_url = vllm_url
        self.model = model
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype

        # Detect assistant token markers for loss masking
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        start_marker_ids, end_marker_ids = detect_assistant_token_markers(tokenizer)

        # Bind markers into standardize_fn so __getitem__ produces accurate masks
        self.standardize_fn = functools.partial(
            standardize_fn,
            start_marker_ids=start_marker_ids,
            end_marker_ids=end_marker_ids,
        )

        self.approx_lengths = self._load_lengths(lengths_file)

    def __len__(self):
        return len(self.text_dataset)

    def _load_lengths(self, lengths_file: str) -> list[int]:
        """Load precomputed sample lengths from sample_lengths.json."""
        sample_lengths_path = Path(lengths_file)
        if not sample_lengths_path.exists():
            raise FileNotFoundError(
                f"Lengths file not found: {lengths_file}. "
                "Run scripts/preprocess_data.py first to generate it."
            )
        with sample_lengths_path.open() as f:
            return json.load(f)

    def __getitem__(self, index) -> BatchType:
        conversation = self.text_dataset[index]
        conversation = _normalize_conversation(conversation["conversations"])

        data = vllm_generate_hidden_states(self.vllm_url, self.model, conversation)

        if data is None:
            return _create_empty_sample()

        data = self.standardize_fn(data)
        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Convert hidden states to the correct dtype
        data = {
            k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
            for k, v in data.items()
        }

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

        # Note: shift_batch will reduce seq_len by 1
        return shift_batch(data)


class Eagle3SampleFileDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.float,
        standardize_fn: StandardizeFnSig = standardize_data_v1,
    ):
        """Initialize the Eagle3SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """
        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list
        self.max_len = max_len
        self.transform = transform
        self.standardize_fn = standardize_fn
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        lengths_0 = self.__getitem__(0)["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def __getitem__(self, index) -> BatchType:
        data = torch.load(
            self.data[index], mmap=True, weights_only=True, map_location="cpu"
        )

        data = self.standardize_fn(data)
        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Convert hidden states to the correct dtype
        data = {
            k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
            for k, v in data.items()
        }

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

        # Note: shift_batch will reduce seq_len by 1
        return shift_batch(data)


def create_collate_fn(max_len: int):
    def collate_fn(batch: list[BatchType]) -> BatchType:
        collated_data = {}
        for key in batch[0]:
            # Concatenate the tensors along the seq (0th) dimension
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)
            # shape: [total_seq_len, ...]

            if key != "lengths":
                # Slice and pad on seq (0th) dimension to max_len
                collated_data[key] = slice_and_pad_to_length(
                    collated_data[key], max_len
                ).unsqueeze(0)
                # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data["lengths"]
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length == 0:
                continue
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        collated_data["lengths"] = torch.tensor(new_lengths, dtype=torch.long)
        return collated_data

    return collate_fn
