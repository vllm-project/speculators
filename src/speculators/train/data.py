# ruff: noqa: ERA001
import json
import math
import os
import random
import shutil
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import openai
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_from_disk
from safetensors.torch import load_file
from torch.utils.data import Dataset

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states,
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


def create_empty_sample(hidden_size: int):
    # data structure: {
    #     "hidden_states": [seq_len, 3 * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }

    return {
        "hidden_states": torch.empty(0, 3 * hidden_size),
        "input_ids": torch.empty(0),
        "verifier_last_hidden_states": torch.empty(0, hidden_size),
        "loss_mask": torch.empty(0),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
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


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.float,
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

        return data


class ArrowDataset(BaseDataset):
    DATASET_TO_OPENAI_FIELDS = {
        "input_ids": "input_ids",
        "_vllm_messages": "messages",
    }

    @classmethod
    def convert_to_openai(cls, dataset_item: dict):
        return {
            openai_field: dataset_item[dataset_field]
            for dataset_field, openai_field in cls.DATASET_TO_OPENAI_FIELDS.items()
            if dataset_field in dataset_item
        }

    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        hidden_states_path: str | PathLike | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.float,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize the ArrowDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory that contains the preprocessed
            arrow dataset.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
        """
        self.data = load_from_disk(datapath)
        if split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.start_file_idx = 0
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

        self.hidden_states_path: Path = (
            Path(datapath) / "hidden_states"
            if hidden_states_path is None
            else Path(hidden_states_path)
        )
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
        # Delay client setup so it runs in dataloader thread if on_missing="generate"
        self.client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = self.client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                "found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _maybe_load_hs_file(self, index: int) -> dict[str, torch.Tensor] | None:
        file_idx = self._map_to_file_idx(index)
        candidate_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        if candidate_path.exists():
            return load_file(candidate_path)

        return None

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        if not self.client:
            self._setup_client()

        dataset_item = self.data[index]
        openai_item = self.convert_to_openai(dataset_item)
        input_ids = openai_item["input_ids"].tolist()
        messages = openai_item.get("messages")

        try:
            hs_filepath = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                input_ids,
                messages=messages,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(str(e), stacklevel=1)
            return None

        loaded_hs = load_file(hs_filepath)

        match self.on_generate:
            case "cache":
                file_idx = self._map_to_file_idx(index)
                target_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                shutil.move(hs_filepath, target_path)
            case "delete":
                Path(hs_filepath).unlink()

        return loaded_hs

    def _get_raw_data(self, index):
        loaded_hs = self._maybe_load_hs_file(index)

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
        #   "hidden_states": [seq_len, 4, hidden_size]
        #   "token_ids": [seq_len]
        # }

        if not torch.equal(loaded_hs["token_ids"], self.data[index]["input_ids"]):
            warnings.warn(
                f"Loaded token ids {loaded_hs['token_ids']} for index {index} don't"
                f"match input ids {self.data[index]['input_ids']}",
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
            "loss_mask": self.data[index]["loss_mask"],  # [seq_len]
        }


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype=None,
    ):
        """Initialize the SampleFileDataset.
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

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

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
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    def collate_fn(batch: list[BatchType | None]) -> BatchType:
        # Apply per-sample preprocessing and filter failed samples
        batch = [preprocess(b) if preprocess else b for b in batch if b is not None]

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found
            batch = [create_empty_sample(hidden_size)]

        collated_data = {}
        for key in batch[0]:  # type: ignore[union-attr]
            # Concatenate the tensors along the seq (0th) dimension
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
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
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        collated_data["lengths"] = torch.tensor(new_lengths, dtype=torch.long)
        return collated_data

    return collate_fn
