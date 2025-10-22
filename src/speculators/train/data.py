# ruff: noqa: ERA001
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset

BatchType = dict[str, Any]


class TransformTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, data):
        for tensor in self.tensors:
            data[tensor] = self.transform(data[tensor])
        return data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class AddGaussianNoise(TransformTensors):
    def __init__(self, mean=0.0, std=0.2, tensors=("hidden_states",)):
        super().__init__(tensors)
        self.mean = mean
        self.std = std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class AddUniformNoise(TransformTensors):
    def __init__(self, std=0.2, tensors=("hidden_states",)):
        super().__init__(tensors)
        self.std = std

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + (torch.rand_like(tensor) - 0.5) * self.std


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
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


class Eagle3SampleFileDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform=None,
        hidden_states_dtype=torch.float,
    ):
        if datapath is not None and file_list is not None:
            raise ValueError("Only one of datapath or file_list may be provided")

        if datapath is not None:
            file_list = list_files(datapath)
        elif file_list is None:
            raise ValueError("Either datapath or file_list must be provided")

        self.data = file_list
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Approximate lengths of the dataset based on the size of the first file"""
        lengths_0 = self.__getitem__(0)["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def __getitem__(self, index) -> BatchType:
        data = torch.load(self.data[index])

        # todo: standardize names during data generation and then remove this
        data["hidden_states"] = data["hidden_state"]
        data["verifier_last_hidden_states"] = data["target"]
        del data["hidden_state"]
        del data["target"]

        # todo: standardize dtypes during data generation and then remove this
        data = {
            k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
            for k, v in data.items()
        }

        seq_len = data["input_ids"].shape[0]
        # Add lengths tensor
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

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

        # Note: shift_batch will reduce seq_len by 1
        return shift_batch(data)


def create_collate_fn(max_len: int):
    def collate_fn(batch: list[BatchType]) -> BatchType:
        collated_data = {}
        for key in batch[0]:
            collated_data[key] = torch.cat([b[key] for b in batch], dim=0)

            if key != "lengths":
                collated_data[key] = slice_and_pad_to_length(
                    collated_data[key], max_len
                ).unsqueeze(0)
                # shape: [1, max_len, ...]

        # Handle lengths update
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
