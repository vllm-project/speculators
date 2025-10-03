from functools import lru_cache
import math
import os
from typing import Any

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

BatchType = dict[str, Any]


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_states"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_states"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_states"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_states"] = noisy_tensor
        return data


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)

    return datapath


def slice_and_pad_to_length(tensor, length):
    sliced_tensor = tensor[:length]
    padding = [0, 0] * sliced_tensor.dim()
    padding[-1] = length - sliced_tensor.shape[0]
    return F.pad(sliced_tensor, padding)


class Eagle3SampleFileDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        max_len: int,
        transform=None,
        hidden_states_dtype=torch.float,
    ):
        self.data = list_files(datapath)
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=1)
    def approx_lengths(self):
        lengths_0 = self.__getitem__(0)["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = os.path.getsize(self.data[0])

        approx_lengths = [
            math.ceil(os.path.getsize(fname) / size_0 * lengths_0)
            for fname in self.data
        ]
        return approx_lengths

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

        # Add lengths tensor
        data["lengths"] = torch.tensor([data["input_ids"].shape[0]], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        # }
        return data


def create_collate_fn(max_len: int):
    def collate_fn(batch: list[BatchType]) -> BatchType:
        collated_data = {}
        for key in batch[0].keys():
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
                cum_length = max_len
                break
            new_lengths.append(length)
            cum_length += length
        if cum_length < max_len:
            # Add extra "padded" sample so that sum(new_lengths) == max_len
            new_lengths.append(max_len - cum_length)
        collated_data["lengths"] = torch.tensor(new_lengths, dtype=torch.long)
        return collated_data

    return collate_fn
