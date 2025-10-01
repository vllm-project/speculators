import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)

    return datapath


def pad_to_length(tensor, dim, length):
    padding = [0, 0] * tensor.dim()
    padding[-dim * 2 - 1] = length - tensor.shape[dim]
    return F.pad(tensor, padding)


class Eagle3SampleFileDataset(Dataset):
    def __init__(self, datapath: str, max_len: int, transform=None):
        self.data = list_files(datapath)
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])

        new_data = {}
        hidden_state = pad_to_length(
            data["hidden_state"][: self.max_len], 0, self.max_len
        )
        # shape: [max_len, 3 * hidden_dim]

        input_ids = pad_to_length(data["input_ids"][: self.max_len], 0, self.max_len)
        # shape: [max_len]
        loss_mask = pad_to_length(data["loss_mask"][: self.max_len], 0, self.max_len)
        # shape: [max_len]
        target = pad_to_length(data["target"][: self.max_len], 0, self.max_len)
        # shape: [max_len, hidden_dim]

        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids

        if self.transform:
            new_data = self.transform(new_data)

        return new_data
