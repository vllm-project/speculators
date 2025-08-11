import argparse
import json
import os
import random
import warnings
from typing import Any

import numpy as np
import safetensors
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from model.configs import EConfig
from model.llama_eagle3_full_grad import Model
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.epoch,
    # Depending on your data and model size, the larger the model,
    # the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "p_w": 0.0,
    "v_w": 0.0,
    "kldiv_w": 1.0,
    "topk_w": args.topk_w,
    "head_w": 0.1,
    "num_workers": 8,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 8192,
    # During training, truncating the training
    # sequences means that the larger the setting,
    # the more training data is used, and the better the effect,
    # but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 1.0,
    "save_freq": 5,
}


torch.backends.cuda.matmul.allow_tf32 = True

set_seed(0)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
)
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


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, dim):
        b, n, s = intensors.shape
        padding_tensor = torch.zeros(b, dim - n, s)
        return torch.cat((intensors, padding_tensor), dim=1)

    def paddingtensor2d(self, intensors, num):
        b, n = intensors.shape

        padding_tensor = torch.zeros(b, num - n, dtype=intensors.dtype)

        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)

        batch_input_ids = torch.cat(
            [self.paddingtensor2d(item["input_ids"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        return {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }



def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target_p, predict, loss_mask, kldiv=None):
    out_head = predict
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    kldiv_loss = kldiv(out_logp, target_p)
    kldiv_loss = torch.sum(torch.sum(loss_mask * kldiv_loss, 2)) / (
        loss_mask.sum() + 1e-5
    )
    return out_head, kldiv_loss

def train(
    data_queue: multiprocessing.Queue,
    gpu_ids: Optional[list[int]] = None,
    **kwargs,
):
    """
    Training function that runs in the main process and handles training
    using the data from the queue.

    :param data_queue: Queue containing training data
    :param gpu_ids: List of GPU IDs to make visible for training process
    :param kwargs: Additional training arguments
    """
    train_dataset = QueueDataset(data_queue=data_queue)
    configure_gpu_visibility(gpu_ids)
    # data=data_queue.get()



    # implement training logic here
