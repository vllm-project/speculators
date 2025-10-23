import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.data import (
    AddUniformNoise,
    Eagle3SampleFileDataset,
    create_collate_fn,
    split_files,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.eagle3.core import Eagle3DraftModel
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.trainer import Trainer
from speculators.train.utils import maybe_destroy_distributed, maybe_setup_distributed

setup_metric_logger(loggers="trackio", run_name=None, output_dir="./logs")
setup_root_logger()

local_rank, world_size, rank, is_distributed = maybe_setup_distributed()

DEVICE = torch.device(local_rank)
EPOCHS = 20
draft_vocab_size = 32000
total_seq_len = 8192
datapath = "./data"
verifier_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
hidden_size = 4096
verifier_vocab_size = 128256
draft_vocab_size = 32000
norm_before_residual = True
ttt_steps = 3


llama_config = LlamaConfig(
    hidden_size=hidden_size, vocab_size=verifier_vocab_size, num_hidden_layers=1
)
llama_config._attn_implementation = "simple_flex_attention"  # noqa: SLF001


d2t = torch.from_numpy(np.load("d2t.npy")).to(DEVICE)
t2d = torch.from_numpy(np.load("t2d.npy")).to(DEVICE)

speculator_config = Eagle3SpeculatorConfig(
    transformer_layer_config=llama_config,
    draft_vocab_size=draft_vocab_size,
    norm_before_residual=norm_before_residual,
    speculators_config=SpeculatorsConfig(
        algorithm="eagle3",
        proposal_methods=[
            GreedyTokenProposalConfig(
                proposal_type="greedy",
                speculative_tokens=ttt_steps,
            )
        ],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path=verifier_model_name_or_path,
            architectures=["LlamaForCausalLM"],
        ),
    ),
)

draft_model = Eagle3DraftModel(
    config=speculator_config, t2d=t2d, d2t=d2t, ttt_steps=ttt_steps
)


noise_transform = AddUniformNoise(
    std=0.2, tensors=("hidden_states", "verifier_last_hidden_states")
)

train_files, val_files = split_files(datapath, ratio=0.9)
train_dataset = Eagle3SampleFileDataset(
    file_list=train_files, max_len=total_seq_len, transform=noise_transform
)
train_batch_sampler = MultipackDistributedBatchSamplerV2(
    batch_max_length=total_seq_len,
    lengths=train_dataset.approx_lengths,
    num_replicas=world_size,
    rank=local_rank,
)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=32,
    prefetch_factor=8,
    pin_memory=True,
    collate_fn=create_collate_fn(total_seq_len),
    persistent_workers=True,
)

val_dataset = Eagle3SampleFileDataset(file_list=val_files, max_len=total_seq_len)
val_batch_sampler = MultipackDistributedBatchSamplerV2(
    batch_max_length=total_seq_len,
    lengths=val_dataset.approx_lengths,
    num_replicas=world_size,
    rank=local_rank,
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_batch_sampler,
    num_workers=32,
    prefetch_factor=8,
    pin_memory=True,
    collate_fn=create_collate_fn(total_seq_len),
    persistent_workers=True,
)


# todo: make config better
config = {
    "num_epochs": EPOCHS,
    "save_path": "./checkpoints",
    "lr": 1e-4,
    "total_seq_len": total_seq_len,
    "datapath": "./data",
    "resume_from_checkpoint": True,
}


trainer = Trainer(
    draft_model,
    config,
    train_loader,
    val_loader,
    is_distributed,
    local_rank,
    world_size,
)
trainer.run_training()

maybe_destroy_distributed()


# RUN WITH:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4  scripts/train.py
