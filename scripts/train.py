import torch
import numpy as np
from transformers import LlamaConfig

from speculators.train.eagle3.core import Eagle3DraftModel, Eagle3VerifierLMHead
from speculators.train.data import Eagle3SampleFileDataset, create_collate_fn
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from torch.utils.data import DataLoader

from speculators.train.utils import maybe_setup_distributed, maybe_destroy_distributed
from speculators.train.trainer import Trainer
from speculators.train.logger import setup_metric_logger, setup_root_logger


local_rank, world_size, rank, is_distributed = maybe_setup_distributed()


DEVICE = torch.device(local_rank)
EPOCHS = 10
draft_vocab_size = 32000
total_seq_len = 4352
datapath = "./data"
verifier_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"


# TEMP MODEL SETUP
llama_config = LlamaConfig.from_pretrained(verifier_model_name_or_path)
hidden_size = llama_config.hidden_size
verifier_vocab_size = llama_config.vocab_size
llama_config = LlamaConfig(hidden_size=hidden_size, vocab_size=verifier_vocab_size)
llama_config._attn_implementation = "simple_flex_attention"

# d2t_vocab = torch.zeros(draft_vocab_size, dtype=torch.long).to(DEVICE)
# t2d_vocab = (
#     torch.cat(
#         [
#             torch.ones(draft_vocab_size),
#             torch.zeros(llama_config.vocab_size - draft_vocab_size),
#         ]
#     )
#     .to(torch.bool)
#     .to(DEVICE)
# )
d2t_vocab = torch.from_numpy(np.load("d2t.npy")).to(DEVICE)
t2d_vocab = torch.from_numpy(np.load("t2d.npy")).to(DEVICE)

setup_metric_logger(loggers="trackio", run_name=None, output_dir="./logs")
setup_root_logger()
# END TEMP MODEL SETUP

draft_model = Eagle3DraftModel(
    verifier_model_name_or_path=verifier_model_name_or_path,
    hidden_size=hidden_size,
    t2d_vocab=t2d_vocab,
    d2t_vocab=d2t_vocab,
    decoder_layer_config=llama_config,
    verifier_vocab_size=verifier_vocab_size,
    verifier_pad_token_id=None,
    num_layers=1,
    ttt_steps=3,
)

verifier_lm_head = Eagle3VerifierLMHead(
    hidden_size=hidden_size, draft_vocab_size=draft_vocab_size
)
verifier_lm_head.load_verifier_lm_head(verifier_model_name_or_path, t2d_vocab)

dataset = Eagle3SampleFileDataset(datapath=datapath, max_len=total_seq_len)
batch_sampler = MultipackDistributedBatchSamplerV2(
    batch_max_length=total_seq_len,
    lengths=dataset.approx_lengths(),
    num_replicas=world_size,
    rank=local_rank,
)
train_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    collate_fn=create_collate_fn(total_seq_len),
)


# todo: make config better
config = {
    "num_epochs": EPOCHS,
    "save_path": "./checkpoints",
    "lr": 1e-5,
    "total_seq_len": total_seq_len,
    "datapath": "./data",
    "resume_from_checkpoint": True,
}


trainer = Trainer(
    draft_model,
    verifier_lm_head,
    config,
    train_loader,
    None,
    is_distributed,
    local_rank,
    world_size,
)
trainer.run_training()

maybe_destroy_distributed()


# RUN WITH:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4  scripts/train.py
