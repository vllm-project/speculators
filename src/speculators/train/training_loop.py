import os
import torch
from transformers import LlamaConfig

from speculators.train.eagle3.core import Eagle3DraftModel
from speculators.train.data import Eagle3SampleFileDataset, create_collate_fn
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from torch.utils.data import DataLoader
from tqdm.rich import tqdm  # todo: requries tqdm and rich

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def log_rank0(message):
    if local_rank != 0:
        return
    print(message)


def maybe_setup_distributed():
    # Based off of https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
    if "LOCAL_RANK" not in os.environ:
        # No distributed training
        return 0, 1, 0, False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()

    print(
        f"Started DDP with local_rank={local_rank}, world_size={world_size}, rank={rank}"
    )
    return local_rank, world_size, rank, True


local_rank, world_size, rank, is_distributed = maybe_setup_distributed()


DEVICE = torch.device(local_rank)
EPOCHS = 10
draft_vocab_size = 5000
verifier_vocab_size = 151936
hidden_size = 5120
total_seq_len = 4096
datapath = "./data"
verifier_model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"


# TEMP MODEL SETUP
llama_config = LlamaConfig(hidden_size=hidden_size)
llama_config._attn_implementation = "simple_flex_attention"

d2t_vocab = torch.zeros(draft_vocab_size, dtype=torch.long).to(DEVICE)
t2d_vocab = (
    torch.cat(
        [
            torch.ones(draft_vocab_size),
            torch.zeros(llama_config.vocab_size - draft_vocab_size),
        ]
    )
    .to(torch.bool)
    .to(DEVICE)
)
# END TEMP MODEL SETUP


draft_model = Eagle3DraftModel(
    hidden_size=hidden_size,
    t2d_vocab=t2d_vocab,
    d2t_vocab=d2t_vocab,
    decoder_layer_config=llama_config,
    verifier_vocab_size=verifier_vocab_size,
    verifier_pad_token_id=151643,
    num_layers=1,
    ttt_steps=3,
).to(DEVICE)

# draft_model.load_verifier_lm_head(verifier_model_name_or_path) # Doesn't work for Qwen2.5 VL, need better head loading method

if is_distributed:
    draft_model = DDP(draft_model, device_ids=[local_rank])
opt = torch.optim.Adam(draft_model.parameters(), lr=1e-4)

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
    num_workers=16,
    pin_memory=True,
    collate_fn=create_collate_fn(total_seq_len),
)


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_path: str, epoch: int
):
    os.makedirs(save_path, exist_ok=True)
    save_path = f"{save_path}/checkpoint_epoch_{epoch}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )
    log_rank0(f"Checkpoint saved to {save_path}")


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_laoder: DataLoader,
    opt: torch.optim.Optimizer,
    epoch: int,
    local_rank: int,
    is_distributed: bool,
):
    model.train()

    if local_rank == 0:
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in train_loader:
        batch = {
            k: v.to(local_rank) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        _, loss = model(**batch, use_off_policy_tokens=True)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss = loss.detach().clone()
        if is_distributed:
            # Note: this is not needed for training, just for logging
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)

        log_rank0(loss.item())


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    opt: torch.optim.Optimizer,
    epochs: int,
    local_rank: int,
    is_distributed: bool,
    save_path: str,
):
    for epoch in range(epochs):
        train_epoch(
            model, train_loader, val_loader, opt, epoch, local_rank, is_distributed
        )

        if is_distributed:
            dist.barrier()

        log_rank0(f"Epoch {epoch} completed")
        if local_rank == 0:
            save_checkpoint(model, opt, save_path, epoch)

    log_rank0(f"Training completed")


train(
    draft_model,
    train_loader,
    None,
    opt,
    EPOCHS,
    local_rank,
    is_distributed,
    "./checkpoints",
)

if is_distributed:
    dist.destroy_process_group()
    print(
        f"Destroyed DDP with local_rank={local_rank}, world_size={world_size}, rank={rank}"
    )


# RUN WITH:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4  src/speculators/train/training_loop.py