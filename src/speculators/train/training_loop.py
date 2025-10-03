import torch
from transformers import LlamaConfig

from speculators.train.eagle3.core import Eagle3DraftModel
from speculators.train.data import Eagle3SampleFileDataset, create_collate_fn
from torch.utils.data import DataLoader


DEVICE = "cuda:0"
EPOCHS = 10
draft_vocab_size = 4096
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


dataset = Eagle3SampleFileDataset(datapath=datapath, max_len=total_seq_len)
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=create_collate_fn(total_seq_len),
)
opt = torch.optim.Adam(draft_model.parameters(), lr=1e-4)


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_laoder: DataLoader,
    opt: torch.optim.Optimizer,
    epoch: int,
    local_rank: int,
):
    model.train()

    for batch in train_loader:
        batch = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        del batch["loss_mask"]

        _, loss = model(**batch, use_off_policy_tokens=True)
        print(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()


for epoch in range(EPOCHS):
    train_epoch(draft_model, train_loader, None, opt, epoch, DEVICE)
