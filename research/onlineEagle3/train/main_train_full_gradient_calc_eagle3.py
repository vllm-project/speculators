# This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
# Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)
# It has been modified to speed up the training function by using dot products
# instead of attention masks when running forward passes.
# And to use Llama 3 instead of Llama 2, along with a few other experiments.
# And to use hidden states as the cache instead of k and v
import argparse
import json
import os
import random
import warnings
from pathlib import Path
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

seed = 42  # or any integer
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, default=None)
parser.add_argument("--configpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--tmpdir", type=str, default=None)
parser.add_argument("--cpdir", type=str, default=None)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--topk_w", type=float, default=1.0)
parser.add_argument("--forward_num_total", type=int, default=3)
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--data_num", type=int, default=68000)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)

    return datapath


datapath = list_files(args.tmpdir)


data_num = len(datapath)
print(f"training on {data_num} examples total")
train_frac = 1.0
total_steps = int(
    data_num
    * train_frac
    * (args.epoch + 1)
    / (args.bs * args.gradient_accumulation_steps)
)
warm_steps = total_steps // 100


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


warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`*."
)


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location="cpu", weights_only=False)
        new_data = {}
        hidden_state = data["hidden_state"][: train_config["max_len"]][None, :]

        input_ids = data["input_ids"][: train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][: train_config["max_len"]][None, :]
        target = data["target"][: train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = target[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


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

def save_offline_debug_info(epoch, step, input_ids, target_states, pred_states, loss_mask, loss, save_dir="offline_debug"):
    """Save debug information for offline training"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Get predictions
    with torch.no_grad():
        target_preds = target_states.argmax(dim=-1)
        drafter_preds = pred_states.argmax(dim=-1)
    
    debug_dict = {
        'epoch': epoch,
        'step': step,
        'input_ids': input_ids.cpu(),
        'target_states': target_states.cpu(),
        'pred_states': pred_states.cpu(),
        'loss_mask': loss_mask.cpu(),
        'loss': loss.item(),
        'target_preds': target_preds.cpu(),
        'drafter_preds': drafter_preds.cpu(),
    }
    
    # Save to file
    filename = f"{save_dir}/offline_debug_e{epoch}_s{step}.pt"
    torch.save(debug_dict, filename)
    
    # Print debug info
    print(f"\nOFFLINE DEBUG INFO (epoch {epoch+1}, step {step}):")
    print(f"Input shape: {input_ids.shape}")
    print(f"Target states shape: {target_states.shape}")
    print(f"Predicted states shape: {pred_states.shape}")
    print(f"Loss mask shape: {loss_mask.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Target predictions: {target_preds[0,:10]}")
    print(f"Drafter predictions: {drafter_preds[0,:10]}")
    print(f"Loss mask (first 10): {loss_mask[0,:10,0].int().tolist()}\n")


def main():
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    try:
        head = torch.nn.Linear(
            baseconfig.hidden_size, baseconfig.vocab_size, bias=False
        )
    except:
        head = torch.nn.Linear(
            baseconfig.text_config.hidden_size,
            baseconfig.text_config.vocab_size,
            bias=False,
        )

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json")) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(
            os.path.join(args.basepath, head_path), framework="pt", device="cpu"
        ) as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json")) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False

    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    traindatapath = datapath[: int(len(datapath) * train_frac)]
    testdatapath = datapath[int(len(datapath) * train_frac) :]

    # Check if we have enough data
    if len(traindatapath) == 0:
        raise ValueError(f"No training data found in {args.tmpdir}. Please check the data directory.")
    
    if accelerator.is_main_process:
        print(f"Training on {len(traindatapath)} files, testing on {len(testdatapath)} files")

    traindataset = CustomDataset(traindatapath, transform=aug)

    testdataset = CustomDataset(testdatapath)
    train_loader = DataLoader(
        traindataset,
        batch_size=train_config["bs"],
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=train_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=train_config["bs"],
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=train_config["num_workers"],
        pin_memory=True,
    )
    tqdm(train_loader)
    if accelerator.is_main_process and (not os.path.exists(args.cpdir)):
        os.makedirs(args.cpdir)

    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(config, load_emb=True, path=args.basepath)

    if args.ckpt_path is not None:
        ea_model_path = args.ckpt_path
        load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.exists(load_model_path):
            ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
        else:
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            ea_layer_state_dict = safetensors.torch.load_file(load_model_path)
        model.load_state_dict(ea_layer_state_dict, strict=True)
        print(f"load model from {load_model_path}")

    kldiv = nn.KLDivLoss(reduction="none")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        betas=(train_config["b1"], train_config["b2"]),
    )

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    if is_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, test_loader, scheduler
        )
    else:
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader
        )

    map_tok = np.load("t2d.npy")
    map_tok = torch.from_numpy(map_tok).bool()

    head = head.to(accelerator.device)

    ##MAIN TRAINING LOOP
    for epoch in range(num_epochs + 1):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        forward_num_total = args.forward_num_total
        for _batch_idx, data in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                hidden_states, input_ids, attention_mask, target, loss_mask = (
                    data["hidden_states"],
                    data["input_ids"],
                    data["attention_mask"],
                    data["target"],
                    data["loss_mask"][..., None],
                )
                loss = 0
                with torch.no_grad():
                    target_head = head(target)

                    target_head = target_head[:, :, map_tok]

                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                    
                hidden_states_history = []

                hidden_states = model.fc(hidden_states.to(torch.bfloat16))
                weight_sum = 0
                for forward_idx in range(forward_num_total):
                    predict = model(
                        hidden_states, input_ids, attention_mask, hidden_states_history
                    )
                    pred = model.lm_head_layernorm(predict)
                    pred = model.lm_head(pred)

                    out_head, kldiv_loss = compute_loss(
                        target_p, pred, loss_mask, kldiv
                    )
                    total_loss = train_config["kldiv_w"] * kldiv_loss
                    weight = forward_idx + 1
                    loss += total_loss
                    weight_sum += weight
                    # Add debug info saving on first forward pass of first batch
                    if _batch_idx == 0 and forward_idx == 0 and epoch == 0:
                        save_offline_debug_info(
                            epoch=epoch,
                            step=_batch_idx,
                            input_ids=input_ids,
                            target_states=target_head,
                            pred_states=pred,
                            loss_mask=loss_mask,
                            loss=total_loss,
                        )
                    hidden_states_history.append(hidden_states)
                    hidden_states = torch.concat(
                        [hidden_states[:, :1, :], predict[:, :-1, :]], dim=1
                    )
                accelerator.backward(loss)
                torch.cuda.empty_cache()
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if is_warmup:
                    scheduler.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[
                    loss_mask.view(-1) == 1
                ]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            if accelerator.is_main_process and ct != 0:
                logdict = {
                    "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                    "train/loss": loss.item(),
                    "train/acc": cc / ct,
                }
                for item, _i in enumerate(top_3acc):
                    logdict[f"train/top_{item + 1}_acc"] = topkacc[item].item() / ct

            epoch_loss += loss.item()
            num_batches += 1

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))

        correct, total = correct.sum().item(), total.sum().item()

        # Safety check to prevent division by zero
        if num_batches > 0:
            epoch_loss /= num_batches
        else:
            epoch_loss = 0.0
            if accelerator.is_local_main_process:
                print(f"Warning: No batches processed in epoch {epoch + 1}")
        
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            if total > 0:
                print(f"Train Accuracy: {100 * correct / total:.2f}%")
            else:
                print("Train Accuracy: N/A (no data processed)")

            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(), f"{args.cpdir}/model{epoch}.safetensors"
            )


if __name__ == "__main__":
    main()
