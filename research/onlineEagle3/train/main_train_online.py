import csv
import logging
import multiprocessing
import os
import queue
from collections.abc import Iterable
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Optional, Union
import argparse
import torch
from typing import Any
import vllm
from vllm import SamplingParams
from datasets import IterableDataset
from datasets import load_dataset
from torch import nn, optim
from datasets import Dataset
from model.configs import EConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
from torch.utils.data import IterableDataset, DataLoader
# from train import train
import random
import numpy as np 
from accelerate.utils import set_seed
from accelerate import Accelerator
from transformers import AutoConfig, get_linear_schedule_with_warmup
import json
from safetensors import safe_open
from model.llama_eagle3_full_grad import Model
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


total_samples=500
data_num = total_samples
print(f"training on {data_num} examples total")
train_frac = 1.0
total_steps = int(
    data_num
    * train_frac
    * (args.epoch + 1)
    / (args.bs * args.gradient_accumulation_steps)
)
warm_steps = total_steps // 100



def build_ds(
    tokenizer,
    split="train",
):
    ds = load_dataset("json", data_files="ShareGPT_V4.3_unfiltered_cleaned_split.json")
    ds = ds[split]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(0, total_samples))

    original_columns1 = ds1.column_names

    def preprocess(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for j in range(len(examples["id"])):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
                    ),
                },
            ]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][j]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for _, sentence in enumerate(source):
                role = roles[sentence["from"]]

                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess,
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1





def configure_gpu_visibility(gpu_ids: Optional[list[int]] = None) -> None:
    """
    Configure GPU visibility for the current process.

    :param gpu_ids: List of GPU IDs to make visible. If None, all GPUs are visible.
    """
    if gpu_ids is not None:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logging.info(f"Configured GPU visibility to: {gpu_str}")
    else:
        logging.info("Using all available GPUs")


def data_batches_loader(data, batch_size: int) :
    """
    Load the source data into prompts to run for online data generation.
    This function should return a generator that yields the prompts for iteration.

    Currently, implemented as a base, simple case with data loaded from a csv file.

    :param data: Path to the source data file or Hugging Face dataset id.
    :param batch_size: Size of the batch to be processed.
    :return: An iterable of prompts.
    """

    batch = []
    while True:  # infinite looping over the data
        for item in data:
            
            batch.append(item)  # Assuming the prompt is in the first column
            if len(batch) >= batch_size:
                yield batch
                batch = []


def online_data_generator(
    data_queue: multiprocessing.Queue,
    shutdown_event: Event,
    verifier: str,
    data: Dataset,
    batch_size: int = 32,
    gpu_ids: Optional[list[int]] = None,
):
    """
    Function used to generate data through vLLM for online training.
    It handles instantiation of vLLM with the provided model,
    loading the data from a source file, running the data through vLLM in batches,
    putting the data into the queue for training, and handling shutdown events.

    :param data_queue: Queue to put generated data into
    :param shutdown_event: Event to signal shutdown
    :param verifier: Model name/path for vLLM
    :param data: Path to source data file
    :param batch_size: Batch size for processing
    :param env_vars: Environment variables to set
    :param gpu_ids: List of GPU IDs to make visible for vLLM process
    """
    # Configure GPU visibility for this process
    print("gpu ids")
    print(gpu_ids)
    # print(1/0)
    configure_gpu_visibility(gpu_ids)

    try:
        tokenizer = AutoTokenizer.from_pretrained(verifier, use_fast=False)

        model = vllm.LLM(model=verifier,enable_prefix_caching=False,speculative_config={"model":"eagle3/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/607d0d5b7871cd4b89395b6af288c070cfa0a168", "num_speculative_tokens":1, "method":"eagle3"})
        data=build_ds(tokenizer)

        sampling_params=SamplingParams(max_tokens=1)
        req_id=1
        for prompt_batch in (data_batches_loader)(
            data=data,
            batch_size=batch_size,
        ):
            text=[x['conversation'] for x in prompt_batch]
            input_ids=[x['input_ids'] for x in prompt_batch]
            loss_mask=[x['loss_mask'] for x in prompt_batch]

        

            try:
                # print(input_ids[0][0].tolist())
                output=model.generate(prompt_token_ids=input_ids[0][0].tolist(),sampling_params=sampling_params)
                req_id+=1

                #output=model.generate(text[0], sampling_params)
                hidden_states=output[0].hidden_states
                aux_hidden_states=output[0].aux_hidden_states

                aux_hidden_states=aux_hidden_states[:len(input_ids[0][0].tolist())]
                hidden_states=hidden_states[:len(input_ids[0][0].tolist())]

                data={
                    "input_ids": input_ids[0][0].unsqueeze(0),
                    "hidden_state": hidden_states[0],
                    "loss_mask": loss_mask[0].cpu()[0],
                    "hidden_states":hidden_states.unsqueeze(0), 
                    "aux_hidden_states":aux_hidden_states.unsqueeze(0),
                }

                data_queue.put(data)
            except Exception as batch_err:
                logging.error(f"Error processing batch: {batch_err}")
                traceback.print_exc()
                break
    except Exception as process_err:
        logging.error(f"Error in online data generator: {process_err}")
        raise process_err



class QueueDataset(IterableDataset):
    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        timeout: float = 1.0,
        eos_on_empty: bool = False,
        batch_size=1
    ):
        """
        Initialize the QueueDataset which sources data from a multiprocessing Queue.

        :param data_queue: The multiprocessing queue containing training data
        :param timeout: Timeout in seconds when getting data from queue
        :param eos_on_empty: Whether to stop iteration when queue is empty
        """
        self.data_queue = data_queue
        self.timeout = timeout
        self.batch_size=batch_size
        self.eos_on_empty = eos_on_empty

    def __iter__(self):
        while True:
            try:
                batch= self.data_queue.get(timeout=self.timeout)

                
                # print(max_len)
                yield {"data": batch}
            except queue.Empty:
                if self.eos_on_empty:
                    break
            except Exception as err:
                logging.error(f"Error getting data from queue: {err}")
                raise err

def collate(batch):

    batch=[b['data'] for b in batch]
    batch_size=len(batch)
    max_len=max([i['input_ids'].shape[-1] for i in batch ])
    input_ids=torch.zeros(batch_size, max_len, dtype=torch.int)
    for i in range(batch_size):
        input_ids[i, :len(batch[i]['input_ids'][0])]=batch[i]['input_ids'][0]
    loss_mask=torch.zeros(batch_size, max_len, dtype=torch.int)
    for i in range(batch_size):
        loss_mask[i, :len(batch[i]['loss_mask'])]=batch[i]['loss_mask']
    aux_hidden_states=torch.zeros(batch_size, max_len, batch[i]['aux_hidden_states'].shape[2],dtype=batch[i]['aux_hidden_states'].dtype )
    for i in range(batch_size):
        aux_hidden_states[i, :len(batch[i]['aux_hidden_states'][0])]=batch[i]['aux_hidden_states'][0]
    hidden_states=torch.zeros(batch_size, max_len,batch[i]['hidden_states'].shape[2], dtype=batch[i]['hidden_states'].dtype )
    for i in range(batch_size):
        hidden_states[i, :len(batch[i]['hidden_states'][0])]=batch[i]['hidden_states'][0]


    new_batch={"input_ids":input_ids, 'loss_mask':loss_mask, 'aux_hidden_states':aux_hidden_states, "hidden_states":hidden_states}
    # print(batch)

    return new_batch
def get_gpu_ids_split(
    verifier_gpus: Union[float, int, list[int]],
    train_gpus: Union[float, int, list[int]],
) -> tuple[list[int], list[int]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("No GPUs available for training.")

    available_gpus = list(range(torch.cuda.device_count()))

    if isinstance(verifier_gpus, int):
        verifier_gpu_ids = available_gpus[:verifier_gpus]
    elif isinstance(verifier_gpus, float):
        verifier_gpu_ids = available_gpus[: int(len(available_gpus) * verifier_gpus)]
    elif isinstance(verifier_gpus, list) and any(
        gpu_id not in available_gpus for gpu_id in verifier_gpus
    ):
        raise ValueError(f"Some verifier GPU IDs {verifier_gpus} are not available.")

    available_gpus = list(set(available_gpus) - set(verifier_gpu_ids))

    if isinstance(train_gpus, int):
        train_gpu_ids = available_gpus[:train_gpus]
    elif isinstance(train_gpus, float):
        train_gpu_ids = available_gpus[ int(len(available_gpus) * train_gpus):]
    elif isinstance(train_gpus, list) and any(
        gpu_id not in available_gpus for gpu_id in train_gpus
    ):
        raise ValueError(f"Some training GPU IDs {train_gpus} are not available.")

    return verifier_gpu_ids, train_gpu_ids



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
    "topk_w": 0,
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

    kldiv_loss = torch.sum(torch.sum(loss_mask.to(kldiv_loss.device) * kldiv_loss, 2)) / (
        loss_mask.sum() + 1e-5
    )
    return out_head, kldiv_loss








def train(data_queue, gpu_ids,args):
    configure_gpu_visibility(gpu_ids)
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(0)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
    )

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
            tensor = tensor_slice[:, :hidden_dim]
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json")) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"]

    head.weight.data = tensor
    head.eval()

    for param in head.parameters():
        param.requires_grad = False

    if accelerator.is_main_process and (not os.path.exists(args.cpdir)):
        os.makedirs(args.cpdir)

    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(config, load_emb=True, path=args.basepath)
    model=nn.DataParallel(model).cuda()
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

        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
    else:
        model, optimizer= accelerator.prepare(
            model, optimizer
        )

    map_tok = np.load("t2d.npy")
    map_tok = torch.from_numpy(map_tok).bool()

    head = head.to(accelerator.device)


    data_loader=DataLoader(QueueDataset(data_queue), batch_size=4, num_workers=0, pin_memory=True, collate_fn=collate)

    for epoch in range(num_epochs + 1):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        forward_num_total = args.forward_num_total
        for data in data_loader:
            print("step")
            print("success")
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                hidden_states, input_ids, attention_mask, target, loss_mask = (
                    data["aux_hidden_states"],
                    data["input_ids"],
                    None,
                    data["hidden_states"],
                    data["loss_mask"][..., None],
                )
                loss = 0
                with torch.no_grad():
                    target_head = head(target.to(accelerator.device).to(torch.bfloat16))

                    target_head = target_head[:,:, map_tok]

                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                hidden_states_history = []

                hidden_states = model.module.fc(hidden_states.to(torch.bfloat16).to(accelerator.device))
                weight_sum = 0
                for forward_idx in range(forward_num_total):
                    predict = model(
                        hidden_states.to(accelerator.device), input_ids.to(accelerator.device), attention_mask, hidden_states_history
                    )
                    pred = model.module.lm_head_layernorm(predict)
                    pred = model.module.lm_head(pred.to(torch.bfloat16))

                    out_head, kldiv_loss = compute_loss(
                        target_p, pred, loss_mask, kldiv
                    )
                    total_loss = train_config["kldiv_w"] * kldiv_loss
                    weight = forward_idx + 1
                    loss += total_loss
                    weight_sum += weight
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
                cc = ((predicted == target) * loss_mask.squeeze().to(accelerator.device)).sum().item()
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

        epoch_loss /= num_batches
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {100 * correct / total:.2f}%")

            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(), f"{args.cpdir}/model{epoch}.safetensors"
            )







    print("finished training")


def main(
    verifier: str,
    data: str,
    verifier_batch_size: int,
    data_cache_limit: int = 1e6,
    verifier_gpus: Union[
        float, int, list[int]
    ] = 0.5,  # Default to half of available GPUs
    train_gpus: Union[float, int, list[int]] = 0.5,  # Default to half of available GPUs
    **train_kwargs,
) -> None:
    """
    Main function that runs in the main process and handles running training.

    :param verifier: Model name/path for vLLM verifier
    :param data: Path to source data file
    :param verifier_batch_size: Batch size for verifier processing
    :param data_cache_limit: Maximum size of data cache queue
    :param verifier_env_vars: Environment variables for verifier process
    :param verifier_gpu_ids: List of GPU IDs for vLLM data generation process
    :param train_gpu_ids: List of GPU IDs for training process
    :param train_kwargs: Additional training arguments
    """
    verifier_gpu_ids, train_gpu_ids = get_gpu_ids_split(
        verifier_gpus,
        train_gpus,
    )
    print(verifier_gpu_ids)
    print(train_gpu_ids)

    with multiprocessing.Manager() as manager:
        data_queue = multiprocessing.Queue(maxsize=int(data_cache_limit))
        data_gen_shutdown_event = manager.Event()

        data_gen_process = multiprocessing.Process(
            target=online_data_generator,
            args=(
                data_queue,
                data_gen_shutdown_event,
                verifier,
                data,
                verifier_batch_size,
                verifier_gpu_ids,
            ),
        )
        data_gen_process.start()

        
        train(data_queue=data_queue, gpu_ids=train_gpu_ids,args=args, **train_kwargs)

        data_gen_shutdown_event.set()
        data_gen_process.join()


if __name__ == "__main__":
    main("meta-llama/Llama-3.1-8B-Instruct","ShareGPT_V4.3_unfiltered_cleaned_split.json", 1)