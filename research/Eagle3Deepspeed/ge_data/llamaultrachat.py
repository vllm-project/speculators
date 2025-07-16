# This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
# Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument("--data_path", type=str, default="0")
parser.add_argument("--model_path", type=str, default="0")
parser.add_argument("--split", type=str, default="sft")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

bigname = args.model_path


def build_ds(
    tokenizer,
    split="train",
):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"{split}_{args.split}")
    ds1 = ds.select(range(args.start, args.end))

    def preprocess(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}

        for j in range(len(examples["messages"])):
            messages = [
                {
                    "role": "system",
                    "content": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
                },
            ]
            messages.extend(examples["messages"][j])
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

    ds1 = ds1.map(preprocess, batched=True, load_from_cache_file=False)

    ds1.set_format(type="torch")
    return ds1


bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
ds = build_ds(bigtokenizer)
bigmodel = AutoModelForCausalLM.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    num_layers = len(bigmodel.model.layers)
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    feature_fusion = [
        outs_big.hidden_states[3],
        outs_big.hidden_states[num_layers // 2 + 1],
        outs_big.hidden_states[-3],
    ]
    hidden_state_big = torch.cat(feature_fusion, dim=-1)
    target = outs_big.hidden_states[-1]

    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
        "target": target.cpu()[0],
    }


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)

outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for item, data in enumerate(ds):
    if item % 100 == 0:
        print(item, end="\t")
    if item % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir, outdata)
