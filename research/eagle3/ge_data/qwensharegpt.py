# This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
# Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)

import argparse
import os
import re

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
parser.add_argument("--split", type=str, default="0")


args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

bigname = args.model_path


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_ds(
    tokenizer,
    split="train",
):
    ds = load_dataset("json", data_files=args.data_path)
    ds = ds[split]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))

    original_columns1 = ds1.column_names

    def preprocess(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            messages = []
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][i]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for _j, sentence in enumerate(source):
                role = roles[sentence["from"]]

                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            input_ids = tokenizer(
                text=conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.zeros_like(input_ids)

            pattern = r"<\|im_start\|>assistant(.*?)<\|im_end\|>"

            responses = re.findall(pattern, conversation, re.DOTALL)

            for response in responses:
                search = tokenizer(
                    text=response,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]

                n = len(search)
                matches = [
                    i
                    for i in range(len(input_ids.tolist()) - n + 1)
                    if input_ids.tolist()[i : i + n] == search.tolist()
                ]
                loss_mask[matches[0] : matches[0] + n] = 1

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


bigtokenizer = AutoTokenizer.from_pretrained(bigname)
ds = build_ds(bigtokenizer)

bigmodel = AutoModelForCausalLM.from_pretrained(
    bigname, device_map="auto", torch_dtype="auto"
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
    target = outs_big.hidden_states[-1]
    hidden_state_big = torch.cat(feature_fusion, dim=-1)

    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
        "target": target.cpu()[0],
    }


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
