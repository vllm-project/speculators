#This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
#Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)

import pandas as pd   
import argparse
import re

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
parser.add_argument('--data_path', type=str, default='0')
parser.add_argument('--model_path', type=str, default='0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset



bigname = args.model_path


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):

    ds =load_dataset('HuggingFaceH4/ultrachat_200k', split='train_gen')
    ds1 = ds.select(range(args.start, args.end))
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }

        for i in range(len(examples['messages'])):

            messages = [
            ]
            messages.extend(examples['messages'][i])
            conversation=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id
            
            input_ids = tokenizer(
                conversation,
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
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
ds = build_dataset_rank(bigtokenizer)
bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)
bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"]
    num_layers=(len(bigmodel.model.layers))
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    featureFusion=[outs_big.hidden_states[3],outs_big.hidden_states[num_layers//2+1],outs_big.hidden_states[-3]]
    hidden_state_big=torch.cat(featureFusion, dim=-1)
    target=outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0], "target":target.cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id,data in enumerate(ds):
    if id%100==0:
        print(id,end="\t", flush=True)
    if id % 1000 == 0:
        print("", flush=True)
    outdata = ge(data)
    writedata(outdir,outdata)