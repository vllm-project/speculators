# ruff: noqa E501

import argparse

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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

import re

from datasets import load_dataset

bigname = args.model_path
system_prompt = """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos."""

system_prompt = """You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was last updated on 2023-10-01. The current date is 2025-06-02.\n\nWhen you\'re not sure about some information, you say that you don\'t have the information and don\'t make up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")"""


def build_dataset_rank(
    tokenizer,
    split="train",  # noqa: ARG001
    select=None,  # noqa: ARG001
):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_{args.split}")
    ds1 = ds.select(range(args.start, args.end))
    num_proc = 4  # noqa: F841

    def preprocess_function(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}

        for i in range(len(examples["messages"])):
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            messages.extend(examples["messages"][i])
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.zeros_like(input_ids)

            pattern = r"\[/INST\](.*?)</s>"
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
                try:
                    loss_mask[matches[0] : matches[0] + n] = 1
                except:
                    print(loss_mask)
            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            # print(loss_mask)
        return new_examples

    ds1 = ds1.map(preprocess_function, batched=True, load_from_cache_file=False)

    ds1.set_format(type="torch")
    return ds1


bigtokenizer = AutoProcessor.from_pretrained(bigname, use_fast=False)
ds = build_dataset_rank(bigtokenizer)
bigmodel = Mistral3ForConditionalGeneration.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)  # noqa: F841
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp = probs[0].max(dim=1).values  # noqa: F841
    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
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


for id_, data in enumerate(ds):
    if id_ % 100 == 0:
        print(id_, end="\t")
    if id_ % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir, outdata)
