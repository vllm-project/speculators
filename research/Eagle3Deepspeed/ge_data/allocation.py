# This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
# Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)


import argparse
import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--outdir", type=str, default="0")
parser.add_argument("--data_path", type=str, default="0")
parser.add_argument("--model_path", type=str, default="0")
parser.add_argument("--dataset", type=str, default="Ultrachat")
parser.add_argument("--total_gpus", type=int, default=8)
parser.add_argument("--gpus_per_model", type=int, default=1)
parser.add_argument("--samples", type=int, default=68000)
parser.add_argument("--split", type=str, default="sft")
parser.add_argument("--chat_template", type=str, default="llama")


args = parser.parse_args()


s = 0
e = args.samples
gpus = [
    [i + j for j in range(args.gpus_per_model)]
    for i in range(0, args.total_gpus, args.gpus_per_model)
]


num_p = len(gpus)
outdir = args.outdir


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append(
                (previous, previous + current_interval - 1)
            )  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = gpus[i]
    gpu_index_str = " ".join(map(str, gpu_index))
    if args.chat_template == "llama":
        command = f"python ge_data/llama{args.dataset}.py --start={start} --end={end} --index={index} --gpu_index {gpu_index_str} --outdir {outdir} --data_path {args.data_path} --model_path {args.model_path} --split {args.split}"
    elif args.chat_template == "qwen":
        command = f"python ge_data/qwen{args.dataset}.py --start={start} --end={end} --index={index} --gpu_index {gpu_index_str} --outdir {outdir} --data_path {args.data_path} --model_path {args.model_path} --split {args.split}"
    else:
        raise NotImplementedError("Only llama and qwen chat templates are supported.")

    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
