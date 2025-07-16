#This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
#Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)


import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--data_path', type=str, default='0')
parser.add_argument('--model_path', type=str, default='0')
parser.add_argument('--dataset', type=str, default='ultrachat')
parser.add_argument('--total_gpus', type=int, default=8)
parser.add_argument('--gpus_per_model', type=int, default=1)
parser.add_argument('--samples', type=int, default=68000)
parser.add_argument('--split', type=str, default="sft")
parser.add_argument('--chat_template', type=str, default="llama")


args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s=0
e = args.samples
gpus=[[i+j for j in range(args.gpus_per_model)] for i in range (0, args.total_gpus, args.gpus_per_model)]


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
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
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
    # gpu_index_str = [str(i) for i in gpu_index]
    # gpu_index_str=','.join(gpu_index_str)
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    # gpu_index_str='['+gpu_index_str+']'
    if args.chat_template=="llama":
        command = "python ge_data/{}.py --start={} --end={} --index={} --gpu_index {} --outdir {} --data_path {} --model_path {} --split {}".format(args.dataset, start, end, index,
                                                                                                gpu_index_str, outdir, args.data_path, args.model_path, args.split)
    elif args.chat_template=='mistral':
        command = "python ge_data/{}Mistral.py --start={} --end={} --index={} --gpu_index {} --outdir {} --data_path {} --model_path {} --split {}".format(args.dataset, start, end, index,
                                                                                                gpu_index_str, outdir, args.data_path, args.model_path, args.split)
    else:
        raise NotImplementedError("Only llama and mistral chat templates are supported.")

    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
