#This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
#Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)


import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--data_path', type=str, default='0')
parser.add_argument('--model_path', type=str, default='0')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

# s = 8400*4
s=0
e = 200000 - 1
# e = 68000 - 1
# e=68
gpus = [[0],[1],[2],[3],[4],[5],[6],[7]]
# gpus = [[0,1],[2,3],[4,5],[6,7]]
# gpus=[[0,1]]
# gpus=[[0]]
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
    command = "python ge_data/ultrachat.py --start={} --end={} --index={} --gpu_index {} --outdir {} --data_path {} --model_path {}".format(start, end, index,
                                                                                                gpu_index_str, outdir, args.data_path, args.model_path)
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
