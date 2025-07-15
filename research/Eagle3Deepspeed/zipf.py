# This file calculates the most commonly used tokens 
# and generates a restricted vocabulary for the draft model based on those frequencies.
import argparse
import os

import numpy as np
import torch

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--dataDir", type=str, default="0")
parser.add_argument("--samples", type=int, default=10000)
parser.add_argument("--vocab", type=int, default=128256)
parser.add_argument("--reduced", type=int, default=32000)


args = parser.parse_args()


all_ids = np.zeros(args.vocab)


n = args.samples

prefix = args.dataDir


def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


files = list_files(args.dataDir)

for i in range(n):
    data = torch.load(files[i])
    ids = data["input_ids"].numpy()
    del data
    unique_values, counts = np.unique(ids, return_counts=True)
    all_ids[unique_values] += counts

vocab_size = args.reduced
keep = np.argsort(-1 * all_ids)[:vocab_size]
t2d = np.full(args.vocab, False, dtype=bool)
t2d[keep] = True
np.save("t2d.npy", t2d)
d2t = (np.arange(0, len(all_ids))[t2d]) - np.arange(0, vocab_size)
np.save("d2t.npy", d2t)
