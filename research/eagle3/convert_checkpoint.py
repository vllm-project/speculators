import os

import numpy as np
import torch
from safetensors.torch import save_file
import argparse
import os
parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--safetensors_file", type=str, default="Eagle3/model4.safetensors")
parser.add_argument("--vocab_dir", type=str, default=".")
parser.add_argument("--save_dir", type=str, default="trained_model")


args = parser.parse_args()




file = args.safetensors_file


f = torch.load(file)
print(f.keys())


state_dict = {}

keys = list(f.keys())
print("Keys in the safetensors file:", keys)
for key, tensor in f.items():
    state_dict[key] = tensor

state_dict["layers.0.hidden_norm.weight"] = state_dict["hidden_norm.weight"]
del state_dict["hidden_norm.weight"]
state_dict["layers.0.input_layernorm.weight"] = state_dict["input_layernorm.weight"]
del state_dict["input_layernorm.weight"]
state_dict["norm.weight"] = state_dict["lm_head_layernorm.weight"]
del state_dict["lm_head_layernorm.weight"]


state_dict["t2d"] = torch.from_numpy(np.load(args.vocab_dir+"/t2d.npy")).bool()

state_dict["d2t"] = torch.from_numpy(np.load(args.vocab_dir+"/d2t.npy"))

# Debug: print working directory
print("Working directory:", os.getcwd())

# Ensure directory exists
model_dir = args.save_dir
os.makedirs(model_dir, exist_ok=True)
print(f"Created or found existing directory: {model_dir}")

# Save your model
save_file(state_dict, os.path.join(model_dir, "model.safetensors"))
print("Model saved.")

print(state_dict.keys())
