file = "Eagle3UnlockedHead5step/model4.safetensors"
import torch

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

import numpy as np

state_dict["t2d"] = torch.from_numpy(np.load("t2d.npy")).bool()

state_dict["d2t"] = torch.from_numpy(np.load("d2t.npy"))

from safetensors.torch import save_file

save_file(state_dict, "eagle3/model.safetensors")

print(state_dict.keys())
