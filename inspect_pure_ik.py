#!/usr/bin/env python3
import numpy as np

data_fname = "dexmachina/assets/retargeter_results/allegro_hand/s01/ketchup_use_01_vector_pure_ik.npy"
print(f"Loading: {data_fname}\n")
data = np.load(data_fname, allow_pickle=True).item()

def describe(val, indent=0):
    pad = "  " * indent
    if isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape} dtype={val.dtype}"
    elif isinstance(val, list):
        return f"list len={len(val)}"
    elif isinstance(val, str):
        return f"str = {val}"
    else:
        return str(type(val).__name__)

def print_structure(d, indent=0):
    pad = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{pad}{k}: dict")
                print_structure(v, indent + 1)
            else:
                print(f"{pad}{k}: {describe(v, indent)}")
    else:
        print(f"{pad}{describe(d, indent)}")

print_structure(data)
