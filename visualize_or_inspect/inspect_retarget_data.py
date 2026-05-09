#!/usr/bin/env python3
"""Inspect retargeting data structure and find duplicate keypoint names."""

import torch
import numpy as np
import os
from collections import Counter

# Load the retargeting file
data_fname = "../dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_pure_ik_para.pt"
out_fname  = os.path.splitext(data_fname)[0] + "_structure.txt"

print(f"Loading: {data_fname}")
data = torch.load(data_fname, weights_only=False)

_lines = []

def emit(s=""):
    print(s)
    _lines.append(s)

def describe(val):
    if isinstance(val, torch.Tensor):
        return f"Tensor shape={tuple(val.shape)} dtype={val.dtype}"
    elif isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape} dtype={val.dtype}"
    elif isinstance(val, list):
        return f"list len={len(val)}"
    elif isinstance(val, str):
        return f"str={val}"
    else:
        return str(type(val).__name__)

def print_structure(d, indent=0):
    pad = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                emit(f"{pad}{k}: dict")
                print_structure(v, indent + 1)
            else:
                emit(f"{pad}{k}: {describe(v)}")
    else:
        emit(f"{pad}{describe(d)}")

emit("\n=== FULL FILE STRUCTURE ===")
print_structure(data)
emit("=" * 60)

retarget_loaded = data["retarget_data"]

for side in ['left', 'right']:
    emit(f"\n{'='*60}")
    emit(f"{side.upper()} HAND")
    emit(f"{'='*60}")

    loaded = retarget_loaded[side]

    # Check kpt_names
    kpt_names = loaded["kpt_names"]
    emit(f"\nKeypoint Names (len={len(kpt_names)}):")
    for i, name in enumerate(kpt_names):
        emit(f"  {i:2d}: {name}")

    # Find duplicates
    name_counts = Counter(kpt_names)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}

    if duplicates:
        emit(f"\nDUPLICATES FOUND:")
        for name, count in duplicates.items():
            indices = [i for i, n in enumerate(kpt_names) if n == name]
            emit(f"  '{name}' appears {count}x at indices {indices}")
    else:
        emit("\nNo duplicates found")

    # Check kpt_pos shape
    kpt_pos = loaded["kpt_pos"]
    emit(f"\nKeypoint Positions shape: {kpt_pos.shape}")
    emit(f"  Expected: (num_frames=600, num_keypoints={len(kpt_names)}, xyz=3)")

    # Check wrist_link_name
    emit(f"\nWrist Link Name: {loaded.get('wrist_link_name', 'NOT SET')}")

    # List joint names
    joint_qpos = loaded["joint_qpos"]
    emit(f"\nJoint Names in retargeting (len={len(joint_qpos)}):")
    for i, jname in enumerate(joint_qpos.keys()):
        emit(f"  {i:2d}: {jname}")

with open(out_fname, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved structure to: {out_fname}")
