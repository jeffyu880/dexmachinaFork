#!/usr/bin/env python3
"""Inspect retargeting data structure and find duplicate keypoint names."""

import torch
import numpy as np
from collections import Counter

# Load the retargeting file
data_fname = "dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_para.pt"

print(f"Loading: {data_fname}")
data = torch.load(data_fname, weights_only=False)

retarget_loaded = data["retarget_data"]

for side in ['left', 'right']:
    print(f"\n{'='*60}")
    print(f"{side.upper()} HAND")
    print(f"{'='*60}")
    
    loaded = retarget_loaded[side]
    
    # Check kpt_names
    kpt_names = loaded["kpt_names"]
    print(f"\nKeypoint Names (len={len(kpt_names)}):")
    for i, name in enumerate(kpt_names):
        print(f"  {i:2d}: {name}")
    
    # Find duplicates
    name_counts = Counter(kpt_names)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    
    if duplicates:
        print(f"\n⚠️  DUPLICATES FOUND:")
        for name, count in duplicates.items():
            indices = [i for i, n in enumerate(kpt_names) if n == name]
            print(f"  '{name}' appears {count}x at indices {indices}")
    else:
        print("\n✓ No duplicates found")
    
    # Check kpt_pos shape
    kpt_pos = loaded["kpt_pos"]
    print(f"\nKeypoint Positions shape: {kpt_pos.shape}")
    print(f"  Expected: (num_frames=600, num_keypoints={len(kpt_names)}, xyz=3)")
    
    # Check wrist_link_name
    print(f"\nWrist Link Name: {loaded.get('wrist_link_name', 'NOT SET')}")
    
    # List joint names
    joint_qpos = loaded["joint_qpos"]
    print(f"\nJoint Names in retargeting (len={len(joint_qpos)}):")
    for i, jname in enumerate(joint_qpos.keys()):
        print(f"  {i:2d}: {jname}")
