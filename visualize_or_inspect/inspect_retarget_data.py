#!/usr/bin/env python3
"""Inspect retargeting data structure and find duplicate keypoint names."""

import torch
import numpy as np
import os
from collections import Counter

# Load the retargeting file
data_fname = "/home/jeffrey/Documents/Manipulation/Genesis/dexmachina/dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_para_1-60dt.pt"
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

# ── Wrist pose inspection ──────────────────────────────────────────────────
print("\n=== WRIST POSE (first 5 frames, both hands) ===")
for side in ['left', 'right']:
    wp = data['retarget_data'][side]['wrist_pose']
    if isinstance(wp, np.ndarray):
        wp = torch.tensor(wp)
    wp = wp.float()
    print(f"\n{side.upper()} wrist_pose shape: {tuple(wp.shape)}")
    for i in range(min(5, len(wp))):
        pos  = wp[i, :3].cpu().data.numpy().round(4)
        quat = wp[i, 3:].cpu().data.numpy().round(4)
        print(f"  frame {i}: pos={pos}  quat=[w={quat[0]:.4f} x={quat[1]:.4f} y={quat[2]:.4f} z={quat[3]:.4f}]")

    # Frame-to-frame rotation distance for first 10 transitions
    print(f"\n  {side.upper()} frame-to-frame rotation jump (first 10):")
    for i in range(min(10, len(wp) - 1)):
        q1, q2 = wp[i, 3:].cpu().data.numpy(), wp[i+1, 3:].cpu().data.numpy()
        dot = abs(np.dot(q1, q2))
        dot = np.clip(dot, 0.0, 1.0)
        diff_rad = 2 * np.arccos(dot)
        flag = " <-- LARGE JUMP" if diff_rad > 0.5 else ""
        print(f"    frame {i}->{i+1}: {np.degrees(diff_rad):.2f} deg ({diff_rad:.4f} rad){flag}")

# ── Wrist rotation DOFs (roll/pitch/yaw) for first 5 frames ───────────────
print("\n=== WRIST ROTATION DOFs (first 5 frames, right hand) ===")
rot_keys = ['R_forearm_roll_link_joint', 'R_forearm_pitch_link_joint', 'R_forearm_yaw_link_joint']
jq = data['retarget_data']['right']['joint_qpos']
for key in rot_keys:
    if key in jq:
        vals = np.array(jq[key].cpu() if hasattr(jq[key], 'cpu') else jq[key])
        print(f"  {key}: {vals[:5].round(4)}")
    else:
        print(f"  {key}: NOT FOUND")
