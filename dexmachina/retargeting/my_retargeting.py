import numpy as np
import yaml
import torch
from pathlib import Path
from copy import deepcopy

from dex_retargeting.retargeting_config import RetargetingConfig
from dexmachina.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

# ── Config ────────────────────────────────────────────────────────────────────
DEMO_PATH   = "../assets/arctic/processed/s01/ketchup_use_01.npy"
CONFIG_PATH = "../assets/allegro_hand/retarget_config.yaml"
SAVE_PATH   = "retargeted/allegro_hand/s01/ketchup_use_01.npy"
HAND_SIDE   = "right"    # "right" or "left"
START_FRAME = 0
END_FRAME   = None       # None = use all frames

# ── MANO 21 keypoint indices ──────────────────────────────────────────────────
# 0         = wrist
# 1, 2, 3, 17   = index  (MCP, PIP, DIP, tip)
# 4, 5, 6, 18   = middle  (MCP, PIP, DIP, tip)
# 7, 8, 9, 20 = pinky
# 10, 11, 12, 19 = ring
# 13, 14, 15, 16 = thumb

def load_arctic_demo(path: str, hand_side: str, start: int, end: int):
    """
    Load processed ARCTIC demo and return hand keypoints.

    The processed .npy file from DexMachina's process_arctic.py contains:
        'right_hand_kps'  : (T, 21, 3)  right hand 3D keypoints (metres, world frame)
        'left_hand_kps'   : (T, 21, 3)  left hand  3D keypoints
        'obj_pose'        : (T, 7)      object pose (pos + quat) per frame
        'obj_angle'       : (T,)        articulation angle per frame
    """
    data = np.load(path, allow_pickle=True).item()

    key = f"{hand_side}_hand_kps"
    if key not in data:
        raise KeyError(
            f"Key '{key}' not found in demo file. "
            f"Available keys: {list(data.keys())}"
        )

    kps = data[key]          # (T, 21, 3)
    end = end or kps.shape[0]
    kps = kps[start:end]

    print(f"Loaded {kps.shape[0]} frames of {hand_side} hand keypoints")
    print(f"  keypoint shape : {kps.shape}")
    print(f"  position range : x=[{kps[...,0].min():.3f}, {kps[...,0].max():.3f}]"
          f"  y=[{kps[...,1].min():.3f}, {kps[...,1].max():.3f}]"
          f"  z=[{kps[...,2].min():.3f}, {kps[...,2].max():.3f}]")
    
    print("loading")
    return kps

def main():
    load_arctic_demo(DEMO_PATH, HAND_SIDE, START_FRAME, END_FRAME)


if __name__ == '__main__':
    main()