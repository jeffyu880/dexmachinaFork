#!/usr/bin/env python3
"""
Extract and save demo trajectories including:
- Demo number and name
- Object trajectories (positions and quaternions)
- Hand wrist positions (left and right)
- Hand joint positions
"""
import os
import sys
import json
import numpy as np
import pickle
import torch
from pathlib import Path
from collections import defaultdict


def load_pkl_data(pkl_path):
    """Load pickle data."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_trajectories_from_demo_data(demo_data, retarget_data, demo_idx, demo_name=None, output_dir='demo_trajectories'):
    """
    Extract trajectories from loaded demo data and save to file.
    
    Args:
        demo_data: Dictionary with 'obj_pos', 'obj_quat', etc.
        retarget_data: Dictionary with hand joint data
        demo_idx: Demo index
        demo_name: Human-readable demo name
        output_dir: Output directory
    
    Returns:
        Dictionary with extracted trajectory data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    trajectory_data = {
        "demo_idx": int(demo_idx),
        "demo_name": str(demo_name) if demo_name else None,
    }
    
    # Object trajectory
    if 'obj_pos' in demo_data:
        obj_pos = demo_data['obj_pos']
        if isinstance(obj_pos, torch.Tensor):
            obj_pos = obj_pos.cpu().numpy()
        trajectory_data['object'] = {
            'trajectory_shape': list(obj_pos.shape),
            'num_frames': int(obj_pos.shape[0]),
            'full_trajectory': obj_pos.tolist(),
            'first_position': obj_pos[0, :3].tolist(),
            'last_position': obj_pos[-1, :3].tolist(),
        }
        if 'obj_quat' in demo_data:
            obj_quat = demo_data['obj_quat']
            if isinstance(obj_quat, torch.Tensor):
                obj_quat = obj_quat.cpu().numpy()
            trajectory_data['object']['full_quaternions'] = obj_quat.tolist()
            trajectory_data['object']['first_quat'] = obj_quat[0].tolist()
            trajectory_data['object']['last_quat'] = obj_quat[-1].tolist()
    
    # Hand trajectories
    trajectory_data['hands'] = {}
    for side in ['left', 'right']:
        side_data = retarget_data.get(side, {})
        if 'residual_qpos' in side_data:
            residual_qpos = side_data['residual_qpos']
            num_frames = side_data.get('num_frames', 0)
            trajectory_data['hands'][side] = {
                'num_frames': int(num_frames),
            }
            # Handle both dict and array formats
            if isinstance(residual_qpos, dict) and len(residual_qpos) > 0:
                first_key = list(residual_qpos.keys())[0]
                qpos_array = residual_qpos[first_key]
                if isinstance(qpos_array, torch.Tensor):
                    qpos_array = qpos_array.cpu().numpy()
                trajectory_data['hands'][side]['residual_qpos_shape'] = list(qpos_array.shape)
                trajectory_data['hands'][side]['full_residual_qpos'] = qpos_array.tolist()
                trajectory_data['hands'][side]['residual_qpos_keys'] = list(residual_qpos.keys())
            elif isinstance(residual_qpos, (np.ndarray, torch.Tensor)):
                if isinstance(residual_qpos, torch.Tensor):
                    residual_qpos = residual_qpos.cpu().numpy()
                trajectory_data['hands'][side]['residual_qpos_shape'] = list(residual_qpos.shape)
                trajectory_data['hands'][side]['full_residual_qpos'] = residual_qpos.tolist()
    
    # Save to JSON
    safe_name = str(demo_name).replace('/', '_').replace(' ', '_') if demo_name else f"demo_{demo_idx}"
    output_file = os.path.join(output_dir, f"{safe_name}_full_trajectories.json")
    
    with open(output_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"[SAVED] Full trajectory data to {output_file}")
    return trajectory_data


def extract_trajectories_from_pkl(pkl_dir='dexmachina/assets', demo_names=None, output_dir='demo_trajectories'):
    """
    Extract trajectories from .pkl demo files.
    
    Args:
        pkl_dir: Directory containing demo .pkl files
        demo_names: List of specific demo names to extract (if None, extract all)
        output_dir: Output directory for trajectory files
    """
    pkl_dir = Path(pkl_dir)
    if not pkl_dir.exists():
        print(f"[ERROR] Directory not found: {pkl_dir}")
        return
    
    # Find all .pkl files
    pkl_files = list(pkl_dir.glob('**/*.pkl'))
    if not pkl_files:
        print(f"[ERROR] No .pkl files found in {pkl_dir}")
        return
    
    print(f"[INFO] Found {len(pkl_files)} .pkl files")
    
    all_trajectories = defaultdict(list)
    
    for pkl_file in pkl_files:
        try:
            data = load_pkl_data(str(pkl_file))
            demo_name = pkl_file.stem  # filename without extension
            
            if demo_names is not None and demo_name not in demo_names:
                continue
            
            print(f"\n[PROCESSING] {demo_name}")
            
            # Extract what we can from the pkl
            traj_info = {
                'demo_name': demo_name,
                'pkl_file': str(pkl_file),
            }
            
            # Check for object data
            if isinstance(data, dict):
                if 'obj_pos' in data:
                    obj_pos = data['obj_pos']
                    if isinstance(obj_pos, torch.Tensor):
                        obj_pos = obj_pos.cpu().numpy()
                    traj_info['object_num_frames'] = len(obj_pos)
                    traj_info['object_first_pos'] = obj_pos[0].tolist() if len(obj_pos) > 0 else None
                
                # Check for hand data
                for side in ['left', 'right']:
                    if side in data:
                        side_data = data[side]
                        if isinstance(side_data, dict) and 'residual_qpos' in side_data:
                            qpos = side_data['residual_qpos']
                            if isinstance(qpos, torch.Tensor):
                                qpos = qpos.cpu().numpy()
                            traj_info[f'{side}_num_frames'] = len(qpos)
                
                all_trajectories[demo_name].append(traj_info)
                print(f"  ✓ Object frames: {traj_info.get('object_num_frames', 'N/A')}")
                print(f"  ✓ Left hand: {traj_info.get('left_num_frames', 'N/A')} frames")
                print(f"  ✓ Right hand: {traj_info.get('right_num_frames', 'N/A')} frames")
        
        except Exception as e:
            print(f"  ✗ Error loading {pkl_file.name}: {e}")
    
    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'trajectory_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(dict(all_trajectories), f, indent=2)
    
    print(f"\n[SAVED] Summary to {summary_file}")
    return all_trajectories


def print_trajectory_summary(trajectory_data):
    """Print formatted summary of trajectory data."""
    print("\n" + "="*60)
    print("TRAJECTORY SUMMARY")
    print("="*60)
    print(f"Demo Index: {trajectory_data.get('demo_idx')}")
    print(f"Demo Name: {trajectory_data.get('demo_name')}")
    
    if 'object' in trajectory_data:
        obj = trajectory_data['object']
        print(f"\nObject Trajectory:")
        print(f"  Frames: {obj.get('num_frames')}")
        print(f"  Shape: {obj.get('trajectory_shape')}")
        print(f"  First position: {obj.get('first_position')}")
        print(f"  Last position: {obj.get('last_position')}")
    
    if 'hands' in trajectory_data:
        print(f"\nHand Trajectories:")
        for side in ['left', 'right']:
            if side in trajectory_data['hands']:
                hand = trajectory_data['hands'][side]
                print(f"  {side.upper()}:")
                print(f"    Frames: {hand.get('num_frames')}")
                print(f"    Joint shape: {hand.get('residual_qpos_shape')}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Extract trajectories from pkl files
    output_dir = 'demo_trajectories'
    
    # You can specify specific demo names or extract all
    demo_names = None  # Set to list like ['ketchup-30-130-s06-u01', 'ketchup-30-130-s01-u01'] to extract specific ones
    
    print(f"Extracting demo trajectories to: {output_dir}")
    trajectories = extract_trajectories_from_pkl(
        pkl_dir='dexmachina/assets',
        demo_names=demo_names,
        output_dir=output_dir
    )
    
    print(f"\n[DONE] Extracted {len(trajectories)} demos")
