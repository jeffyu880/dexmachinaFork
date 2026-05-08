#!/usr/bin/env python3
"""
Analyze and display saved demo trajectory data.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_trajectory_file(traj_file):
    """Load a trajectory JSON file."""
    with open(traj_file, 'r') as f:
        return json.load(f)


def display_trajectories(traj_dir='demo_trajectories'):
    """Display all trajectories in the directory with nice formatting."""
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        print(f"[ERROR] Directory not found: {traj_dir}")
        return
    
    traj_files = sorted(traj_dir.glob('*_trajectories.json'))
    if not traj_files:
        print(f"[WARNING] No trajectory files found in {traj_dir}")
        return
    
    print("\n" + "="*80)
    print("SAVED DEMO TRAJECTORIES")
    print("="*80)
    
    for i, traj_file in enumerate(traj_files, 1):
        try:
            traj = load_trajectory_file(traj_file)
            
            print(f"\n[{i}] {traj_file.name}")
            print("-" * 80)
            print(f"    Demo Index: {traj.get('demo_idx')}")
            print(f"    Demo Name:  {traj.get('demo_name')}")
            
            # Object info
            if 'object' in traj:
                obj = traj['object']
                print(f"\n    Object Trajectory:")
                print(f"      • Frames:           {obj.get('num_frames')}")
                print(f"      • Shape:            {obj.get('trajectory_shape')}")
                print(f"      • Start position:   {obj.get('first_position')}")
                print(f"      • End position:     {obj.get('last_position')}")
                if 'first_quat' in obj:
                    print(f"      • Start quaternion: {obj.get('first_quat')}")
            
            # Hand info
            if 'hands' in traj:
                print(f"\n    Hand Trajectories:")
                for side in ['left', 'right']:
                    if side in traj['hands']:
                        hand = traj['hands'][side]
                        print(f"      {side.upper()}:")
                        print(f"        • Frames:        {hand.get('num_frames')}")
                        print(f"        • Joint shape:   {hand.get('qpos_shape')}")
                        if 'residual_qpos_keys' in hand:
                            keys = hand['residual_qpos_keys']
                            if isinstance(keys, list):
                                print(f"        • Joints:        {len(keys)} ({', '.join(keys[:3])}...)")
        
        except Exception as e:
            print(f"    [ERROR] Failed to load: {e}")
    
    print("\n" + "="*80 + "\n")


def compare_trajectories(traj_dir='demo_trajectories'):
    """Compare multiple trajectory files."""
    traj_dir = Path(traj_dir)
    traj_files = sorted(traj_dir.glob('*_trajectories.json'))
    
    if len(traj_files) < 2:
        print("[INFO] Need at least 2 trajectory files to compare")
        return
    
    print("\n" + "="*80)
    print("TRAJECTORY COMPARISON")
    print("="*80)
    
    trajectories = {}
    for traj_file in traj_files:
        try:
            traj = load_trajectory_file(traj_file)
            name = traj.get('demo_name', traj_file.stem)
            trajectories[name] = traj
        except:
            pass
    
    if not trajectories:
        return
    
    # Compare object trajectories
    print("\nObject Trajectory Comparison:")
    print("-" * 80)
    print(f"{'Demo Name':<40} {'Frames':<10} {'Start Pos':<30}")
    print("-" * 80)
    
    for name, traj in trajectories.items():
        if 'object' in traj:
            obj = traj['object']
            frames = obj.get('num_frames', 'N/A')
            start_pos = obj.get('first_position', [0, 0, 0])
            start_str = f"[{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]"
            print(f"{name:<40} {frames:<10} {start_str:<30}")
    
    # Compare hand trajectories
    print("\n\nHand Joint Comparison:")
    print("-" * 80)
    print(f"{'Demo Name':<40} {'Left Hand':<20} {'Right Hand':<20}")
    print("-" * 80)
    
    for name, traj in trajectories.items():
        left_info = "N/A"
        right_info = "N/A"
        
        if 'hands' in traj:
            if 'left' in traj['hands']:
                left_frames = traj['hands']['left'].get('num_frames', 0)
                left_shape = traj['hands']['left'].get('qpos_shape', [0, 0])
                left_info = f"{left_frames} frames, {left_shape[1]} joints"
            
            if 'right' in traj['hands']:
                right_frames = traj['hands']['right'].get('num_frames', 0)
                right_shape = traj['hands']['right'].get('qpos_shape', [0, 0])
                right_info = f"{right_frames} frames, {right_shape[1]} joints"
        
        print(f"{name:<40} {left_info:<20} {right_info:<20}")
    
    print("\n" + "="*80 + "\n")


def extract_wrist_distances(traj_dir='demo_trajectories'):
    """
    Calculate and display object displacement for each demo.
    """
    traj_dir = Path(traj_dir)
    traj_files = sorted(traj_dir.glob('*_trajectories.json'))
    
    if not traj_files:
        print("[INFO] No trajectory files to analyze")
        return
    
    print("\n" + "="*80)
    print("OBJECT DISPLACEMENT ANALYSIS")
    print("="*80)
    print(f"{'Demo Name':<40} {'Displacement':<15} {'Distance':<12}")
    print("-" * 80)
    
    for traj_file in traj_files:
        try:
            traj = load_trajectory_file(traj_file)
            name = traj.get('demo_name', traj_file.stem)
            
            if 'object' in traj:
                obj = traj['object']
                start = np.array(obj.get('first_position', [0, 0, 0]))
                end = np.array(obj.get('last_position', [0, 0, 0]))
                
                # Calculate displacement vector
                displacement = end - start
                magnitude = np.linalg.norm(displacement)
                
                displacement_str = f"[{displacement[0]:7.4f}, {displacement[1]:7.4f}, {displacement[2]:7.4f}]"
                distance_str = f"{magnitude:.6f} m"
                
                print(f"{name:<40} {displacement_str:<15} {distance_str:<12}")
        except Exception as e:
            print(f"{name:<40} [ERROR] {str(e):<27}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    traj_dir = 'demo_trajectories'
    
    # Display all trajectories
    display_trajectories(traj_dir)
    
    # Compare trajectories
    compare_trajectories(traj_dir)
    
    # Analyze object displacement
    extract_wrist_distances(traj_dir)
    
    print("\n[TIP] Load JSON files directly for full trajectory data:")
    print("      import json")
    print("      with open('demo_trajectories/demo_name_trajectories.json') as f:")
    print("          data = json.load(f)")
