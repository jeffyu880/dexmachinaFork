#!/usr/bin/env python3
"""Verify that duplicate keypoint names have identical position values."""

import torch
import numpy as np
from collections import defaultdict

# Load the retargeting file
data_fname = "dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_para.pt"

print(f"Loading: {data_fname}")
data = torch.load(data_fname, weights_only=False)

retarget_loaded = data["retarget_data"]

for side in ['left', 'right']:
    print(f"\n{'='*70}")
    print(f"{side.upper()} HAND")
    print(f"{'='*70}")
    
    loaded = retarget_loaded[side]
    kpt_names = loaded["kpt_names"]
    kpt_pos = loaded["kpt_pos"]  # shape (600, 25, 3)
    
    # Group indices by name
    name_to_indices = defaultdict(list)
    for i, name in enumerate(kpt_names):
        name_to_indices[name].append(i)
    
    # Find duplicates
    duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
    
    if not duplicates:
        print("✓ No duplicate keypoint names found")
        continue
    
    print(f"Found {len(duplicates)} duplicate keypoint names:")
    all_identical = True
    
    for name, indices in sorted(duplicates.items()):
        print(f"\n  '{name}' appears at indices: {indices}")
        
        # Extract positions for each occurrence
        positions = [kpt_pos[:, idx, :] for idx in indices]  # List of (600, 3) tensors
        
        # Check if all positions are identical across all frames
        identical = True
        for i in range(1, len(positions)):
            # Compare with first occurrence
            diff = torch.max(torch.abs(positions[i] - positions[0])).item()
            if diff > 1e-6:  # Allow tiny numerical differences
                identical = False
                all_identical = False
                print(f"    ✗ Indices {indices[0]} vs {indices[i]}: MAX DIFF = {diff:.2e}")
                # Show a few examples
                print(f"      Frame 0 - Index {indices[0]}: {positions[0][0].numpy()}")
                print(f"      Frame 0 - Index {indices[i]}: {positions[i][0].numpy()}")
                break
        
        if identical:
            print(f"    ✓ All {len(indices)} occurrences are IDENTICAL across all 600 frames")
            # Verify this for all frames
            max_diff = torch.max(torch.abs(positions[1] - positions[0])).item()
            print(f"      Max numerical difference: {max_diff:.2e}")
    
    print(f"\n{'─'*70}")
    if all_identical:
        print(f"✓ All duplicate keypoints have IDENTICAL position values")
        print(f"  Safe to deduplicate!")
    else:
        print(f"✗ WARNING: Some duplicate keypoints have DIFFERENT position values")
        print(f"  Check duplicates manually before deduplicating!")
