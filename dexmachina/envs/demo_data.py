import os
import sys
import torch
import numpy as np
from os.path import join
from functools import lru_cache
from dexmachina.asset_utils import get_asset_path

ARCTIC_PROCESSED_DIR = get_asset_path("arctic/processed")
RETARGET_DIR = get_asset_path("retargeted")
IK_RETARGET_DIR=get_asset_path("retargeter_results")
RETARGET_CONTACT_DIR= get_asset_path("contact_retarget")

def get_demo_data(
    obj_name="box", 
    frame_start=10, 
    frame_end=30, 
    hand_name='inspire_hand', 
    subject_name="s01", 
    use_clip="01",
    load_retarget_contact=False, 
):
    """ This is only processed Arctic data, not the raw data. Not including dexterous hand retargeting data. """
    demo_fname = f"{ARCTIC_PROCESSED_DIR}/{subject_name}/{obj_name}_use_{use_clip}.npy"
    demo_data = np.load(demo_fname, allow_pickle=True).item() 
    world_coord = demo_data["world_coord"] 
    # this contains dict_keys(['joints.left', 'joints.right', 'contacts.left', 'valid_contacts.left', 'contacts.right', 'valid_contacts.right', 'contact_threshold', 'contact_links_left', 'contact_links_right'])
    demo_data = demo_data["params"] 
    
    demo_data = {
        "obj_pos": demo_data["obj_trans"][frame_start:frame_end], 
        "obj_quat": demo_data["obj_quat"][frame_start:frame_end],
        "obj_arti": demo_data["obj_arti"][frame_start:frame_end], 
        "contact_links_left": world_coord["contact_links_left"][frame_start:frame_end],
        "contact_links_right": world_coord["contact_links_right"][frame_start:frame_end],
    }
    if load_retarget_contact:
        retar_contact = load_contact_retarget_data(
            obj_name=obj_name, hand_name=hand_name, frame_start=frame_start, frame_end=frame_end,
            use_clip=use_clip, subject_name=subject_name,
        )
        print(f"Replacing demo_data with retarget contact data")
        demo_data.update(retar_contact) 
    return demo_data
 
def get_joint_init_limits(joint_pos_dict):
    limits = dict()
    default_qpos = dict()
    default_margin = 0.15
    for k, v in joint_pos_dict.items():        
        margin = default_margin
        if 'tx' in k or 'ty' in k or 'tz' in k:
            margin = 0.2 # 20 cm
        if 'roll' in k or 'pitch' in k or 'yaw' in k:
            # print(f"Using 30 degrees margin for {k}")
            margin = 0.5 # 30 degrees
        limits[k] = (min(v) - margin, max(v) + margin)
        default_qpos[k] = v[0] 
    return limits, default_qpos
 
@lru_cache(maxsize=None)
def load_genesis_retarget_data(
    obj_name="box",
    hand_name='inspire_hand',
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
    given_data_fname=None,
    use_ik_retarget=False,      # use the trajectory coming from the pure IK retargeting instead of the smoothed IK
):
    ''' retarget_data is a dict containing a dict for left and right hands
        for each hand there is  init_qpos=init_pos, 
                                limits=limits, 
                                residual_qpos=sliced_qpos,
                                qpos_targets=qpos_targets,
                                num_frames=num_frames,
                                kpts_data=kpt_info,
                                wrist_pose=wrist_pose, # need this for contact frame reward
    '''
    """ data saved from new retargeting code """
    ret_type = "vector"
    if 'shadow' in hand_name:
        print(f"Using position retargeting for {hand_name}")
        ret_type = "position"
    if given_data_fname is not None:
        data_fname = given_data_fname
    elif use_ik_retarget:       # using only the trajectories from the pure IK retargeting
        data_fname = f"{RETARGET_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}_{ret_type}_pure_ik_{save_name}.npy"
        print("Training with just IK retargeting data")
    else:
        # ex: retargeted/allegro_hand/s01/ketchup_use_u01_vector_para.npy
        data_fname = f"{RETARGET_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}_{ret_type}_{save_name}.npy"
        print("Training with just IK and smoothign data")

    loaded_tensor = False
    
    if not os.path.exists(data_fname):
        # try .pt extension
        data_fname = data_fname.replace(".npy", ".pt")
        assert os.path.exists(data_fname), f"File {data_fname} not found"

    if data_fname.endswith(".npy"):
        data = np.load(data_fname, allow_pickle=True).item()
    else:
        data = torch.load(data_fname, weights_only=False)
        loaded_tensor = True 

    demo_data = data["demo_data"] 
    demo_data = {k: v[frame_start:frame_end] for k, v in demo_data.items()}
    if len(demo_data['obj_arti'].shape) > 1:
        demo_data['obj_arti'] = demo_data['obj_arti'][:, 0] # shape (num_frames,)

    retarget_loaded = data["retarget_data"] 
    retarget_data = dict()
    for side in ['left', 'right']:
        loaded = retarget_loaded[side]
        residual_qpos = loaded["joint_qpos"]
        sliced_qpos = {k: v[frame_start:frame_end] for k, v in residual_qpos.items()}
        
        qpos_targets = None 
        if 'joint_targets' in loaded:
            print("Using joint_targets")
            qpos_targets = loaded["joint_targets"]
            qpos_targets = {k: v[frame_start:frame_end] for k, v in qpos_targets.items()}
        limits, init_pos = get_joint_init_limits(sliced_qpos) # this is a dict 
        kpt_pos = loaded["kpt_pos"]
        if len(kpt_pos.shape) > 3:
            print("Omitting the first dimension of kpt_pos")
            kpt_pos = kpt_pos[0]
        kpt_pos = kpt_pos[frame_start:frame_end]
        
        kpt_names_raw = loaded["kpt_names"]
        # Deduplicate keypoint names while preserving order
        seen = set()
        kpt_names = []
        unique_idxs = []
        for i, name in enumerate(kpt_names_raw):
            if name not in seen:
                kpt_names.append(name)
                unique_idxs.append(i)
                seen.add(name)
        
        # Verify that duplicate keypoints have identical position values
        if len(unique_idxs) < len(kpt_names_raw):
            from collections import defaultdict
            name_to_indices = defaultdict(list)
            for i, name in enumerate(kpt_names_raw):
                name_to_indices[name].append(i)
            
            for name, indices in name_to_indices.items():
                if len(indices) > 1:
                    # Check all occurrences match the first
                    ref_pos = kpt_pos[:, indices[0], :]  # shape (T, 3)
                    for idx in indices[1:]:
                        max_diff = torch.max(torch.abs(kpt_pos[:, idx, :] - ref_pos)).item()
                        assert max_diff < 1e-5, (
                            f"[{side}] Duplicate keypoint '{name}' at indices {indices[0]} and {idx} "
                            f"have DIFFERENT position values (max_diff={max_diff:.2e}). "
                            f"Cannot safely deduplicate!"
                        )
        
        # Filter kpt_pos to only keep unique keypoints
        if len(unique_idxs) < kpt_pos.shape[1]:
            print(f"[{side}] Filtering kpt_pos: {kpt_pos.shape[1]} → {len(unique_idxs)} unique keypoints")
            kpt_pos = kpt_pos[:, unique_idxs, :]  # (T, unique_kpts, 3)
        
        kpt_info = dict(
            kpt_pos=kpt_pos,
            kpt_names=kpt_names,
        )
        # print("LOADING IN KEYBPOITN DICT: ", kpt_info['kpt_names'])
        wrist_pose = loaded[f"wrist_pose"]
        if len(wrist_pose.shape) > 2:
            print("Omitting the first dimension of wrist_pose")
            wrist_pose = wrist_pose[0]
        wrist_pose = wrist_pose[frame_start:frame_end]
        num_frames = wrist_pose.shape[0]
        retarget_data[side] = dict(
            init_qpos=init_pos, 
            limits=limits, 
            residual_qpos=sliced_qpos,
            qpos_targets=qpos_targets,
            num_frames=num_frames,
            kpts_data=kpt_info,
            wrist_pose=wrist_pose, # need this for contact frame reward
            ) 
    return demo_data, retarget_data         # returns (2,) one per hand

@lru_cache(maxsize=None)
def load_contact_retarget_data(
    obj_name="box",
    hand_name='inspire_hand',
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
):
    # e.g. assets/contact_retarget/ability_hand/s01/box_use_01.npy
    fname = f"{RETARGET_CONTACT_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}.npy"
    assert os.path.exists(fname), f"File {fname} not found"
    loaded = np.load(fname, allow_pickle=True).item()
    retar_contact = dict()
    for side in ['left', 'right']:
        data = loaded[side] 
        for source_key, target_key in zip(
            ["dexlink_contacts", "dexlink_valid_contacts"],
            [f"contact_links_{side}", f"contact_links_valid_{side}"]
        ):
            retar_contact[target_key] = data[source_key][frame_start:frame_end]
        retar_contact[side] = {key: data[key] for key in ["collision_link_names", "collision_link_local_idxs"]}
        
    return retar_contact

# class Multi_Demo():
#     def __init__():
#         # stores a list of the retargeted contacts, demo_data, retargeted_data