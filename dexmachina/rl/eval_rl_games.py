import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
from datetime import datetime 

# Monkey-patch torch.load to use weights_only=False by default
# This is needed for loading RL-Games checkpoints with PyTorch 2.6+
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string, get_all_env_cfg  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv

  
from collections import defaultdict
import shutil
import moviepy
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from scipy.spatial.transform import Rotation


def remap_paths_in_config(config, server_username='jsyu', local_base_path=None):
    """
    Recursively remap server paths to local paths in configuration dictionaries.
    Handles the case where a checkpoint trained on a server has hard-coded paths
    that need to be mapped to the local machine.
    
    Args:
        config: Configuration dictionary (or nested structure) to remap
        server_username: Username on the server (default 'jsyu')
        local_base_path: Local base path to use for remapping (auto-detected if None)
    
    Returns:
        The config with paths remapped
    """
    from pathlib import Path
    
    # Auto-detect local base path if not provided
    if local_base_path is None:
        # Get current working directory and find the appropriate base
        cwd = os.getcwd()
        if 'dexmachina' in cwd:
            # Extract the Genesis prefix
            if '/Genesis/' in cwd:
                local_base_path = cwd.split('/Genesis/')[0] + '/Genesis'
            else:
                local_base_path = '/home/jeffrey/Documents/Manipulation/Genesis'
        else:
            local_base_path = '/home/jeffrey/Documents/Manipulation/Genesis'
    
    if isinstance(config, dict):
        remapped = {}
        for key, value in config.items():
            remapped[key] = remap_paths_in_config(value, server_username, local_base_path)
        return remapped
    elif isinstance(config, list):
        return [remap_paths_in_config(item, server_username, local_base_path) for item in config]
    elif isinstance(config, (str, Path)):
        # Convert Path to string for processing
        config_str = str(config)
        # Check if this is a server path
        if f'/home/{server_username}/' in config_str:
            # Replace server path with local path
            # Extract the part after Genesis/
            if '/Genesis/' in config_str:
                genesis_part = config_str.split('/Genesis/', 1)[1]
                # Try to construct the path
                local_path = os.path.join(local_base_path, genesis_part)
                
                # If file doesn't exist, try alternative folder (dexmachina <-> dexmachinaFork)
                if not os.path.exists(local_path):
                    if 'dexmachinaFork' in local_path:
                        alt_path = local_path.replace('dexmachinaFork', 'dexmachina', 1)
                    else:
                        alt_path = local_path.replace('dexmachina', 'dexmachinaFork', 1)
                    
                    if os.path.exists(alt_path):
                        local_path = alt_path
                
                print(f"[Path Remap] {config_str} -> {local_path}")
                return local_path
        # Always return as string
        return config_str
    else:
        return config



def compute_auc_add3(obj_states, obj_demo_states, object_models=None, num_timesteps=-1):
    """
    Compute AUC-ADD3 metric for object pose estimation.
    
    ADD (Average Distance of Model Points) measures the average L2 distance between
    transformed object points under estimated and ground truth poses.
    
    Args:
        obj_states: (T, 8) agent's object states [pos(3), quat(4), arti(1)]
        obj_demo_states: (T, 8) ground truth object states [pos(3), quat(4), arti(1)]
        object_models: dict with 'vertices' key containing object point cloud or None
        num_timesteps: int, number of timesteps to use for ADD calculation (-1 for all, default -1)
    
    Returns:
        dict with keys:
            - 'add_errors': (T,) ADD error at each timestep
            - 'add3_errors': (T,) ADD-3 errors (computed with 3 thresholds)
            - 'auc_add3': float, Area Under Curve for ADD-3
            - 'add_auc': float, Average ADD error
    """
    T = obj_states.shape[0]
    
    # Select timesteps if specified
    if num_timesteps > 0:
        obj_states = obj_states[:num_timesteps]
        obj_demo_states = obj_demo_states[:num_timesteps]
        T = num_timesteps
        print(f"Using first {num_timesteps} timesteps for ADD metric calculation")
    
    device = torch.device('cuda:0')

    obj_states = torch.tensor(obj_states, dtype=torch.float32, device=device)
    obj_states = torch.squeeze(obj_states)

    # Extract poses
    agent_pos = torch.tensor(obj_states[:, :3], dtype=torch.float32, device=device)
    agent_quat = torch.tensor(obj_states[:, 3:7], dtype=torch.float32, device=device)  # (T, 4)

    # print(agent_pos)
    # print(agent_quat)   

    demo_pos = torch.tensor(obj_demo_states[:, :3], dtype=torch.float32, device=device)
    demo_quat = torch.tensor(obj_demo_states[:, 3:7], dtype=torch.float32, device=device)  # (T, 4)
    
    # print(demo_pos)
    # print(demo_quat)

    # Use object vertices if provided
    if object_models is not None and 'vertices' in object_models:
        vertices = torch.tensor(object_models['vertices'], dtype=torch.float32, device=device)
        vertices = vertices.unsqueeze(0).repeat(T, 1, 1)  # (T, N, 3)
        print("using vertices")
    else:
        print("No vertices to compute ADD with")
        return None

    
    # Transform vertices using agent pose
    agent_R = Rotation.from_quat(agent_quat.cpu().numpy()).as_matrix()  # (T, 3, 3)
    agent_vertices = torch.tensor(agent_R, dtype=torch.float32, device=device) @ vertices.transpose(1, 2)
    agent_vertices = agent_vertices.transpose(1, 2) + agent_pos.unsqueeze(1)  # (T, N, 3)
    
    # Transform vertices using demo pose (ground truth)
    demo_R = Rotation.from_quat(demo_quat.cpu().numpy()).as_matrix()  # (T, 3, 3)
    demo_vertices = torch.tensor(demo_R, dtype=torch.float32, device=device) @ vertices.transpose(1, 2)
    demo_vertices = demo_vertices.transpose(1, 2) + demo_pos.unsqueeze(1)  # (T, N, 3)
    
    # Average distance between corresponding points
    point_distances = torch.norm(agent_vertices - demo_vertices, dim=2)  # (T, N)
    
    # Check if we have per-part information
    if 'num_bottom' in object_models and 'num_top' in object_models:
        num_bottom = object_models['num_bottom']
        num_top = object_models['num_top']
        
        # Split point distances into bottom and top
        bottom_distances = point_distances[:, :num_bottom]  # (T, num_bottom)
        top_distances = point_distances[:, num_bottom:]      # (T, num_top)
        
        print("bottom distances shape: ", bottom_distances.shape)
        print("top distances shape: ", top_distances.shape)

        # Compute ADD for each part for each timestep
        bottom_add = torch.mean(bottom_distances, dim=1)  # (T,)
        top_add = torch.mean(top_distances, dim=1)        # (T,)

        print("Top ADD error: ", (top_add))
        print("Bottom ADD error: ", (bottom_add))

        # Average per-part ADD
        add_avg_errors = (torch.sum(bottom_add.unsqueeze(dim=1) + top_add.unsqueeze(dim=1), dim=1) / 2.0)  # (T,)
        
        print("add avg errors: ", add_avg_errors)

        print(f"Computing ADD per-part:")
        print(f"  Bottom vertices: {num_bottom}")
        print(f"  Top vertices: {num_top}")

        # Fall back to overall ADD if per-part info not available
        add_errors = torch.mean(point_distances, dim=1)  # (T,) - average over points
       
    
    # Compute ADD-3 (typically uses 3 different distance thresholds)
    thresholds = [0.03, 0.05, 0.1]  # 3cm, 5cm, 10cm
    add3_scores = []
    add3_avg_scores = []
    
    for threshold in thresholds:
        # Metric 1: Per-point ADD-3
        # For each timestep, what % of individual points are within threshold?
        within_threshold = (point_distances <= threshold).float()  # (T, N)
        accuracy = torch.mean(within_threshold, dim=1)  # (T,) - avg across points
        add3_scores.append(accuracy.cpu().numpy())

        # Metric 2: Per-part averaged ADD-3
        # For each timestep, is the averaged ADD (bottom+top)/2 within threshold?
        within_threshold = (add_avg_errors.squeeze() <= threshold).float()  # (T,) - squeeze out dim 1
        add3_avg_scores.append(within_threshold.cpu().numpy())  # (# threshold, T)
    
    # AUC-ADD3: area under curve of the ADD-3 metric
    # Average the three threshold scores
    add3_mean = np.mean(add3_scores, axis=0)  # (T,)
    auc_add3 = np.mean(add3_mean)

    # AUC-ADD3: area under curve for ADD-3 using the average ADD from the top and bottom
    avg_add3_mean = np.mean(add3_avg_scores, axis=0)
    auc_avg_add3 = np.mean(avg_add3_mean)

    return {
        'mean_add_errors': np.mean(add_errors.cpu().numpy()),         # the mean across all timesteps for the errors calculated for every point on the object
        'mean_avg_add_errors': np.mean(add_avg_errors.cpu().numpy()),        # the mean across all timesteps for the average ADD error, taken between top and bottom objects
        'auc_avg_add3': auc_avg_add3,           # the auc calculated taken from the mean add from the top and bottom object parts
        'auc_add3': float(auc_add3),        # the auc calculated taken from all the points
        'episode_length': T,                # Include episode length
        # 'add_mean': float(np.mean(add_errors.cpu().numpy())),
        # 'add_max': float(np.max(add_errors.cpu().numpy())),
        # 'add_min': float(np.min(add_errors.cpu().numpy())),
    }


def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

def eval_one_episode(env, agent, obj_state_tensor, print_rew=False, record_video=False, show_reference=False, save_traj=False):
    obs = env.reset() 
    
    # print("Obs: ", obs)
    
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    joint_target_left = left_hand.residual_qpos
    joint_target_right = right_hand.residual_qpos
    
    eval_data = defaultdict(list)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
            demo_state = obj_state_tensor[env_step]
                    
            if show_reference: # visualize the demo traj and set zero action
                if num_envs < 2:
                    print(f"ERROR: show_reference requires at least 2 environments, but got {num_envs}")
                    print("Re-run with --num_envs 2")
                    show_reference = False
                else:
                    obj.set_object_state(
                        root_pos=demo_state[:3][None],
                        root_quat=demo_state[3:7][None],
                        joint_qpos=demo_state[7:][None],
                        env_idxs=torch.tensor([1], dtype=torch.int32, device=device),
                    )
                    actions[-1, :] = -1.0 
                    for robot, joints in zip([left_hand, right_hand], [joint_target_left, joint_target_right]):
                        robot.set_joint_position(
                            joint_targets=joints[env_step][None],
                            env_idxs=[1],
                        ) 
            obs, rew, dones, infos = env.step(actions) 
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())

            # Collect robot observations
            left_hand_obs = left_hand.get_observations()
            right_hand_obs = right_hand.get_observations()
            for key, val in left_hand_obs.items():
                eval_data[f"left_hand_{key}"].append(val.cpu().numpy() if hasattr(val, 'cpu') else val)
            for key, val in right_hand_obs.items():
                eval_data[f"right_hand_{key}"].append(val.cpu().numpy() if hasattr(val, 'cpu') else val)
            
            # Collect object observations
            obj_obs = obj.get_observations()
            for key, val in obj_obs.items():
                eval_data[f"obj_{key}"].append(val.cpu().numpy() if hasattr(val, 'cpu') else val)

            if env_step == 0:
                print(f"\n========== EVAL DEBUG STEP 0 ==========")
                print(f"obs shape: {obs.shape}, min: {obs.min():.3f}, max: {obs.max():.3f}")
                print(f"actions shape: {actions.shape}, min: {actions.min():.3f}, max: {actions.max():.3f}")
                print(f"obj_pos (sim):  {obj_pos[0].cpu().numpy().round(3)}")
                print(f"obj_pos (demo): {demo_state[:3].cpu().numpy().round(3)}")
                print(f"obj_quat (sim):  {obj_quat[0].cpu().numpy().round(3)}")
                print(f"obj_quat (demo): {demo_state[3:7].cpu().numpy().round(3)}")
                print(f"left  residual_qpos shape: {joint_target_left.shape}, step0: {joint_target_left[0].cpu().numpy().round(3)}")
                print(f"right residual_qpos shape: {joint_target_right.shape}, step0: {joint_target_right[0].cpu().numpy().round(3)}")
                print(f"left  curr_targets[0]: {left_hand.curr_targets[0].cpu().numpy().round(3)}")
                print(f"right curr_targets[0]: {right_hand.curr_targets[0].cpu().numpy().round(3)}")
                print(f"left  wrist_pose[0]: {left_hand.wrist_pose[0].cpu().numpy().round(3)}")
                print(f"right wrist_pose[0]: {right_hand.wrist_pose[0].cpu().numpy().round(3)}")
                ep_buf = uenv.episode_length_buf
                demo_wrist_left  = uenv.reward_module.match_demo_state("wrist_pose_left",  ep_buf)
                demo_wrist_right = uenv.reward_module.match_demo_state("wrist_pose_right", ep_buf)
                print(f"left  wrist_pose target[0]: {demo_wrist_left[0].cpu().numpy().round(3)}")
                print(f"right wrist_pose target[0]: {demo_wrist_right[0].cpu().numpy().round(3)}")
                print(f"========================================\n")
            # rew_dict = uenv.rew_dict
            # for key in ['pos_dist', 'rot_dist', 'arti_dist']:
            #     eval_data[key].append(rew_dict[key].cpu().numpy())
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0 
        
        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes 
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")

    # print("OBJ: ")
    # print(eval_data["obj_state"])
    # print("DEMO: ")
    # print(eval_data["demo_state"])

    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    frames = uenv.get_recorded_frames()
    return frames, eval_data


def get_camera_config(angle='front', res=1024):
    """Get camera configuration by angle preset
    
    Args:
        angle: Camera angle ('front', 'back', 'top', 'side', 'isometric')
        res: Resolution in pixels (width and height, e.g., 1024, 2048)
    """
    cameras = {
        'front': dict(
            res=(res, res),
            fov=30,
            pos=(0.0, -1.6, 2.2),
            lookat=(0.0, -0.1, 1.2),
        ),
        'back': dict(
            res=(res, res),
            fov=25,
            pos=(0.4, 1.5, 1.8),
            lookat=(0.0, -0.15, 1.0),
        ),
        'top': dict(
            res=(res, res),
            fov=30,
            pos=(0.0, 0.0, 5.0),
            lookat=(0.0, 0.0, 1.0),
        ),
        'side': dict(
            res=(res, res),
            fov=30,
            pos=(3.5, -0.1, 1.5),
            lookat=(0.0, -0.1, 1.0),
        ),
        'isometric': dict(
            res=(res, res),
            fov=30,
            pos=(2.2, -2.2, 2.6),
            lookat=(0.0, -0.1, 1.0),
        ),
    }
    return cameras.get(angle, cameras['front'])


def load_object_model_for_evaluation(obj_name):
    """
    Load object mesh vertices for ADD metric calculation.
    
    Args:
        obj_name: str, object name from ARCTIC dataset
        
    Returns:
        dict with 'vertices' key containing (N, 3) vertex positions
    """
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed. Install with: pip install trimesh")
        return None
    
    from dexmachina.envs.object import get_arctic_object_cfg
    from pathlib import Path
    
    try:
        obj_cfg = get_arctic_object_cfg(name=obj_name)
        
        # Load both parts of the articulated object
        bottom_path = obj_cfg['bottom_mesh_fname']
        top_path = obj_cfg['top_mesh_fname']
        
        # Load meshes
        bottom_mesh = trimesh.load(bottom_path)
        top_mesh = trimesh.load(top_path)
        
        # Combine vertices from both parts
        all_vertices = np.vstack([
            bottom_mesh.vertices,
            top_mesh.vertices
        ])
        
        print(f"[Object Mesh] Loaded {obj_name}")
        print(f"  Bottom vertices: {bottom_mesh.vertices.shape[0]}")
        print(f"  Top vertices: {top_mesh.vertices.shape[0]}")
        print(f"  Total vertices: {all_vertices.shape[0]}")
        
        return {
            'vertices': all_vertices,
            'bottom_vertices': bottom_mesh.vertices,
            'top_vertices': top_mesh.vertices,
            'num_bottom': bottom_mesh.vertices.shape[0],
            'num_top': top_mesh.vertices.shape[0]
        }
        
    except Exception as e:
        print(f"Warning: Could not load object mesh: {e}")
        return None

def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--npy_name', '-npy', type=str, default=None, help='Base name for the saved .npy file (e.g. "my_eval" -> my_eval_ep0.npy). Defaults to "eval".')
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--reference_clip', '-ref_clip', type=str, default=None, help='Alternative demonstration clip to use as reference trajectory (e.g., "box-0-100")')
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--camera_angle', '-cam', type=str, default='front', choices=['front', 'top', 'side', 'back', 'isometric'], help='Camera angle for video recording')
    parser.add_argument('--resolution', '-res', type=int, default=1024, help='Video resolution in pixels (512-4096, default 1024)')
    parser.add_argument('--demo_idx', '-di', type=int, default=None, help='Index of demo to use as reference (for multi-demo checkpoints). If not set, prompts interactively.')
    parser.add_argument('--save_traj', action='store_true', help='Record object and hand policy trajectory and demo trajectory')
    args = parser.parse_args()

    # Convert checkpoint path to absolute path if relative
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.abspath(args.checkpoint)
        print(f"[Path] Converted to absolute path: {args.checkpoint}")
    
    # Remove duplicate 'dexmachina' in path if present
    # e.g., /path/to/dexmachina/dexmachina/dexmachina/izar_logs -> /path/to/dexmachina/dexmachina/izar_logs
    while '/dexmachina/dexmachina/dexmachina/' in args.checkpoint:
        args.checkpoint = args.checkpoint.replace('/dexmachina/dexmachina/dexmachina/', '/dexmachina/dexmachina/')
        print(f"[Path] Removed duplicate 'dexmachina': {args.checkpoint}")

    # Auto-enable video recording if output_render is requested
    if args.output_render:
        args.record_video = True
        print("[INFO] Video recording enabled (output_render requested)")

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    os.makedirs(ckpt_data_folder, exist_ok=True) 

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = ckpt_data_folder
        # render_dir = os.path.join(ckpt_data_folder, run_name)
        # print('Saving video to a different folder')
        video_fname = os.path.join(ckpt_data_folder, ckpt_name)
        os.makedirs(ckpt_data_folder, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    # load to pkl
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    # Remap server paths to local paths
    print("[INFO] Remapping server paths to local paths...")
    env_kwargs = remap_paths_in_config(env_kwargs, server_username='jsyu')
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"

    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to {args.resolution}x{args.resolution} with camera angle: {args.camera_angle}") 
        # Get the camera config with specified resolution
        camera_cfg = get_camera_config(args.camera_angle, res=args.resolution)
        # Make sure the camera_kwargs has this angle defined
        if 'camera_kwargs' not in env_kwargs['env_cfg']:
            env_kwargs['env_cfg']['camera_kwargs'] = {}
        env_kwargs['env_cfg']['camera_kwargs'][args.camera_angle] = camera_cfg
        # Set the render_camera to use the specified angle
        env_kwargs['env_cfg']['render_camera'] = args.camera_angle
        print(f"Camera config set: pos={camera_cfg['pos']}, lookat={camera_cfg['lookat']}, res={camera_cfg['res']}") 
    for name, cfg in env_kwargs['object_cfgs'].items():
        print("Setting eval time obj gains to 0.0")
        cfg['actuated'] = False

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    print("Removing any curriculum config during eval")
    env_kwargs.pop("curriculum_cfg")
    
    # Load reference clip BEFORE creating environment so demo_data is correct from the start
    demo_tag = None   # set below for multi-demo checkpoints; used in output filenames
    all_demo_names = env_kwargs.get('all_demo_names', None)
    if all_demo_names and len(all_demo_names) > 1:
        if args.reference_clip is not None:
            print("Warning, using multi-demo loading, so ignoring the argument reference_clip")
        # Multi-demo checkpoint: prompt user or use --demo_idx
        print("\nMultiple demos found in checkpoint:")
        for i, name in enumerate(all_demo_names):
            print(f"  [{i}] {name}")
        chosen = args.demo_idx if args.demo_idx is not None else int(input(f"Select demo index [0-{len(all_demo_names)-1}]: "))
        assert 0 <= chosen < len(all_demo_names), f"Invalid demo index {chosen}"
        print(f"[INFO] Using demo [{chosen}]: {all_demo_names[chosen]}")
        # collapse to single demo so _setup_multi_demo doesn't round-robin across all demos
        env_kwargs['demo_data'] = env_kwargs['all_demo_data'][chosen]
        env_kwargs['retarget_data'] = env_kwargs['all_retarget_data'][chosen]
        # env_kwargs['all_demo_data'] = [env_kwargs['all_demo_data'][chosen]]
        # env_kwargs['all_retarget_data'] = [env_kwargs['all_retarget_data'][chosen]]
        # env_kwargs['all_demo_names'] = [all_demo_names[chosen]]
        demo_tag = all_demo_names[chosen].replace("/", "_")
        video_fname = join(ckpt_data_folder, f"video_{demo_tag}.mp4")
        if args.output_render:
            video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + f"_{demo_tag}" + args.video_fname)

    elif args.reference_clip is not None: # using a single but alternative demo than the one stored in the env.pkl
        from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
        print(f"\n[INFO] Loading alternative reference clip: {args.reference_clip}")
        # Parse reference clip
        obj_name_ref, start, end, subject_name, use_clip = parse_clip_string(args.reference_clip)
        # Use the hand type from the training checkpoint, not command-line default
        checkpoint_hand = env_kwargs['env_cfg'].get('hand', args.hand)
        print(f"[DEBUG] Using hand from checkpoint: {checkpoint_hand} (command-line was: {args.hand})")
        # Load reference demo data (object data only)
        demo_data = get_demo_data(
            obj_name=obj_name_ref, 
            frame_start=start, 
            frame_end=end, 
            hand_name=checkpoint_hand,
            subject_name=subject_name, 
            use_clip=use_clip,
            load_retarget_contact=True, 
        )
        # Also load retargeted hand trajectories
        _, ref_retarget_data = load_genesis_retarget_data(
            obj_name=obj_name_ref,
            hand_name=checkpoint_hand,
            frame_start=start,
            frame_end=end,
            save_name="para",  # Match the training retarget variant
            use_clip=use_clip,
            subject_name=subject_name,
        )

        print(f"[INFO] Loaded reference clip: {args.reference_clip}")
        print(f"       Subject: {subject_name}, Frames: {start}-{end} ({int(end)-int(start)} frames)")
        print(f"[DEBUG] Reference demo_data keys: {list(demo_data.keys())}, hand: {checkpoint_hand}")
        # Update env_kwargs to use the alternative reference demo
        env_kwargs['demo_data'] = demo_data
        env_kwargs['retarget_data'] = ref_retarget_data
    else:
        print(f"[INFO] Using training clip reference trajectory")
            
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    
    # # Load reference clip (either alternative or default training clip)
    # if args.reference_clip is not None:
    #     print(f"\n[INFO] Loading alternative reference clip: {args.reference_clip}")
    #     from dexmachina.envs.constructors import get_all_env_cfg
    #     # Parse reference clip
    #     obj_name, start, end, subject_name, use_clip = parse_clip_string(args.reference_clip)
    #     # Create temporary args for loading reference clip
    #     ref_args = argparse.Namespace(**env_kwargs['env_cfg'])
    #     ref_args.arctic_object = obj_name
    #     ref_args.frame_start = start
    #     ref_args.frame_end = end
    #     ref_args.arctic_subject = subject_name  # Override subject if different from training
    #     ref_args.use_clip = use_clip  # Override use_clip if different
    #     # Load reference clip data
    #     ref_env_cfg = get_all_env_cfg(ref_args, device='cuda:0')
    #     demo_data = ref_env_cfg['demo_data']
    #     print(f"[INFO] Loaded reference clip: {args.reference_clip}")
    #     print(f"       Subject: {subject_name}, Frames: {start}-{end} ({int(end)-int(start)} frames)")
    # else:

    # Extract object name from object_cfgs (it's the key in the dictionary)
    obj_name = list(env_kwargs['object_cfgs'].keys())[0]
    assert obj_name is not None, "ERROR: obj_name not found in object_cfgs. Object was not saved correctly in the environment!"
    
    print(f"[INFO] Loading object: {obj_name}")
    # Print the saved object position from training config
    obj_init_pos = env_kwargs['object_cfgs'][obj_name]['base_init_pos']
    obj_init_quat = env_kwargs['object_cfgs'][obj_name]['base_init_quat']
    print(f"[INFO] Saved ketchup init position: {obj_init_pos}")
    print(f"[INFO] Saved ketchup init quaternion: {obj_init_quat}")
    # Load object mesh for ADD metric
    object_models = load_object_model_for_evaluation(obj_name)
    obj_state_tensor = gather_object_state_tensor(env_kwargs['demo_data'])

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped 
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint) 
    agent.restore(resume_path)
    agent.reset()

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, 
            obj_state_tensor, 
            args.print_rew, 
            args.record_video, 
            args.show_reference,
            args.save_traj
            )
        
        # if object_models is not None:
        #     # Compute AUC-ADD3 metric
        #     print("\n" + "="*60)
        #     print("Computing AUC-ADD3 Metric")
        #     print("="*60)
        #     add3_metrics = compute_auc_add3(
        #         eval_data['obj_state'][:, 0, :],        # take just the first policy demonstration
        #         eval_data['demo_state'],
        #         object_models=object_models,  # Can pass object vertices if available
        #         num_timesteps=80            # number of timesteps to use for ADD calculations
        #     )
            
        #     # Add AUC-ADD3 metrics to eval_data
        #     eval_data['mean_add_errors'] = add3_metrics['mean_add_errors']
        #     eval_data['mean_avg_add_errors'] = add3_metrics['mean_avg_add_errors']
        #     eval_data['auc_add3'] = add3_metrics['auc_add3']
        #     eval_data['avg_auc3_add_errors'] = add3_metrics['auc_avg_add3']
        #     # eval_data['auc_add3'] = add3_metrics['auc_add3']
            
        #     # Print AUC-ADD3 results
        #     print(f"ADD Errors: {add3_metrics['mean_add_errors']}")
        #     print(f"Average ADD Errors: {add3_metrics['mean_avg_add_errors']}")
        #     print(f"AUC-ADD3 Score: {add3_metrics['auc_add3']:.6f}")
        #     print(f"Average AUC3 ADD Errors: {add3_metrics['auc_avg_add3']:.6f}")
        #     print("="*60 + "\n")

        #     # Save ADD metrics to JSON file with checkpoint name
        #     try:
        #         add_metrics_fname = os.path.join(ckpt_data_folder, f"{ckpt_name.split('.')[0]}_add_metrics_ep{eps}.json")
                
        #         # Ensure parent directory exists
        #         os.makedirs(os.path.dirname(add_metrics_fname), exist_ok=True)
                
        #         metrics_to_save = {
        #             'mean_add_errors': float(add3_metrics['mean_add_errors']),
        #             'mean_avg_add_errors': float(add3_metrics['mean_avg_add_errors']),
        #             'auc_add3': float(add3_metrics['auc_add3']),
        #             'auc_avg_add3': float(add3_metrics['auc_avg_add3']),
        #             'episode_length': int(add3_metrics['episode_length']),
        #             'checkpoint': args.checkpoint,
        #         }
        #         with open(add_metrics_fname, 'w') as f:
        #             json.dump(metrics_to_save, f, indent=2)
        #         print(f"✓ Saved ADD metrics to {add_metrics_fname}")
        #     except Exception as e:
        #         print(f"✗ Error saving ADD metrics JSON: {e}")
        
        npy_base = args.npy_name if args.npy_name is not None else (demo_tag if demo_tag is not None else "eval")
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"{npy_base}")
        np.save(os.path.join(ckpt_eval_fname, ".npy"), eval_data)
        print(f"Saved eval data to {ckpt_eval_fname}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item()

        if args.save_traj:
            def to_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy()
                if isinstance(obj, dict):
                    return {k: to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(to_cpu(v) for v in obj)
                return obj

            pkl_data = to_cpu({
                'policy_obj_state': eval_data['obj_state'],
                'policy_left_hand': {k.removeprefix('left_hand_'): v
                                      for k, v in eval_data.items() if k.startswith('left_hand_')},
                'policy_right_hand': {k.removeprefix('right_hand_'): v
                                       for k, v in eval_data.items() if k.startswith('right_hand_')},
                'demo_obj': env_kwargs['demo_data'],
                'demo_robot': env_kwargs.get('retarget_data', {}),
                'demo_state': eval_data['demo_state'],
            })
            pkl_fname = os.path.join(ckpt_eval_fname, ".pkl")
            with open(pkl_fname, 'wb') as f:
                pickle.dump(pkl_data, f)
            print(f"Saved pkl data to {pkl_fname}")

        if args.record_video:
            # save video with opencv (cv2)
            import cv2
            fps = int(1/uenv.dt/2)
            if len(frames) > 0:
                frame_height, frame_width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_fname, fourcc, fps, (frame_width, frame_height))
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                print(f"Saved video to {video_fname}")
            else:
                print("No frames recorded, skipping video save")
    
    print("Done evaluating")


if __name__ == '__main__':
    main()

