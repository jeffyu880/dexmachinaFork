"""
Playback retargeted hand animation from parallel_retarget.py output
"""
import os
import sys
import time
import yaml 
import torch
import argparse
import numpy as np
import genesis as gs
from os.path import join
from pathlib import Path
from copy import deepcopy
import cv2

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.demo_data import get_demo_data 
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg 
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

def set_entities_to_step(hand_entities, retargeter_results, step, device):
    for side, hand in hand_entities.items():
        hand_qpos = retargeter_results[side]["hand_qpos"][step]
        hand_qpos = torch.tensor(hand_qpos).to(device)[None]
        joint_idxs = [joint.dof_idx_local for joint in hand.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        hand.set_dofs_position(position=hand_qpos, dofs_idx_local=joint_idxs)
    return 

def set_init_object_states(obj, obj_pos, obj_quat, obj_arti, joint_only=False):
    """ demo_data: dict of shape (num_demo_step, k) """   
    print(f"Initial object states:")
    print(f"  obj_pos shape: {obj_pos.shape}, first: {obj_pos[0]}")
    print(f"  obj_quat shape: {obj_quat.shape}, first: {obj_quat[0]}")
    print(f"  obj_arti shape: {obj_arti.shape}, first: {obj_arti[0]}")
    num_demo_steps = obj_pos.shape[0]
    env_idxs = [i for i in range(obj.num_envs)]   
    if joint_only:
        obj.entity.set_dofs_position(position=obj_arti, dofs_idx_local=obj.dof_idxs, zero_velocity=True, envs_idx=env_idxs)
        return 
    obj.set_object_state(
        root_pos=obj_pos,
        root_quat=obj_quat,
        joint_qpos=obj_arti,
        env_idxs=env_idxs
    )

def get_obj_demo_tensors(demo_data, device=torch.device("cuda")):
    obj_pos = torch.tensor(demo_data['obj_pos'], device=device)
    obj_quat = torch.tensor(demo_data['obj_quat'], device=device)
    obj_arti = torch.tensor(demo_data['obj_arti'], device=device)[:, None] # shape (num_demo_steps, 1)
    return obj_pos, obj_quat, obj_arti

def create_scene(args, object_name, urdfs, demo_data):
    import genesis as gs
    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0,0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_joint_limit=True,
        ),
        show_viewer=args.vis,
        use_visualizer=(args.vis or args.save_video),
        show_FPS=False,
        vis_options = gs.options.VisOptions( 
            plane_reflection = True,
            ambient_light    = (0.4, 0.4, 0.4),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
            ]
        ),
        viewer_options=gs.options.ViewerOptions( 
            camera_pos=(2, 1.2, 2.1),
            camera_lookat=(0.0, -0.1, 1.1),
            camera_fov=25,
        ),
    )
    plane_urdf = 'urdf/plane/plane.urdf' # NOTE this is Genesis default plane
    if args.raytrace:
        scene_cfg['renderer'] = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=10.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 1.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
        plane_urdf = join(get_asset_path('plane'), 'plane_custom.urdf') # use custom plane with texture
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    
    # Add ground plane
    ground = scene.add_entity(gs.morphs.URDF(file=plane_urdf, fixed=True))
    
    cam = None 
    if args.save_video:
        if args.raytrace:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(1024, 1024), fov=20, GUI=False) 
        else:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(512, 512), fov=20, GUI=False)

    hand_entities = dict()
    for side, urdf_path in urdfs.items():
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path, 
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=True, # has to be true for get_AABB to work
                # collision=False, 
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.8
                ),
            # surface=gs.surfaces.Smooth(color=(0, 0, 0.8, 0.5)),            
        )
        hand_entities[side] = hand 
    obj = None
    # if args.show_object:
    cardbox_size = (0.2,0.2,0.1)
    cardboard_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, -0.08, 0.90),
            size=cardbox_size,
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            roughness=0.1, 
        ),
    ) 

    obj_cfg = get_arctic_object_cfg(object_name, convexify=False)
    obj_cfg['fixed'] = False
    obj_cfg['disable_collision'] = False
    obj_cfg['color'] = (1.0, 0.423, 0.039, 0.3)
    obj = ArticulatedObject(obj_cfg, device=device, scene=scene, num_envs=1)
        
    return scene, hand_entities, obj, cam

def main(args):
    num_envs = 1
    
    urdfs = dict()
    robot_dir = get_asset_path(args.hand)
    config_path = join(robot_dir, "retarget_config.yaml") 
    for side in ['left', 'right']:
        config = yaml.safe_load(open(config_path, 'r'))
        urdf_path = config[side]['urdf_path']
        urdfs[side] = join(robot_dir, urdf_path)  

    # get path to retargeted demonstration and load demo_data
    assert os.path.exists(args.load_fname), f"load_fname={args.load_fname} does not exist"
    subject_name = args.load_fname.split("/")[-2]
    hand_name = args.hand if 'hand' in args.hand else f"{args.hand}_hand"
    retarget_type = 'position' if hand_name == 'shadow_hand' else 'vector'
    
    # Construct path to saved .pt file from parallel_retarget
    traj_name = args.load_fname.split("/")[-1].replace(".npy", "")
    save_fname = f"dexmachina/assets/retargeted/{hand_name}/{subject_name}/{traj_name}_{retarget_type}_para.pt"
    
    if os.path.exists(save_fname):
        loaded_data = torch.load(save_fname, weights_only=False)
        demo_data = loaded_data.get('demo_data', {})
        print(f"Loaded demo_data from {save_fname}")
    else:
        raise FileNotFoundError(f"Retargeted file not found: {save_fname}. Run parallel_retarget.py first with --save flag.")
    
    # create the manipulation scene
    scene, hand_entities, obj, cam = create_scene(args, args.obj_name, urdfs, demo_data)

    device = torch.device('cuda:0')

    # Load retargeter results
    retarget_fname = join(
        f"dexmachina/assets/retargeter_results/{hand_name}/{subject_name}", 
        args.load_fname.split("/")[-1].replace(".npy", f"_{retarget_type}.npy")
    )
    retargeter_results = np.load(retarget_fname, allow_pickle=True).item()

    # Build scene
    scene.build(n_envs=num_envs, env_spacing=(2.0, 2.0))
    scene.reset()
    
    # Now set initial object states after scene is built (use first frame only)
    obj_pos, obj_quat, obj_arti = get_obj_demo_tensors(demo_data, device=device)
    # Only use first frame since we have 1 environment
    obj_pos_init = obj_pos[0:1]  # shape (1, 3)
    obj_quat_init = obj_quat[0:1]  # shape (1, 4)
    obj_arti_init = obj_arti[0:1]  # shape (1, 1)
    set_init_object_states(obj, obj_pos_init, obj_quat_init, obj_arti_init, joint_only=False)
    
    # Setup video directory if recording
    video_dir = None
    if args.save_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save video frames to {video_dir}")
    
    # Get number of frames
    total_frames = retargeter_results['left']["hand_qpos"].shape[0]
    
    # Parse frames argument
    if args.frames:
        parts = args.frames.split('-')
        start_frame = int(parts[0])
        end_frame = int(parts[1])
    else:
        start_frame = 0
        end_frame = total_frames
    
    num_steps = end_frame - start_frame
    print(f"Total frames available: {total_frames}")
    print(f"Playback range: {start_frame} to {end_frame} ({num_steps} frames)")
    
    # Main playback loop
    step = start_frame
    frame_delay = 1.0 / args.playback_fps
    last_frame_time = time.time()
    
    print("Starting playback... Press Ctrl+C to stop")
    
    try:
        while True:
            if step >= end_frame:
                step = start_frame
                scene.reset()
                if obj:
                    obj.post_scene_build_setup()
                    # Now set initial object states after scene is built (use first frame only)
                    obj_pos, obj_quat, obj_arti = get_obj_demo_tensors(demo_data, device=device)
                    # Only use first frame since we have 1 environment
                    obj_pos_init = obj_pos[0:1]  # shape (1, 3)
                    obj_quat_init = obj_quat[0:1]  # shape (1, 4)
                    obj_arti_init = obj_arti[0:1]  # shape (1, 1)
                    set_init_object_states(obj, obj_pos_init, obj_quat_init, obj_arti_init, joint_only=False)
                print(f"Looping back to frame {start_frame}...")
            
            set_entities_to_step(hand_entities, retargeter_results, step, device)
            
            # Print hand z position for first step in range
            if step == start_frame:
                for side, hand in hand_entities.items():
                    hand_pos = hand.get_links_pos()[0, 0, :]  # Get root link position (x, y, z) for env 0
                    print(f"[Step {step}] {side.capitalize()} hand root z position: {hand_pos[2]:.4f}")
            
            scene.step()
            
            # Render and save camera frame if recording
            if args.save_video and cam:
                render_result = cam.render()
                # Handle variable number of returns from cam.render()
                if isinstance(render_result, tuple):
                    rgb = render_result[0]
                else:
                    rgb = render_result
                
                # Convert to numpy if needed
                if hasattr(rgb, 'cpu'):
                    rgb_np = rgb[0].cpu().numpy().astype(np.uint8)
                else:
                    rgb_np = rgb[0].astype(np.uint8) if len(rgb.shape) > 3 else rgb.astype(np.uint8)
                
                # OpenCV expects BGR, convert from RGB
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                frame_path = video_dir / f"frame_{step:06d}.png"
                cv2.imwrite(str(frame_path), bgr_np)
            
            step += 1
            
            # Frame rate control
            elapsed = time.time() - last_frame_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()
    
    except KeyboardInterrupt:
        print("\nPlayback stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback retargeted hand animation")
    parser.add_argument('--load_fname', '-lf', type=str, default='dexmachina/assets/contact_retarget/allegro_hand/s01/ketchup_use_01.npy')
    parser.add_argument('--hand', type=str, default='allegro_hand', help='Hand model name')
    parser.add_argument("--obj_name", type=str, help="Name of the object being manipulated")
    parser.add_argument('--retarget_name', type=str, default='para', help='Retargeting save name')
    parser.add_argument('--vis', action='store_true', help='Show viewer')
    parser.add_argument('--playback_fps', type=float, default=30, help='Playback FPS')
    parser.add_argument('--save_video', action='store_true', help='Save camera frames to video')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save video frames')
    parser.add_argument('--raytrace', action='store_true', help='Whether to use raytracer')
    parser.add_argument('--frames', type=str, default=None, help='Frame range for playback (e.g., "30-130")')
    args = parser.parse_args()

    
    main(args)
