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
import pickle


from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data 
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg 
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.constructors import get_common_argparser
from dexmachina.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

RETARGETER_RESULTS_DIR=get_asset_path("retargeter_results")


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
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_joint_limit=True,
        ),
        show_viewer=args.vis,
        use_visualizer=(args.vis or args.record_video),
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
    if args.record_video:
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
    cardbox_size = (0.3,0.3,0.1)
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

    obj_cfg = get_arctic_object_cfg(object_name, convexify=True)
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

    # load_fname is directly the retargeted file:
    #   .pt  → dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_para.pt
    #   .npy → dexmachina/assets/retargeter_results/allegro_hand/s01/ketchup_use_01_vector.npy
    # if not os.path.isabs(args.load_fname):
    #     args.load_fname = os.path.join(get_asset_path(".."), args.load_fname)
    assert os.path.exists(args.load_fname), f"load_fname={args.load_fname} does not exist"
    hand_name = args.hand if 'hand' in args.hand else f"{args.hand}_hand"
    retarget_type = 'position' if hand_name == 'shadow_hand' else 'vector'

    subject_name = args.load_fname.split("/")[-2]  # e.g. s01
    fname_stem = os.path.basename(args.load_fname)  # e.g. ketchup_use_01_vector_para.pt

    is_ik = args.load_fname.endswith(".npy")

    # parse traj_name: strip retarget_type suffix (and save_name for .pt)
    # e.g. ketchup_use_01_vector_para → ketchup_use_01
    traj_name = fname_stem.replace(f"_{retarget_type}_para.pt", "").replace(f"_{retarget_type}.npy", "")
    use_clip_str = traj_name.split("_use_")[-1] if "_use_" in traj_name else "01"
    obj_name = traj_name.split("_use_")[0] if "_use_" in traj_name else traj_name

    def print_struct(obj, indent=0):
        struct_lines = []
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    struct_lines.append(f"{prefix}{k}: dict")
                    struct_lines.extend(print_struct(v, indent + 1))
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    struct_lines.append(f"{prefix}{k}: {type(v).__name__} shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, list):
                    struct_lines.append(f"{prefix}{k}: list len={len(v)}")
                else:
                    struct_lines.append(f"{prefix}{k}: {type(v).__name__} = {v}")
        else:
            struct_lines.append(f"{prefix}{type(obj).__name__}")
        return struct_lines

    if is_ik:
        retargeter_results = np.load(args.load_fname, allow_pickle=True).item()
        # struct_lines = [f"Source: .npy (pure kinematic IK)"] + print_struct(retargeter_results)
        # struct_out = args.load_fname.replace(".npy", "_structure.txt")
        # with open(struct_out, "w") as f:
        #     f.write("\n".join(struct_lines) + "\n")
        #     f.write("left hand qpos: \n")
        #     for i in range(600):
        #         line = retargeter_results['left']['hand_qpos'][i]
        #         f.write(" ".join(str(x) for x in line) + "\n")
        #     for i in range(600):
        #         line = retargeter_results['right']['hand_qpos'][i]
        #         f.write(" ".join(str(x) for x in line) + "\n")
        # print(f"Saved structure to {struct_out}")
        num_frames = retargeter_results['left']['hand_qpos'].shape[0]
        demo_data = get_demo_data(
            obj_name=obj_name,
            hand_name=hand_name,
            frame_start=0,
            frame_end=num_frames,
            use_clip=use_clip_str,
            subject_name=subject_name,
        )
    else:
        loaded_data = torch.load(args.load_fname, weights_only=False)
        # struct_lines = [f"Source: .pt (physics-settled)"] + print_struct(loaded_data)
        # struct_out = args.load_fname.replace(".pt", "_structure.txt")
        # with open(struct_out, "w") as f:
        #     f.write("\n".join(struct_lines) + "\n")
        #     f.write("left hand qpos: \n")
        #     for i in range(600):
        #         line = loaded_data['retargeter_results']['left']['hand_qpos'][i]
        #         f.write(" ".join(str(x) for x in line) + "\n")
        #     for i in range(600):
        #         line = loaded_data['retargeter_results']['right']['hand_qpos'][i]
        #         f.write(" ".join(str(x) for x in line) + "\n")
        # print(f"Saved structure to {struct_out}")
        demo_data = loaded_data.get('demo_data', {})
        # Build retargeter_results from physics-settled retarget_data (joint_qpos dict → flat array)
        pt_retarget_data = loaded_data.get('retarget_data', {})
        retargeter_results = {}
        for side in ['left', 'right']:
            side_data = pt_retarget_data.get(side, {})
            joint_qpos_dict = side_data.get('joint_qpos', {})
            if joint_qpos_dict:
                # stack named joints into (T, num_dofs) array, converting tensors to numpy
                arrays = []
                for v in joint_qpos_dict.values():
                    arr = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
                    arrays.append(arr)
                hand_qpos = np.stack(arrays, axis=1)  # (T, num_dofs)
                retargeter_results[side] = {
                    'hand_qpos': hand_qpos,
                    'actuated_dof_names': list(joint_qpos_dict.keys()),
                }
            else:
                retargeter_results[side] = loaded_data.get('retargeter_results', {}).get(side, {})
    print(f"Loaded from {args.load_fname} ({'IK npy' if is_ik else 'physics pt'})")

    # create the manipulation scene
    scene, hand_entities, obj, cam = create_scene(args, obj_name, urdfs, demo_data)

    device = torch.device('cuda:0')

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
    
    # Setup video collection if recording
    if args.record_video:
        print(f"Will collect frames for MP4 video output")
    
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
    
    # Collect playback trajectory
    playback_trajectory = {
        'obj_state': [],     # actual object state during playback
        'object_demo_state': [],    # ground truth object demonstration state
        'hand_qpos': {side: [] for side in hand_entities},  # hand joint positions at each step
    }
    
    # Initialize frame collection for video
    frames = []
    
    # Get demo data tensors
    obj_pos, obj_quat, obj_arti = get_obj_demo_tensors(demo_data, device=device)
    
    print("Starting playback... Press Ctrl+C to stop")
    
    try:
        while True:
            # if step >= end_frame:
                # step = start_frame
                # scene.reset()
                # if obj:
                #     obj.post_scene_build_setup()
                #     # Now set initial object states after scene is built (use first frame only)
                #     obj_pos, obj_quat, obj_arti = get_obj_demo_tensors(demo_data, device=device)
                #     # Only use first frame since we have 1 environment
                #     obj_pos_init = obj_pos[0:1]  # shape (1, 3)
                #     obj_quat_init = obj_quat[0:1]  # shape (1, 4)
                #     obj_arti_init = obj_arti[0:1]  # shape (1, 1)
                #     set_init_object_states(obj, obj_pos_init, obj_quat_init, obj_arti_init, joint_only=False)
                # print(f"Looping back to frame {start_frame}...")
            
            set_entities_to_step(hand_entities, retargeter_results, step, device)
            
            # Collect object state at this step
            obj.update_value_buffers()
            obj_state = np.concatenate([obj.root_pos[0].cpu().numpy(), obj.root_quat[0].cpu().numpy(), obj.dof_pos[0].cpu().numpy()])
            playback_trajectory['obj_state'].append(obj_state)
            
            # Get object demo state at this step (ground truth)
            demo_state = np.concatenate([obj_pos[step].cpu().numpy(), obj_quat[step].cpu().numpy(), obj_arti[step].cpu().numpy().flatten()])
            playback_trajectory['object_demo_state'].append(demo_state)

            # Record hand joint positions
            for side, hand in hand_entities.items():
                joint_idxs = [joint.dof_idx_local for joint in hand.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
                qpos = hand.get_dofs_position(joint_idxs)[0].cpu().numpy()
                playback_trajectory['hand_qpos'][side].append(qpos)

            # Print hand z position for first step in range
            if step == start_frame:
                for side, hand in hand_entities.items():
                    hand_pos = hand.get_links_pos()[0, 0, :]  # Get root link position (x, y, z) for env 0
                    print(f"[Step {step}] {side.capitalize()} hand root z position: {hand_pos[2]:.4f}")
            
            scene.step()
            
            # Render and collect camera frame if recording
            if args.record_video and cam:
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
                
                # Collect frame (keep in RGB, will convert to BGR when saving)
                frames.append(rgb_np)
            
            step += 1
            if step == end_frame:
                break
            
            # Frame rate control
            elapsed = time.time() - last_frame_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()
    
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    
    # Save video from collected frames
    if args.record_video and len(frames) > 0:
        import cv2
        fps = int(1 / (1/60) / 2)  # Approximate frame rate (half of simulation step rate)
        frame_height, frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video directory if needed
        video_dir = Path(os.path.join(args.video_dir, args.hand, subject_name))
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video file path
        if is_ik:
            video_fname = video_dir/f"{args.hand}_{subject_name}_u{use_clip_str}_pure_IK_{obj_name}_{start_frame}_{end_frame}.mp4"
        else:    
            video_fname = video_dir / f"{args.hand}_{subject_name}_u{use_clip_str}_IK_and_smoothing_{obj_name}_{start_frame}_{end_frame}.mp4"
        out = cv2.VideoWriter(str(video_fname), fourcc, fps, (frame_width, frame_height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"Saved video to {video_fname}")
    elif args.record_video:
        print("No frames recorded, skipping video save")
    
    # Save playback trajectory to .npy file
    playback_trajectory = {
        'obj_state': np.array(playback_trajectory['obj_state']),     # shape (T, 8)
        'object_demo_state': np.array(playback_trajectory['object_demo_state']),   # shape (T, 8)
        'hand_qpos': {side: np.array(qpos_list) for side, qpos_list in playback_trajectory['hand_qpos'].items()},  # shape (T, num_dofs) per side
    }
    
    # Create output directory structure: kinematic_playback/{hand_name}/{subject_name}/
    output_base_dir = "/home/jeffrey/Documents/Manipulation/Genesis/dexmachina/dexmachina/assets/kinematic_playback"
    hand_folder = os.path.join(output_base_dir, args.hand, subject_name)
    os.makedirs(hand_folder, exist_ok=True)
    
    src_tag = "npy" if is_ik else "pt"
    output_fname = os.path.join(hand_folder, f"playback_{obj_name}_use_{use_clip_str}_{start_frame}_{end_frame}_{src_tag}.pkl")
    with open(output_fname, "wb") as f:
        pickle.dump(playback_trajectory, f)
    print(f"\nSaved playback trajectory to {output_fname}")
    print(f"  obj_state shape: {playback_trajectory['obj_state'].shape}")
    print(f"  demo_state shape: {playback_trajectory['object_demo_state'].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback retargeted hand animation")
    parser.add_argument('--load_fname', '-lf', required=True, help="Trajectory to playback, either a .pt or .npy file")
    parser.add_argument('--hand', type=str, default='allegro_hand', help='Hand model name')
    parser.add_argument('--retarget_name', type=str, default='para', help='Retargeting save name')
    parser.add_argument('--vis', action='store_true', help='Show viewer')
    parser.add_argument('--playback_fps', type=float, default=30, help='Playback FPS')
    parser.add_argument('--record_video', action='store_true', help='Save camera frames to video')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save video frames')
    parser.add_argument('--raytrace', action='store_true', help='Whether to use raytracer')
    parser.add_argument('--frames', type=str, default=None, help='Frame range for playback (e.g., "30-130")')
    args = parser.parse_args()

    
    main(args)
