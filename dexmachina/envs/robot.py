import os  
import sys 
from os.path import join 
import numpy as np
import torch
import genesis as gs
from collections import defaultdict
from typing import Dict
import importlib 
from dexmachina.asset_utils import get_urdf_path

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower + 1e-5)

@torch.jit.script
def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # [w, x, y, z] convention
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

@torch.jit.script
def perturb_quat(quat: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Apply isotropic rotation noise via exponential map. quat is [w, x, y, z]."""
    omega = torch.randn(quat.shape[0], 3, device=quat.device) * noise_std
    theta = omega.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    half_theta = theta / 2.0
    q_noise = torch.cat([torch.cos(half_theta), torch.sin(half_theta) * axis], dim=-1)
    return quat_multiply(q_noise, quat)

def get_hand_specific_cfg(name="inspire_hand"):
    hand_prefix = name.replace("_hand", "") # xhand still stays xhand
    try:
        module_name = f"dexmachina.envs.hand_cfgs.{hand_prefix}"
        hand_cfg_module = importlib.import_module(module_name)
        # Assume the hand cfgs all ends with *_CFGs inside the module
        cfg_attr = next(
            attr for attr in dir(hand_cfg_module)
            if attr.upper().endswith("_CFGS")
        )
        hand_cfgs = getattr(hand_cfg_module, cfg_attr)
        assert isinstance(hand_cfgs, dict), f"{cfg_attr} should be a dict"
        return hand_cfgs

    except (ImportError, StopIteration, AttributeError, KeyError) as e:
        raise ValueError(f"Invalid hand name or configuration: {name} - {e}")

def get_default_robot_cfg(name="inspire_hand", side="left", wrist_only=True, group_collisions=False):
    robot_cfg = {
        "name": f"{name}_{side}",
        "gravity_compensation": 0.8,
        "actuators": {
            "all": dict(
                joint_exprs=['.*'],
                kp=100.0,
                kv=10.0,
                force_range=100.0,
            )
        },
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_moving_avg": 1.0,
        "action_mode": "residual",  
        'wrist_only': False,
        "kpt_link_names": [], # NOTE this can only contact non-collision links, the collision links would be added automatically
        "collect_data": False, # if True, save kpt pos and joint qpos after each episode
        "use_saved_targets": False, # if True, use saved targets for residual action, which is not always achievable
        "hybrid_scales": (0.04, 0.5),
        "res_cap": False, # if True, use hybrid_scales to also cap residual wrist actions
        "show_keypoints": False,
        "visualization": True,
        "randomize_observations": False
    }
    assert side in ["left", "right"], f"Invalid side {side}"
    # NOTE robot_cfg['wrist_link_name'] should match retargeting results 
    
    hand_cfg = get_hand_specific_cfg(name=name)
    assert hand_cfg is not None, f"Hand config for {name} not found"
    assert hand_cfg.get(side, None) is not None, f"Hand config for {name} {side} not found"
    robot_cfg.update(hand_cfg[side].copy())
    
    if group_collisions:
        assert robot_cfg.get("collision_groups", None) is not None, "Need to set collision_groups"
    
    urdf_path = robot_cfg["urdf_path"] 
    assert os.path.exists(urdf_path), f"{urdf_path} does not exist"  
    return robot_cfg


class BaseRobot:
    def __init__(
        self, 
        robot_cfg, 
        device, 
        scene, 
        num_envs,
        obs_scale={'dof_vel': 0.1, 'root_ang_vel': 0.1, 'contact_norm': 0.1, 'kpt_vel': 0.1,
                   'goal_wrist_vel': 0.1, 'goal_wrist_ang_vel': 0.1, 'goal_kpt_vel': 0.1},
        retarget_data=dict(),
        visualize_contact=False,
        is_eval=False, 
        disable_collision=False,
    ):

        """
        NOTE need to set fixed=True so the root link is fixed and the 6 wrist joints are actuated
        """

        self.name = robot_cfg["name"]
        self.cfg = robot_cfg 
        self.device = device
        self.initialized = False
        self.entity = None 
        self.obs_scale = obs_scale # a dict of scales for each observation
         
        self.init_pos = torch.tensor(
            robot_cfg["base_init_pos"], dtype=torch.float32, device=self.device
        )
        
        print("INITIAL POSITION FOR ROBOT: ", self.init_pos)
        self.init_quat = torch.tensor(
            robot_cfg["base_init_quat"], dtype=torch.float32, device=self.device
        )
        print("INITIAL ROTATION FOR ROBOT: ", self.init_quat)

        self.action_mode = robot_cfg.get("action_mode", "residual")
        assert self.action_mode in ["residual", "absolute", "relative", "hybrid", "kinematic"], f"Invalid action mode {self.action_mode}"
        self.hybrid_scales = robot_cfg.get("hybrid_scales", (0.04, 0.5))
        self.res_cap = robot_cfg.get("res_cap", False)
        self.num_envs = num_envs
        self.scene = scene
        assert not self.initialized, "Robot already initialized"
        self.entity = scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg["urdf_path"],
                pos=self.init_pos.cpu().numpy(),
                quat=self.init_quat.cpu().numpy(),
                convexify=True,
                fixed=True,
                merge_fixed_links=False, # NOTE: need to keep this to track the fingertip links
                recompute_inertia=True,
                collision=(not disable_collision), 
                visualization=self.cfg.get("visualization", True),
            ),
            material=gs.materials.Rigid(gravity_compensation=robot_cfg["gravity_compensation"]),
            visualize_contact=visualize_contact,
        )
        
        # mimic joints are controlled together as a group, one action input controls multiple physical joints with optional scaling factors
        all_joints = self.entity.joints  
        self.actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        self.actuated_dof_names = [joint.name for joint in self.actuated_joints]
        print("Actuated joints: ", self.actuated_dof_names)
        self.actuated_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints]
        self.ndof = len(self.actuated_joints) # NOTE this is NOT necessarily action dim due to mimic joints
        self.wrist_only = robot_cfg.get("wrist_only", False)
        self.wrist_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm' in joint.name]
        self.finger_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm' not in joint.name]
        assert len(self.wrist_dof_idxs) == 6, f"Found {len(self.wrist_dof_idxs)} wrist dofs"
        # setup joint limits 
        dof_limits = []
        custom_dof_limits = robot_cfg.get("joint_limits", dict()).copy()
        for joint in self.actuated_joints:
            if joint.name in custom_dof_limits:
                limit = custom_dof_limits.pop(joint.name)
                dof_limits.append(
                    np.array(limit)[None,:]# shape (1, 2)
                    )
            else:
                dof_limits.append(joint.dofs_limit)
        assert len(custom_dof_limits) == 0, f"Custom limits {custom_dof_limits} not used"
        dof_limits = np.concatenate(dof_limits) # shape (ndof, 2) # NOTE that joint.dofs_limit is already shape (1, 2)
        # check the joint range is not less than zero 
        assert (dof_limits[:, 1] - dof_limits[:, 0] >= 0).all(), f"Joint limits have negative range: {dof_limits}"
        self.dof_limits = torch.tensor(dof_limits, dtype=torch.float32, device=self.device) # shape (num_joints, 2)
        self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (num_joints,)
        
        self.action_moving_avg = self.cfg["action_moving_avg"]  

        # setup mimic joint mapping, i.e. map the same action input index to multiple joints
        self.mimic_joint_map = self.cfg.get("mimic_joint_map", dict())
        self.setup_action_mapping(self.actuated_joints, self.mimic_joint_map)
        print("mimic joints: ", self.mimic_joint_map)

        # setup default joint qpos based on mimic joint mapping
        qposes = np.concatenate([joint.init_qpos for joint in self.actuated_joints]) # shape (ndof,) 
        default_qpos = robot_cfg.get("default_qpos", None) 
        if default_qpos is not None:
            assert len(default_qpos) == len(qposes), f"len(default_qpos)={len(default_qpos)} != {self.ndof}"
            qposes[:] = np.array(default_qpos)
        qposes = torch.tensor(qposes, dtype=torch.float32, device=self.device)
        # repeat for each env
        
        self.is_eval = is_eval
        
        self.init_qpos = qposes.unsqueeze(0).repeat(self.num_envs, 1) 
        if 'init_qpos' in retarget_data and not robot_cfg.get('multi_demo', False): 
            print("Using custom init qpos")
            self.set_custom_init_qpos(retarget_data['init_qpos'])
        elif self.is_eval:
            self.set_custom_init_qpos(retarget_data['init_qpos'])
            print("evaluating, setting custom init qpos")
        else:
            print("Skip settting custom init qpos, using multiple demos")
      
        
        # if self.is_eval and self.num_envs > 1:
        #     print('WARNING: setting robot.is_eval to True, setting the init_qpos for last env to 0s, as this is the one that the reference demo is using')
        #     self.init_qpos[-1, :] = 0.0
        
        # only take collision geoms for contact forces!
        coll_idxs_local, coll_idxs_global = [], []
        coll_link_names = []
        for i, link in enumerate(self.entity.links):
            if len(link.geoms) > 0:
                coll_idxs_local.append(i)
                coll_idxs_global.append(link.idx) # NOTE this is global!!
                coll_link_names.append(link.name)
        self.coll_idxs_local = coll_idxs_local # NOTE this is local!!
        self.coll_idxs_global = coll_idxs_global
        self.n_coll_links = len(coll_idxs_local)
        self.coll_link_names = coll_link_names
        
        # setup link names to track
        link_names = self.cfg.get("kpt_link_names", [])
        link_names += self.coll_link_names
        
        if 'kpts_data' in retarget_data:
            print(f"Overwrite kpt_link_names with saved retarget data")
            raw_kpt_names = retarget_data['kpts_data']['kpt_names']
            # Deduplicate while preserving order: keep first occurrence
            seen = set()
            link_names = []
            for name in raw_kpt_names:
                if name not in seen:
                    link_names.append(name)
                    seen.add(name)
            print(f"[{self.name}] Deduped keypoint names: {len(raw_kpt_names)} → {len(link_names)} unique")
        
        print("Side: ", self.name)           # get the hand side
        print("LINK NAMES: ", link_names)
        self.set_kpt_links(link_names)

        self.kpt_markers = []
        if self.cfg.get("show_keypoints", False):
            KPT_COLOR = [x/255.0 for x in (229, 152, 155)] + [1.0] # pink
            KPT_RADIUS = 0.007
            for _ in range(self.n_kpts):
                marker = self.scene.add_entity(
                    gs.morphs.Sphere(
                        radius=KPT_RADIUS, fixed=False, collision=False, # no collision
                        ),
                    surface=gs.surfaces.Rough(color=KPT_COLOR),
                )
                self.kpt_markers.append(marker)


        self.wrist_link_name = self.cfg.get("wrist_link_name", "base_link")
        wrist_link_idxs = [i for i, link in enumerate(self.entity.links) if link.name == self.wrist_link_name]
        assert len(wrist_link_idxs) == 1, f"Found {len(wrist_link_idxs)} wrist links"
        self.wrist_link_idx = wrist_link_idxs[0]

        self.residual_qpos = None       # stores a trajectory of joint positions over time (T, ndof)
        self.residual_num_frames = None  # number of frames in single demo trajectory
      
        self.env_demo_idx = None  # which demo each environment follows
        if 'residual_qpos' in retarget_data:
            qpos_targets = None
            if self.cfg.get("use_saved_targets", False):
                qpos_targets = retarget_data.get('qpos_targets', None)          # qpos_targets are the PD targets, which are coming from the physics simulator
                assert qpos_targets is not None, "Need to set qpos_targets for residual action mode"
            self.set_residual_qpos(
                num_frames=retarget_data['num_frames'],
                residual_qpos_dict=retarget_data['residual_qpos'],          # residual_qpos is the kinematic pose, the inverse kinematics retargeting
                qpos_targets_dict=qpos_targets,
            )
        if self.action_mode == "relative":
            assert self.residual_qpos is not None, "Need to set residual qpos for relative action mode"
            self.set_relative_step_size(self.residual_qpos)

        if 'limits' in retarget_data:
            print("custom limits")
            self.set_custom_joint_limits(retarget_data['limits'])
         
        self.n_links = self.entity.n_links 
        
        self.initialized = True
        self.initialize_value_buffers()

        
        self.obs_dim, self.obs_dim_info = self.compute_obs_dim()
        
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self.collect_data = self.cfg.get("collect_data", False)
        if self.collect_data:
            print(f"Collecting data for {self.name}")
        self.episode_data = defaultdict(list) 

        # randomize observations
        self.randomize_observations = self.cfg.get("randomize_observations", False)
        if self.randomize_observations:
            self.max_joint_angle_noise = 0.06 # radians ~ 5 degrees
            self.max_joint_pos_noise = 0.0005 # meters   ~ mocap is sub milimeter error
            self.max_joint_vel_noise = 0.002 # meters/second        # at assumed 1/30 fps mocap

        
        ### MULTI DEMO DATA STRUCTURES
        self.all_residual_qpos = None       # stores all demo trajectories of joint positions over time (ndemos, T, ndof)
        self.all_residual_num_frames = None  # list of frame counts for each demo (the same)
        self.all_init_qpos = torch.zeros(robot_cfg.get("num_demos", 1), qposes.shape[0], device=self.device, dtype=torch.float32)  # (num_demos, ndof)
        # self.all_dof_limits = torch.tensor(dof_limits, dtype=torch.float32, device=self.device)      # stores all joint limits for each demo   shape (ndemos, num_joints, 2), should be the same per demo
        # self.all_dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (ndemos, num_joints,)
        # torch.set_printoptions(threshold=10000, linewidth=200, profile='full')
        # print("\n" + "="*80)
        # print("Robot:", self.name)
        # if self.residual_qpos is not None:
        #     print(f"Residual QPos Shape: {self.residual_qpos.shape} (T={self.residual_num_frames}, ndof={self.ndof})")
        #     print(f"Residual QPos (all elements):\n{self.residual_qpos}")
        # else:
        #     print("Residual QPos: None")
        # print("="*80 + "\n")
        
    def get_collision_groups(self):
        return self.cfg.get("collision_groups", dict())
    
    def set_kpt_links(self, kpt_link_names):
        self.kpt_link_names, self.kpt_link_idxs = [], []
        link_names = [link.name for link in self.entity.links]
        for name in kpt_link_names:
            if name in link_names:
                self.kpt_link_names.append(name)
                self.kpt_link_idxs.append(link_names.index(name))
            else:
                print(f"WARNING - Link {name} not found in the URDF")
        print("Keypoint link names: ", self.kpt_link_names)
        self.n_kpts = len(self.kpt_link_names)
        return 
    
    def set_custom_init_qpos(self, joint_qpos: Dict, demo_idx=None):                # takes the initial 
        curr_qpos = self.init_qpos.clone() # shape (num_envs, ndof)
        for jname, qpos in joint_qpos.items():
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                continue
            idx = self.actuated_dof_names.index(jname) 
            if not isinstance(qpos, torch.Tensor):
                qpos = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            curr_qpos[:, idx] = qpos.clone().to(self.device)
        if demo_idx is not None:
            self.all_init_qpos[demo_idx] = curr_qpos[0]     # set_custom_init_qpos sets for all the enviornments 
            # print("SET ALL INIT QPOS SUCCESSFULLY")
            # self.all_curr_targets[demo_idx] = curr_qpos.clone()       # this is set at reset_idx
        else:
            self.init_qpos = curr_qpos 
            self.curr_targets = curr_qpos.clone()        
    
    def set_residual_qpos(self, num_frames, residual_qpos_dict: Dict, qpos_targets_dict=None): # v.shape is (num_frames, 1)
        # assert self.initialized, "Robot not initialized" 
        # need to set shape (num_frames, ndof), don't repeat for the env dim
        # init_qpos is shape (num_envs, ndof)!!
        residual_qpos = self.init_qpos[0].clone().unsqueeze(0).repeat(num_frames, 1)     # (num_frames, ndof)   # initializes a list of poses that are just the repeated of the first pose
        qpos_dict = residual_qpos_dict      # kinematically retargeted joint positoins
        if qpos_targets_dict is not None:
            qpos_dict = qpos_targets_dict       # joint controller target trajectories
        touched_dof_idxs = set()
        for jname, qpos in qpos_dict.items():
            assert qpos.shape[0] == num_frames, f"qpos.shape[0]={qpos.shape[0]} != {num_frames}"
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                breakpoint()
            dof_idx = self.actuated_dof_names.index(jname)
            if isinstance(qpos, np.ndarray):
                residual_qpos[:, dof_idx] = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            else:
                residual_qpos[:, dof_idx] = qpos.to(self.device)
            touched_dof_idxs.add(dof_idx)
        untouched = [self.actuated_dof_names[i] for i in range(self.ndof) if i not in touched_dof_idxs]
        if untouched:
            print(f"[set_residual_qpos] {len(untouched)} joints frozen at init_qpos: {untouched}")
        self.residual_qpos = residual_qpos
        self.residual_num_frames = num_frames

    def set_all_residual_qpos(self, all_retarget_data, side):
        """Pre-load all demos' residual qpos as (num_demos, T, ndof) for per-env switching.
        CRITICAL: Must maintain 1-to-1 indexing with demo order (NO FILTERING).
        
        Args:
            all_retarget_data: List of retarget data dicts for all demos
            side: 'left' or 'right' hand
        """
        # Step 1: Collect trajectory data from each demo
        per_demo_qpos = []          # List of (T, ndof) tensors for each demo
        per_demo_lengths = []       # List of trajectory lengths per demo
        per_demo_init_qpos = []     # List of init positions (frame 0) per demo
        per_demo_dof_limits = []    # List of (ndof, 2) joint limit tensors per demo
        first_valid_qpos = None     # Placeholder trajectory for missing demos
        first_valid_init_qpos = None  # Placeholder init position for missing demos
        first_valid_dof_limits = None  # Placeholder limits for missing demos
        missing_demos = []          # Track which demos have missing data

        original_init_qpos = self.init_qpos.clone()
        original_dof_limits = self.dof_limits.clone()

        for demo_idx, retarget_data in enumerate(all_retarget_data):
            side_data = retarget_data.get(side, {})

            # Handle missing trajectory data (e.g., retargeting failed for this demo)
            if 'residual_qpos' not in side_data:
                per_demo_qpos.append(None)
                per_demo_lengths.append(0)
                per_demo_init_qpos.append(None)  # Keep lists aligned with demo indices
                per_demo_dof_limits.append(None)
                missing_demos.append(demo_idx)
                continue

            # Reset to base limits/init, then apply this demo's overrides
            self.dof_limits = original_dof_limits.clone()
            if 'init_qpos' in side_data:
                self.set_custom_init_qpos(side_data['init_qpos'], demo_idx)
            if 'limits' in side_data:
                self.set_custom_joint_limits(side_data['limits'])

            # Load trajectory data for this demo
            qpos_targets = None
            if self.cfg.get("use_saved_targets", False):
                qpos_targets = side_data.get('qpos_targets', None)
            self.set_residual_qpos(
                num_frames=side_data['num_frames'],
                residual_qpos_dict=side_data['residual_qpos'],
                qpos_targets_dict=qpos_targets,
            )

            # Store trajectory and its metadata
            per_demo_qpos.append(self.residual_qpos.clone())
            per_demo_lengths.append(side_data['num_frames'])
            per_demo_init_qpos.append(self.residual_qpos[0].clone())  # First frame = init position
            # print("First element inside demo_init_qpos: ", self.residual_qpos[0].clone())
            per_demo_dof_limits.append(self.dof_limits.clone())     # the limits are the same across the same hand regardless of demo
            

        # Restore to pre-loop state
        self.init_qpos = original_init_qpos
        self.dof_limits = original_dof_limits
        self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0]

        # Early return if all demos are missing
        if all(q is None for q in per_demo_qpos):
            return

        self.all_residual_qpos = torch.stack(per_demo_qpos, dim=0)   # (num_demos, T, ndof)
        self.all_residual_num_frames = per_demo_lengths
        self.assign_env_demos_round_robin(len(per_demo_qpos))
        print("NO PADDING NEEDED")
        return
        
        # Step 3: Restore single-demo references for backward compatibility
        # These point to the first demo by default
        self.residual_qpos = self.all_residual_qpos[0]
        self.residual_num_frames = self.all_residual_num_frames[0]

        # Step 4: Log diagnostics
        if len(missing_demos) > 0:
            print(f"[{side.upper()} QPOS] WARNING: Demos {missing_demos} are MISSING trajectory data - padded with demo 0")
        print(f"[{side.upper()} QPOS] Loaded {len(per_demo_qpos)} demos, all_residual_qpos shape={self.all_residual_qpos.shape}, all_init_qpos shape={self.all_init_qpos.shape}, lengths={self.all_residual_num_frames}")

    def assign_env_demos_round_robin(self, num_demos):
        """Assign each env a fixed demo index via round-robin: env i -> demo i % num_demos."""
        self.env_demo_idx = torch.arange(self.num_envs, device=self.device) % num_demos

    def smooth_residual_qpos(self, joint_idxs=None):
        """
        return a smoothed version of residual_qpos along time dimension
        """
        from scipy.ndimage import gaussian_filter1d
        def gaussian_smooth(joint_values):
            joint_values = joint_values.cpu().numpy()
            smoothed = gaussian_filter1d(joint_values, sigma=1.0, axis=0)
            return torch.tensor(smoothed, dtype=torch.float32, device=self.device)            

        smoothed_qpos = self.residual_qpos.clone()
        if joint_idxs is None:
            joint_idxs = [i for i in range(smoothed_qpos.shape[1])]
        smoothed_qpos[:, joint_idxs] = gaussian_smooth(smoothed_qpos[:, joint_idxs])
        return smoothed_qpos
    
    def interpolate_residual_qpos(self, mutiplier=1.0):
        """ interpolates residual qpos by mutiplier times, so it goes from (T, njoints) -> (T*multiplier, njoints) """
        assert self.residual_qpos is not None, "Need to set residual qpos"
        from scipy.interpolate import interp1d
        new_qpos = []
        old_T = self.residual_num_frames
        new_T = int(self.residual_num_frames * mutiplier)
        for j in range(self.residual_qpos.shape[1]):
            qpos = self.residual_qpos[:, j].cpu().numpy() # shape (num_frames,)
            x = np.linspace(0, old_T-1, old_T)
            func = interp1d(x, qpos, kind='linear')
            qpos_int = func(np.linspace(0, old_T-1, new_T))
            new_qpos.append(qpos_int)
        new_qpos = torch.tensor(new_qpos, dtype=torch.float32, device=self.device).T # shape (new_T, ndof)
        assert new_qpos.shape[0] == new_T, f"new_qpos.shape[0]={new_qpos.shape[0]} != {new_T}"
        self.residual_qpos = new_qpos
        self.residual_num_frames = new_T

    def set_relative_step_size(self, residual_qpos):
        # take the max of absolute values of all the per-step joint value changes 
        deltas = torch.abs(residual_qpos[1:] - residual_qpos[:-1])
        max_delta = torch.max(deltas, dim=0).values
        self.relative_step_size = max_delta * 2.0 # shape (ndof,)

    def set_custom_joint_limits(self, joint_limits_dict: Dict):
        """ NOTE: skip finger joints """
        joint_limits = self.dof_limits.clone()
        for jname, limit in joint_limits_dict.items():
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                continue
            if 'forearm' not in jname:
                # only setting wrist joints
                continue 
            idx = self.actuated_dof_names.index(jname)
            joint_limits[idx] = torch.tensor(limit, dtype=torch.float32, device=self.device)
        
        # if demo_idx is not None:
            
        # self.dof_limits = joint_limits
        # self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (num_joints,) 
        
    def multi_demo_change_custom_joint_limits(self, joint_limits_dict: Dict):       # not needed as the joint limits are the same per hand
        joint_limits = self.dof_limits.clone()
        for jname, limit in joint_limits_dict.items():
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                continue
            if 'forearm' not in jname:
                # only setting wrist joints
                continue 
            idx = self.actuated_dof_names.index(jname)
            joint_limits[idx] = torch.tensor(limit, dtype=torch.float32, device=self.device)
        self.dof_limits = joint_limits
        self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (num_joints,) 

    def setup_action_mapping(self, actuated_joints, mimic_joint_map):
        """Setup so that action translation happens like actions[from_idx] controls joint targets"""
        # if no mimic joint, each action idx maps to a single joint
        if len(mimic_joint_map) == 0:
            self.action_dim = len(actuated_joints)
            self.action_from_idxs = [i for i in range(self.action_dim)] 
            self.joint_from_idxs = [i for i in range(self.action_dim)]
            self.joint_multipliers = torch.tensor(
                [1.0 for i in range(self.action_dim)],
                dtype=torch.float32,
                device=self.device, 
            )
            return
        # if mimic joint, each action idx maps to multiple joints
        action_dim = 0
        action_from_idxs = [] 
        joint_from_idxs = []
        joint_multipliers = []
        joint_name_to_action_idx = dict()
        joint_name_to_dof_idx = dict()
        for joint in actuated_joints: # assume the underlying model is fully actuated
            if joint.name not in mimic_joint_map:
                dof_idx = joint.dof_idx_local
                action_from_idxs.append(action_dim)
                joint_from_idxs.append(dof_idx) 
                joint_name_to_action_idx[joint.name] = action_dim
                joint_name_to_dof_idx[joint.name] = dof_idx
                joint_multipliers.append(1.0)
                action_dim += 1
        for joint in actuated_joints:
            if joint.name in mimic_joint_map:
                parent, ratio = mimic_joint_map[joint.name] 
                action_from_idxs.append(
                    joint_name_to_action_idx[parent]
                ) 
                joint_from_idxs.append(
                    joint_name_to_dof_idx[parent]
                )
                joint_multipliers.append(ratio)
        self.action_dim = action_dim
        self.action_from_idxs = action_from_idxs 
        self.joint_from_idxs = joint_from_idxs

        self.joint_multipliers = torch.tensor(
            joint_multipliers, dtype=torch.float32, device=self.device, 
        )
        return 
    
    def get_action_dim(self):
        assert self.initialized, "Robot not initialized"
        return self.action_dim
    
    def set_joint_gains(self, kp, kv, fr, joint_idxs=None):
        if joint_idxs is None:
            joint_idxs = self.actuated_dof_idxs
        num_joints = len(joint_idxs)
        batched_kp = torch.tensor(
            [kp]*num_joints, dtype=torch.float32, device=self.device
            ) 
        batched_kv = torch.tensor(
            [kv]*num_joints, dtype=torch.float32, device=self.device
            ) 
            
        self.entity.set_dofs_kp(
            batched_kp,
            joint_idxs,
        )
        self.entity.set_dofs_kv(
            batched_kv,
            joint_idxs,
        )
        fr = torch.tensor(
            [fr]*num_joints, dtype=torch.float32, device=self.device
            )
        self.entity.set_dofs_force_range(
            -1.0 * fr,
            fr,
            joint_idxs,
        )

    def set_inspire_gains(self):
        """ set the tuned values """
        kp, kv = 300.0, 30.0 
        # kp, kv = 5000.0, 50.0
        forearm_trans = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm_t' in joint.name]
        self.set_kp_kv_joints(kp, kv, forearm_trans)

        kp, kv = 300.0, 30.0
        # kp, kv = 5000.0, 50.0 
        forearm_rot = [joint.dof_idx_local for joint in self.actuated_joints if 'roll' in joint.name or 'pitch' in joint.name or 'yaw' in joint.name]
        self.set_kp_kv_joints(kp, kv, forearm_rot)

        kp, kv = 20.0, 2.0
        # kp, kv = 100.0, 10.0
        fingers = [joint.dof_idx_local for joint in self.actuated_joints if '_J1' in joint.name or '_J2' in joint.name or '_J3' in joint.name or '_J4' in joint.name]
        self.set_kp_kv_joints(kp, kv, fingers)
        
    def post_scene_build_setup(self):
        assert self.initialized, "Robot not initialized" 
        self.set_dof_gains_by_group(self.cfg["actuators"])
        
    def find_joints_in_group(self, joint_exprs):
        import re 
        joint_idxs = []
        for joint in self.actuated_joints:
            matched = False 
            jname = joint.name
            for expr in joint_exprs:
                if re.match(expr, jname):
                    matched = True
                    break
            if matched:
                joint_idxs.append(joint.dof_idx_local)
        if len(joint_idxs) == 0:
            print(f"No joints found for {joint_exprs}")
            breakpoint()
        return joint_idxs
    
    def set_dof_gains_by_group(self, actuator_cfgs):
        """ pass in a list of joint_groups """
        for group_name, group in actuator_cfgs.items():
            joint_idxs = self.find_joints_in_group(group["joint_exprs"])
            kp = group.get("kp", 200.0)
            kv = group.get("kv", 20.0)
            fr = group.get("force_range", 50.0)
            print("group_name: ", group_name)
            print("kp: ", kp)
            print("kv: ", kv)
            self.set_joint_gains(kp, kv, fr, joint_idxs)

    def initialize_value_buffers(self):
        assert self.initialized, "Robot not initialized"  
        self.dof_pos = self.init_qpos.clone()
        self.dof_vel = torch.zeros((self.num_envs, self.ndof), dtype=torch.float32, device=self.device)
        # repeat default init pos
        self.curr_targets = self.init_qpos.clone()
        self.prev_targets = self.init_qpos.clone()
        self.prev_dof_pos = self.init_qpos.clone()
        self.curr_res_qpos = self.init_qpos.clone()

        # track keypoint links
        self.kpt_pos = torch.zeros((self.num_envs, self.n_kpts, 3), dtype=torch.float32, device=self.device)
        self.kpt_vel = torch.zeros((self.num_envs, self.n_kpts, 3), dtype=torch.float32, device=self.device)
        self.wrist_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device) # 4 for quat, 3 for pos
        self.control_forces = torch.zeros((self.num_envs, self.ndof), dtype=torch.float32, device=self.device)
        # tracks envs that were teleported this step; their velocities should be zeroed
        self.just_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # goal state from demo (set each step via set_goal_state)
        self.goal_wrist_pose    = torch.zeros((self.num_envs, 7),           dtype=torch.float32, device=self.device)
        self.goal_kpt_pos       = torch.zeros((self.num_envs, self.n_kpts, 3), dtype=torch.float32, device=self.device)
        self.goal_wrist_vel     = torch.zeros((self.num_envs, 3),           dtype=torch.float32, device=self.device)
        self.goal_wrist_ang_vel = torch.zeros((self.num_envs, 3),           dtype=torch.float32, device=self.device)
        self.goal_kpt_vel       = torch.zeros((self.num_envs, self.n_kpts, 3), dtype=torch.float32, device=self.device)

    def update_value_buffers(self):
        assert self.initialized, "Robot not initialized"
        entity = self.entity
        self.dof_pos[:] = entity.get_dofs_position(self.actuated_dof_idxs)
        self.dof_vel[:] = entity.get_dofs_velocity(self.actuated_dof_idxs)

        link_pos = entity.get_links_pos()
        link_vel = entity.get_links_vel()
        self.kpt_pos[:] = link_pos[:, self.kpt_link_idxs, :]
        self.kpt_vel[:] = link_vel[:, self.kpt_link_idxs, :]
        # zero velocity for envs that were teleported this step to suppress jump artifacts
        if self.just_reset_mask.any():
            self.kpt_vel[self.just_reset_mask] = 0.0
            self.dof_vel[self.just_reset_mask] = 0.0
            self.just_reset_mask[:] = False
        # self.contact_forces[:] = entity.get_links_net_contact_force()[:, self.coll_idxs_local, :]
        self.control_forces[:] = entity.get_dofs_control_force(self.actuated_dof_idxs)
        self.wrist_pose[:, :3] = link_pos[:, self.wrist_link_idx, :]
        self.wrist_pose[:, 3:] = entity.get_links_quat()[:, self.wrist_link_idx, :]
        if len(self.kpt_markers) > 0:
            # update the kpt pos
            kpt_pos = self.entity.get_links_pos()[:, self.kpt_link_idxs, :]
            for i, marker in enumerate(self.kpt_markers):
                marker.set_pos(kpt_pos[:, i, :])

    def set_goal_state(self, wrist_pose, kpt_pos, wrist_vel, wrist_ang_vel, kpt_vel):
        self.goal_wrist_pose[:]    = wrist_pose
        self.goal_kpt_pos[:]       = kpt_pos
        self.goal_wrist_vel[:]     = wrist_vel
        self.goal_wrist_ang_vel[:] = wrist_ang_vel
        self.goal_kpt_vel[:]       = kpt_vel

    def get_nan_envs(self):
        """ check along the env dim if self.dof_pos, self.dof_vel, self.kpt_pos have NaNs """
        assert self.initialized, "Robot not initialized"
        nan_mask = torch.isnan(self.dof_pos).any(dim=-1)
        for values in [self.dof_vel, self.kpt_pos, self.wrist_pose, self.control_forces]:
            nan_mask |= torch.isnan(values.flatten(start_dim=1)).any(dim=-1)
        return nan_mask
 
    def get_observations(self):
        # Returns obs_dict with shapes (num_envs, dim):
        #   dof_target_pos     (num_envs, ndof)        — goal minus current joint angles
        #   dof_pos            (num_envs, ndof)        — current joint angles, unscaled to [-1, 1]
        #   dof_vel            (num_envs, ndof)        — current joint velocities
        #   kpt_pos            (num_envs, n_kpts * 3)  — fingertip/keypoint 3-D world positions, flattened
        #   kpt_vel            (num_envs, n_kpts * 3)  — fingertip/keypoint 3-D world velocities, flattened
        #   wrist_pose         (num_envs, 7)           — wrist position (3) + quaternion (4)
        #   goal_pos           (num_envs, ndof)        — target joint angles (curr_targets)
        #   previous_pos       (num_envs, ndof)        — joint angles from the previous step
        #   goal_wrist_pose    (num_envs, 7)           — demo target wrist position (3) + quaternion (4)
        #   goal_kpt_pos       (num_envs, n_kpts * 3)  — demo target keypoint 3-D positions, flattened
        #   goal_wrist_vel     (num_envs, 3)           — demo target wrist linear velocity
        #   goal_wrist_ang_vel (num_envs, 3)           — demo target wrist angular velocity
        #   goal_kpt_vel       (num_envs, n_kpts * 3)  — demo target keypoint 3-D velocities, flattened
        assert self.initialized, "Robot not initialized"
        target_pos_diff = self.curr_targets - self.dof_pos            # diff bn goal and curr joint states
        obs_dict = {
            "dof_target_pos":     target_pos_diff,
            "dof_pos": unscale(
                self.dof_pos,
                self.dof_limits[:, 0],
                self.dof_limits[:, 1],
            ),
            "dof_vel":            self.dof_vel,
            "kpt_pos":            self.kpt_pos.view(self.num_envs, -1),
            "kpt_vel":            self.kpt_vel.view(self.num_envs, -1),
            "wrist_pose":         self.wrist_pose,
            "goal_pos":           self.curr_targets,
            "previous_pos":       self.prev_dof_pos,
            "goal_wrist_pose":    self.goal_wrist_pose,
            "goal_kpt_pos":       self.goal_kpt_pos.view(self.num_envs, -1),
            "goal_wrist_vel":     self.goal_wrist_vel,
            "goal_wrist_ang_vel": self.goal_wrist_ang_vel,
            "goal_kpt_vel":       self.goal_kpt_vel.view(self.num_envs, -1),
        }

        if self.randomize_observations:
            # randomize the kpt_pos, wrist_pos, dof_pos
            # print("randomize observations")
            noisy_dof_pos = self.dof_pos + torch.randn_like(self.dof_pos) * self.max_joint_angle_noise
            obs_dict["dof_pos"] = unscale(noisy_dof_pos, self.dof_limits[:, 0], self.dof_limits[:, 1])
            obs_dict["dof_target_pos"] = obs_dict["dof_target_pos"] + torch.randn_like(obs_dict["dof_target_pos"]) * self.max_joint_angle_noise
            obs_dict["kpt_pos"] = obs_dict["kpt_pos"] + torch.randn_like(obs_dict["kpt_pos"]) * self.max_joint_pos_noise
            wrist_pos_noise = torch.randn(self.num_envs, 3, device=self.device) * self.max_joint_pos_noise
            noisy_quat = perturb_quat(self.wrist_pose[:, 3:], self.max_joint_angle_noise)
            obs_dict["wrist_pose"] = torch.cat([self.wrist_pose[:, :3] + wrist_pos_noise, noisy_quat], dim=-1)
            # print(f"[obs noise] dof_pos[0]:       clean={self.dof_pos[0].cpu().numpy().round(4)}  noisy={noisy_dof_pos[0].cpu().numpy().round(4)}")
            # print(f"[obs noise] dof_target_pos[0]: clean={(self.curr_targets - self.dof_pos)[0].cpu().numpy().round(4)}  noisy={obs_dict['dof_target_pos'][0].cpu().numpy().round(4)}")
            # print(f"[obs noise] kpt_pos[0]:        clean={self.kpt_pos.view(self.num_envs, -1)[0].cpu().numpy().round(4)}  noisy={obs_dict['kpt_pos'][0].cpu().numpy().round(4)}")
            # print(f"[obs noise] wrist_pos[0]:      clean={self.wrist_pose[0, :3].cpu().numpy().round(4)}  noisy={obs_dict['wrist_pose'][0, :3].cpu().numpy().round(4)}")
            # print(f"[obs noise] wrist_quat[0]:     clean={self.wrist_pose[0, 3:].cpu().numpy().round(4)}  noisy={noisy_quat[0].cpu().numpy().round(4)}")
            self.prev_dof_pos[:] = noisy_dof_pos        # update the previous_dof_pos
        else:
            self.prev_dof_pos[:] = self.dof_pos

        for k, scale in self.obs_scale.items():         # scales the observations to make different magnitudes closer to be more policy friendly 
            if k in obs_dict:
                obs_dict[k] *= scale 
        return obs_dict  
    
    def compute_obs_dim(self): 
        n_kpts = len(self.kpt_link_names)
        dims = dict(
            qpos_dim           = self.ndof,
            qpos_target_dim    = self.ndof,
            qvel_dim           = self.ndof,
            kpt_dim            = n_kpts * 3,
            kpt_vel_dim        = n_kpts * 3,
            wrist_dim          = 7,
            goal_dim           = self.ndof,
            prev_goal_dim      = self.ndof,
            goal_wrist_pose_dim = 7,
            goal_kpt_pos_dim   = n_kpts * 3,
            goal_wrist_vel_dim     = 3,
            goal_wrist_ang_vel_dim = 3,
            goal_kpt_vel_dim   = n_kpts * 3,
        )
        # print("observation dimensionss: ", dims)
        return sum(dims.values()), dims

    def translate_actions(self, actions, episode_length_buf):
        assert self.initialized, "Robot not initialized"
        assert actions.shape[-1] == self.action_dim, f"actions.shape={actions.shape} != {self.action_dim}" 
        # first map low-dim action to joint targets 
        joint_actions = actions[:, self.action_from_idxs]  # shape (N,ndof) because self.action_from_idxs may contain repeated idxs
        # assume actions are in [-1, 1], scale based on default init pos and limits
        upper_limit = self.dof_limits[:, 1] # shape (n_envs,)
        lower_limit = self.dof_limits[:, 0] # shape shape (n_envs,) 
        if self.residual_qpos is not None:
            next_buf = episode_length_buf + 1
            if self.env_demo_idx is not None and self.all_residual_qpos is not None:
                demo_lengths = torch.tensor(self.all_residual_num_frames, device=self.device, dtype=episode_length_buf.dtype)
                per_env_len = demo_lengths[self.env_demo_idx]
                demo_t = torch.minimum(next_buf, per_env_len - 1)
                res_qpos = self.all_residual_qpos[self.env_demo_idx, demo_t]
            else:
                demo_t = torch.where(
                    next_buf >= self.residual_num_frames,
                    self.residual_num_frames - 1,
                    next_buf
                    )
                res_qpos = self.residual_qpos[demo_t]
                # print(res_qpos)
            self.curr_res_qpos[:] = res_qpos
            
        if self.action_mode == "residual":
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"   
            # NOTE there's an implicit broadcast here bc actions is shape (n_envs, ndof). init_qpos is also shape (ndof,)
            # scale action to add to centering default init pos 
            upper_margin = upper_limit - res_qpos  # shape (n_envs, 1)
            lower_margin = res_qpos - lower_limit  # shape (n_envs, 1)
            if self.res_cap:
                scale_trans, scale_rot = self.hybrid_scales
                upper_margin[:, self.wrist_dof_idxs[:3]] = scale_trans
                upper_margin[:, self.wrist_dof_idxs[3:]] = scale_rot
                lower_margin[:, self.wrist_dof_idxs[:3]] = -scale_trans
                lower_margin[:, self.wrist_dof_idxs[3:]] = -scale_rot
            # joint_actions is -1, 1, make it center around init_qpos
            upper = joint_actions >= 0 
            scaled = torch.where(upper, joint_actions * upper_margin, joint_actions * lower_margin) # joint_actions has sign +-1!!
            joint_targets = res_qpos  + scaled  # DEBUG

        elif self.action_mode == "kinematic": # just all zeros
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"
            joint_targets = res_qpos    

        elif self.action_mode == "relative":
            # translates policy action to a delta, then add to previous target & clamp 
            deltas = joint_actions * self.relative_step_size # shape (n_envs, ndof) 
            joint_targets = self.dof_pos + deltas 

        elif self.action_mode == "absolute":
            joint_targets = lower_limit + (upper_limit - lower_limit) * (joint_actions + 1) / 2 # shape (n_envs, ndof)
        
        elif self.action_mode == "hybrid": # only residual on wrist joints, absolute on finger joints, use self.wrist_dof_idxs and self.finger_dof_idxs
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"
            # from objdex paper: wrist delta actions ±4 centimeters for transition and ±0.5 radian for rotation 
            wrist_actions = torch.clamp(joint_actions[:, self.wrist_dof_idxs], -1, 1) # shape (n_envs, 6)
            wrist_trans_actions = joint_actions[:, self.wrist_dof_idxs[:3]] # shape (n_envs, 3)
            wrist_rot_actions = joint_actions[:, self.wrist_dof_idxs[3:]] # shape (n_envs, 3)
            scale_trans, scale_rot = self.hybrid_scales
            wrist_trans = self.curr_res_qpos[:, self.wrist_dof_idxs[:3]] + scale_trans * wrist_actions[:, :3] # shape (n_envs, 3)
            wrist_rot = self.curr_res_qpos[:, self.wrist_dof_idxs[3:]] + scale_rot * wrist_actions[:, 3:6] # shape (n_envs, 3)

            finger_actions = joint_actions[:, self.finger_dof_idxs]
            finger_targets = lower_limit[self.finger_dof_idxs] + (upper_limit[self.finger_dof_idxs] - lower_limit[self.finger_dof_idxs]) * (finger_actions + 1) / 2
            
            joint_targets = torch.concatenate([wrist_trans, wrist_rot, finger_targets], dim=-1)
            
        else:
            raise NotImplementedError  
        # ignore the mimic values!
        target_dof_pos = joint_targets[:, self.joint_from_idxs] * self.joint_multipliers # shape (n_envs, ndof), 
        target_dof_pos = torch.clamp(target_dof_pos, lower_limit, upper_limit) 
        new_targets = self.action_moving_avg * target_dof_pos + (1 - self.action_moving_avg) * self.curr_targets 
        new_targets = torch.clamp(new_targets, lower_limit, upper_limit)
        self.prev_targets[:] = self.curr_targets
        self.curr_targets[:] = new_targets 
        return new_targets
    
    def map_joint_targets_to_actions(self, joint_targets):
        """
        Translate joint targets to [-1, 1] normalized actions
        """
        assert self.initialized, "Robot not initialized"
        assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
        assert self.action_mode != 'residual', "Residual action not supported"
        upper_limit = self.dof_limits[:, 1]
        lower_limit = self.dof_limits[:, 0]
        # given the desired joint targets, properly scale and shift to map to correct actions
        joint_targets = torch.clamp(joint_targets, lower_limit, upper_limit)
        joint_targets = (joint_targets - lower_limit) / (upper_limit - lower_limit) * 2 - 1
        # map to actions
        actions = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        for i, idx in enumerate(self.action_from_idxs):
            actions[:, idx] = joint_targets[:, i]
        return actions

    def reset_idx(self, env_idxs=None, episode_start=None):
        assert self.initialized, "Robot not initialized" 
        if env_idxs is None:
            env_idxs = range(self.num_envs) # reset all!
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs] 
        if len(env_idxs) == 0:
            return
        if episode_start is not None:
            assert episode_start.shape[0] == len(env_idxs), f"episode_start.shape={episode_start.shape} != {len(env_idxs)}" 
        # reset value buffers 
        if episode_start is not None and self.action_mode in ['residual', 'kinematic']:
            if self.env_demo_idx is not None and self.all_residual_qpos is not None:
                # multi-demo: index into per-demo trajectory at the correct start frame
                demo_lengths = torch.tensor(self.all_residual_num_frames, device=self.device, dtype=episode_start.dtype)
                per_env_len = demo_lengths[self.env_demo_idx[env_idxs]]
                starts = torch.minimum(episode_start, per_env_len - 1)
                init_qpos = self.all_residual_qpos[self.env_demo_idx[env_idxs], starts]
                demo_ids = self.env_demo_idx[env_idxs]
                self.residual_qpos = self.all_residual_qpos[demo_ids]
            elif self.residual_qpos is not None:
                # single-demo
                init_qpos = self.residual_qpos[episode_start]
                print("Init qpos: ", init_qpos)
            # elif self.env_demo_idx is not None and self.all_init_qpos is not None:
            #     init_qpos = self.all_init_qpos[self.env_demo_idx[env_idxs]].float()
            # else:
            #     init_qpos = self.init_qpos[env_idxs]
        elif self.env_demo_idx is not None and self.all_init_qpos is not None:
            # multi-demo: index into per-demo trajectory at the correct start frame
            init_qpos = self.all_init_qpos[self.env_demo_idx[env_idxs]].float()
        else:
            # single demo init
            init_qpos = self.init_qpos[env_idxs]

        self.dof_pos[env_idxs, :] = init_qpos
        self.dof_vel[env_idxs, :] = 0.0
        if self.is_eval and self.num_envs > 1:
            self.dof_pos[-1, :] = 0.0
        
        self.curr_targets[env_idxs, :] = init_qpos.clone()
        self.prev_targets[env_idxs, :] = init_qpos.clone()
        self.prev_dof_pos[env_idxs, :] = init_qpos.clone()
        self.curr_res_qpos[env_idxs, :] = init_qpos.clone()
        # avoid doing this it it might NaN  
        self.entity.set_dofs_position(
            position=self.dof_pos[env_idxs],
            dofs_idx_local=self.actuated_dof_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        ) 
  
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)
        
        prev_kpt_pos = self.kpt_pos[env_idxs, :].clone()
        new_kpt_pos = self.entity.get_links_pos()[:, self.kpt_link_idxs, :] 
        self.kpt_pos[env_idxs] = new_kpt_pos[env_idxs]
        # reset wrist pose
        self.wrist_pose[env_idxs, :3] = new_kpt_pos[env_idxs, self.wrist_link_idx, :]
        self.wrist_pose[env_idxs, 3:] = self.entity.get_links_quat()[env_idxs, self.wrist_link_idx, :]
   
        # self.contact_forces[env_idxs, :] = 0.0
        self.control_forces[env_idxs, :] = 0.0
        self.episode_length_buf[env_idxs] = 0
        if episode_start is not None:
            self.episode_length_buf[env_idxs] = episode_start
        # mark these envs as just teleported so update_value_buffers zeros their velocity
        self.just_reset_mask[env_idxs] = True
        # self.episode_data = defaultdict(list) NOTE: only clear this after flush is called 
    
    def check_env_idxs(self, env_idxs):
        if env_idxs is None:
            env_idxs = range(self.num_envs)
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        return env_idxs

    def set_joint_position(self, joint_targets, joint_idxs=[], env_idxs=None):
        assert self.initialized, "Robot not initialized"
        env_idxs = self.check_env_idxs(env_idxs)
        if len(joint_idxs) == 0:
            assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
            joint_idxs = self.actuated_dof_idxs
        self.entity.set_dofs_position(
            position=joint_targets,
            dofs_idx_local=joint_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        ) 
        self.dof_pos[env_idxs] = joint_targets
        self.prev_targets[env_idxs] = joint_targets
        self.curr_targets[env_idxs] = joint_targets #self.entity.get_dofs_position(envs_idx=env_idxs)
        self.update_value_buffers()

    def get_wrist_xyz_joints(self):
        # return joint_idxs for wrist joints xyz 
        idxs = []
        for word in ['forearm_tx', 'forearm_ty', 'forearm_tz']:
            for joint in self.actuated_joints:
                if word in joint.name:
                    idxs.append(joint.dof_idx_local)
        return idxs
    
    def get_control_force(self):
        return self.entity.get_dofs_control_force(dofs_idx_local=self.actuated_dof_idxs)

    def control_joint_position(self, joint_targets, joint_idxs=[], env_idxs=None):
        assert self.initialized, "Robot not initialized"
        env_idxs = self.check_env_idxs(env_idxs)
        if len(joint_idxs) == 0:
            assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
            joint_idxs = self.actuated_dof_idxs
        upper_limit = self.dof_limits[:, 1] # shape (n_envs,)
        lower_limit = self.dof_limits[:, 0] # shape shape (n_envs,)  
        joint_targets = torch.clamp(joint_targets, lower_limit, upper_limit) 
        self.entity.control_dofs_position(
            joint_targets,
            dofs_idx_local=joint_idxs,
            envs_idx=env_idxs,
        ) 
        self.prev_targets[:] = self.curr_targets
        self.curr_targets[:] = joint_targets
        return  
    
    def step(self, actions, env_idxs=None):
        assert self.initialized, "Robot not initialized"
        target_dof_pos = self.translate_actions(actions, self.episode_length_buf)
        if env_idxs is not None:
            target_dof_pos = target_dof_pos[env_idxs]
        
        if self.wrist_only:
            self.entity.control_dofs_position(
                target_dof_pos[:, :6], 
                self.wrist_dof_idxs,
                envs_idx=env_idxs
            )
        else: 
            self.entity.control_dofs_position(
                target_dof_pos, 
                self.actuated_dof_idxs,
                envs_idx=env_idxs
                ) 
        # NOTE: step the scene in the main thread 
        self.episode_length_buf += 1    
        return 
    
    def get_bc_dist(self):
        # returns current joint target and residual qpos distance
        err = 0.5 * (self.curr_targets - self.curr_res_qpos).pow(2) # shape (n_envs, num_joints)
        # normalize with joint range 
        err /= self.dof_range
        return err

    def flush_episode_data(self):
        if len(self.episode_data) == 0:
            return dict()
        jnames = self.actuated_dof_names    
        kpt_names = self.kpt_link_names
        qpos_dict = dict()
        qpos_targets_dict = dict()
        for i, name in enumerate(jnames):  
            joint_qpos = torch.stack(
                [step_data[:, i] for step_data in self.episode_data['qpos']], dim=0
            )
            if joint_qpos.shape[0] == 1: # shape (num_steps=1, num_envs)
                joint_qpos = joint_qpos[0]
            qpos_dict[name] = joint_qpos
            joint_target = torch.stack(
                [step_data[:, i] for step_data in self.episode_data['qpos_targets']], dim=0
            )
            if joint_target.shape[0] == 1: # shape (num_steps=1, num_envs)
                joint_target = joint_target[0]
            qpos_targets_dict[name] = joint_target # shape (num_step, num_envs) 

        _data = dict(
            joint_qpos=qpos_dict,
            joint_targets=qpos_targets_dict,
            kpt_pos=torch.stack(self.episode_data['kpt_pos'], dim=0), # (num_frames, num_kpts, 3)
            kpt_names=kpt_names, 
            wrist_pose=torch.stack(self.episode_data['wrist_pose'], dim=0), # (num_frames, 7)
            wrist_link_name=self.wrist_link_name,
        )
        self.episode_data = defaultdict(list)
        return _data
    
    def get_control_errors(self):
        err = torch.norm(self.curr_targets - self.dof_pos, p=2, dim=-1) # shape (n_envs,)
        return err

    def collect_data_step(self, collect_all_envs=False):
        if not self.collect_data:
            print("Not collecting data")
            return 
        self.update_value_buffers() 
        
        env_ids = [0]
        if collect_all_envs:
            env_ids = range(self.num_envs)
        # save torch tensors, should be more numerically accurate
        self.episode_data['qpos'].append(self.dof_pos[env_ids].clone())
        self.episode_data['qpos_targets'].append(self.curr_targets[env_ids].clone())
        self.episode_data['kpt_pos'].append(self.kpt_pos[env_ids].clone())
        self.episode_data['wrist_pose'].append(self.wrist_pose[env_ids].clone()) 