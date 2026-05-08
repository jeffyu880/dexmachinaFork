import torch
from typing import Dict, Tuple
from dexmachina.envs.reward_utils import rotation_distance
from dexmachina.envs.hand_cfgs.allegro import ALLEGRO_LEFT_CFG, ALLEGRO_RIGHT_CFG

def _hand_reward(
        wrist_pose:   torch.Tensor,
        kpts:         torch.Tensor,
        dof_vel:      torch.Tensor,
        demo_wrist:   torch.Tensor,
        demo_kpts:    torch.Tensor,
        scale_factor: float,
        running_progress_buf,
        keypoint_idx,
        wrist_force:  torch.Tensor,   # (N, 6)  wrist DOF control forces
        finger_force: torch.Tensor,   # (N, 16) finger DOF control forces
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        # wrist calculation
        diff_wrist_pos = torch.norm(demo_wrist[:, :3] - wrist_pose[:, :3], dim=-1)

        diff_wrist_rot = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])

        # fingertip calculation
        diff_kpts_dist = torch.norm(demo_kpts - kpts, dim=-1)   # (N, n_kpts)
        n_kpts = kpts.shape[1]

        print("keypoint indices corresponding to the different hand links: ", keypoint_idx)
        
        # calculating keypoint rewards
        diff_thumb_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["thumb_tip"]]].mean(dim=-1)
        diff_index_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["index_tip"]]].mean(dim=-1)
        diff_middle_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["middle_tip"]]].mean(dim=-1)
        if "ring_tip" in keypoint_idx:
            diff_ring_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["ring_tip"]]].mean(dim=-1)
        else:
            diff_ring_tip_pos_dist = torch.zeros_like(diff_thumb_tip_pos_dist)
        diff_pinky_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["pinky_tip"]]].mean(dim=-1)
        diff_level_1_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["level_1_joints"]]].mean(dim=-1)
        diff_level_2_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["level_2_joints"]]].mean(dim=-1)

        rew_wrist_pos  = torch.exp(-40.0 * diff_wrist_pos)
        rew_wrist_rot  = torch.exp(-1.0 * diff_wrist_rot)
        reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
        reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
        reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
        reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
        reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
        reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
        reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

        # there is also no keypoint joint velocities right now
        # # velocity of keypoint difference (there is no ground truth cartesian velocity right now)
        joints_vel = states["joints_state"][:, 1:, 7:10]
        target_joints_vel = target_states["joints_vel"]
        diff_joints_vel = target_joints_vel - joints_vel

        # ── velocity sanity check ────────────────────────────────────────
        error_buf = torch.abs(dof_vel).mean(dim=-1) > 200.0

        # ── failed_execute: large tracking error after warm-up ───────────
        failed_execute = (
        (
            (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
        )  & (running_progress_buf >= 20) ) | error_buf

        # ── weighted reward sum ──────────────────────────────────────────
        hand_reward = (
            0.1 * rew_wrist_pos
            + 0.6 * rew_wrist_rot
            + 0.9 * reward_thumb_tip_pos
            + 0.8 * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos
            + 0.6 * reward_pinky_tip_pos
            + 0.6 * reward_ring_tip_pos
            + 0.5 * reward_level_1_pos
            + 0.3 * reward_level_2_pos
            # + 0.1 * reward_eef_vel
            # + 0.05 * reward_eef_ang_vel
            # + 0.1 * reward_joints_vel
            # + 0.5 * reward_power
            # + 0.5 * reward_wrist_power
         )

        reward_dict = {
            "reward_eef_pos": rew_wrist_pos,
            "reward_eef_rot": rew_wrist_rot,
            # "reward_eef_vel": reward_eef_vel,
            # "reward_eef_ang_vel": reward_eef_ang_vel,
            # "reward_joints_vel": reward_joints_vel,
            "reward_joints_pos": (
                    reward_thumb_tip_pos
                    + reward_index_tip_pos
                    + reward_middle_tip_pos
                    + reward_pinky_tip_pos
                    + reward_ring_tip_pos
                    + reward_level_1_pos
                    + reward_level_2_pos
        ),
            # "reward_power": reward_power,
            # "reward_wrist_power": reward_wrist_power,
        }

        return hand_reward, failed_execute, reward_dict

def compute_no_obj_imitation_reward(
    wrist_pose_left:    torch.Tensor,   # (N, 7)  [x, y, z, qw, qx, qy, qz]
    wrist_pose_right:   torch.Tensor,
    kpts_left:          torch.Tensor,   # (N, n_kpts, 3)
    kpts_right:         torch.Tensor,
    demo_wrist_left:    torch.Tensor,   # (N, 7)  already time-matched
    demo_wrist_right:   torch.Tensor,
    demo_kpts_left:     torch.Tensor,   # (N, n_kpts, 3)
    demo_kpts_right:    torch.Tensor,
    dof_vel_left:       torch.Tensor,   # (N, ndof)
    dof_vel_right:      torch.Tensor,
    wrist_force_left:   torch.Tensor,   # (N, 6)  wrist DOF control forces
    wrist_force_right:  torch.Tensor,
    finger_force_left:  torch.Tensor,   # (N, 16) finger DOF control forces
    finger_force_right: torch.Tensor,
    running_progress_buf: torch.Tensor,  # (N,) steps since last reset
) -> Tuple[torch.Tensor, Dict]:
    """
    Imitation reward for no-object training, modeled on the reference
    compute_imitation_reward from ManipTrans / DexMachina IsaacGym.

    """
    # Print shapes of all inputs
    # print(f"[compute_no_obj_imitation_reward] wrist_pose_left: {wrist_pose_left.shape}")
    # print(f"[compute_no_obj_imitation_reward] wrist_pose_right: {wrist_pose_right.shape}")
    # print(f"[compute_no_obj_imitation_reward] kpts_left: {kpts_left.shape}")
    # print(f"[compute_no_obj_imitation_reward] kpts_right: {kpts_right.shape}")
    # print(f"[compute_no_obj_imitation_reward] demo_wrist_left: {demo_wrist_left.shape}")
    # print(f"[compute_no_obj_imitation_reward] demo_wrist_right: {demo_wrist_right.shape}")
    # print(f"[compute_no_obj_imitation_reward] demo_kpts_left: {demo_kpts_left.shape}")
    # print(f"[compute_no_obj_imitation_reward] demo_kpts_right: {demo_kpts_right.shape}")
    # print(f"[compute_no_obj_imitation_reward] dof_vel_left: {dof_vel_left.shape}")
    # print(f"[compute_no_obj_imitation_reward] dof_vel_right: {dof_vel_right.shape}")
    # print(f"[compute_no_obj_imitation_reward] running_progress_buf: {running_progress_buf.shape}")

    left_rew,  left_failed,  left_dict  = _hand_reward(
                                            wrist_pose_left,  
                                            kpts_left,  
                                            dof_vel_left,  
                                            demo_wrist_left,  
                                            demo_kpts_left,  
                                            1.0, 
                                            running_progress_buf, 
                                            ALLEGRO_LEFT_CFG['keypoint_idx'],  
                                            wrist_force_left,  
                                            finger_force_left,
                                            )
    right_rew, right_failed, right_dict = _hand_reward(
                                            wrist_pose_right, 
                                            kpts_right, 
                                            dof_vel_right, 
                                            demo_wrist_right, 
                                            demo_kpts_right, 
                                            1.0, 
                                            running_progress_buf, 
                                            ALLEGRO_RIGHT_CFG['keypoint_idx'], 
                                            wrist_force_right, 
                                            finger_force_right,
                                            )

    total_rew      = (left_rew + right_rew) / 2.0
    failed_execute = left_failed | right_failed

    rew_dict: Dict = {}
    for k, v in left_dict.items():
        rew_dict[f"no_obj_imi/left/{k}"] = v
    for k, v in right_dict.items():
        rew_dict[f"no_obj_imi/right/{k}"] = v

    rew_dict["task_rew"]       = total_rew  # alias, not accumulated (no_object skips cumulative_task_rew)
    rew_dict["imi_rew"]        = total_rew.clone()
    rew_dict["failed_execute"] = failed_execute
    return total_rew, rew_dict
