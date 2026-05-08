import torch
from typing import Dict, Tuple
from dexmachina.envs.reward_utils import rotation_distance
from dexmachina.envs.hand_cfgs.allegro import ALLEGRO_LEFT_CFG, ALLEGRO_RIGHT_CFG

def _hand_reward(
        wrist_pose:           torch.Tensor,   # (N, 7)  [x,y,z, qw,qx,qy,qz]
        kpts:                 torch.Tensor,   # (N, n_kpts, 3)
        kpts_vel:             torch.Tensor,   # (N, n_kpts, 3)  robot kpt linear vel (world frame)
        dof_vel:              torch.Tensor,   # (N, 22)  [wrist0-5, finger0-15]
        demo_wrist:           torch.Tensor,   # (N, 7)
        demo_kpts:            torch.Tensor,   # (N, n_kpts, 3)
        demo_kpts_vel:        torch.Tensor,   # (N, n_kpts, 3)
        demo_wrist_vel:       torch.Tensor,   # (N, 3)  demo wrist linear vel (DOF space)
        demo_wrist_ang_vel:   torch.Tensor,   # (N, 3)  demo wrist angular vel (DOF space)
        scale_factor:         float,
        running_progress_buf,
        keypoint_idx,
        wrist_force:          torch.Tensor,   # (N, 6)
        finger_force:         torch.Tensor,   # (N, 16)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        # ── wrist pose ──────────────────────────────────────────────────
        diff_wrist_pos = torch.norm(demo_wrist[:, :3] - wrist_pose[:, :3], dim=-1)
        diff_wrist_rot = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])

        # ── keypoint positions ───────────────────────────────────────────
        diff_kpts_dist = torch.norm(demo_kpts - kpts, dim=-1)   # (N, n_kpts)

        diff_thumb_tip_pos_dist  = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["thumb_tip"]]].mean(dim=-1)
        diff_index_tip_pos_dist  = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["index_tip"]]].mean(dim=-1)
        diff_middle_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["middle_tip"]]].mean(dim=-1)
        if "ring_tip" in keypoint_idx:
            diff_ring_tip_pos_dist = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["ring_tip"]]].mean(dim=-1)
        else:
            diff_ring_tip_pos_dist = torch.zeros_like(diff_thumb_tip_pos_dist)
        diff_pinky_tip_pos_dist  = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["pinky_tip"]]].mean(dim=-1)
        diff_level_1_pos_dist    = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["level_1_joints"]]].mean(dim=-1)
        diff_level_2_pos_dist    = diff_kpts_dist[:, [k - 1 for k in keypoint_idx["level_2_joints"]]].mean(dim=-1)

        # ── velocities ───────────────────────────────────────────────────
        # wrist DOF space: first 3 = translation, last 3 = rotation
        wrist_vel     = dof_vel[:, :3]
        wrist_ang_vel = dof_vel[:, 3:6]
        finger_vel    = dof_vel[:, 6:]

        diff_eef_vel     = demo_wrist_vel     - wrist_vel      # (N, 3)
        diff_eef_ang_vel = demo_wrist_ang_vel - wrist_ang_vel  # (N, 3)
        diff_joints_vel  = demo_kpts_vel      - kpts_vel       # (N, n_kpts, 3)

        # ── power ────────────────────────────────────────────────────────
        power       = (finger_force.abs() * finger_vel.abs()).mean(dim=-1)
        wrist_power = (wrist_force.abs()  * dof_vel[:, :6].abs()).mean(dim=-1)

        # ── reward terms ─────────────────────────────────────────────────
        rew_wrist_pos        = torch.exp(-40.0 * diff_wrist_pos)
        rew_wrist_rot        = torch.exp(-1.0  * diff_wrist_rot)
        reward_thumb_tip_pos  = torch.exp(-100  * diff_thumb_tip_pos_dist)
        reward_index_tip_pos  = torch.exp(-90   * diff_index_tip_pos_dist)
        reward_middle_tip_pos = torch.exp(-80   * diff_middle_tip_pos_dist)
        reward_pinky_tip_pos  = torch.exp(-60   * diff_pinky_tip_pos_dist)
        reward_ring_tip_pos   = torch.exp(-60   * diff_ring_tip_pos_dist)
        reward_level_1_pos    = torch.exp(-50   * diff_level_1_pos_dist)
        reward_level_2_pos    = torch.exp(-40   * diff_level_2_pos_dist)
        reward_eef_vel        = torch.exp(-1.0  * diff_eef_vel.abs().mean(dim=-1))
        reward_eef_ang_vel    = torch.exp(-1.0  * diff_eef_ang_vel.abs().mean(dim=-1))
        reward_joints_vel     = torch.exp(-1.0  * diff_joints_vel.abs().mean(dim=-1).mean(dim=-1))
        reward_power          = torch.exp(-10.0 * power)
        reward_wrist_power    = torch.exp(-2.0  * wrist_power)

        # ── sanity / failure checks ──────────────────────────────────────
        error_buf = (
            (torch.norm(wrist_vel,     dim=-1) > 100)
            | (torch.norm(wrist_ang_vel, dim=-1) > 200)
            | (torch.norm(kpts_vel,      dim=-1).mean(dim=-1) > 100)
            | (torch.abs(dof_vel).mean(dim=-1) > 200)
        )
        
        if error_buf.any():
            print(f"[error_buf] {error_buf.sum().item()} envs hit velocity/force limits")

        failed_execute = (
            (
                (diff_thumb_tip_pos_dist  > 0.04  / 0.7 * scale_factor)
                | (diff_index_tip_pos_dist  > 0.045 / 0.7 * scale_factor)
                | (diff_middle_tip_pos_dist > 0.05  / 0.7 * scale_factor)
                | (diff_pinky_tip_pos_dist  > 0.06  / 0.7 * scale_factor)
                | (diff_ring_tip_pos_dist   > 0.06  / 0.7 * scale_factor)
                | (diff_level_1_pos_dist    > 0.07  / 0.7 * scale_factor)
                | (diff_level_2_pos_dist    > 0.08  / 0.7 * scale_factor)
            ) & (running_progress_buf >= 20)
        )
        if failed_execute.any():
            print(f"[failed_execute] {failed_execute.sum().item()} envs failed (large displacement)")
        
        failed_execute = failed_execute| error_buf

        # ── weighted reward sum ──────────────────────────────────────────
        hand_reward = (
            0.1  * rew_wrist_pos
            + 0.6  * rew_wrist_rot
            + 0.9  * reward_thumb_tip_pos
            + 0.8  * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos
            + 0.6  * reward_pinky_tip_pos
            + 0.6  * reward_ring_tip_pos
            + 0.5  * reward_level_1_pos
            + 0.3  * reward_level_2_pos
            + 0.1  * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1  * reward_joints_vel
            + 0.5  * reward_power
            + 0.5  * reward_wrist_power
        )

        reward_dict = {
            "reward_eef_pos":     rew_wrist_pos,
            "reward_eef_rot":     rew_wrist_rot,
            "reward_eef_vel":     reward_eef_vel,
            "reward_eef_ang_vel": reward_eef_ang_vel,
            "reward_joints_vel":  reward_joints_vel,
            "reward_power":       reward_power,
            "reward_wrist_power": reward_wrist_power,
            "reward_joints_pos": (
                reward_thumb_tip_pos
                + reward_index_tip_pos
                + reward_middle_tip_pos
                + reward_pinky_tip_pos
                + reward_ring_tip_pos
                + reward_level_1_pos
                + reward_level_2_pos
            ),
        }

        return hand_reward, failed_execute, reward_dict

# retargeter_results: dict
#   left/right: dict
#     hand_qpos:      (600, 22) float32   finger joint positions
#     wrist_qpos:     (600, 6)  float64   wrist DOF positions
#     hand_vel_qpos:  (600, 22) float32   finger DOF velocities
#     wrist_vel_qpos: (600, 6)  float64   wrist DOF velocities  [:3]=linear, [3:6]=angular
#     kpt_pos:        (600, 25, 3) float32
#     kpt_vel:        (600, 25, 3) float32


def compute_no_obj_imitation_reward(
    wrist_pose_left:          torch.Tensor,   # (N, 7)
    wrist_pose_right:         torch.Tensor,
    kpts_left:                torch.Tensor,   # (N, n_kpts, 3)
    kpts_right:               torch.Tensor,
    demo_wrist_left:          torch.Tensor,   # (N, 7)
    demo_wrist_right:         torch.Tensor,
    demo_kpts_left:           torch.Tensor,   # (N, n_kpts, 3)
    demo_kpts_right:          torch.Tensor,
    kpts_vel_left:            torch.Tensor,   # (N, n_kpts, 3)
    kpts_vel_right:           torch.Tensor,
    demo_kpts_vel_left:       torch.Tensor,   # (N, n_kpts, 3)
    demo_kpts_vel_right:      torch.Tensor,
    demo_wrist_vel_left:      torch.Tensor,   # (N, 3)
    demo_wrist_vel_right:     torch.Tensor,
    demo_wrist_ang_vel_left:  torch.Tensor,   # (N, 3)
    demo_wrist_ang_vel_right: torch.Tensor,
    dof_vel_left:             torch.Tensor,   # (N, 22)
    dof_vel_right:            torch.Tensor,
    wrist_force_left:         torch.Tensor,   # (N, 6)
    wrist_force_right:        torch.Tensor,
    finger_force_left:        torch.Tensor,   # (N, 16)
    finger_force_right:       torch.Tensor,
    running_progress_buf:     torch.Tensor,   # (N,)
    scale_factor:             float = 1.0,
) -> Tuple[torch.Tensor, Dict]:

    left_rew,  left_failed,  left_dict  = _hand_reward(
        wrist_pose_left,
        kpts_left,
        kpts_vel_left,
        dof_vel_left,
        demo_wrist_left,
        demo_kpts_left,
        demo_kpts_vel_left,
        demo_wrist_vel_left,
        demo_wrist_ang_vel_left,
        scale_factor,
        running_progress_buf,
        ALLEGRO_LEFT_CFG['keypoint_idx'],
        wrist_force_left,
        finger_force_left,
    )
    right_rew, right_failed, right_dict = _hand_reward(
        wrist_pose_right,
        kpts_right,
        kpts_vel_right,
        dof_vel_right,
        demo_wrist_right,
        demo_kpts_right,
        demo_kpts_vel_right,
        demo_wrist_vel_right,
        demo_wrist_ang_vel_right,
        scale_factor,
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

    rew_dict["task_rew"]       = 0.0
    rew_dict["imi_rew"]        = total_rew.clone()
    rew_dict["failed_execute"] = failed_execute
    return total_rew, rew_dict
