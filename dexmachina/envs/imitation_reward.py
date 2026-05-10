import torch
import numpy as np
from typing import Dict, Tuple
from dexmachina.envs.reward_utils import rotation_distance
from dexmachina.envs.hand_cfgs.allegro import ALLEGRO_LEFT_CFG, ALLEGRO_RIGHT_CFG
import matplotlib.pyplot as plt


_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
_POS_GT_KEYS  = ["thumb", "index", "middle", "pinky", "wrist_pos"]
_TRACK_KEYS   = (_FINGER_NAMES + ["wrist_pos", "wrist_rot"]
                 + [f"{k}_robot" for k in _POS_GT_KEYS]
                 + [f"{k}_demo"  for k in _POS_GT_KEYS])
_finger_diffs: Dict[str, Dict[str, list]] = {
    "left":  {k: [] for k in _TRACK_KEYS},
    "right": {k: [] for k in _TRACK_KEYS},
}

def plot_finger_histograms(save_path=None, first_n=None):
    _plot_cols = ["thumb", "index", "middle", "pinky", "wrist_pos", "wrist_rot"]
    _units     = {"wrist_pos": "m", "wrist_rot": "rad"}
    fig, axes = plt.subplots(2, 6, figsize=(22, 7), sharey=False)
    title = f"Tracking error distribution — first {first_n} steps" if first_n else "Tracking error distribution"
    fig.suptitle(title, fontsize=13)
    colors = {"left": "#3498db", "right": "#e74c3c"}
    for row, side in enumerate(["left", "right"]):
        for col, key in enumerate(_plot_cols):
            ax = axes[row, col]
            data = _finger_diffs[side][key][:first_n] if first_n else _finger_diffs[side][key]
            if data:
                ax.hist(data, bins=50, color=colors[side], alpha=0.8, edgecolor="none")
                mean_val = np.mean(data)
                ax.axvline(mean_val, color="k", linestyle="--", linewidth=1.2,
                           label=f"mean={mean_val:.4f}")
                ax.legend(fontsize=7)
            unit = _units.get(key, "m")
            ax.set_title(f"{side} {key}", fontsize=9)
            ax.set_xlabel(f"error ({unit})", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved finger histogram to {save_path}")
    else:
        plt.show()

def clear_finger_diffs():
    for side in _finger_diffs:
        for k in _finger_diffs[side]:
            _finger_diffs[side][k].clear()


_KPT3D_KEYS = ["robot_tips", "demo_tips", "robot_wrist", "demo_wrist",
               "robot_wrist_vel", "demo_wrist_vel"]
_kpt_pos_3d: Dict[str, Dict[str, list]] = {
    "left":  {k: [] for k in _KPT3D_KEYS},
    "right": {k: [] for k in _KPT3D_KEYS},
}

_VEL_BASE_KEYS   = ["eef_vel", "eef_ang_vel", "kpt_vel"]
_VEL_FINGER_KEYS = ["thumb_vel", "index_vel", "middle_vel", "pinky_vel"]
_VEL_KEYS = (_VEL_BASE_KEYS
             + [f"{k}_robot" for k in _VEL_BASE_KEYS]
             + [f"{k}_demo"  for k in _VEL_BASE_KEYS]
             + [f"{k}_robot" for k in _VEL_FINGER_KEYS]
             + [f"{k}_demo"  for k in _VEL_FINGER_KEYS])
_vel_diffs: Dict[str, Dict[str, list]] = {
    "left":  {k: [] for k in _VEL_KEYS},
    "right": {k: [] for k in _VEL_KEYS},
}

def plot_pos_tip_keypoint_error(save_path=None, first_n=None):
    _plot_keys = ["thumb", "index", "middle", "pinky", "wrist_pos", "wrist_rot"]
    _units     = {"wrist_rot": "rad"}
    colors = {"left": "#3498db", "right": "#e74c3c"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
    title = f"Position tracking error — first {first_n} steps" if first_n else "Position tracking error over time"
    fig.suptitle(title, fontsize=13)
    for i, key in enumerate(_plot_keys):
        ax = axes[i // 3, i % 3]
        for side in ["left", "right"]:
            data = _finger_diffs[side][key][:first_n] if first_n else _finger_diffs[side][key]
            if data:
                ax.plot(data, color=colors[side], alpha=0.8, linewidth=0.8, label=side)
        unit = _units.get(key, "m")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(f"error ({unit})", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved position timeseries to {save_path}")
    else:
        plt.show()

def plot_vel_wrist_error(save_path=None, first_n=None):
    labels = {"eef_vel": "EEF lin vel err (m/s)",
              "eef_ang_vel": "EEF ang vel err (rad/s)",
              "kpt_vel": "Fingertip vel err (m/s)"}
    colors = {"left": "#3498db", "right": "#e74c3c"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
    title = f"Velocity tracking error — first {first_n} steps" if first_n else "Velocity tracking error over time"
    fig.suptitle(title, fontsize=13)
    for col, key in enumerate(_VEL_BASE_KEYS):
        ax = axes[col]
        for side in ["left", "right"]:
            data = _vel_diffs[side][key][:first_n] if first_n else _vel_diffs[side][key]
            if data:
                ax.plot(data, color=colors[side], alpha=0.8, linewidth=0.8, label=side)
        ax.set_title(labels[key], fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved velocity timeseries to {save_path}")
    else:
        plt.show()

def clear_vel_diffs():
    for side in _vel_diffs:
        for k in _vel_diffs[side]:
            _vel_diffs[side][k].clear()

def plot_pos_gt_timeseries(save_path=None, first_n=None):
    side_colors = {"left": "#3498db", "right": "#e74c3c"}
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
    title = f"Position robot vs demo — first {first_n} steps" if first_n else "Position robot vs demo"
    fig.suptitle(title, fontsize=13)
    for row, side in enumerate(["left", "right"]):
        for col, key in enumerate(_POS_GT_KEYS):
            ax = axes[row, col]
            c  = side_colors[side]
            sl = lambda buf: buf[:first_n] if first_n else buf
            rob = sl(_finger_diffs[side][f"{key}_robot"])
            dem = sl(_finger_diffs[side][f"{key}_demo"])
            if rob: ax.plot(rob, color=c,       linewidth=0.8, label="robot")
            if dem: ax.plot(dem, color=c, alpha=0.45, linewidth=0.8, linestyle="--", label="demo")
            ax.set_title(f"{side} {key}", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("pos norm (m)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved position GT timeseries to {save_path}")
    else:
        plt.show()

def plot_wrist_vel_gt_timeseries(save_path=None, first_n=None):
    labels = {"eef_vel": "EEF lin vel (m/s)",
              "eef_ang_vel": "EEF ang vel (rad/s)"}
    side_colors = {"left": "#3498db", "right": "#e74c3c"}
    plot_keys = ["eef_vel", "eef_ang_vel"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    title = f"Velocity robot vs demo — first {first_n} steps" if first_n else "Velocity robot vs demo"
    fig.suptitle(title, fontsize=13)
    for row, side in enumerate(["left", "right"]):
        for col, key in enumerate(plot_keys):
            ax = axes[row, col]
            c  = side_colors[side]
            sl = lambda buf: buf[:first_n] if first_n else buf
            rob = sl(_vel_diffs[side][f"{key}_robot"])
            dem = sl(_vel_diffs[side][f"{key}_demo"])
            if rob: ax.plot(rob, color=c,       linewidth=0.8, label="robot")
            if dem: ax.plot(dem, color=c, alpha=0.45, linewidth=0.8, linestyle="--", label="demo")
            ax.set_title(f"{side} {labels[key]}", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved velocity GT timeseries to {save_path}")
    else:
        plt.show()

def plot_fingertip_vel_gt(save_path=None, first_n=None):
    side_colors = {"left": "#3498db", "right": "#e74c3c"}
    # per-finger entries use "{fname}_vel_robot/demo"; "all" uses "kpt_vel_robot/demo"
    finger_cols = [("thumb", "thumb_vel"), ("index", "index_vel"),
                   ("middle", "middle_vel"), ("pinky", "pinky_vel"),
                   ("all", "kpt_vel")]
    fig, axes = plt.subplots(2, 5, figsize=(26, 7), sharey=False)
    title = f"Fingertip velocity robot vs demo — first {first_n} steps" if first_n else "Fingertip velocity robot vs demo"
    fig.suptitle(title, fontsize=13)
    for row, side in enumerate(["left", "right"]):
        for col, (label, key) in enumerate(finger_cols):
            ax = axes[row, col]
            c  = side_colors[side]
            sl = lambda buf: buf[:first_n] if first_n else buf
            rob = sl(_vel_diffs[side][f"{key}_robot"])
            dem = sl(_vel_diffs[side][f"{key}_demo"])
            if rob: ax.plot(rob, color=c,       linewidth=0.8, label="robot")
            if dem: ax.plot(dem, color=c, alpha=0.45, linewidth=0.8, linestyle="--", label="demo")
            ax.set_title(f"{side} {label} tip vel", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("vel norm (m/s)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved fingertip velocity GT to {save_path}")
    else:
        plt.show()

def plot_wrist_gt(save_path=None, first_n=None):
    side_colors = {"left": "#3498db", "right": "#e74c3c"}
    cols = [
        ("wrist_pos",  _finger_diffs, "pos norm (m)"),
        ("eef_vel",    _vel_diffs,    "lin vel (m/s)"),
        ("eef_ang_vel",_vel_diffs,    "ang vel (rad/s)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharey=False)
    title = f"Wrist position & velocity robot vs demo — first {first_n} steps" if first_n else "Wrist position & velocity robot vs demo"
    fig.suptitle(title, fontsize=13)
    for row, side in enumerate(["left", "right"]):
        for col, (key, buf_dict, ylabel) in enumerate(cols):
            ax = axes[row, col]
            c  = side_colors[side]
            sl = lambda buf: buf[:first_n] if first_n else buf
            rob = sl(buf_dict[side][f"{key}_robot"])
            dem = sl(buf_dict[side][f"{key}_demo"])
            if rob: ax.plot(rob, color=c,       linewidth=0.8, label="robot")
            if dem: ax.plot(dem, color=c, alpha=0.45, linewidth=0.8, linestyle="--", label="demo")
            ax.set_title(f"{side} {key}", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved wrist GT to {save_path}")
    else:
        plt.show()

def plot_3d_fingertip_trajectory(side="right", first_n=None, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.widgets as widgets

    buf = _kpt_pos_3d[side]
    if not buf["robot_tips"]:
        print(f"[plot_3d_fingertip_trajectory] no data for side={side}")
        return

    robot_tips      = np.stack(buf["robot_tips"])       # (T, 4, 3)
    demo_tips       = np.stack(buf["demo_tips"])         # (T, 4, 3)
    robot_wrist     = np.stack(buf["robot_wrist"])       # (T, 3)
    demo_wrist_pos  = np.stack(buf["demo_wrist"])        # (T, 3)
    robot_wrist_vel = np.stack(buf["robot_wrist_vel"])   # (T, 3)
    demo_wrist_vel  = np.stack(buf["demo_wrist_vel"])    # (T, 3)

    if first_n:
        robot_tips      = robot_tips[:first_n]
        demo_tips       = demo_tips[:first_n]
        robot_wrist     = robot_wrist[:first_n]
        demo_wrist_pos  = demo_wrist_pos[:first_n]
        robot_wrist_vel = robot_wrist_vel[:first_n]
        demo_wrist_vel  = demo_wrist_vel[:first_n]

    T = robot_tips.shape[0]
    FINGER_COLORS = ['tomato', 'gold', 'limegreen', 'violet']
    FINGER_NAMES  = ['thumb', 'index', 'middle', 'pinky']
    ARROW_SCALE   = 0.3

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.12)
    fig.suptitle(f"{side} hand — fingertip + wrist trajectory (robot vs demo)", fontsize=12)

    # static wrist paths
    # ax.plot(*robot_wrist.T,    color='steelblue', alpha=0.2, linewidth=1, label='Robot wrist')
    ax.plot(*demo_wrist_pos.T, color='darkorange', alpha=0.2, linewidth=1, label='Demo wrist')

    # dynamic scatter: fingertips (circle=robot, triangle=demo)
    # robot_sc = [ax.scatter(*robot_tips[0, i], c=FINGER_COLORS[i], s=35,
    #                         label=f'R {FINGER_NAMES[i]}', depthshade=False)
    #             for i in range(4)]
    demo_sc  = [ax.scatter(*demo_tips[0, i],  c=FINGER_COLORS[i], s=35, marker='^',
                            alpha=0.5, label=f'D {FINGER_NAMES[i]}', depthshade=False)
                for i in range(4)]
    # robot_wrist_sc = ax.scatter(*robot_wrist[0],    c='steelblue',  s=120, marker='*',
    #                              label='R wrist', depthshade=False)
    demo_wrist_sc  = ax.scatter(*demo_wrist_pos[0], c='darkorange', s=120, marker='*',
                                 label='D wrist', depthshade=False)
    quivers = [None, None]

    all_pts = np.concatenate([robot_tips.reshape(-1, 3), demo_tips.reshape(-1, 3),
                               robot_wrist, demo_wrist_pos], axis=0)
    pad = 0.05
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=6, ncol=2)

    def update(t):
        t = int(t)
        for i in range(4):
            # robot_sc[i]._offsets3d = (robot_tips[t, i, :1], robot_tips[t, i, 1:2], robot_tips[t, i, 2:3])
            demo_sc[i]._offsets3d  = (demo_tips[t, i, :1],  demo_tips[t, i, 1:2],  demo_tips[t, i, 2:3])
        rw = robot_wrist[t]
        dw = demo_wrist_pos[t]
        # robot_wrist_sc._offsets3d = ([rw[0]], [rw[1]], [rw[2]])
        demo_wrist_sc._offsets3d  = ([dw[0]], [dw[1]], [dw[2]])

        for qi, (wp, wv, col) in enumerate([
            (rw, robot_wrist_vel[t], 'steelblue'),
            (dw, demo_wrist_vel[t],  'darkorange'),
        ]):
            if quivers[qi] is not None:
                quivers[qi].remove()
            speed = np.linalg.norm(wv)
            arrow = wv * ARROW_SCALE if speed > 1e-4 else np.zeros(3)
            quivers[qi] = ax.quiver(
                wp[0], wp[1], wp[2], arrow[0], arrow[1], arrow[2],
                color=col, linewidth=2, arrow_length_ratio=0.3,
            )

        ax.set_title(
            f"frame {t}/{T-1}  |  robot wrist speed: {np.linalg.norm(robot_wrist_vel[t]):.3f} m/s  "
            f"demo: {np.linalg.norm(demo_wrist_vel[t]):.3f} m/s",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.15, 0.04, 0.70, 0.025])
    slider = widgets.Slider(ax_slider, 'Frame', 0, T - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    if save_path:
        update(T - 1)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved 3D trajectory to {save_path}")
    else:
        update(0)
        plt.show()

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
        side:                 str = "right",
        debug:                bool = False,   # whether or not to store values for debugging  
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        # # ── wrist pose ──────────────────────────────────────────────────
        # print(f"  wrist_pose pos:  {wrist_pose[0, :3].cpu().numpy().round(3)}  quat: {wrist_pose[0, 3:].cpu().numpy().round(3)}")
        # print(f"  demo_wrist pos:  {demo_wrist[0, :3].cpu().numpy().round(3)}  quat: {demo_wrist[0, 3:].cpu().numpy().round(3)}")
        # for i in range(4):
        #     print(f"  kpts[{i}]:      robot={kpts[0, i].cpu().numpy().round(3)}  demo={demo_kpts[0, i].cpu().numpy().round(3)}  diff={float((kpts[0,i]-demo_kpts[0,i]).norm()):.4f}")
        diff_wrist_pos = torch.norm(demo_wrist[:, :3] - wrist_pose[:, :3], dim=-1)
        diff_wrist_rot = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
        for i in range(len(running_progress_buf)):
            print(f"    env{i}: step={running_progress_buf[i].item():3d}  rot_err={diff_wrist_rot[i].item():.4f} rad  pos_err={diff_wrist_pos[i].item():.4f}")

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

        if debug: 
            # ── accumulate diffs for histogram ───────────────────────────────
            _finger_diffs[side]["thumb"].extend(diff_thumb_tip_pos_dist.cpu().numpy().tolist())
            _finger_diffs[side]["index"].extend(diff_index_tip_pos_dist.cpu().numpy().tolist())
            _finger_diffs[side]["middle"].extend(diff_middle_tip_pos_dist.cpu().numpy().tolist())
            _finger_diffs[side]["ring"].extend(diff_ring_tip_pos_dist.cpu().numpy().tolist())
            _finger_diffs[side]["pinky"].extend(diff_pinky_tip_pos_dist.cpu().numpy().tolist())
            _finger_diffs[side]["wrist_pos"].extend(diff_wrist_pos.cpu().numpy().tolist())
            _finger_diffs[side]["wrist_rot"].extend(diff_wrist_rot.cpu().numpy().tolist())

            # ── accumulate raw robot / demo positions for ground-truth plot ──
            for fname, kidxs in [("thumb",  keypoint_idx["thumb_tip"]),
                                ("index",  keypoint_idx["index_tip"]),
                                ("middle", keypoint_idx["middle_tip"]),
                                ("pinky",  keypoint_idx["pinky_tip"])]:
                idx = [k - 1 for k in kidxs]
                _finger_diffs[side][f"{fname}_robot"].extend(kpts[:, idx].norm(dim=-1).mean(dim=-1).cpu().numpy().tolist())
                _finger_diffs[side][f"{fname}_demo"].extend(demo_kpts[:, idx].norm(dim=-1).mean(dim=-1).cpu().numpy().tolist())
            _finger_diffs[side]["wrist_pos_robot"].extend(wrist_pose[:, :3].norm(dim=-1).cpu().numpy().tolist())
            _finger_diffs[side]["wrist_pos_demo"].extend(demo_wrist[:, :3].norm(dim=-1).cpu().numpy().tolist())

        # ── velocities ───────────────────────────────────────────────────
        # wrist DOF space: first 3 = translation, last 3 = rotation
        wrist_vel     = dof_vel[:, :3]
        wrist_ang_vel = dof_vel[:, 3:6]
        finger_vel    = dof_vel[:, 6:]

        diff_eef_vel     = demo_wrist_vel     - wrist_vel      # (N, 3)
        diff_eef_ang_vel = demo_wrist_ang_vel - wrist_ang_vel  # (N, 3)
        diff_joints_vel  = demo_kpts_vel      - kpts_vel       # (N, n_kpts, 3)
        
        if debug:
            # ── accumulate velocity diffs + raw robot/demo for timeseries ───
            _vel_diffs[side]["eef_vel"].append(diff_eef_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["eef_ang_vel"].append(diff_eef_ang_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["kpt_vel"].append(diff_joints_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["eef_vel_robot"].append(wrist_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["eef_vel_demo"].append(demo_wrist_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["eef_ang_vel_robot"].append(wrist_ang_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["eef_ang_vel_demo"].append(demo_wrist_ang_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["kpt_vel_robot"].append(kpts_vel.norm(dim=-1).mean().item())
            _vel_diffs[side]["kpt_vel_demo"].append(demo_kpts_vel.norm(dim=-1).mean().item())
            for fname, kidxs in [("thumb",  keypoint_idx["thumb_tip"]),
                                ("index",  keypoint_idx["index_tip"]),
                                ("middle", keypoint_idx["middle_tip"]),
                                ("pinky",  keypoint_idx["pinky_tip"])]:
                idx = [k - 1 for k in kidxs]
                _vel_diffs[side][f"{fname}_vel_robot"].append(kpts_vel[:, idx].norm(dim=-1).mean().item())
                _vel_diffs[side][f"{fname}_vel_demo"].append(demo_kpts_vel[:, idx].norm(dim=-1).mean().item())

            # ── accumulate 3-D positions for interactive trajectory plot ─────
            tips_robot = np.stack([
                kpts[:, [k - 1 for k in keypoint_idx["thumb_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                kpts[:, [k - 1 for k in keypoint_idx["index_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                kpts[:, [k - 1 for k in keypoint_idx["middle_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                kpts[:, [k - 1 for k in keypoint_idx["pinky_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
            ])  # (4, 3)
            tips_demo = np.stack([
                demo_kpts[:, [k - 1 for k in keypoint_idx["thumb_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                demo_kpts[:, [k - 1 for k in keypoint_idx["index_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                demo_kpts[:, [k - 1 for k in keypoint_idx["middle_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
                demo_kpts[:, [k - 1 for k in keypoint_idx["pinky_tip"]]].mean(dim=1).mean(dim=0).cpu().numpy(),
            ])  # (4, 3)
            _kpt_pos_3d[side]["robot_tips"].append(tips_robot)
            _kpt_pos_3d[side]["demo_tips"].append(tips_demo)
            _kpt_pos_3d[side]["robot_wrist"].append(wrist_pose[:, :3].mean(dim=0).cpu().numpy())
            _kpt_pos_3d[side]["demo_wrist"].append(demo_wrist[:, :3].mean(dim=0).cpu().numpy())
            _kpt_pos_3d[side]["robot_wrist_vel"].append(wrist_vel.mean(dim=0).cpu().numpy())
            _kpt_pos_3d[side]["demo_wrist_vel"].append(demo_wrist_vel.mean(dim=0).cpu().numpy())

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
        
        # if error_buf.any():
        #     print(f"[error_buf] {error_buf.sum().item()} envs hit velocity/force limits")

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

        # def _fmt(t): return f"{t.mean().item():.4f}"
        # print(
        #     f"  diff_wrist_pos={_fmt(diff_wrist_pos)}  diff_wrist_rot={_fmt(diff_wrist_rot)} (rad)\n"
        #     f"  diff_thumb={_fmt(diff_thumb_tip_pos_dist)}  diff_index={_fmt(diff_index_tip_pos_dist)}"
        #     f"  diff_middle={_fmt(diff_middle_tip_pos_dist)}  diff_pinky={_fmt(diff_pinky_tip_pos_dist)}"
        #     f"  diff_ring={_fmt(diff_ring_tip_pos_dist)}\n"
        #     f"  diff_lvl1={_fmt(diff_level_1_pos_dist)}  diff_lvl2={_fmt(diff_level_2_pos_dist)}"
        # )

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
    debug:                    bool = False,   # used to store intermediate values
) -> Tuple[torch.Tensor, Dict]:

    # for name, val in [
    #     ("wrist_pose_left",          wrist_pose_left),
    #     ("wrist_pose_right",         wrist_pose_right),
    #     ("kpts_left",                kpts_left),
    #     ("kpts_right",               kpts_right),
    #     ("demo_kpts_left",           demo_kpts_left),
    #     ("demo_kpts_right",          demo_kpts_right),
    # ]:
    #     print(f"  {name:30s}: {tuple(val.shape)}")
    # print(f"  {'scale_factor':30s}: {scale_factor}")

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
        side="left",
        debug=debug
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
        side="right",
        debug=debug
    )

    total_rew      = (left_rew + right_rew) / 2.0
    failed_execute = left_failed | right_failed

    rew_dict: Dict = {}
    for k, v in left_dict.items():
        rew_dict[f"no_obj_imi/left/{k}"] = v
    for k, v in right_dict.items():
        rew_dict[f"no_obj_imi/right/{k}"] = v

    rew_dict["task_rew"]       = torch.zeros_like(total_rew)
    rew_dict["imi_rew"]        = total_rew.clone()
    rew_dict["failed_execute"] = failed_execute
    return total_rew, rew_dict
