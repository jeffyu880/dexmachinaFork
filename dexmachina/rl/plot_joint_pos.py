"""
Visualize trajectories saved by eval_rl_games.py --save_traj.

Usage:
    python plot_joint_pos.py <path/to/eval_ep0.pkl>
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_obj_xyz(data: dict):
    """
    Returns (T, 3) arrays for policy and demo object xyz positions.
    policy_obj_state: (T, num_envs, 8) — take env 0
    demo_state:       (T, 8)
    """
    policy = data["policy_obj_state"]       # (T, num_envs, 8)
    demo   = data["demo_state"]              # (T, 8)

    if policy.ndim == 3:
        policy = policy[:, 0, :]            # take env 0 -> (T, 8)

    return policy[:, :3], demo[:, :3]       # (T, 3), (T, 3)


# ---------------------------------------------------------------------------
# 3-D trajectory plot with time slider
# ---------------------------------------------------------------------------

def plot_3d_trajectory(pkl_path: str):
    data = load_pkl(pkl_path)
    policy_xyz, demo_xyz = extract_obj_xyz(data)
    T = len(policy_xyz)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Object XYZ Trajectory — Policy vs Demo", fontsize=13)

    # Leave room at bottom for the slider
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.15)

    # Active trajectory lines (updated by slider)
    (policy_line,) = ax.plot([], [], [], color="steelblue",  linewidth=2,   label="Policy")
    (demo_line,)   = ax.plot([], [], [], color="darkorange", linewidth=2,   label="Demo")

    # Current-position markers
    (policy_dot,) = ax.plot([], [], [], "o", color="steelblue",  markersize=7)
    (demo_dot,)   = ax.plot([], [], [], "o", color="darkorange", markersize=7)

    # Axis limits with a little padding
    all_xyz = np.concatenate([policy_xyz, demo_xyz], axis=0)
    pad = 0.02
    for setter, col in zip(
        [ax.set_xlim, ax.set_ylim, ax.set_zlim], [0, 1, 2]
    ):
        lo, hi = all_xyz[:, col].min(), all_xyz[:, col].max()
        setter(lo - pad, hi + pad)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left")

    def update(t_val):
        t = int(t_val)
        # Slice trajectories from 0..t
        px, py, pz = policy_xyz[:t, 0], policy_xyz[:t, 1], policy_xyz[:t, 2]
        dx, dy, dz = demo_xyz[:t, 0],   demo_xyz[:t, 1],   demo_xyz[:t, 2]
        policy_line.set_data(px, py);        policy_line.set_3d_properties(pz)
        demo_line.set_data(dx, dy);          demo_line.set_3d_properties(dz)
        # Current tip
        if t > 0:
            policy_dot.set_data([px[-1]], [py[-1]]); policy_dot.set_3d_properties([pz[-1]])
            demo_dot.set_data([dx[-1]], [dy[-1]]);   demo_dot.set_3d_properties([dz[-1]])
        ax.set_title(f"t = {t} / {T - 1}", fontsize=10)
        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.15, 0.04, 0.70, 0.03])
    slider = Slider(ax_slider, "t", 0, T - 1, valinit=T - 1, valstep=1)
    slider.on_changed(update)

    # Draw full trajectory initially
    update(T - 1)

    return fig, slider  # keep slider alive


# ---------------------------------------------------------------------------
# Wrist pose trajectories (static, one subplot per hand)
# ---------------------------------------------------------------------------

def plot_wrist_trajectories(pkl_path: str):
    data = load_pkl(pkl_path)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Wrist Position Trajectory — Policy vs Demo", fontsize=13)

    for i, side in enumerate(["left", "right"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        # Policy wrist: (T, num_envs, 7) or (T, 7)
        policy_wrist = data[f"policy_{side}_hand"]["wrist_pose"]
        if policy_wrist.ndim == 3:
            policy_wrist = policy_wrist[:, 0, :]   # take env 0 -> (T, 7)
        policy_xyz = policy_wrist[:, :3]            # pos is first 3

        # Demo wrist: (T, 7)
        demo_wrist = data["demo_robot"][side]["wrist_pose"]
        if isinstance(demo_wrist, np.ndarray) is False:
            demo_wrist = np.array(demo_wrist)
        demo_xyz = demo_wrist[:, :3]

        # Align lengths in case demo and policy have different T
        T = min(len(policy_xyz), len(demo_xyz))
        policy_xyz, demo_xyz = policy_xyz[:T], demo_xyz[:T]

        ax.plot(*policy_xyz.T, color="steelblue",  linewidth=2, label="Policy")
        ax.plot(*demo_xyz.T,   color="darkorange", linewidth=2, label="Demo")

        # Markers at start and end
        for xyz, color in [(policy_xyz, "steelblue"), (demo_xyz, "darkorange")]:
            ax.plot(*xyz[0],  "o", color=color, markersize=6)
            ax.plot(*xyz[-1], "s", color=color, markersize=6)

        all_xyz = np.concatenate([policy_xyz, demo_xyz], axis=0)
        pad = 0.02
        for setter, col in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [0, 1, 2]):
            lo, hi = all_xyz[:, col].min(), all_xyz[:, col].max()
            setter(lo - pad, hi + pad)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{side.capitalize()} Hand")
        ax.legend(loc="upper left")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wrist XYZ components over time — 3 rows (x, y, z) × 2 cols (left, right)
# ---------------------------------------------------------------------------

def plot_wrist_xyz_components(pkl_path: str):
    data = load_pkl(pkl_path)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle("Wrist Position Components Over Time — Policy vs Demo", fontsize=13)

    labels = ["X (m)", "Y (m)", "Z (m)"]

    for col, side in enumerate(["left", "right"]):
        # Policy wrist: (T, num_envs, 7) or (T, 7)
        policy_wrist = data[f"policy_{side}_hand"]["wrist_pose"]
        if policy_wrist.ndim == 3:
            policy_wrist = policy_wrist[:, 0, :]
        policy_xyz = policy_wrist[:, :3]

        # Demo wrist: (T, 7)
        demo_wrist = data["demo_robot"][side]["wrist_pose"]
        if not isinstance(demo_wrist, np.ndarray):
            demo_wrist = np.array(demo_wrist)
        demo_xyz = demo_wrist[:, :3]

        T = min(len(policy_xyz), len(demo_xyz))
        policy_xyz, demo_xyz = policy_xyz[:T], demo_xyz[:T]
        t = np.arange(T)

        for row, (label, dim) in enumerate(zip(labels, range(3))):
            ax = axes[row, col]
            ax.plot(t, policy_xyz[:, dim], color="steelblue",  linewidth=1.5, label="Policy")
            ax.plot(t, demo_xyz[:, dim],   color="darkorange", linewidth=1.5, label="Demo")
            ax.set_ylabel(label)
            ax.grid(True, linestyle="--", alpha=0.5)
            if row == 0:
                ax.set_title(f"{side.capitalize()} Hand")
                ax.legend(loc="upper right", fontsize=8)
            if row == 2:
                ax.set_xlabel("Timestep")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Object XYZ components over time — 3 rows (x, y, z) × 1 col
# ---------------------------------------------------------------------------

def plot_obj_xyz_components(pkl_path: str):
    data = load_pkl(pkl_path)
    policy_xyz, demo_xyz = extract_obj_xyz(data)

    T = min(len(policy_xyz), len(demo_xyz))
    policy_xyz, demo_xyz = policy_xyz[:T], demo_xyz[:T]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Object Position Components Over Time — Policy vs Demo", fontsize=13)

    for row, label in enumerate(["X (m)", "Y (m)", "Z (m)"]):
        ax = axes[row]
        ax.plot(t, policy_xyz[:, row], color="steelblue",  linewidth=1.5, label="Policy")
        ax.plot(t, demo_xyz[:, row],   color="darkorange", linewidth=1.5, label="Demo")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Timestep")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", help="Path to the .pkl file saved by eval_rl_games.py --save_traj")
    parser.add_argument("--save_figs", action='store_true', help="store trajectories ")
    args = parser.parse_args()

    

    # Interactive Graphs
    # fig, slider = plot_3d_trajectory(args.pkl)      # plotting the 3D trajectory of the bottle over time

    # Static Graphs
    figs = {
        "wrist_3d":  plot_wrist_trajectories(args.pkl),    # left & right wrist 3D trajectory
        "wrist_xyz": plot_wrist_xyz_components(args.pkl),  # 3×2 grid: x/y/z rows × left/right cols
        "obj_xyz":   plot_obj_xyz_components(args.pkl),    # 3×1 grid: x/y/z rows for object
    }

    if args.save_figs:
        pkl_stem = os.path.splitext(os.path.basename(args.pkl))[0]
        os.makedirs(pkl_stem, exist_ok=True)
        save_dir = os.path.join(os.path.dirname(os.path.abspath(args.pkl)), "plots", pkl_stem)
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            out = os.path.join(save_dir, f"{pkl_stem}_{name}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()
