#!/usr/bin/env python3
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

data_fname = sys.argv[1] if len(sys.argv) > 1 else \
    "../dexmachina/assets/retargeted/allegro_hand/s01/ketchup_use_01_vector_pure_ik_para.pt"

print(f"Loading: {data_fname}\n")

if data_fname.endswith(".pt"):
    data = torch.load(data_fname, weights_only=False)
else:
    data = np.load(data_fname, allow_pickle=True).item()

# ── extract left hand data ────────────────────────────────────────────────────
retar_res  = data['retargeter_results']['left']
retar_data = data['retarget_data']['left']

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().float().numpy()
    return np.array(x, dtype=np.float32)

kpt_pos_all = to_numpy(retar_res['kpt_pos'])   # (T, 25, 3)
kpt_vel_all = to_numpy(retar_res['kpt_vel'])   # (T, 25, 3)

# filter to fingertip keypoints only
kpt_names = list(retar_data.get('kpt_names', []))
if not kpt_names:
    kpt_names = list(data['retarget_data']['left'].get('kpt_names', []))
tip_idxs = [i for i, n in enumerate(kpt_names) if 'tip' in n.lower()]
if not tip_idxs:
    tip_idxs = list(range(kpt_pos_all.shape[1]))  # fallback: all
print(f"Fingertip keypoints ({len(tip_idxs)}): {[kpt_names[i] for i in tip_idxs]}")

kpt_pos = kpt_pos_all[:, tip_idxs, :]   # (T, n_tips, 3)
kpt_vel = kpt_vel_all[:, tip_idxs, :]   # (T, n_tips, 3)

wrist_pose = to_numpy(retar_data['wrist_pose'])
if wrist_pose.ndim == 3:
    wrist_pose = wrist_pose[0]             # (T, 7)
wrist_pos = wrist_pose[:, :3]             # (T, 3)
wrist_vel = np.gradient(wrist_pos, axis=0) * 30.0  # finite-diff at 30fps (m/s)

T = kpt_pos.shape[0]
print(f"Frames: {T}  |  Keypoints: {kpt_pos.shape[1]}")

# ── figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
ax  = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.12)

# static: full wrist trajectory
ax.plot(wrist_pos[:, 0], wrist_pos[:, 1], wrist_pos[:, 2],
        color='steelblue', alpha=0.25, linewidth=1, label='Wrist path')

# dynamic placeholders
kpt_scatter  = ax.scatter(*kpt_pos[0].T,  c='tomato',    s=18,  label='Keypoints', depthshade=False)
wrist_scatter = ax.scatter(*wrist_pos[0], c='steelblue', s=100, marker='*', label='Wrist', depthshade=False)
quiver_ref   = [None]

# axis limits from full trajectory
pad = 0.05
all_pts = np.concatenate([kpt_pos_all.reshape(-1, 3), wrist_pos], axis=0)
ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend(loc='upper left')

# ── update function ───────────────────────────────────────────────────────────
ARROW_SCALE = 0.3    # metres per (m/s) — tune visually

def update(t):
    t = int(t)

    # keypoints
    kpts = kpt_pos[t]
    kpt_scatter._offsets3d = (kpts[:, 0], kpts[:, 1], kpts[:, 2])

    # wrist dot
    wp = wrist_pos[t]
    wrist_scatter._offsets3d = ([wp[0]], [wp[1]], [wp[2]])

    # wrist velocity arrow
    if quiver_ref[0] is not None:
        quiver_ref[0].remove()
    wv = wrist_vel[t]
    speed = np.linalg.norm(wv)
    if speed > 1e-4:
        arrow = wv * ARROW_SCALE
    else:
        arrow = np.zeros(3)
    quiver_ref[0] = ax.quiver(
        wp[0], wp[1], wp[2],
        arrow[0], arrow[1], arrow[2],
        color='limegreen', linewidth=2, arrow_length_ratio=0.3,
    )

    ax.set_title(f'Left hand — frame {t}/{T-1}   |   wrist speed {speed:.3f} m/s')
    fig.canvas.draw_idle()

# ── slider ────────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.15, 0.04, 0.70, 0.025])
slider = widgets.Slider(ax_slider, 'Frame', 0, T - 1, valinit=0, valstep=1)
slider.on_changed(update)

update(0)
plt.show()
