"""
distill_rl_games.py

Policy distillation: train a single student policy to imitate N teacher policies,
one per demonstration clip. Uses DAgger-style collection (teacher drives the env)
and MSE action supervision.

The student checkpoint is saved in rl_games format and is loadable by eval_rl_games.py.

Requirements:
  - All teacher clips must use the same hand type (same action_dim).
  - All teacher clips must produce the same obs_dim (same object type and same
    training config). If obs_dims differ, the script will pad smaller observations
    with zeros up to max_obs_dim.

Usage:
  python -m dexmachina.rl.distill_rl_games \\
      --checkpoints logs/rl_games/inspire_hand/run1/nn/ep5000.pth \\
                    logs/rl_games/inspire_hand/run2/nn/ep5000.pth \\
      --clips ketchup-0-100-s01-u01 ketchup-0-100-s01-u02 \\
      --hand inspire_hand \\
      --exp_name student_ketchup \\
      --num_envs 64 --num_epochs 5000
"""

import os
import math
import yaml
import torch
import pickle
import argparse
import numpy as np
import genesis as gs
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from dexmachina.rl.train_rl_multi_demo import load_multi_demo_data_separate, dump_yaml

# Patch torch.load for PyTorch 2.6+ compatibility with rl_games checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# Teacher: loads the actor MLP directly from an rl_games checkpoint.
# No rl_games env registration needed — just pure PyTorch inference.
#
# rl_games actor_critic state_dict keys (separate=False, fixed_sigma=True):
#   a2c_network.actor_mlp.0.weight  [512, obs_dim]
#   a2c_network.actor_mlp.0.bias    [512]
#   a2c_network.actor_mlp.2.weight  [512, 512]
#   a2c_network.actor_mlp.4.weight  [256, 512]
#   a2c_network.actor_mlp.6.weight  [128, 256]
#   a2c_network.mu.weight           [action_dim, 128]
#   a2c_network.mu.bias             [action_dim]
#   a2c_network.sigma               [action_dim]   (log std, not used for inference)
# ---------------------------------------------------------------------------

class TeacherActor(nn.Module):
    """
    Extracts and runs only the actor (mean) head from an rl_games checkpoint.
    Works without any rl_games env registration.
    """

    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()
        ckpt = torch.load(ckpt_path)
        weights = ckpt['model']

        first_key = 'a2c_network.actor_mlp.0.weight'
        mu_key = 'a2c_network.mu.weight'
        assert first_key in weights, (
            f"Expected key '{first_key}' not found. "
            f"Available keys (first 10): {list(weights.keys())[:10]}"
        )
        assert mu_key in weights, f"Expected key '{mu_key}' not found."

        self.obs_dim = weights[first_key].shape[1]
        self.action_dim = weights[mu_key].shape[0]

        # Collect linear layer indices from the actor MLP sequential
        mlp_idxs = sorted([
            int(k.split('.')[2])
            for k in weights
            if k.startswith('a2c_network.actor_mlp.') and k.endswith('.weight')
        ])

        layers = []
        for idx in mlp_idxs:
            w = weights[f'a2c_network.actor_mlp.{idx}.weight']
            b = weights[f'a2c_network.actor_mlp.{idx}.bias']
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            layers.append(nn.ELU())

        # mu (mean action) head — raw linear output, no activation
        mu_w = weights[mu_key]
        mu_b = weights['a2c_network.mu.bias']
        mu_lin = nn.Linear(mu_w.shape[1], mu_w.shape[0])
        mu_lin.weight.data.copy_(mu_w)
        mu_lin.bias.data.copy_(mu_b)
        layers.append(mu_lin)

        self.net = nn.Sequential(*layers)
        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Student network that mirrors the rl_games actor_critic architecture so its
# state_dict can be saved as an rl_games checkpoint and loaded by eval_rl_games.
# ---------------------------------------------------------------------------

class StudentNetwork(nn.Module):
    """
    Mirrors the rl_games actor_critic MLP (separate=False, fixed_sigma=True).
    Produces the same state_dict key structure so the checkpoint is loadable
    by eval_rl_games.py via agent.restore().

    State dict keys produced:
      a2c_network.actor_mlp.{0,2,4,6}.weight / .bias
      a2c_network.mu.weight / .bias
      a2c_network.value.weight / .bias
      a2c_network.sigma                        (fixed log-std parameter)
    """

    def __init__(self, obs_dim: int, action_dim: int, mlp_units: list):
        super().__init__()

        # Build shared actor/critic trunk (rl_games uses same trunk when separate=False)
        mlp_layers = []
        in_dim = obs_dim
        for out_dim in mlp_units:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ELU())
            in_dim = out_dim

        # Wrap in a sub-module named 'a2c_network' to match rl_games key structure
        class _A2CNetwork(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.actor_mlp = nn.Sequential(*mlp_layers)
                self_inner.mu = nn.Linear(in_dim, action_dim)
                self_inner.value = nn.Linear(in_dim, 1)
                self_inner.sigma = nn.Parameter(torch.zeros(action_dim))

            def forward(self_inner, obs):
                x = self_inner.actor_mlp(obs)
                return self_inner.mu(x), self_inner.value(x)

        self.a2c_network = _A2CNetwork()

    def forward(self, obs: torch.Tensor):
        """Returns (mu, value) — student is trained on mu only."""
        return self.a2c_network(obs)

    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return mu


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_student_checkpoint(student: StudentNetwork, optimizer, epoch: int,
                             obs_dim: int, action_dim: int, mlp_units: list,
                             teacher_checkpoints: list, path: str):
    """Save student in rl_games checkpoint format (loadable by eval_rl_games.py)."""
    torch.save({
        'model': student.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'distill_meta': {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'mlp_units': mlp_units,
            'teacher_checkpoints': teacher_checkpoints,
        },
    }, path)
    print(f"[Checkpoint] Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_common_argparser()
    parser.add_argument('--checkpoints', '-ck', nargs='+', required=True,
                        help='Teacher checkpoint .pth files, one per --clips entry')
    parser.add_argument('--clips', nargs='+', required=True,
                        help='Demo clips matching each checkpoint, e.g. ketchup-0-100-s01-u01')
    parser.add_argument('--exp_name', '-exp', type=str, default='distill')
    parser.add_argument('--horizon', '-ho', type=int, default=32,
                        help='Steps collected per gradient update per teacher')
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--save_freq', '-sf', type=int, default=500)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.clips), (
        f"--checkpoints ({len(args.checkpoints)}) and --clips ({len(args.clips)}) "
        "must have the same number of entries."
    )
    for ck in args.checkpoints:
        assert os.path.exists(ck), f"Checkpoint not found: {ck}"

    device = torch.device('cuda:0')
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    exp_name = f"{args.exp_name}_distill_{timestamp}"

    # -----------------------------------------------------------------------
    # 1. Load teacher actors (pure PyTorch, no Genesis needed)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading {len(args.checkpoints)} teacher(s)")
    print(f"{'='*60}")

    teachers = []
    teacher_obs_dims = []
    for i, ck in enumerate(args.checkpoints):
        teacher = TeacherActor(ck, device)
        teachers.append(teacher)
        teacher_obs_dims.append(teacher.obs_dim)
        print(f"  Teacher {i}: obs_dim={teacher.obs_dim}, "
              f"action_dim={teacher.action_dim}  [{ck}]")

    action_dim = teachers[0].action_dim
    assert all(t.action_dim == action_dim for t in teachers), (
        "All teachers must have the same action_dim (same hand type)."
    )

    student_obs_dim = max(teacher_obs_dims)
    if len(set(teacher_obs_dims)) > 1:
        print(f"\n[WARN] obs_dims differ across teachers: {teacher_obs_dims}")
        print(f"       Using max obs_dim={student_obs_dim}, padding smaller obs with zeros.")
    else:
        print(f"\nAll teachers share obs_dim={student_obs_dim}. No padding needed.")

    # -----------------------------------------------------------------------
    # 2. Build multi-demo student environment
    # -----------------------------------------------------------------------
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding='utf-8') as f:
        agent_cfg = yaml.full_load(f)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    mlp_units = agent_cfg["params"]["network"]["mlp"]["units"]

    all_demo_data, all_retarget_data, base_env_cfg, demo_info_lines = \
        load_multi_demo_data_separate(
            args.clips, args, 'cuda:0',
            exp_name=exp_name, timestamp=timestamp
        )

    base_env_cfg['env_cfg']['use_rl_games'] = True
    base_env_cfg['env_cfg']['rand_init_ratio'] = getattr(args, 'rand_init_ratio', 0.5)
    base_env_cfg['env_cfg']['demo_sampling'] = getattr(args, 'demo_sampling', 'random')
    base_env_cfg.pop('curriculum_cfg', None)

    gs.init(backend=gs.gpu, logging_level='warning')
    base_env = BaseEnv(**base_env_cfg)

    env = RlGamesVecEnvWrapper(base_env, 'cuda:0', clip_obs, clip_actions)
    vecenv.register(
        'IsaacRlgWrapper',
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register('rlgpu', {
        'vecenv_type': 'IsaacRlgWrapper',
        'env_creator': lambda **kwargs: env,
    })

    uenv = base_env  # unwrapped, for current_demo_idx access

    print(f"\nStudent env: obs_dim={base_env.obs_dim}, action_dim={base_env.num_actions}")
    assert base_env.obs_dim == student_obs_dim, (
        f"Student env obs_dim ({base_env.obs_dim}) != max teacher obs_dim ({student_obs_dim}). "
        "Make sure the clips use the same object type and training config as the teachers."
    )

    # -----------------------------------------------------------------------
    # 3. Build student network
    # -----------------------------------------------------------------------
    student = StudentNetwork(student_obs_dim, action_dim, mlp_units).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)

    total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student network: {mlp_units} MLP, {total_params:,} parameters")

    # -----------------------------------------------------------------------
    # 4. Setup logging directories
    # -----------------------------------------------------------------------
    log_root = os.path.abspath(os.path.join('logs', 'distill', args.hand))
    run_dir = os.path.join(log_root, exp_name)
    ckpt_dir = os.path.join(run_dir, 'nn')
    param_dir = os.path.join(run_dir, 'params')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)

    # Save env config so eval_rl_games.py can load the student
    env_save_kwargs = base_env_cfg.copy()
    env_save_kwargs.pop('demo_data', None)
    env_save_kwargs.pop('retarget_data', None)
    env_save_kwargs.pop('all_demo_data', None)
    env_save_kwargs.pop('all_retarget_data', None)
    with open(os.path.join(param_dir, 'env.pkl'), 'wb') as f:
        pickle.dump(base_env_cfg, f)
    dump_yaml(os.path.join(param_dir, 'env.yaml'), env_save_kwargs)
    dump_yaml(os.path.join(param_dir, 'agent.yaml'), agent_cfg)
    with open(os.path.join(run_dir, 'demo_info.txt'), 'w') as f:
        f.write('\n'.join(demo_info_lines))

    print(f"\n{'='*60}")
    print(f"Distillation: {len(teachers)} teacher(s) -> 1 student")
    print(f"obs_dim={student_obs_dim}, action_dim={action_dim}")
    print(f"Horizon={args.horizon}, Epochs={args.num_epochs}, LR={args.learning_rate}")
    print(f"Saving to: {run_dir}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 5. Distillation training loop (DAgger-style: teacher drives the env)
    # -----------------------------------------------------------------------
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs['obs']

    running_loss = 0.0
    loss_count = 0

    for epoch in range(1, args.num_epochs + 1):
        obs_buf = []
        teacher_action_buf = []

        for step in range(args.horizon):
            # Which demo (and thus teacher) is currently active?
            demo_idx = int(uenv.current_demo_idx)
            teacher = teachers[demo_idx]

            # Pad obs if this teacher was trained with a smaller obs_dim
            t_obs = obs
            if teacher.obs_dim < student_obs_dim:
                pad = torch.zeros(obs.shape[0], student_obs_dim - teacher.obs_dim, device=device)
                t_obs = torch.cat([obs[:, :teacher.obs_dim], pad], dim=-1)

            with torch.inference_mode():
                teacher_actions = teacher(t_obs[:, :teacher.obs_dim])

            obs_buf.append(obs.clone())
            teacher_action_buf.append(teacher_actions.clone())

            # Advance the environment using teacher actions (DAgger)
            obs, _, dones, _ = env.step(teacher_actions)
            if isinstance(obs, dict):
                obs = obs['obs']

        # Stack collected rollout: (horizon * num_envs, obs_dim)
        batch_obs = torch.cat(obs_buf, dim=0)
        batch_teacher_actions = torch.cat(teacher_action_buf, dim=0)

        # Student forward pass
        student.train()
        student_actions = student.get_actions(batch_obs)
        loss = F.mse_loss(student_actions, batch_teacher_actions.detach())

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        optimizer.step()

        running_loss += loss.item()
        loss_count += 1

        if epoch % 50 == 0:
            avg_loss = running_loss / loss_count
            print(f"Epoch {epoch:5d} | loss={avg_loss:.6f} | demo={int(uenv.current_demo_idx)}")
            running_loss = 0.0
            loss_count = 0

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{args.hand}_ep{epoch}.pth")
            save_student_checkpoint(
                student, optimizer, epoch,
                student_obs_dim, action_dim, mlp_units,
                args.checkpoints, ckpt_path
            )

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"{args.hand}_final.pth")
    save_student_checkpoint(
        student, optimizer, args.num_epochs,
        student_obs_dim, action_dim, mlp_units,
        args.checkpoints, final_path
    )
    print(f"\nDone. Final student checkpoint: {final_path}")
    print(f"Evaluate with: python -m dexmachina.rl.eval_rl_games --checkpoint {final_path} ...")


if __name__ == '__main__':
    main()