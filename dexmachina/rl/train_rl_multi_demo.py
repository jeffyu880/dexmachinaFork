import os 
import math
import yaml
import torch  
import wandb 
import shutil
import pickle
import argparse
import numpy as np
import genesis as gs
from datetime import datetime 

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner 

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True) 
    # make all the numpy arrays into lists, recursively
    def to_list(d):
        for k, v in d.items():
            if isinstance(v, dict):
                to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
    
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)

def load_multi_demo_data_separate(clip_list, args, device, exp_name, timestamp):
    """Load multiple demonstration clips separately.

    Builds base_env_cfg once from the first clip, then loads only demo/retarget
    data for the remaining clips without constructing a new BaseEnv per clip.
    """
    all_demo_data = []
    all_retarget_data = []
    base_env_cfg = None
    total_frames = 0

    demo_info_lines = [
        "=" * 80,
        "Multi-Demonstration Training Info",
        "=" * 80,
        f"Experiment: {exp_name}",
        f"Timestamp: {timestamp}",
        f"Hand: {args.hand}",
        f"Total demos: {len(clip_list)}",
        "",
        "Demonstrations used:",
        "-" * 80,
    ]

    args.multi_demo = True
    args.num_demos = len(clip_list)

    print(f"\n[INFO] Loading {len(clip_list)} demonstrations (separate)...")
    for i, clip in enumerate(clip_list, 1):
        print(f"  [{i}] Loading {clip}...")

        obj_name, start, end, subject_name, use_clip = parse_clip_string(clip)

        if i == 1:
            # Build full env config once from the first clip
            args.clip = clip
            args.arctic_object = obj_name
            args.frame_start = start
            args.frame_end = end
            env_cfg = get_all_env_cfg(args, device=device)
            args.randomize_observations = True
            base_env_cfg = {k: v for k, v in env_cfg.items() if k not in ['demo_data', 'retarget_data']}
            demo_data = env_cfg['demo_data']
            retarget_data = env_cfg['retarget_data']
        else:
            # Only load demo/retarget data for subsequent clips
            _, retarget_data = load_genesis_retarget_data(
                obj_name=obj_name,
                hand_name=args.hand,
                frame_start=start,
                frame_end=end,
                save_name=args.retarget_name,
                use_clip=use_clip,
                subject_name=subject_name,
            )
            
            demo_data = get_demo_data(
                obj_name=obj_name,
                hand_name=args.hand,
                frame_start=start,
                frame_end=end,
                use_clip=use_clip,
                subject_name=subject_name,
                load_retarget_contact=args.use_retarget_contact,
            )

        all_demo_data.append(demo_data)
        all_retarget_data.append(retarget_data)

        num_frames = int(end) - int(start)
        total_frames += num_frames
        demo_info_lines.append(f"{i}. {clip}")
        demo_info_lines.append(f"   Object: {obj_name}, Frames: {start}-{end} ({num_frames} frames)")
        demo_info_lines.append(f"   Subject: {subject_name}, Use_clip: {use_clip}")
        demo_info_lines.append("")
        print(f"       Loaded {num_frames} frames")

    print(f"[INFO] Loaded {len(all_demo_data)} separate demonstrations")
    print(f"[INFO] Demos loaded for epoch-level sampling")
    demo_info_lines.extend([
        "-" * 80,
        f"Total frames from all demos: {total_frames}",
        "=" * 80,
    ])

    # Package all demo references directly into env kwargs template.
    base_env_cfg['demo_data'] = all_demo_data[0]
    base_env_cfg['retarget_data'] = all_retarget_data[0]
    base_env_cfg['all_demo_data'] = all_demo_data
    base_env_cfg['all_retarget_data'] = all_retarget_data
    base_env_cfg['all_demo_names'] = list(clip_list)
    print("\n" + "\n".join(demo_info_lines))

    return all_demo_data, all_retarget_data, base_env_cfg, demo_info_lines


def main():
    parser = get_common_argparser() 
    # now add RL training args 
    parser.add_argument("--exp_name", "-exp", type=str, default="multi_demo", help="Experiment name.") 
    parser.add_argument("--clips", nargs='+', required=True, help="List of demonstration clips (e.g., 'ketchup-0-100-s01-u01 ketchup-0-100-s01-u02')")
    parser.add_argument("--horizon", '-ho', type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", '-ck', type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate for the agent.") 
    parser.add_argument("--wandb_project", "-wp", type=str, default="dexmachina", help="WandB project name.")
    parser.add_argument("--save_freq", "-sf", type=int, default=500)
    args = parser.parse_args()

    # Generate simplified experiment name: task_name_combined_MMDD_HHMMSS
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    
    # Extract task names from all clips
    task_names = []
    for clip in args.clips:
        obj_name, _, _, _, _ = parse_clip_string(clip)
        task_names.append(obj_name)
    
    # Create task name: combine unique object names
    task_name = "_".join(sorted(set(task_names)))
    exp_name = f"{task_name}_combined_{timestamp}"

    num_envs = args.num_envs   
    
    # Load multiple demonstrations
    all_demo_data, all_retarget_data, base_env_cfg, demo_info_lines = load_multi_demo_data_separate(
        args.clips, args, 'cuda:0', exp_name=exp_name, timestamp=timestamp
    )

    base_env_cfg['env_cfg']['use_rl_games'] = True
    env_kwargs = base_env_cfg
    
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    base_env = BaseEnv(**env_kwargs)
    
    # Now wrap the base environment for RL-Games
    env = base_env
    
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["name"] = args.hand
    
    log_root_path = os.path.join("logs", "rl_games", args.hand)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args.checkpoint is not None: 
        assert os.path.exists(args.checkpoint), f"Checkpoint file not found: {args.checkpoint}"
    
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    
    # register the environment to rl-games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    print("env_unwrapped: ", env.unwrapped.num_envs)
    print("args num_envs: ", args.num_envs)
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    if args.multi_demo:
        agent_cfg["params"]["config"]["mini_epochs"] = min(5, max(1, int(args.num_envs / 4096 * 5))) 
        # print("Mini epochs: ", agent_cfg["params"]["config"]["mini_epochs"])       
        # limiting the number of mini_epochs to 5 so that the policy does not drift too far when training with multiple demonstrations
    else:
        agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5))    
    agent_cfg["params"]["config"]["num_steps_per_env"] = args.horizon
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate
    agent_cfg["params"]["config"]["early_stop_reward_threshold"] = 0 if env_kwargs['env_cfg'].get('no_object', False) else 35
        
    env_save_kwargs = env_kwargs.copy()
    # pop the demo data and retargeted data (too large to save)
    env_save_kwargs.pop('demo_data')
    env_save_kwargs.pop('retarget_data')
    
    # convert agent_cfg to dict:
    wandb_cfg = agent_cfg.copy()
    wandb_cfg['env_kwargs'] = env_save_kwargs
    # also save args
    wandb_cfg['clips'] = args.clips
    wandb_cfg['hand'] = args.hand
    
    wandb.login(key="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")
    
    run = wandb.init(
        project=args.wandb_project, 
        config=wandb_cfg,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )

    # get wandb run name and id
    run_name = run.name
    run_id = run.id
    env_save_kwargs['wandb'] = dict(
        run_name=run_name,
        run_id=run_id,
    )
    
    # Save configuration
    ckpt_data_folder = os.path.join(log_root_path, f"{exp_name}")
    os.makedirs(ckpt_data_folder, exist_ok=True)
    
    param_folder = os.path.join(ckpt_data_folder, "params")
    os.makedirs(param_folder, exist_ok=True)
    
    with open(os.path.join(param_folder, "env.pkl"), "wb") as f:
        pickle.dump(env_kwargs, f)
    
    dump_yaml(os.path.join(param_folder, "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(param_folder, "agent.yaml"), agent_cfg)
    
    # Save demo info file
    demo_info_path = os.path.join(ckpt_data_folder, "demo_info.txt")
    with open(demo_info_path, "w") as f:
        f.write("\n".join(demo_info_lines))
    
    print(f"\n{'='*80}")
    print(f"Starting training with {len(args.clips)} demonstrations")
    print(f"Experiment: {exp_name}")
    print(f"Demo switching: On each episode reset ({args.demo_sampling} sampling)")
    print(f"Demo info saved to: {demo_info_path}")
    print("="*80)
    print(f"{'='*80}\n")
    
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint) 
    runner.run(runner_args)
    wandb.finish()

    # close the simulator
    exit()


if __name__ == "__main__":
    main() 
    exit()
