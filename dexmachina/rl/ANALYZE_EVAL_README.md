# Evaluation Analysis Script

This script analyzes RL training evaluation results and generates comprehensive visualization plots comparing agent performance against demonstrations.

## Features

The analysis script generates the following plots:

### 1. **Object State Comparison** (`object_state_comparison.png`)
- **Position XY/Z axis**: Agent vs demo position over time
- **Position error**: L2 norm of position difference
- **Rotation distance**: Quaternion distance metric

### 2. **Distance Metrics** (`distance_metrics.png`)
- **Position distance**: L2 norm error in object position
- **Rotation distance**: Angular distance between agent and demo rotations
- **Articulation distance**: Error in object joint positions

### 3. **Object Articulation** (`object_articulation.png`)
- Tracks object joint angle over time
- Compares agent-controlled state vs demonstration

### 4. **Object Trajectories** (`object_trajectories.png`)
- 3D visualization of object center trajectory
- XY and YZ plane views
- Shows start/end positions for both agent and demo

### 5. **Summary Statistics** (`summary_statistics.png`)
- Position error distribution (histogram)
- Cumulative position error over episode
- Mean values of all metrics
- Episode summary text box

## Usage

### Basic Usage
```bash
python dexmachina/rl/analyze_eval.py logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/eval_ep0.npy
```

### With Explicit Configuration Path
```bash
python dexmachina/rl/analyze_eval.py \
  logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/eval_ep0.npy \
  --env_pkl logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/params/env.pkl \
  --hand inspire_hand
```

### Supported Hand Models
```bash
--hand inspire_hand      # Default
--hand dex3_hand
--hand schunk_hand
--hand allegro_hand
--hand xhand
```

## Output

All plots are saved to a directory next to the eval data:
```
logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/
├── eval_ep0.npy
└── analysis/
    ├── object_state_comparison.png
    ├── distance_metrics.png
    ├── object_articulation.png
    ├── object_trajectories.png
    └── summary_statistics.png
```

## Typical Workflow

### 1. Run Evaluation
```bash
python dexmachina/rl/eval_rl_games.py \
  --checkpoint logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/model_ep5000.pth \
  --eval_episodes 1 \
  --show_reference \
  --record_video
```

This saves `eval_ep0.npy` with the evaluation data.

### 2. Analyze Results
```bash
python dexmachina/rl/analyze_eval.py \
  logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/eval_ep0.npy
```

### 3. Review Generated Plots
Open the PNG files in the `analysis/` directory to inspect:
- How well the agent follows the demonstration
- Where position/rotation errors occur
- Overall episode trajectory

## Data Structure

The eval data (`.npy` file) contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `obj_state` | (T, 8) | Agent's object state: pos(3) + quat(4) + articulation(1) |
| `demo_state` | (T, 8) | Reference demo object state: pos(3) + quat(4) + articulation(1) |
| `pos_dist` | (T,) or (T, num_envs) | L2 distance in object position |
| `rot_dist` | (T,) or (T, num_envs) | Rotation distance (quaternion) |
| `arti_dist` | (T,) or (T, num_envs) | Object joint position error |

Where:
- **T**: Number of timesteps in episode
- **pos**: Position (x, y, z) in meters
- **quat**: Quaternion (x, y, z, w) for rotation
- **articulation**: Object joint angle in radians

## Interpretation Guide

### Good Performance
- Position error stays small and consistent (< 0.05 m)
- Agent trajectory closely overlaps with demo
- Rotation distance remains low
- Early convergence with small variance

### Poor Performance
- Large position errors (> 0.1 m)
- Agent trajectory diverges from demo
- High rotation/articulation errors
- Errors accumulate over time

## Notes

- The script automatically infers the `env.pkl` path if not provided
- Uses non-interactive matplotlib backend ('Agg') for headless environments
- Handles multi-environment eval data by averaging metrics
- All plots are saved at 150 DPI for high quality

