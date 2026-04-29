"""
Analysis script to generate plots for evaluation results.
Compares agent's hand joints, keypoints, and object state against demonstrations.

Usage:
    python analyze_eval.py logs/rl_games/inspire_hand/box_combined_0323_150000_stage0/eval_ep0.npy

    NEEEEED TO TEST AND VERIFY!
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path
import pickle
import torch

# Add dexmachina to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dexmachina.envs.robot import get_hand_specific_cfg
from dexmachina.envs.constructors import parse_clip_string
from dex_retargeting.kinematics_adaptor import KinematicAdaptor


class EvalAnalyzer:
    def __init__(self, eval_data_path, env_pkl_path=None, hand_name='inspire_hand'):
        """
        Initialize analyzer with eval data.
        
        Args:
            eval_data_path: Path to eval_ep*.npy file
            env_pkl_path: Path to env.pkl for configuration
            hand_name: Hand model name (e.g., 'inspire_hand', 'dex3_hand')
        """
        self.eval_data_path = eval_data_path
        self.hand_name = hand_name
        
        # Load eval data
        self.eval_data = np.load(eval_data_path, allow_pickle=True).item()
        print(f"Loaded eval data from {eval_data_path}")
        print(f"  Keys: {self.eval_data.keys()}")
        print(f"  Shapes: {[(k, v.shape) for k, v in self.eval_data.items()]}")
        
        # Load environment config if provided
        self.env_cfg = None
        if env_pkl_path and os.path.exists(env_pkl_path):
            with open(env_pkl_path, 'rb') as f:
                env_kwargs = pickle.load(f)
                self.env_cfg = env_kwargs.get('env_cfg', {})
                print(f"Loaded env config from {env_pkl_path}")
        
        # Get hand configuration
        self.hand_cfg = get_hand_specific_cfg(hand_name)
        print(f"Hand config: {hand_name}")
        
        # Create output directory
        self.output_dir = Path(eval_data_path).parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def plot_object_state_comparison(self):
        """Plot comparison of agent vs demo object state over time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Object State Comparison: Agent vs Demo', fontsize=16, fontweight='bold')
        
        obj_state = self.eval_data['obj_state']  # (T, 8) - pos(3) + quat(4) + arti(1)
        demo_state = self.eval_data['demo_state']  # (T, 8)
        
        T = obj_state.shape[0]
        timesteps = np.arange(T)
        
        # Position comparison
        ax = axes[0, 0]
        ax.plot(timesteps, obj_state[:, 0], 'b-', label='Agent X', linewidth=2)
        ax.plot(timesteps, demo_state[:, 0], 'r--', label='Demo X', linewidth=2)
        ax.set_ylabel('X Position (m)')
        ax.set_title('Position - X Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(timesteps, obj_state[:, 1], 'b-', label='Agent Y', linewidth=2)
        ax.plot(timesteps, demo_state[:, 1], 'r--', label='Demo Y', linewidth=2)
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Position - Y Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position error
        ax = axes[1, 0]
        pos_error = np.linalg.norm(obj_state[:, :3] - demo_state[:, :3], axis=1)
        ax.plot(timesteps, pos_error, 'g-', linewidth=2)
        ax.fill_between(timesteps, pos_error, alpha=0.3, color='g')
        ax.set_ylabel('Error (m)')
        ax.set_xlabel('Timestep')
        ax.set_title('Position Error (L2 norm)')
        ax.grid(True, alpha=0.3)
        
        # Quaternion error (using the rot_dist if available)
        ax = axes[1, 1]
        if 'rot_dist' in self.eval_data:
            rot_dist = self.eval_data['rot_dist'].squeeze()
            ax.plot(timesteps, rot_dist, 'orange', linewidth=2)
            ax.fill_between(timesteps, rot_dist, alpha=0.3, color='orange')
            ax.set_ylabel('Rotation Error')
            ax.set_title('Rotation Distance')
        else:
            # Compute quaternion distance manually
            quat_agent = obj_state[:, 3:7]
            quat_demo = demo_state[:, 3:7]
            quat_error = 2 * np.arccos(np.clip(np.abs(np.sum(quat_agent * quat_demo, axis=1)), 0, 1))
            ax.plot(timesteps, quat_error, 'orange', linewidth=2)
            ax.fill_between(timesteps, quat_error, alpha=0.3, color='orange')
            ax.set_ylabel('Quaternion Distance (rad)')
            ax.set_title('Rotation Distance')
        ax.set_xlabel('Timestep')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'object_state_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_distance_metrics(self):
        """Plot distance metrics over time."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Distance Metrics Over Time', fontsize=16, fontweight='bold')
        
        T = self.eval_data['obj_state'].shape[0]
        timesteps = np.arange(T)
        
        metrics = ['pos_dist', 'rot_dist', 'arti_dist']
        colors = ['blue', 'red', 'green']
        titles = ['Position Distance', 'Rotation Distance', 'Articulation Distance']
        
        for ax, metric, color, title in zip(axes, metrics, colors, titles):
            if metric in self.eval_data:
                data = self.eval_data[metric].squeeze()
                if data.ndim > 1:
                    # If multi-environment, take mean
                    data = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
                
                ax.plot(timesteps, data, color=color, linewidth=2)
                ax.fill_between(timesteps, data, alpha=0.3, color=color)
                ax.set_title(title)
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Distance')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(data)
                max_val = np.max(data)
                ax.axhline(mean_val, color=color, linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.4f}')
                ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / 'distance_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_object_articulation(self):
        """Plot object articulation state (joint positions)."""
        obj_state = self.eval_data['obj_state']  # Last dimension is articulation
        demo_state = self.eval_data['demo_state']
        
        T = obj_state.shape[0]
        timesteps = np.arange(T)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Object typically has 1 articulation DOF
        if obj_state.shape[1] >= 8:
            agent_arti = obj_state[:, 7]
            demo_arti = demo_state[:, 7]
            
            ax.plot(timesteps, agent_arti, 'b-', label='Agent', linewidth=2)
            ax.plot(timesteps, demo_arti, 'r--', label='Demo', linewidth=2)
            ax.fill_between(timesteps, agent_arti, demo_arti, alpha=0.2, color='gray')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Joint Position (rad)')
            ax.set_title('Object Articulation (Joint Position)')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / 'object_articulation.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
    
    def plot_summary_statistics(self):
        """Plot summary statistics across the episode."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Summary Statistics', fontsize=16, fontweight='bold')
        
        T = self.eval_data['obj_state'].shape[0]
        timesteps = np.arange(T)
        
        # Position error distribution
        obj_state = self.eval_data['obj_state']
        demo_state = self.eval_data['demo_state']
        pos_error = np.linalg.norm(obj_state[:, :3] - demo_state[:, :3], axis=1)
        
        ax = axes[0, 0]
        ax.hist(pos_error, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(pos_error), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pos_error):.4f}')
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Cumulative error
        ax = axes[0, 1]
        cum_error = np.cumsum(pos_error)
        ax.plot(timesteps, cum_error, 'b-', linewidth=2)
        ax.fill_between(timesteps, cum_error, alpha=0.3, color='blue')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cumulative Error (m)')
        ax.set_title('Cumulative Position Error')
        ax.grid(True, alpha=0.3)
        
        # All metrics comparison
        ax = axes[1, 0]
        metrics_data = {}
        for metric in ['pos_dist', 'rot_dist', 'arti_dist']:
            if metric in self.eval_data:
                data = self.eval_data[metric].squeeze()
                if data.ndim > 1:
                    data = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
                metrics_data[metric] = np.mean(data)
        
        if metrics_data:
            ax.bar(metrics_data.keys(), metrics_data.values(), color=['blue', 'red', 'green'], alpha=0.7)
            ax.set_ylabel('Mean Distance')
            ax.set_title('Mean Distance Metrics')
            ax.grid(True, alpha=0.3, axis='y')
            for i, (k, v) in enumerate(metrics_data.items()):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Episode summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
Episode Summary
{'='*40}
Duration: {T} timesteps
Position Error:
  Mean: {np.mean(pos_error):.4f} m
  Max: {np.max(pos_error):.4f} m
  Min: {np.min(pos_error):.4f} m
  Std: {np.std(pos_error):.4f} m

Final Position Error: {pos_error[-1]:.4f} m
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / 'summary_statistics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_agent_vs_demo_trajectories(self):
        """Plot 3D trajectories of object center in world frame."""
        obj_state = self.eval_data['obj_state']
        demo_state = self.eval_data['demo_state']
        
        fig = plt.figure(figsize=(14, 6))
        
        # XY plane view
        ax1 = fig.add_subplot(121)
        ax1.plot(obj_state[:, 0], obj_state[:, 1], 'b-', label='Agent', linewidth=2, alpha=0.7)
        ax1.plot(demo_state[:, 0], demo_state[:, 1], 'r--', label='Demo', linewidth=2, alpha=0.7)
        ax1.scatter(obj_state[0, 0], obj_state[0, 1], color='blue', s=100, marker='o', label='Agent Start', zorder=5)
        ax1.scatter(obj_state[-1, 0], obj_state[-1, 1], color='blue', s=100, marker='s', label='Agent End', zorder=5)
        ax1.scatter(demo_state[0, 0], demo_state[0, 1], color='red', s=100, marker='o', alpha=0.5, zorder=5)
        ax1.scatter(demo_state[-1, 0], demo_state[-1, 1], color='red', s=100, marker='s', alpha=0.5, zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Object Trajectory - XY Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # YZ plane view
        ax2 = fig.add_subplot(122)
        ax2.plot(obj_state[:, 1], obj_state[:, 2], 'b-', label='Agent', linewidth=2, alpha=0.7)
        ax2.plot(demo_state[:, 1], demo_state[:, 2], 'r--', label='Demo', linewidth=2, alpha=0.7)
        ax2.scatter(obj_state[0, 1], obj_state[0, 2], color='blue', s=100, marker='o', label='Agent Start', zorder=5)
        ax2.scatter(obj_state[-1, 1], obj_state[-1, 2], color='blue', s=100, marker='s', label='Agent End', zorder=5)
        ax2.scatter(demo_state[0, 1], demo_state[0, 2], color='red', s=100, marker='o', alpha=0.5, zorder=5)
        ax2.scatter(demo_state[-1, 1], demo_state[-1, 2], color='red', s=100, marker='s', alpha=0.5, zorder=5)
        ax2.set_xlabel('Y (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Object Trajectory - YZ Plane')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        save_path = self.output_dir / 'object_trajectories.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_add3_metric(self):
        """Plot AUC-ADD3 metric if available."""
        if 'add_errors' not in self.eval_data or 'add3_errors' not in self.eval_data:
            print("   ADD3 metric not found in eval data")
            return
        
        T = self.eval_data['obj_state'].shape[0]
        timesteps = np.arange(T)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('AUC-ADD3 Metric: Object Pose Estimation Accuracy', fontsize=16, fontweight='bold')
        
        # ADD errors over time
        ax = axes[0, 0]
        add_errors = self.eval_data['add_errors']
        ax.plot(timesteps, add_errors, 'b-', linewidth=2)
        ax.fill_between(timesteps, add_errors, alpha=0.3, color='blue')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('ADD Error (m)')
        ax.set_title('Average Distance of Model Points (ADD)')
        ax.grid(True, alpha=0.3)
        
        # ADD3 scores over time
        ax = axes[0, 1]
        add3_errors = self.eval_data['add3_errors']
        ax.plot(timesteps, add3_errors, 'g-', linewidth=2)
        ax.fill_between(timesteps, add3_errors, alpha=0.3, color='green')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('ADD-3 Score')
        ax.set_title('ADD-3 Accuracy (Multi-threshold)')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # ADD error distribution
        ax = axes[1, 0]
        ax.hist(add_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(add_errors), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(add_errors):.4f} m')
        ax.set_xlabel('ADD Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('ADD Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # AUC-ADD3 summary
        ax = axes[1, 1]
        ax.axis('off')
        auc_add3 = self.eval_data.get('auc_add3', None)
        summary_text = f"""
AUC-ADD3 Metric Summary
{'='*40}
"""
        if auc_add3 is not None:
            summary_text += f"""
AUC-ADD3 Score: {auc_add3:.6f}

ADD Statistics:
  Mean: {np.mean(add_errors):.6f} m
  Max:  {np.max(add_errors):.6f} m
  Min:  {np.min(add_errors):.6f} m
  Std:  {np.std(add_errors):.6f} m

ADD-3 Statistics:
  Mean: {np.mean(add3_errors):.6f}
  Max:  {np.max(add3_errors):.6f}
  Min:  {np.min(add3_errors):.6f}

Interpretation:
  - Higher AUC-ADD3 is better
  - Lower ADD error is better
  - Range: 0.0 (worst) to 1.0 (best)
        """
        else:
            summary_text += "AUC-ADD3 score not computed"
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / 'add3_metric.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all analysis plots."""
        print("\n" + "="*60)
        print("Generating Analysis Plots")
        print("="*60 + "\n")
        
        try:
            print("1. Plotting object state comparison...")
            self.plot_object_state_comparison()
        except Exception as e:
            print(f"   Warning: {e}")
        
        try:
            print("2. Plotting distance metrics...")
            self.plot_distance_metrics()
        except Exception as e:
            print(f"   Warning: {e}")
        
        try:
            print("3. Plotting object articulation...")
            self.plot_object_articulation()
        except Exception as e:
            print(f"   Warning: {e}")
        
        try:
            print("4. Plotting summary statistics...")
            self.plot_summary_statistics()
        except Exception as e:
            print(f"   Warning: {e}")
        
        try:
            print("5. Plotting object trajectories...")
            self.plot_agent_vs_demo_trajectories()
        except Exception as e:
            print(f"   Warning: {e}")
        
        try:
            print("6. Plotting AUC-ADD3 metric...")
            self.plot_add3_metric()
        except Exception as e:
            print(f"   Warning: {e}")
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results and generate plots')
    parser.add_argument('eval_data_path', type=str, help='Path to eval_ep*.npy file')
    parser.add_argument('--env_pkl', '-pkl', type=str, default=None, 
                       help='Path to env.pkl file for configuration')
    parser.add_argument('--hand', type=str, default='inspire_hand',
                       help='Hand model name (e.g., inspire_hand, dex3_hand)')
    
    args = parser.parse_args()
    
    # If eval_pkl not provided, try to infer it from eval_data_path
    env_pkl_path = args.env_pkl
    if not env_pkl_path:
        # Try common location: parent_dir/params/env.pkl
        eval_path = Path(args.eval_data_path)
        inferred_pkl = eval_path.parent.parent / 'params' / 'env.pkl'
        if inferred_pkl.exists():
            env_pkl_path = str(inferred_pkl)
            print(f"Inferred env.pkl path: {env_pkl_path}")
    
    analyzer = EvalAnalyzer(args.eval_data_path, env_pkl_path, args.hand)
    analyzer.generate_all_plots()


if __name__ == '__main__':
    main()
