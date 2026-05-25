import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401 — registers projection='3d'

def quat_angle_diff(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)  # radians

def quat_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def visualize_quats(q_robot, q_demo, title="Wrist orientation comparison", save_path=None):
    """Visualize two [w,x,y,z] quaternions as 3D coordinate frames side by side."""
    R_robot = quat_to_rotmat(q_robot)
    R_demo  = quat_to_rotmat(q_demo)
    diff_deg = np.degrees(quat_angle_diff(q_robot, q_demo))

    fig = plt.figure(figsize=(11, 5))
    fig.suptitle(f"{title}  |  angle diff = {diff_deg:.1f}°", fontsize=12)

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    labels = ['X', 'Y', 'Z']

    for col_idx, (R, subtitle) in enumerate([(R_robot, 'Robot'), (R_demo, 'Demo target')]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection='3d')
        for i, (col, lab) in enumerate(zip(colors, labels)):
            ax.quiver(0, 0, 0, R[0, i], R[1, i], R[2, i],
                      color=col, linewidth=3, label=lab, arrow_length_ratio=0.15)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_title(subtitle, fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    q1 = np.array([ 0.342,  0.044, -0.926, -0.151])
    q2 = np.array([ 0.116,  0.369, -0.423, -0.820])

    print(f"angle diff: {np.degrees(quat_angle_diff(q1, q2)):.2f}°")
    visualize_quats(q1, q2, title="Wrist orientation comparison")
