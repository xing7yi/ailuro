import matplotlib.pyplot as plt
import numpy as np

def plot_particle_trajectory(iteration, positions):
    """
    绘制粒子在二维空间中的轨迹

    参数:
    positions: np.ndarray, 形状为 (n_steps, 2)，表示粒子在每个时间步的位置
    title: str, 图表标题
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1], marker='o')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)
    ax.axis('equal')
    fig.savefig(f"particle_trajectory_iter_{iteration:04d}.png", dpi=300)