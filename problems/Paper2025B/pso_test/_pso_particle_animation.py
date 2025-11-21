import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simuopt.pso import PSOOptimizer

def create_pso_animation(pso: PSOOptimizer, 
                         xlabel='$x_0$', ylabel='$x_1$',
                         zlabel='Cost',
                         output_file='pso_animation.mp4', frames_per_iter=5):
    """创建 PSO 优化过程的动画
    """

    start_time = time.time()
    print("正在准备动画数据...")
    
    # 从 particle_history 中重建每次迭代的位置和成本
    df = pd.DataFrame(pso.particle_history)
    max_iter = df['iteration'].max()
    n_particles = pso.cfg.n_particles
    
    # 预先提取所有迭代的数据（避免每帧重复查询）
    positions_by_iter = []
    costs_by_iter = []
    particle_id = np.arange(n_particles) + 1
    for iter_idx in range(max_iter + 1):
        iter_data = df[df['iteration'] == iter_idx]
        pos = iter_data[[f'x{i}' for i in range(pso.cfg.n_dims)]].values
        cost = iter_data['cost'].values
        positions_by_iter.append(pos)
        costs_by_iter.append(cost)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('PSO Optimization Animation')
    ax.grid(True, linestyle='--', alpha=0.3)

    lb, ub = np.array(pso.cfg.lb), np.array(pso.cfg.ub)
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])

    # 初始化散点图
    scatter = ax.scatter([], [], c=[], s=80, alpha=1, 
                        edgecolors='black', linewidths=0.5, 
                        cmap='tab20', vmin=particle_id.min(), vmax=particle_id.max())

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(zlabel, rotation=270, labelpad=15)
    
    # 使用 tight_layout 自动调整布局，防止标签被截断
    fig.tight_layout()
    # ax.legend()
    
    def update_frame(frame):
        """更新动画帧"""
        # 计算当前迭代和子帧
        iter_idx = frame // frames_per_iter
        sub_frame = frame % frames_per_iter
        
        if iter_idx > max_iter:
            iter_idx = max_iter
        
        # 从预提取的数据中获取位置和成本
        positions = positions_by_iter[iter_idx]
        costs = costs_by_iter[iter_idx]
        
        # 如果有下一次迭代，进行插值以实现平滑动画
        if iter_idx < max_iter and sub_frame > 0:
            next_positions = positions_by_iter[iter_idx + 1]
            
            # 线性插值
            alpha = sub_frame / frames_per_iter
            positions = positions * (1 - alpha) + next_positions * alpha
        
        # 更新散点图
        scatter.set_offsets(positions)
        scatter.set_array(particle_id) # 使用粒子ID作为颜色映射
        
        # 更新标题
        ax.set_title(f'PSO Optimization Animation - Iteration {iter_idx}/{max_iter}')
        
        return scatter,  # 注意：必须返回 tuple 或 list（即使只有一个元素）
    
    # 创建动画
    total_frames = (max_iter + 1) * frames_per_iter
    print(f"正在创建动画 ({total_frames} 帧)...")
    
    ani = FuncAnimation(fig, update_frame, frames=total_frames, 
                        blit=True, repeat=True)
    
    # 保存动画
    output_path = pso.output_dir / output_file
    print(f"正在保存动画到 {output_path}...")
    # fps=20: 100帧在5秒播放完成
    # 如果想更快：fps=30 (3.3秒)，fps=50 (2秒)
    # 使用 ffmpeg writer 生成 mp4 视频，比 gif 体积小很多
    ani.save(output_path, writer='ffmpeg', fps=100, dpi=300, 
             bitrate=3600, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    print(f"动画已保存！")
    
    plt.close(fig)  # 关闭图形释放内存
    end_time = time.time()
    print(f"动画绘制耗时: {end_time - start_time:.2f} 秒")
    return ani
