import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import numpy as np


def create_pso_callback(xlabel='$x_0$', ylabel='$x_1$', zlabel='Cost'):
    """
    创建PSO回调函数的工厂函数（优化版，避免闪烁）
    
    Args:
        save_dir: 保存图片的目录
    
    Returns:
        callback函数
    """
    import os

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(4, 3.6))

    # ax1.set_ylabel('Average Cost')
    # ax2.set_ylabel('Best Cost')

    ax2.set_xlabel('Iteration')
    ax1.set_title('PSO Iteration Progress')

    # 隐藏x轴刻度标签
    ax1.set_xticklabels([])

    # 初始化线条对象（仅创建一次）
    line_avg, = ax1.plot([], [], 'r-', linewidth=1, label='Average Cost')
    line_best, = ax1.plot([], [], 'b-', linewidth=1, label='Best Cost')
    line_best_log, = ax2.plot([], [], 'b-', linewidth=1, label='Best Cost (log)')

    # set log scale for y-axis
    ax2.set_yscale('log')

    ax1.legend(loc='upper right',fontsize=10,frameon=False)
    ax2.legend(loc='upper right',fontsize=10,frameon=False)

    fig3, ax3 = plt.subplots(figsize=(5.2, 4))
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)

    # 初始化散点图对象（将会被更新）
    marker_s_current = 80  # 当前粒子的 marker 大小
    marker_s_prev = 10      # 上一个位置粒子的 marker 大小    
    prev_particles_scatter = ax3.scatter([], [], c='white', s=marker_s_prev, alpha=1, 
                                   edgecolors="#C9C9C9FF", linewidths=0.5, label='Previous Position',zorder=2)

    particles_scatter = ax3.scatter([], [], c=[], s=marker_s_current, alpha=1, 
                                   edgecolors='black', linewidths=0.5, 
                                   cmap='coolwarm', norm=None, 
                                   label='Current Position',zorder=50)  # _r 表示反转颜色映射
    

    best_scatter = ax3.scatter([], [], c='orange', s=60, marker='*', 
                              edgecolors='black', linewidths=0.5, label='Global Best', zorder=100)
    
    # 图例放在标题下方，坐标区域上方，水平排列
    # columnspacing: 列之间的间距（默认2.0）
    # handletextpad: 图例标记和文字之间的间距（默认0.8）
    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), 
              ncol=3, fontsize=8, frameon=True, fancybox=True, shadow=False,
              columnspacing=0.7, handletextpad=0)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 添加 colorbar（初始为空，后续更新）
    cbar = plt.colorbar(particles_scatter, ax=ax3)
    # 设置为对数刻度，初始范围会在回调函数中更新
    # cbar = plt.colorbar(particles_scatter, ax=ax3, format=LogFormatter())
    cbar.set_label(zlabel, rotation=270, labelpad=15)
    
    
    def pso_callback(optimizer, iteration):
        """PSO 回调函数"""
        
        # nonlocal arrow_patches  # 使用 nonlocal 以便在内部函数中修改外部变量
        arrow_patches = []  # 存储箭头对象

        iteration = optimizer.iter_history['iteration']
        best_cost = optimizer.iter_history['best_cost']
        avg_cost = optimizer.iter_history['avg_cost']

        # 更新线条数据而不是重新绘制
        line_avg.set_data(iteration, avg_cost)
        line_best.set_data(iteration, best_cost)
        line_best_log.set_data(iteration, best_cost)
        
        # 自动调整坐标轴范围
        ax1.relim()
        ax2.relim()
        ax1.autoscale_view()
        ax2.autoscale_view()
        
        fig.tight_layout()
        fig.savefig(optimizer.output_dir / f"pso_iter_convergence.png", dpi=300)


        # 更新粒子位置散点图
        position = optimizer.swarm.position
        prev_position = optimizer.swarm.prev_position
        cost = optimizer.swarm.cost
        
        # 更新粒子位置和颜色（根据 cost）
        particles_scatter.set_offsets(position)
    
        particles_scatter.set_array(cost)  # 设置颜色映射的数据

        # 更新前一位置的散点图
        if prev_position is not None and len(prev_position) > 0:
            prev_particles_scatter.set_offsets(prev_position)
        else:
            prev_particles_scatter.set_offsets(np.empty((0, 2)))

        # 绘制新的箭头（从上一位置到当前位置）
        if prev_position is not None and len(prev_position) > 0:

            # 计算 marker 半径（单位：points）用于 shrinkA 和 shrinkB
            radius_current = np.sqrt(marker_s_current) / 2.0
            radius_prev = np.sqrt(marker_s_prev) / 2.0
            
            # 获取参数空间范围，用于计算相对距离阈值
            lb, ub = np.array(optimizer.cfg.lb), np.array(optimizer.cfg.ub)
            param_range = ub - lb
            # 距离阈值：参数空间对角线的 2%（可调整）
            distance_threshold = 0.02 * np.linalg.norm(param_range)
            
            for i in range(len(position)):
                prev_pos = prev_position[i]
                curr_pos = position[i]
                
                # 计算移动距离
                distance = np.linalg.norm(curr_pos - prev_pos)
                
                # 只有当距离超过阈值时才绘制箭头
                if distance > distance_threshold:
                    # 箭头大小控制参数
                    # mutation_scale: 控制箭头头部大小，值越小箭头越小（建议范围：3-15）
                    # linewidth: 控制箭头线条粗细，值越小线条越细（建议范围：0.3-1.0）
                    mut_scale = 6  # 原值：max(6.0, radius_current * 3.0)，改小使箭头更精致
                    arrow = FancyArrowPatch(
                        posA=(prev_pos[0], prev_pos[1]),
                        posB=(curr_pos[0], curr_pos[1]),
                        arrowstyle='-|>',
                        mutation_scale=mut_scale,
                        linewidth=0.3,  # 原值：0.5，改小使线条更细
                        color="#C9C9C9FF",
                        alpha=1,
                        shrinkA=radius_prev,
                        shrinkB=radius_current,
                        transform=ax3.transData,
                        zorder=1  # 设为1，确保箭头在散点下方
                    )
                    ax3.add_patch(arrow)
                    arrow_patches.append(arrow)
        
        # 更新 colorbar 范围（对数刻度）
        if len(cost) > 0:
            # 对数刻度需要确保最小值大于0
            vmin = min(cost.min(), 0)
            vmax = cost.max()
            particles_scatter.set_clim(vmin=vmin, vmax=vmax)
        
        # 更新最优位置
        best_scatter.set_offsets([[optimizer.swarm.best_pos[0], optimizer.swarm.best_pos[1]]])
        
        lb, ub = np.array(optimizer.cfg.lb), np.array(optimizer.cfg.ub)
        ax3.set_xlim(lb[0], ub[0])
        ax3.set_ylim(lb[1], ub[1])
        ax3.set_title(f'PSO Particle Trajectory (iteration:{iteration[-1]:3d})', pad=25)  # 增加标题与图例之间的间距

        
        fig3.tight_layout()

        particle_plot_out_dir = optimizer.output_dir / "particle_plots"
        particle_plot_out_dir.mkdir(parents=True, exist_ok=True)

        fig3.savefig(particle_plot_out_dir / f"pso_particle_trajectory_{iteration[-1]:04d}.png", dpi=300)

        # 清空箭头
        for arrow in arrow_patches:
            arrow.remove()
        arrow_patches.clear()

    return pso_callback
