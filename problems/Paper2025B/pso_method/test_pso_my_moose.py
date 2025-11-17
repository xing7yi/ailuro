import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import yaml
from simuopt.pso import PSOOptimizer, PSOConfig
from simuopt.moose_interface import MOOSEObjectiveFunction,MOOSEConfig
import time

xlabel = 'Yield Strength (MPa)'
ylabel = 'Tangent Modulus (MPa)'
zlabel = 'RMSE'


def create_pso_animation(pso, output_file='pso_animation.gif', frames_per_iter=5):
    """
    创建 PSO 优化过程的动画（优化版，不绘制等高线）
    
    Args:
        pso: PSO 优化器对象（已运行完成）
        func: 目标函数（可选，不再使用）
        output_file: 输出 GIF 文件名
        frames_per_iter: 每次迭代的帧数（用于平滑动画）
    """

    
    print("正在准备动画数据...")
    
    # 从 particle_history 中重建每次迭代的位置和成本
    df = pd.DataFrame(pso.particle_history)
    max_iter = df['iteration'].max()
    n_particles = pso.cfg.n_particles
    
    # 预先提取所有迭代的数据（避免每帧重复查询）
    positions_by_iter = []
    costs_by_iter = []
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
    scatter = ax.scatter([], [], c=[], s=80, alpha=0.6, 
                        edgecolors='black', linewidths=0.5, 
                        cmap='coolwarm', vmin=df['cost'].min(), vmax=df['cost'].max())

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
        scatter.set_array(costs)
        
        # 更新全局最优位置
        # best_pos = pso.iter_history['best_pos'][min(iter_idx, len(pso.iter_history['best_pos']) - 1)]
        # best_scatter.set_offsets([best_pos])
        
        # 更新标题
        ax.set_title(f'PSO Optimization Animation - Iteration {iter_idx}/{max_iter}')
        
        return scatter,  # 注意：必须返回 tuple 或 list（即使只有一个元素）
    
    # 创建动画
    total_frames = (max_iter + 1) * frames_per_iter
    print(f"正在创建动画 ({total_frames} 帧)...")
    
    # 100次迭代，frames_per_iter=1时，总共约100帧
    # 要在5秒内播放完，fps需要设置为 100/5 = 20
    ani = FuncAnimation(fig, update_frame, frames=total_frames, 
                       interval=10, blit=True, repeat=True)
    
    # 保存动画
    output_path = pso.output_dir / output_file
    print(f"正在保存动画到 {output_path}...")
    # fps=20: 100帧在5秒播放完成
    # 如果想更快：fps=30 (3.3秒)，fps=50 (2秒)
    ani.save(output_path, writer='pillow', fps=20 , dpi=300)
    print(f"动画已保存！")
    
    plt.close(fig)  # 关闭图形释放内存
    
    return ani


def create_pso_callback(save_dir=None):
    """
    创建PSO回调函数的工厂函数（优化版，避免闪烁）
    
    Args:
        save_dir: 保存图片的目录
    
    Returns:
        callback函数
    """
    import os

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('PSO Optimization Progress')

    # 初始化线条对象（仅创建一次）
    line_best, = ax.plot([], [], 'b-', label='Best Cost')
    line_avg, = ax.plot([], [], 'r-', label='Average Cost')
    ax.legend(loc='upper right',fontsize=10)  # 只设置一次图例


    fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title('PSO Particle Trajectory', pad=25)  # 增加标题与图例之间的间距

    # 初始化散点图对象（将会被更新）
    marker_s_current = 80  # 当前粒子的 marker 大小
    marker_s_prev = 25      # 上一个位置粒子的 marker 大小    
    prev_particles_scatter = ax2.scatter([], [], c='grey', s=marker_s_prev, alpha=0.5, 
                                   edgecolors='grey', linewidths=0.5, label='Previous Position')

    particles_scatter = ax2.scatter([], [], c=[], s=marker_s_current, alpha=0.6, 
                                   edgecolors='black', linewidths=0.5, 
                                   cmap='coolwarm', label='Current Position')  # _r 表示反转颜色映射

    best_scatter = ax2.scatter([], [], c='orange', s=150, marker='*', 
                              edgecolors='black', linewidths=0.5, label='Global Best', zorder=10)
    
    # 图例放在标题下方，坐标区域上方，水平排列
    # columnspacing: 列之间的间距（默认2.0）
    # handletextpad: 图例标记和文字之间的间距（默认0.8）
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), 
              ncol=3, fontsize=8, frameon=True, fancybox=True, shadow=False,
              columnspacing=0.7, handletextpad=0)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 添加 colorbar（初始为空，后续更新）
    cbar = plt.colorbar(particles_scatter, ax=ax2)
    cbar.set_label(zlabel, rotation=270, labelpad=15)
    
    # 存储箭头对象列表（用于更新）
    arrow_patches = []
    
    # 标记是否已清理过PNG文件
    png_cleaned = False
    
    def pso_callback(optimizer, iteration):
        """PSO 回调函数"""
        nonlocal arrow_patches, png_cleaned  # 使用外部作用域的箭头列表和清理标记
        
        out_dir = optimizer.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 第一次调用时清理旧的PNG文件
        if not png_cleaned:
            import glob
            from pathlib import Path
            png_pattern = out_dir / "pso_particle_trajectory_*.png"
            old_pngs = glob.glob(str(png_pattern))
            for old_png in old_pngs:
                try:
                    Path(old_png).unlink()
                except Exception:
                    pass  # 忽略删除错误
            if old_pngs:
                print(f"已清理 {len(old_pngs)} 个旧PNG文件")
            png_cleaned = True

        iteration = optimizer.iter_history['iteration']
        best_cost = optimizer.iter_history['best_cost']
        avg_cost = optimizer.iter_history['avg_cost']

        # 更新线条数据而不是重新绘制
        line_best.set_data(iteration, best_cost)
        line_avg.set_data(iteration, avg_cost)
        
        # 自动调整坐标轴范围
        ax.relim()
        ax.autoscale_view()
        
        fig.tight_layout()
        fig.savefig(out_dir / f"pso_iter_convergence.png", dpi=300)


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
        
        # 清除旧的箭头
        for arrow in arrow_patches:
            arrow.remove()
        arrow_patches.clear()
        
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
                        color='gray',
                        alpha=0.5,
                        shrinkA=radius_prev,
                        shrinkB=radius_current,
                        transform=ax2.transData,
                        zorder=3
                    )
                    ax2.add_patch(arrow)
                    arrow_patches.append(arrow)
        
        # 更新 colorbar 范围
        if len(cost) > 0:
            particles_scatter.set_clim(vmin=cost.min(), vmax=cost.max())
        
        # 更新最优位置
        best_scatter.set_offsets([[optimizer.swarm.best_pos[0], optimizer.swarm.best_pos[1]]])
        
        lb, ub = np.array(optimizer.cfg.lb), np.array(optimizer.cfg.ub)
        ax2.set_xlim(lb[0], ub[0])
        ax2.set_ylim(lb[1], ub[1])

        
        fig2.tight_layout()
        fig2.savefig(out_dir / f"pso_particle_trajectory_{iteration[-1]:04d}.png", dpi=300)
    
    return pso_callback

def rastrigin_2d(x):
    x1, x2 = x
    A = 10
    return (
        A * 2 + 
        (x1**2 - A * np.cos(2 * np.pi * x1)) + 
        (x2**2 - A * np.cos(2 * np.pi * x2)) 
        ,'success'
    )


def moose_eval():

    work_dir = 'sim_validation_biso'
    moose_config = MOOSEConfig(
        work_dir=work_dir,
        input_file='particle_friction_plastic_voce.i',
        param_names=['p0','p1'],
        param_paths=['p0','p1'],
        filebase_param='out_name',
        csv_col_x='cmpr_ratio',
        csv_col_y='force_norm_MPa',
        reference_csv='obj_csv/E100_p0_600_p1_200_p2_300_p3_50.csv',
    )

    moose_func = MOOSEObjectiveFunction(moose_config)

    moose_func.mode = 'multithreading'

    pso_config = PSOConfig(
        n_dims=2,
        n_particles=25,
        max_iters=200,
        w=0.8,
        c1=1.5,
        c2=1.5,
        lb=[10, 0],
        ub=[2000, 4000],
        cmode='avg_cost',
        cthreshold=100,
        cwindow=10,
    )
    callback = create_pso_callback()
    pso = PSOOptimizer(moose_func, pso_config, 
                       output_dir=work_dir, 
                       callback=callback)
    return pso


def test_eval():
    pso_config_rastrigin = PSOConfig(
        n_dims=2,
        n_particles=20,
        max_iters=150,
        w=0.7,
        c1=1.5,
        c2=1.5,
        lb=[-5.12, -5.12],
        ub=[5.12, 5.12],
        cmode='avg_cost',
        cthreshold=1,
        cwindow=30,
    )
    callback = create_pso_callback()
    pso = PSOOptimizer(rastrigin_2d, pso_config_rastrigin, 
                       output_dir='test_rastrigin_2d', 
                       callback=callback,callback_interval=20)
    return pso

if __name__ == "__main__":

    eval_method = 'moose'
    if eval_method == 'moose':
        pso = moose_eval()
    elif eval_method == 'test':
        pso = test_eval()
    pso.run()
    # 输出结果到yaml文件
    result = {
        'total evaluations': pso.particle_history['eval_id'][-1] + 1,
        'total iterations': pso.iter_history['iteration'][-1] + 1,
        'total elapsed time (s)': pso.elapsed_time,
        'best pos': pso.swarm.best_pos.tolist(),
        'best cost': float(pso.swarm.best_cost),  # 转换为 Python float
        'best eval_id': int(pso.swarm.best_id)   # 转换为 Python int
    }
    with open(pso.output_dir / 'result.yaml', 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    # print(result)

    # 绘制gif动画（优化版，不需要传入 func）
    start_time = time.time()
    # frames_per_iter=1: 不使用插值，每次迭代1帧
    # 100次迭代 = 100帧，fps=20意味着5秒播放完成
    create_pso_animation(pso, output_file='pso_animation.gif', frames_per_iter=1)
    end_time = time.time()
    print(f"动画绘制耗时: {end_time - start_time:.2f} 秒")