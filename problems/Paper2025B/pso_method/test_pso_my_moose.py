import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
from simuopt.pso import PSOOptimizer, PSOConfig
from simuopt.moose_interface import MOOSEObjectiveFunction,MOOSEConfig
import time

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
    ax.legend()  # 只设置一次图例


    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    ax2.set_title('PSO Particle Trajectory')

    # 初始化散点图对象（将会被更新）
    prev_particles_scatter = ax2.scatter([], [], c='grey', s=25, alpha=0.3, 
                                   edgecolors='grey', linewidths=0.5, label='')

    particles_scatter = ax2.scatter([], [], c=[], s=80, alpha=1, 
                                   edgecolors='grey', linewidths=0.5, 
                                   cmap='coolwarm', label='Particles')
    

    best_scatter = ax2.scatter([], [], c='blue', s=120, marker='H', 
                              edgecolors='grey', linewidths=0.5, label='Global Best', zorder=5)
    
    # 添加 colorbar（初始为空，后续更新）
    cbar = plt.colorbar(particles_scatter, ax=ax2)
    cbar.set_label('Cost', rotation=270, labelpad=15)
    
    def pso_callback(optimizer, iteration):
        """PSO 回调函数"""
        out_dir = optimizer.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

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

        prev_particles_scatter.set_offsets(prev_position) if prev_position is not None else np.empty((0, 2))
        
        # 更新 colorbar 范围
        particles_scatter.set_clim(vmin=cost.min(), vmax=cost.max())
        
        # 更新最优位置
        best_scatter.set_offsets([[optimizer.swarm.best_pos[0], optimizer.swarm.best_pos[1]]])
        
        lb, ub = np.array(optimizer.cfg.lb), np.array(optimizer.cfg.ub)
        ax2.set_xlim(lb[0], ub[0])
        ax2.set_ylim(lb[1], ub[1])

        ax2.legend()  # 只设置一次图例
        
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
    moose_config = MOOSEConfig(
        work_dir='2d_particle',
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
        n_particles=8,
        max_iters=5,
        w=0.8,
        c1=1.5,
        c2=1.5,
        lb=[10, 0],
        ub=[2000, 4000],
        cmode='avg_cost',
        cthreshold=20,
        cwindow=20,
    )

    pso = PSOOptimizer(moose_func, pso_config, output_dir='2d_particle')
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


def create_pso_animation(pso, output_file='pso_animation.gif', frames_per_iter=5):
    """
    创建 PSO 优化过程的动画（优化版，不绘制等高线）
    
    Args:
        pso: PSO 优化器对象（已运行完成）
        func: 目标函数（可选，不再使用）
        output_file: 输出 GIF 文件名
        frames_per_iter: 每次迭代的帧数（用于平滑动画）
    """
    import pandas as pd
    
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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_title('PSO Optimization Animation')

    lb, ub = np.array(pso.cfg.lb), np.array(pso.cfg.ub)
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])

    # 初始化散点图
    scatter = ax.scatter([], [], c=[], s=100, alpha=0.8, 
                        edgecolors='black', linewidths=1, 
                        cmap='coolwarm', vmin=df['cost'].min(), vmax=df['cost'].max())

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cost', rotation=270, labelpad=15)
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
    ani = FuncAnimation(fig, update_frame, frames=total_frames, 
                       interval=50, blit=True, repeat=True)
    
    # 保存动画
    output_path = pso.output_dir / output_file
    print(f"正在保存动画到 {output_path}...")
    ani.save(output_path, writer='pillow', fps=20)
    print(f"动画已保存！")
    
    plt.close(fig)  # 关闭图形释放内存
    
    return ani


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
    create_pso_animation(pso, output_file='pso_animation.gif', frames_per_iter=5)
    end_time = time.time()
    print(f"动画绘制耗时: {end_time - start_time:.2f} 秒")