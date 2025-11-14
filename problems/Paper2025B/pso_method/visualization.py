"""
PSO 可视化模块

该模块包含 PSO 优化过程的可视化函数。

主要功能：
- plot_particle_distribution: 绘制粒子在参数空间中的分布（2D）
- plot_convergence_curve: 绘制收敛曲线
- generate_particle_animation: 生成粒子运动轨迹的 GIF 动画
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from glob import glob


def plot_particle_distribution(iteration_eval, iteration_state, swarm,  
                               axis_labels, axis_bounds, output_dir):
    """
    绘制粒子在2D参数空间中的分布图（使用新的数据类简化参数）
    
    参数：
        iteration_eval: IterationEvaluation，当前迭代的评估数据
        iteration_state: IterationStateSummary，当前迭代的状态信息
        swarm: SwarmState，粒子群状态（用于获取上次位置绘制轨迹）    
        axis_labels: List[str]，参数轴标签
        axis_bounds: tuple (lower_bounds, upper_bounds)，参数边界
        output_dir: Path，输出目录
    
    返回：
        scatter_file: Path，保存的图片文件路径
    """
    lower_bounds, upper_bounds = axis_bounds
    xmin, xmax = lower_bounds[0], upper_bounds[0]
    ymin, ymax = lower_bounds[1], upper_bounds[1]

    x_label, y_label = axis_labels

    # 从 iteration_eval 提取粒子位置和成本
    x_values = iteration_eval.parameters[:, 0]
    y_values = iteration_eval.parameters[:, 1]
    costs = iteration_eval.costs


    
    # 使用固定布局参数，避免尺寸变化
    fig = plt.figure(figsize=(8, 6))
    # 手动设置子图位置：左、下、宽、高（相对于figure的比例）
    # 为 colorbar 预留右侧空间
    ax_scatter = fig.add_axes([0.10, 0.15, 0.70, 0.70])  # [left, bottom, width, height]
    

    # 绘制轨迹和箭头（从上一个位置到当前位置）
    marker_s_current = 100  # 当前粒子的 marker 大小
    marker_s_prev = 20      # 上一个位置粒子的 marker 大小
    
    # 计算 marker 半径（单位：points）用于 FancyArrowPatch 的 shrinkA 和 shrinkB
    radius_current = np.sqrt(marker_s_current) / 2.0  # 当前位置粒子半径
    radius_prev = np.sqrt(marker_s_prev) / 2.0        # 上一个位置粒子半径

    # 使用scatter绘制当前位置，颜色表示cost
    scatter = ax_scatter.scatter(x_values, y_values, c=costs, 
                                cmap='viridis', s=marker_s_current, alpha=0.6, 
                                edgecolors='black', linewidth=0.5, zorder=5)
    
    # 绘制上一个位置（小点）和运动轨迹箭头
    if swarm.prev_position is not None:
        x_values_prev = swarm.prev_position[:, 0]
        y_values_prev = swarm.prev_position[:, 1]
        ax_scatter.scatter(x_values_prev, y_values_prev, c='gray', s=marker_s_prev, alpha=0.5, zorder=2)
        
        # 绘制箭头（显示粒子运动轨迹）
        for i in range(len(x_values)):
            prev_pos = (x_values_prev[i], y_values_prev[i])
            curr_pos = (x_values[i], y_values[i])
            
            mut_scale = max(6.0, radius_current * 3.0)
            arr = FancyArrowPatch(
                posA=prev_pos,
                posB=curr_pos,
                arrowstyle='->',
                mutation_scale=mut_scale,
                linewidth=0.5,
                color='gray',
                alpha=0.5,
                shrinkA=radius_prev,
                shrinkB=radius_current,
                transform=ax_scatter.transData,
                zorder=3
            )
            ax_scatter.add_patch(arr)


    # 标记全局最优点（从 swarm 获取）
    ax_scatter.scatter(swarm.best_pos[0], swarm.best_pos[1], 
                      marker='*', s=250, c='red', edgecolors='black', 
                      linewidth=0.5, 
                      label=f'Global Best (cost={swarm.best_cost:.2f})', 
                      zorder=10)
    
    # 标记本迭代最优点（从 iteration_state 获取）
    ax_scatter.scatter(iteration_state.iter_best_params[0], iteration_state.iter_best_params[1], 
                      marker='D', s=80, c='orange', edgecolors='black', 
                      linewidth=0.5, 
                      label=f'Iter Best (cost={iteration_state.iter_best_cost:.2f})', 
                      zorder=9)
    
    # 添加颜色条并设置标签
    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label('Objective (Cost)', rotation=270, labelpad=15)
    
    # 固定坐标轴范围（使用参数边界）
    ax_scatter.set_xlim(xmin, xmax)
    ax_scatter.set_ylim(ymin, ymax)
    
    # 设置标签和标题
    ax_scatter.set_xlabel(x_label, fontsize=12)
    ax_scatter.set_ylabel(y_label, fontsize=12)
    ax_scatter.set_title(f'Particle Distribution - Iteration {iteration_state.iteration}', 
                        fontsize=14, pad=35)
    ax_scatter.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), 
                     ncol=2, fontsize=10, frameon=True, fancybox=True, shadow=False)
    ax_scatter.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图片到指定输出目录
    scatter_file = output_dir / f"pso_particles_iter_{iteration_state.iteration:04d}.png"
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return scatter_file


def update_gif_realtime(work_dir):
    """
    实时更新 GIF 动画（在每次迭代后调用）
    
    参数：
        work_dir: Path，工作目录
    
    返回：
        success: bool，是否成功更新
    """
    try:
        import imageio.v2 as imageio
        
        # 查找所有已生成的粒子分布图（按顺序）
        particle_images = sorted(glob(str(work_dir / "pso_particles_iter_*.png")))
        
        if len(particle_images) > 0:
            # 读取所有图片
            images = []
            for img_file in particle_images:
                images.append(imageio.imread(img_file))
            
            # 更新 GIF（覆盖之前的版本）
            gif_file = work_dir / "pso_particle_animation.gif"
            imageio.mimsave(gif_file, images, duration=1.0, loop=0)  # 每帧 1 秒
            return True
        return False
    except ImportError:
        return False  # imageio 未安装
    except Exception as e:
        print(f"  GIF 更新失败: {e}")
        return False


def plot_convergence_curve(cost_history, avg_cost_history, iteration, work_dir, final=False):
    """
    绘制收敛曲线
    
    参数：
        cost_history: List[float]，全局最优成本历史
        avg_cost_history: List[float]，平均成本历史
        iteration: int，当前迭代次数
        work_dir: Path，工作目录
        final: bool，是否为最终曲线（高分辨率）
    
    返回：
        plot_file: Path，保存的图片文件路径
    """
    # final=True -> larger figure and higher dpi for publication-quality
    if final:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        save_dpi = 300
    else:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        save_dpi = 150
    
    # 绘制全局最优成本曲线
    ax.plot(range(1, len(cost_history)+1), cost_history, 
            color='blue', linewidth=1.5, label='Best Cost')
    ax.scatter(len(cost_history), cost_history[-1], color='blue', s=50, zorder=5)
    
    # 绘制平均成本曲线
    if avg_cost_history and len(avg_cost_history) > 0:
        ax.plot(range(1, len(avg_cost_history)+1), avg_cost_history, 
                color='orange', linewidth=1.0,
                linestyle='--', label='Average Cost', alpha=0.95)
        ax.scatter(len(avg_cost_history), avg_cost_history[-1], 
                  color='orange', s=50, zorder=5)
    
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Cost", fontsize=10)
    
    # 使用统一的收敛曲线文件，每次迭代覆盖更新
    ax.set_title(f"PSO Convergence (Iter {iteration})", fontsize=11)
    plot_file = work_dir / "pso_convergence.png"

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()

    plt.savefig(plot_file, dpi=save_dpi, bbox_inches='tight')
    plt.close(fig)
    
    return plot_file


def generate_particle_animation(work_dir, duration=1.0):
    """
    从历史 CSV 数据生成粒子运动轨迹的 GIF 动画（内存模式，不保存中间图片）
    
    参数：
        work_dir: Path，工作目录
        duration: float，每帧持续时间（秒）
    
    返回：
        gif_file: Path 或 None，生成的 GIF 文件路径（如果成功）
        num_frames: int，生成的帧数
    """
    try:
        import imageio.v2 as imageio
        import io
        
        # 读取历史数据
        history_file = work_dir / "pso_evaluation_history.csv"
        if not history_file.exists():
            print("未找到历史数据文件，无法生成 GIF")
            return None, 0
        
        df_all = pd.read_csv(history_file)
        
        # 检查是否是 2D 参数空间
        param_names = [col for col in df_all.columns if col not in 
                      ['eval_id', 'particle_id', 'iteration', 'objective', 'elapsed_time', 'status']]
        
        if len(param_names) != 2:
            print(f"参数空间不是 2D ({len(param_names)}D)，无法生成粒子分布 GIF")
            return None, 0
        
        # 获取所有迭代编号
        iterations = sorted(df_all['iteration'].unique())
        if len(iterations) == 0:
            print("未找到迭代数据，无法生成 GIF")
            return None, 0
        
        # 获取参数边界（从所有数据中计算）
        lower_bounds = np.array([df_all[param_names[0]].min(), df_all[param_names[1]].min()])
        upper_bounds = np.array([df_all[param_names[0]].max(), df_all[param_names[1]].max()])
        
        # 添加一些边距
        margin = 0.1
        range_0 = upper_bounds[0] - lower_bounds[0]
        range_1 = upper_bounds[1] - lower_bounds[1]
        lower_bounds[0] -= margin * range_0
        upper_bounds[0] += margin * range_0
        lower_bounds[1] -= margin * range_1
        upper_bounds[1] += margin * range_1
        
        print(f"  从历史数据生成 {len(iterations)} 帧...")
        
        # 生成每一帧的图片（存储在内存中）
        images = []
        particle_history = {}  # 存储粒子轨迹
        
        for iter_idx, iteration in enumerate(iterations):
            # 获取该迭代的数据
            df_iter = df_all[df_all['iteration'] == iteration]
            
            if len(df_iter) == 0:
                continue
            
            # 提取粒子位置和成本
            p0_values = df_iter[param_names[0]].values
            p1_values = df_iter[param_names[1]].values
            costs = df_iter['objective'].values
            particle_ids = df_iter['particle_id'].values
            
            # 更新粒子历史位置
            for i, pid in enumerate(particle_ids):
                if pid not in particle_history:
                    particle_history[pid] = []
                particle_history[pid].append((p0_values[i], p1_values[i]))
            
            # 找到全局最优和当前迭代最优
            global_best = df_all[df_all['iteration'] <= iteration].loc[
                df_all[df_all['iteration'] <= iteration]['objective'].idxmin()
            ]
            iter_best = df_iter.loc[df_iter['objective'].idxmin()]
            
            global_best_params = [global_best[param_names[0]], global_best[param_names[1]]]
            global_best_cost = global_best['objective']
            iter_best_params = [iter_best[param_names[0]], iter_best[param_names[1]]]
            iter_best_cost = iter_best['objective']
            
            # 绘制该帧（在内存中）
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 绘制轨迹和箭头
            marker_s_current = 100
            marker_s_prev = 20
            radius_current = np.sqrt(marker_s_current) / 2.0
            radius_prev = np.sqrt(marker_s_prev) / 2.0
            
            for pid in particle_ids:
                if len(particle_history[pid]) >= 2:
                    prev_pos = particle_history[pid][-2]
                    curr_pos = particle_history[pid][-1]
                    
                    ax.scatter(prev_pos[0], prev_pos[1], 
                             c='gray', s=marker_s_prev, alpha=0.5, zorder=2)
                    
                    mut_scale = max(6.0, radius_current * 3.0)
                    arr = FancyArrowPatch(
                        posA=(prev_pos[0], prev_pos[1]),
                        posB=(curr_pos[0], curr_pos[1]),
                        arrowstyle='->',
                        mutation_scale=mut_scale,
                        linewidth=0.5,
                        color='gray',
                        alpha=0.5,
                        shrinkA=radius_prev,
                        shrinkB=radius_current,
                        transform=ax.transData,
                        zorder=3
                    )
                    ax.add_patch(arr)
            
            # 绘制当前粒子
            scatter = ax.scatter(p0_values, p1_values, c=costs, 
                               cmap='viridis', s=100, alpha=0.6, 
                               edgecolors='black', linewidth=0.5, zorder=5)
            
            # 标记全局最优和迭代最优
            ax.scatter(global_best_params[0], global_best_params[1], 
                      marker='*', s=250, c='red', edgecolors='black', 
                      linewidth=0.5, label=f'Global Best (cost={global_best_cost:.2f})', 
                      zorder=10)
            ax.scatter(iter_best_params[0], iter_best_params[1], 
                      marker='D', s=80, c='orange', edgecolors='black', 
                      linewidth=0.5, label=f'Iter Best (cost={iter_best_cost:.2f})', 
                      zorder=9)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Objective (Cost)', rotation=270, labelpad=20)
            
            # 设置坐标轴和标题
            ax.set_xlim(lower_bounds[0], upper_bounds[0])
            ax.set_ylim(lower_bounds[1], upper_bounds[1])
            ax.set_xlabel(f'{param_names[0]}', fontsize=12)
            ax.set_ylabel(f'{param_names[1]}', fontsize=12)
            ax.set_title(f'Particle Distribution - Iteration {iteration}', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            
            # 将图片保存到内存缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            images.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)
            
            if (iter_idx + 1) % 10 == 0:
                print(f"    已生成 {iter_idx + 1}/{len(iterations)} 帧")
        
        if len(images) == 0:
            print("未生成任何图片帧，无法生成 GIF")
            return None, 0
        
        # 生成 GIF
        gif_file = work_dir / "pso_particle_animation.gif"
        imageio.mimsave(gif_file, images, duration=duration, loop=0)
        
        return gif_file, len(images)
        
    except ImportError:
        print("需要安装 imageio 库才能生成 GIF: pip install imageio")
        return None, 0
    except Exception as e:
        print(f"生成 GIF 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def export_particles_to_vtk(work_dir):
    """
    将粒子历史数据导出为 VTK 时间序列文件，可在 ParaView 中可视化
    
    ParaView 可以高效处理大规模数据，支持交互式 3D 可视化和动画
    
    参数：
        work_dir: Path，工作目录
    
    返回：
        success: bool，是否成功导出
        num_files: int，导出的文件数量
    """
    try:
        # 读取历史数据
        history_file = work_dir / "pso_evaluation_history.csv"
        if not history_file.exists():
            print("未找到历史数据文件，无法导出 VTK")
            return False, 0
        
        df_all = pd.read_csv(history_file)
        
        # 提取参数名称
        param_names = [col for col in df_all.columns if col not in 
                      ['eval_id', 'particle_id', 'iteration', 'objective', 'elapsed_time', 'status']]
        
        if len(param_names) < 2:
            print(f"参数空间维度不足 ({len(param_names)}D)，需要至少 2D")
            return False, 0
        
        # 创建 VTK 输出目录
        vtk_dir = work_dir / "vtk_particles"
        vtk_dir.mkdir(exist_ok=True)
        
        # 获取所有迭代编号
        iterations = sorted(df_all['iteration'].unique())
        
        print(f"  导出 {len(iterations)} 个时间步到 VTK 格式...")
        
        # 为每个迭代生成一个 VTK 文件
        for iter_idx, iteration in enumerate(iterations):
            df_iter = df_all[df_all['iteration'] == iteration]
            
            if len(df_iter) == 0:
                continue
            
            # 提取粒子数据
            n_particles = len(df_iter)
            
            # VTK 文件路径
            vtk_file = vtk_dir / f"particles_{iteration:04d}.vtk"
            
            with open(vtk_file, 'w') as f:
                # VTK 文件头
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"PSO Particles Iteration {iteration}\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                
                # 点坐标（将参数空间映射到 3D 空间）
                f.write(f"POINTS {n_particles} float\n")
                for idx, row in df_iter.iterrows():
                    # 2D 参数空间：(p0, p1, 0)
                    # 3D+ 参数空间：(p0, p1, p2) 或 (p0, p1, objective)
                    if len(param_names) >= 3:
                        x, y, z = row[param_names[0]], row[param_names[1]], row[param_names[2]]
                    else:
                        # 2D 情况，使用 objective 作为 z 坐标
                        x, y = row[param_names[0]], row[param_names[1]]
                        z = row['objective']
                    f.write(f"{x} {y} {z}\n")
                
                # 点数据（标量和向量）
                f.write(f"\nPOINT_DATA {n_particles}\n")
                
                # 标量：objective（成本函数值）
                f.write("SCALARS objective float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for idx, row in df_iter.iterrows():
                    f.write(f"{row['objective']}\n")
                
                # 标量：particle_id
                f.write("\nSCALARS particle_id int 1\n")
                f.write("LOOKUP_TABLE default\n")
                for idx, row in df_iter.iterrows():
                    f.write(f"{int(row['particle_id'])}\n")
                
                # 如果有速度信息，添加向量场（需要从前后两帧计算）
                if iter_idx > 0:
                    prev_iter = iterations[iter_idx - 1]
                    df_prev = df_all[df_all['iteration'] == prev_iter]
                    
                    # 计算速度（当前位置 - 前一位置）
                    f.write("\nVECTORS velocity float\n")
                    for pid in df_iter['particle_id'].values:
                        curr_row = df_iter[df_iter['particle_id'] == pid].iloc[0]
                        prev_row = df_prev[df_prev['particle_id'] == pid]
                        
                        if len(prev_row) > 0:
                            prev_row = prev_row.iloc[0]
                            vx = curr_row[param_names[0]] - prev_row[param_names[0]]
                            vy = curr_row[param_names[1]] - prev_row[param_names[1]]
                            if len(param_names) >= 3:
                                vz = curr_row[param_names[2]] - prev_row[param_names[2]]
                            else:
                                vz = curr_row['objective'] - prev_row['objective']
                            f.write(f"{vx} {vy} {vz}\n")
                        else:
                            f.write("0 0 0\n")
                else:
                    # 第一帧，速度为零
                    f.write("\nVECTORS velocity float\n")
                    for _ in range(n_particles):
                        f.write("0 0 0\n")
            
            if (iter_idx + 1) % 20 == 0:
                print(f"    已导出 {iter_idx + 1}/{len(iterations)} 个文件")
        
        # 生成 PVD 文件（ParaView 时间序列索引文件）
        pvd_file = vtk_dir / "particles_series.pvd"
        with open(pvd_file, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            for iteration in iterations:
                vtk_filename = f"particles_{iteration:04d}.vtk"
                # 使用迭代编号作为时间值
                f.write(f'    <DataSet timestep="{iteration}" file="{vtk_filename}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        print(f"  VTK 文件已导出到: {vtk_dir}")
        print(f"  在 ParaView 中打开: {pvd_file.name}")
        
        return True, len(iterations)
        
    except Exception as e:
        print(f"导出 VTK 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
