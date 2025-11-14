"""
PSO 运行器模块

该模块包含粒子群优化 (PSO) 算法的核心运行逻辑。

主要功能：
- run_pso_optimization: PSO 优化主函数
- iteration_callback: 迭代回调函数，保存历史和绘图
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

from objective_function import MOOSEObjectiveFunction
from visualization import (
    plot_particle_distribution,
    plot_convergence_curve,
    update_gif_realtime,
    generate_particle_animation
)


def _build_iteration_dataframe(eval_records, param_names):
    """
    从评估记录构建 DataFrame（辅助函数）
    
    参数：
        eval_records: list, 评估记录列表
        param_names: list, 参数名称列表
    
    返回：
        pd.DataFrame: 包含评估数据的 DataFrame
    """
    df_data = []
    for record in eval_records:
        row = {
            'eval_id': record['eval_id'],
            'particle_id': record.get('particle_id'),
            'iteration': record.get('iteration')
        }
        for (name, val) in zip(param_names, record['parameters']):
            row[name] = val
        row['objective'] = record['objective']
        row['elapsed_time'] = record['elapsed_time']
        row['status'] = record['status']
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def run_pso_optimization(config: dict):
    """
    运行 PSO 优化
    
    参数：
        config (dict): 配置字典，包含以下关键字段：
            - parameters: 参数配置（名称、边界等）
            - pso: PSO 算法参数（粒子数、迭代次数、系数等）
            - 其他目标函数相关配置
    
    返回：
        result (dict): 优化结果，包含：
            - optimal_parameters: 最优参数
            - optimal_cost: 最优成本
            - optimal_eval_id: 最优评估ID
            - n_iterations: 实际迭代次数
            - elapsed_time: 运行时间
            - optimizer: 优化器对象（包含 cost_history 等）
            - objective_func: 目标函数对象
    """
    from pyswarms.single.global_best import GlobalBestPSO

    print("="*80)
    print(f"Particle Swarm Optimization (MOOSE)")
    print("="*80)

    print(f"\nRunning PSO Optimization...")

    # 创建目标函数
    objective_func = MOOSEObjectiveFunction(config)
    
    # 可视化配置
    save_plot_interval = config.get('save_plot_interval', 0)  # 默认每次都保存

    # 创建用于保存粒子分布图的文件夹
    particle_plots_dir = objective_func.work_dir / "particle_plots"
    particle_plots_dir.mkdir(parents=True, exist_ok=True)

    # 参数配置
    param_config = config['parameters']
    lower_bounds = np.array(param_config['lower_bounds'])
    upper_bounds = np.array(param_config['upper_bounds'])
    bounds = (lower_bounds, upper_bounds)
    print(f"\nParameter Configuration:")
    for i, name in enumerate(param_config['names']):
        print(f"  {name}: [{lower_bounds[i]}, {upper_bounds[i]}]")

    # PSO 参数
    pso_config = config['pso']
    n_particles = pso_config['n_particles']
    max_iters = pso_config['max_iterations']
    # 收敛参数 - 确保转换为正确的数值类型
    convergence_threshold = float(pso_config.get('convergence_threshold'))
    convergence_window = int(pso_config.get('convergence_window'))


    options = {
        'c1': pso_config['cognitive_param'],  # 认知参数
        'c2': pso_config['social_param'],     # 社会参数
        'w': pso_config['inertia_weight']     # 惯性权重
    }

    print(f"\nPSO Configuration:")
    print(f"  Number of particles: {n_particles}")
    print(f"  Maximum iterations:  {max_iters}")
    print(f"  Cognitive parameter: {options['c1']}")
    print(f"  Social parameter:    {options['c2']}")
    print(f"  Inertia weight:      {options['w']}")

    # 创建 PSO 优化器
    print("\nStarting optimization...")
    print("-"*80)

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(param_config['names']),
        options=options,
        bounds=bounds,
        ftol=0.1,
        ftol_iter=10
    )

    # 定义迭代回调函数，每次迭代后保存历史和绘图
    last_saved_count = 0
    printed_header = False
    particle_history = {}  # 存储每个粒子的历史位置 {particle_id: [(p0, p1), ...]}
    last_iter_time = None
    optimization_start_time = None
    
    # 统一的历史记录文件路径
    history_file = objective_func.work_dir / "pso_evaluation_history.csv"
    
    def iteration_callback(iteration, cost_history):
        """
        在每次迭代后调用，保存历史和绘制收敛曲线
        
        参数：
            iteration: int，当前迭代次数
            cost_history: list，成本历史记录
        """
        nonlocal last_saved_count, printed_header, last_iter_time, optimization_start_time
        
        # 计算时间
        current_time = time.time()
        total_elapsed = current_time - optimization_start_time
        
        if last_iter_time is not None:
            iter_elapsed = current_time - last_iter_time
        else:
            iter_elapsed = total_elapsed  # 第一次迭代
        
        last_iter_time = current_time
        
        # 判断是否需要保存（根据 save_plot_interval）
        should_save = (save_plot_interval == 0 or 
                       iteration % save_plot_interval == 0 or 
                       iteration == 1)
        
        # 保存该迭代的新评估记录（仅保存新增部分）
        current_count = len(objective_func.eval_history)
        new_records = objective_func.eval_history[last_saved_count:current_count]
        
        if new_records:
            # 构建当前迭代的 DataFrame
            df = _build_iteration_dataframe(new_records, objective_func.param_names)
            
            # 找到该迭代的最优值及对应参数和eval_id
            iter_best_idx = df['objective'].idxmin()
            iter_best_cost = df.loc[iter_best_idx, 'objective']
            iter_best_params = [df.loc[iter_best_idx, name] for name in objective_func.param_names]
            iter_best_eval_id = df.loc[iter_best_idx, 'eval_id']
            
            # 从全局历史中找到全局最优值对应的参数和eval_id
            df_all_history = pd.DataFrame([
                {'eval_id': rec['eval_id'], 'objective': rec['objective'], 
                 **{name: rec['parameters'][i] for i, name in enumerate(objective_func.param_names)}}
                for rec in objective_func.eval_history
            ])
            global_best_idx = df_all_history['objective'].idxmin()
            global_best_cost = df_all_history.loc[global_best_idx, 'objective']
            global_best_params = [df_all_history.loc[global_best_idx, name] for name in objective_func.param_names]
            global_best_eval_id = df_all_history.loc[global_best_idx, 'eval_id']
            

            # 追加到统一的历史记录文件（第一次迭代创建文件，之后追加）
            if iteration == 1 or not history_file.exists():
                df.to_csv(history_file, index=False, mode='w')
            else:
                df.to_csv(history_file, index=False, mode='a', header=False)
            
            # 打印表格形式的统计信息
            if not printed_header:
                # 打印表头
                param_headers = ' '.join([f"{name:>8}" for name in objective_func.param_names])
                print(f"\n{'Iter':>4} {'GbCost':>10} {'GbID':>6} {param_headers} {'ItCost':>10} {'ItID':>6} {param_headers} {'ItT(s)':>8} {'Total(s)':>8}")
                print("-" * (4 + 10 + 6 + 8*len(objective_func.param_names) + 10 + 6 + 8*len(objective_func.param_names) + len(objective_func.param_names)*2 + 8 + 8 + 2))
                printed_header = True
            
            # 打印数据行
            global_params_str = ' '.join([f"{val:8.2f}" for val in global_best_params])
            iter_params_str = ' '.join([f"{val:8.2f}" for val in iter_best_params])
            print(f"{iteration:4d} {global_best_cost:10.4f} {int(global_best_eval_id):6d} {global_params_str} {iter_best_cost:10.4f} {int(iter_best_eval_id):6d} {iter_params_str} {iter_elapsed:8.3f} {total_elapsed:8.2f}")
            
            # 仅在需要保存时绘图
            if should_save:
                # 如果是2D参数空间，绘制粒子分布图
                if len(objective_func.param_names) == 2:
                    # 使用可视化模块绘制粒子分布
                    scatter_file = plot_particle_distribution(
                        df, iteration, objective_func, lower_bounds, upper_bounds,
                        global_best_params, global_best_cost,
                        iter_best_params, iter_best_cost,
                        particle_history
                    )
                    
                    # 根据配置决定是否实时更新 GIF 动画
                    if objective_func.realtime_gif:
                        update_gif_realtime(objective_func.work_dir)
                
                # 绘制收敛曲线（传入平均成本历史）
                plot_convergence_curve(cost_history, iteration, objective_func.work_dir, 
                                     avg_cost_history=optimizer.avg_cost_history)
    
            # 更新已保存计数
            last_saved_count = current_count


    # 执行优化
    start_time = time.time()
    # 初始化优化开始时间
    optimization_start_time = start_time
    
    # 使用自定义优化循环，手动控制 PSO 迭代过程
    # 初始化粒子群（第一次迭代）
    optimizer.swarm.position = np.random.uniform(
        low=lower_bounds, 
        high=upper_bounds, 
        size=(n_particles, len(param_config['names']))
    )
    velocity_factor = 0.4
    optimizer.swarm.velocity = np.random.uniform(
        low=-velocity_factor*np.abs(upper_bounds - lower_bounds),
        high=velocity_factor*np.abs(upper_bounds - lower_bounds),
        size=(n_particles, len(param_config['names']))
    )
    optimizer.swarm.pbest_pos = optimizer.swarm.position.copy()
    optimizer.swarm.pbest_cost = np.full(n_particles, np.inf)
    optimizer.swarm.best_pos = None
    optimizer.swarm.best_cost = np.inf
    
    # 初始化 optimizer 的平均成本历史记录
    optimizer.avg_cost_history = []
    
    # 手动迭代
    actual_iterations = 0
    for i in range(max_iters):
        # 评估当前位置
        costs = objective_func(optimizer.swarm.position, iteration=i+1)
        # 记录本次迭代的平均成本（浮点数）
        mean_cost = float(np.mean(costs))
        optimizer.avg_cost_history.append(mean_cost)
        
        # 更新个体最优
        better_mask = costs < optimizer.swarm.pbest_cost
        optimizer.swarm.pbest_cost[better_mask] = costs[better_mask]
        optimizer.swarm.pbest_pos[better_mask] = optimizer.swarm.position[better_mask]
        
        # 更新全局最优
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < optimizer.swarm.best_cost:
            optimizer.swarm.best_cost = costs[min_cost_idx]
            optimizer.swarm.best_pos = optimizer.swarm.position[min_cost_idx].copy()
        
        # 更新 cost_history
        if not hasattr(optimizer, 'cost_history'):
            optimizer.cost_history = []
        optimizer.cost_history.append(optimizer.swarm.best_cost)
        
        # 调用回调函数
        iteration_callback(i + 1, optimizer.cost_history)
        
        actual_iterations = i + 1
        

        # 检查收敛（使用最近若干次迭代所有粒子平均成本的相对变化）
        if (convergence_window is not None and convergence_threshold is not None and 
            len(optimizer.avg_cost_history) >= convergence_window):
            recent_avgs = optimizer.avg_cost_history[-convergence_window:]
            max_avg = max(recent_avgs)
            min_avg = min(recent_avgs)
            mean_avg = sum(recent_avgs) / convergence_window
            # 绝对判断
            absolute_change = max_avg - min_avg
            if mean_avg < convergence_threshold:
                print(f"\nConverged at iteration {i+1}: Absolute change in average cost over last {convergence_window} iterations is {absolute_change:.2e} < {convergence_threshold}")
                break
            # # 避免除以零，当最大值接近零时退化为绝对判断
            # if max_avg > 1e-12:
            #     relative_change = (max_avg - min_avg) / max_avg
            #     if relative_change < convergence_threshold:
            #         print(f"\n收敛于迭代 {i+1}: 最近{convergence_window}次迭代平均成本相对变化 {relative_change:.2e} < {convergence_threshold}")
            #         break
            # else:
            #     absolute_change = max_avg - min_avg
            #     if absolute_change < convergence_threshold:
            #         print(f"\n收敛于迭代 {i+1}: 最近{convergence_window}次迭代平均成本绝对变化 {absolute_change:.2e} < {convergence_threshold}")
            #         break
        
        # 更新速度和位置（PSO 公式）
        if i < max_iters - 1:  # 最后一次迭代不需要更新
            # 动态惯性权重：线性递减 w = 0.9 - (0.9-0.2) * iter/max_iter
            w_max = 0.8
            w_min = 0.8
            w_dynamic = w_max - (w_max - w_min) * (i / max_iters)
            
            # 认知分量
            cognitive = (options['c1'] * np.random.random(optimizer.swarm.position.shape) * 
                        (optimizer.swarm.pbest_pos - optimizer.swarm.position))
            
            # 社会分量
            social = (options['c2'] * np.random.random(optimizer.swarm.position.shape) * 
                     (optimizer.swarm.best_pos - optimizer.swarm.position))
            
            # 更新速度（使用动态惯性权重）
            optimizer.swarm.velocity = (w_dynamic * optimizer.swarm.velocity + 
                                       cognitive + social)
            
            # 更新位置
            new_position = optimizer.swarm.position + optimizer.swarm.velocity
            
            # 边界处理：反弹法（Reflecting）
            # 粒子撞到边界时，位置反弹回边界内，速度反向
            for dim in range(len(lower_bounds)):
                # 检查下边界
                lower_violation = new_position[:, dim] < lower_bounds[dim]
                if np.any(lower_violation):
                    # 位置反弹：超出部分镜像回边界内
                    new_position[lower_violation, dim] = (
                        2 * lower_bounds[dim] - new_position[lower_violation, dim]
                    )
                    # 速度反向
                    optimizer.swarm.velocity[lower_violation, dim] *= -0.75
                
                # 检查上边界
                upper_violation = new_position[:, dim] > upper_bounds[dim]
                if np.any(upper_violation):
                    # 位置反弹：超出部分镜像回边界内
                    new_position[upper_violation, dim] = (
                        2 * upper_bounds[dim] - new_position[upper_violation, dim]
                    )
                    # 速度反向
                    optimizer.swarm.velocity[upper_violation, dim] *= -0.75
            
            # 最终裁剪（防止反弹后仍超出边界）
            optimizer.swarm.position = np.clip(new_position, lower_bounds, upper_bounds)
    
    cost = optimizer.swarm.best_cost
    pos = optimizer.swarm.best_pos
    
    elapsed_time = time.time() - start_time

    # 输出结果
    print("-"*80)
    print(f"Best cost (objective function value): {cost:.6e}")
    print(f"Best position (parameters):")
    for i, name in enumerate(param_config['names']):
        print(f"  {name} = {pos[i]:.6f}")

    print(f"Optimization completed in {elapsed_time:.2f} seconds.")
    
    # 打印缓存统计信息（从评估历史中统计）
    objective_func.cache.print_stats(eval_history=objective_func.eval_history)
    
    # 从评估历史中找到最优评估的 eval_id
    best_eval = min(objective_func.eval_history, key=lambda x: x['objective'])
    optimal_eval_id = best_eval['eval_id']
    
    # 保存最优参数
    result_file = objective_func.work_dir / "pso_optimal_parameters.yaml"
    
    result = {
        'optimal_parameters': {name: float(val) for name, val in zip(param_config['names'], pos)},
        'optimal_cost': float(cost),
        'optimal_eval_id': int(optimal_eval_id),
        'n_iterations': actual_iterations,
        'elapsed_time': elapsed_time,
        'n_evaluations': len(objective_func.eval_history),
        'success_count': objective_func.success_count,
        'fail_count': objective_func.fail_count
    }
    
    # ========================================================================
    # 绘制最终状态图
    # ========================================================================
    print("\n绘制最终状态图...")
    
    # 1. 绘制最终收敛曲线
    plot_convergence_curve(
        optimizer.cost_history,
        actual_iterations,
        objective_func.work_dir,
        avg_cost_history=optimizer.avg_cost_history
    )
    print(f"  ✓ 收敛曲线: pso_convergence.png")
    
    # 2. 绘制最终粒子分布图（仅2D参数空间）
    if len(param_config['names']) == 2:
        # 从评估历史中提取最后一次迭代的数据
        final_records = [rec for rec in objective_func.eval_history 
                        if rec.get('iteration') == actual_iterations]
        
        if final_records:
            # 构建 DataFrame（复用辅助函数）
            df_final = _build_iteration_dataframe(final_records, objective_func.param_names)
            
            # 找到最后一次迭代的最优值
            iter_best_idx = df_final['objective'].idxmin()
            iter_best_cost = df_final.loc[iter_best_idx, 'objective']
            iter_best_params = [df_final.loc[iter_best_idx, name] 
                               for name in objective_func.param_names]
            
            # 全局最优参数（从 result 中获取）
            global_best_params = [result['optimal_parameters'][name] 
                                 for name in param_config['names']]
            global_best_cost = result['optimal_cost']
            
            # particle_history 只需要最后一个位置即可（用于绘图）
            final_particle_history = {
                row['particle_id']: [(row[param_config['names'][0]], 
                                     row[param_config['names'][1]])]
                for _, row in df_final.iterrows()
            }
            
            plot_particle_distribution(
                df_final, actual_iterations, objective_func, 
                lower_bounds, upper_bounds,
                global_best_params, global_best_cost,
                iter_best_params, iter_best_cost,
                final_particle_history
            )
            print(f"  ✓ 粒子分布: pso_particles.png")
    
    return result, optimizer, objective_func
