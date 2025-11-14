"""
PSO 运行器类模块

该模块包含面向对象的 PSO 运行器实现。

主要类：
- PSORunner: PSO 优化运行器类

"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

from typing import List

import yaml

from objective_function import MOOSEObjectiveFunction
from visualization import (
    plot_particle_distribution,
    plot_convergence_curve,
    update_gif_realtime,
    generate_particle_animation
)

from data_class import Parameters, PSOConfig, SwarmState, IterationStateSummary, IterationEvaluation, OptimizationResult

class PSORunner:
    """
    粒子群优化运行器
    
    将 PSO 优化过程封装为一个类，便于状态管理和功能扩展。
    """
    
    def __init__(self, config: dict):
        """
        初始化 PSO 运行器
        
        参数：
            config: dict, 配置字典
        """
        print("="*80)
        print(f"Particle Swarm Optimization")
        print("="*80)

        # 创建目标函数
        self.objective_func = MOOSEObjectiveFunction(config)
        
        # Parameter Configuration
        self.parameters = Parameters.from_dict(config['parameters'])

        # 创建 PSO 配置对象
        self.pso_config = PSOConfig.from_dict(config['pso'], self.parameters)

        # 粒子群状态
        self.swarm = None  # 将在 _initialize_swarm 中初始化

        # 迭代评估历史（每次迭代的详细评估结果）
        self.evaluation_history: List[IterationEvaluation] = []
        
        # 迭代摘要历史（每次迭代的摘要信息）
        self.iteration_history: List[IterationStateSummary] = []
        
        # 迭代状态变量（用于进度输出）
        self.printed_header = False
        self.last_iter_time = None
        self.pso_start_time = time.time()
        
        # 收敛状态
        self.converged = False
        self.convergence_iteration = None

        # 绘图间隔
        self.plot_interval = config.get('save_plot_interval', 0)

        # 缓存设置
        self.cache_enabled = config.get('enable_cache', False)

        # 输出目录设置
        self.work_dir = Path(config.get('work_dir'))

        self.evaluation_file = self.work_dir / "pso_evaluation_history.csv"
        self.iteration_file = self.work_dir / "pso_iteration_history.csv"
        self.optimal_result_file = self.work_dir / "pso_optimal_result.yaml"

        self.particle_plots_dir = self.work_dir / "particle_plots"
        self.particle_plots_dir.mkdir(parents=True, exist_ok=True)

        # 打印配置
        print(self.parameters)
        print(self.pso_config)

        # 初始粒子群
        self._initialize_swarm()        
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        params = self.pso_config.parameters
        (lb, ub) = (params.lower_bounds, params.upper_bounds)
        n_particles = self.pso_config.n_particles
        n_dims = self.pso_config.n_dims

        initial_position = np.random.uniform(
            low=lb, high=ub, size=(n_particles, n_dims)
        )
        
        velocity_factor = 0.4
        initial_velocity = np.random.uniform(
            low=-velocity_factor * np.abs(ub - lb),
            high=velocity_factor * np.abs(ub - lb),
            size=(n_particles, n_dims)
        )
        
        pbest_pos = initial_position.copy()
        pbest_cost = np.full(n_particles, np.inf)
        current_cost = np.full(n_particles, np.inf)
        
        self.swarm = SwarmState(
            position=initial_position,
            velocity=initial_velocity,
            cost=current_cost,
            pbest_pos=pbest_pos,
            pbest_cost=pbest_cost,
            best_id=None,
            best_pos=None,
            best_cost=np.inf,
            prev_position=initial_position.copy()  # 初始化为当前位置
        )
    
    def _check_convergence(self, iteration):
        """
        检查是否收敛
        
        返回：
            bool: 是否收敛
        """
        conv_window = self.pso_config.convergence_window
        conv_threshold = self.pso_config.convergence_threshold

        if (conv_window is None or 
            conv_threshold is None or
            len(self.iteration_history) < conv_window):
            return False
        
        # 从 iteration_history 提取最近的平均成本
        recent_avgs = [state.avg_cost for state in self.iteration_history[-conv_window:]]
        max_avg = max(recent_avgs)
        min_avg = min(recent_avgs)
        mean_avg = sum(recent_avgs) / conv_window
        
        if mean_avg < conv_threshold:
            print(f"\nConverged at iteration {iteration}: "
                  f"Average cost {mean_avg:.2e} < {conv_threshold}")
            self.converged = True
            self.convergence_iteration = iteration
            return True
        
        return False
    
    def _update_particles(self, iteration):
        """更新粒子速度和位置"""
        params = self.pso_config.parameters
        (lb, ub) = (params.lower_bounds, params.upper_bounds)
        n_dims = self.pso_config.n_dims
        max_iters = self.pso_config.max_iterations

        if iteration >= max_iters - 1:
            return
        
        # 动态惯性权重
        w_max = 0.8
        w_min = 0.8
        w_dynamic = w_max - (w_max - w_min) * (iteration / max_iters)
        
        # 认知分量和社会分量
        options = self.pso_config.get_options_dict()
        cognitive = (options['c1'] * np.random.random(self.swarm.position.shape) * 
                    (self.swarm.pbest_pos - self.swarm.position))
        social = (options['c2'] * np.random.random(self.swarm.position.shape) * 
                 (self.swarm.best_pos - self.swarm.position))
        
        # 更新速度
        self.swarm.velocity = (w_dynamic * self.swarm.velocity + cognitive + social)
        
        # 保存上次位置（用于绘制轨迹）
        self.swarm.prev_position = self.swarm.position.copy()
        
        # 更新位置
        new_position = self.swarm.position + self.swarm.velocity
        
        # 边界处理（反弹法）
        for dim in range(n_dims):
            # 下边界
            lower_violation = new_position[:, dim] < lb[dim]
            if np.any(lower_violation):
                new_position[lower_violation, dim] = (
                    2 * lb[dim] - new_position[lower_violation, dim]
                )
                self.swarm.velocity[lower_violation, dim] *= -0.75
            
            # 上边界
            upper_violation = new_position[:, dim] > ub[dim]
            if np.any(upper_violation):
                new_position[upper_violation, dim] = (
                    2 * ub[dim] - new_position[upper_violation, dim]
                )
                self.swarm.velocity[upper_violation, dim] *= -0.75
        
        # 最终裁剪
        self.swarm.position = np.clip(new_position, lb, ub)
        
    def _output_plots(self, iteration: int, iter_state: IterationStateSummary, iter_eval: IterationEvaluation):
        """绘制并保存当前迭代的图表
        
        参数：
            iteration: int, 当前迭代次数
            iter_state: IterationStateSummary, 迭代状态对象
            iter_eval: IterationEvaluation, 迭代评估对象
        """
        
        # 从 iteration_history 提取历史数据
        cost_history = [state.global_best_cost for state in self.iteration_history]
        avg_cost_history = [state.avg_cost for state in self.iteration_history]
        
        # 绘制收敛曲线
        plot_convergence_curve(
            cost_history,
            avg_cost_history,
            iteration, 
            self.objective_func.work_dir
        )

        # 绘制粒子分布（仅2D）
        if self.pso_config.n_dims == 2:
            params = self.pso_config.parameters
            
            plot_particle_distribution(
                iter_eval,      # IterationEvaluation 对象
                iter_state,     # IterationState 对象
                self.swarm,     # SwarmState 对象
                params.names,   # 参数名称
                (params.lower_bounds, params.upper_bounds),  # 参数边界
                self.particle_plots_dir  # 输出目录
            )
            
            if self.objective_func.realtime_gif:
                update_gif_realtime(self.objective_func.work_dir)
    
    def _output_stats(self):
        """输出当前迭代的统计信息

        参数：
            iteration: int, 当前迭代次数
            iter_state: IterationStateSummary, 迭代状态对象
            iter_eval: IterationEvaluation, 迭代评估对象
        """

        # 保存评估历史（从 evaluation_history 构建 DataFrame）
        eval_df = [iter_eval.to_dataframe(self.parameters.names)
                   for iter_eval in self.evaluation_history]
        eval_df = pd.concat(eval_df, ignore_index=True)
        if not eval_df.empty:
            eval_df.to_csv(self.evaluation_file, index=False)

        # 保存迭代历史
        iter_df = pd.DataFrame([
            {
                'iteration': s.iteration,
                'global_best_cost': s.global_best_cost,
                'global_best_eval_id': s.global_best_eval_id,
                'iter_best_cost': s.iter_best_cost,
                'iter_best_eval_id': s.iter_best_eval_id,
                'avg_cost': s.avg_cost,
                'iter_elapsed': s.iter_elapsed,
                'total_elapsed': s.total_elapsed,
                **{f'global_best_{name}': s.global_best_params[i] 
                   for i, name in enumerate(self.parameters.names)},
                **{f'iter_best_{name}': s.iter_best_params[i] 
                   for i, name in enumerate(self.parameters.names)}
            }
            for s in self.iteration_history
        ])
        if not iter_df.empty:
            iter_df.to_csv(self.iteration_file, index=False)


    def run(self):
        """
        运行 PSO 优化
        
        返回：
            OptimizationResult: 优化结果对象
        """
        print("\nStarting optimization...")
        print("-"*80)

        # 主循环
        for i in range(1, self.pso_config.max_iterations):
            iter_start_time = time.time()
            
            # ========== 1. 评估所有粒子 ==========
            eval_results = self.objective_func(self.swarm.position, i)
            self.swarm.cost = eval_results['cost']
            
            # ========== 2. 保存评估结果 ==========
            particle_ids = np.arange(self.pso_config.n_particles)
            eval_ids = (i - 1) * self.pso_config.n_particles + particle_ids
            
            iter_eval = IterationEvaluation(
                iteration=i,
                eval_ids=eval_ids,
                particle_ids=particle_ids,
                parameters=self.swarm.position.copy(),
                costs=self.swarm.cost.copy(),
                elapsed_times=eval_results['elapsed_time'],
                statuses=np.array(eval_results['status'])
            )
            self.evaluation_history.append(iter_eval)
            
            # ========== 3. 更新粒子群状态 ==========
            # 更新个体最优
            better_mask = self.swarm.cost < self.swarm.pbest_cost
            self.swarm.pbest_cost[better_mask] = self.swarm.cost[better_mask]
            self.swarm.pbest_pos[better_mask] = self.swarm.position[better_mask]

            
            # 更新全局最优
            min_cost_idx = np.argmin(self.swarm.cost)
            if self.swarm.cost[min_cost_idx] < self.swarm.best_cost:
                self.swarm.best_cost = self.swarm.cost[min_cost_idx]
                self.swarm.best_pos = self.swarm.position[min_cost_idx].copy()
                self.swarm.best_id = int(eval_ids[min_cost_idx])

            # 保存迭代状态摘要
            iter_state = IterationStateSummary(
                iteration=i,
                global_best_cost=float(self.swarm.best_cost),
                global_best_eval_id= self.swarm.best_id,
                global_best_params=self.swarm.best_pos.tolist(),
                iter_best_cost=float(np.min(self.swarm.cost)),
                iter_best_eval_id=int(eval_ids[np.argmin(self.swarm.cost)]),
                iter_best_params=self.swarm.position[np.argmin(self.swarm.cost)].tolist(),
                iter_elapsed=time.time() - iter_start_time,
                total_elapsed=time.time() - self.pso_start_time,
                avg_cost= float(np.mean(self.swarm.cost)),
            )
            self.iteration_history.append(iter_state)
            
            # 打印迭代信息
            if not self.printed_header:
                print(IterationStateSummary.get_header(self.parameters.names))
                print(IterationStateSummary.get_separator(len(self.parameters.names)))
                self.printed_header = True
            print(iter_state)

            self.converged = self._check_convergence(i)

            should_output = (
                self.plot_interval == 0 or 
                i % self.plot_interval == 0 or 
                i == 1 or self.converged
             )
            
            if should_output:
                self._output_plots(i, iter_state, iter_eval)
                self._output_stats()

            # 检查收敛
            if self.converged:
                break

            # 更新粒子位置和速度
            self._update_particles(i)
        
        # ========== 构建优化结果 ==========


        # ========== 保存结果 ==========
        # 保存优化结果摘要 到 YAML文件
        optimal_result = {
            'best_cost': float(self.swarm.best_cost),
            'best_params': self.swarm.best_pos.tolist(),
            'best_eval_id': self.swarm.best_id,
            'total_iterations': len(self.iteration_history),
            'total_evaluations': len(self.evaluation_history) * self.pso_config.n_particles,
            'total_time': time.time() - self.pso_start_time,
            'converged': self.converged,
            'convergence_iteration': self.convergence_iteration,
            'cache_enabled': self.cache_enabled,
            'cache_stats': self.cache_stats if self.cache_enabled else None
        }

        with open(self.optimal_result_file, 'w') as f:
            yaml.dump(optimal_result, f, default_flow_style=False)
        print(f"  ✓ 优化结果摘要: {self.optimal_result_file.name}")
        
        # ========== 打印摘要 ==========
        print("\n" + "="*80)

        # self.objective_func.cache.print_stats()
        
        return self.evaluation_history


def run_pso_optimization(config: dict) -> OptimizationResult:
    """
    运行 PSO 优化（函数式接口，内部调用 PSORunner 类）
    
    参数：
        config: dict, 配置字典
    
    返回：
        OptimizationResult: 优化结果对象，包含：
            - 最优解信息（best_cost, best_params, best_eval_id）
            - 迭代历史（cost_history, avg_cost_history, iteration_history）
            - 优化统计（total_iterations, total_evaluations, total_time, converged）
            - 缓存统计（cache_enabled, cache_hits, cache_misses, cache_hit_rate）
            - 评估历史（evaluation_history DataFrame）
    """

    
    runner = PSORunner(config)
    return runner.run()

