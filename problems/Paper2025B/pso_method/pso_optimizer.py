"""
PSO 优化器重构版本

主要改进：
1. 修复迭代从 0 开始（而非 1）
2. 拆分主循环为多个小方法，提高可读性
3. 增量保存 CSV，避免重复构建 DataFrame
4. 使用属性方法简化代码
5. 收敛后仍然输出一次结果
6. 修复缓存统计获取错误
"""

import numpy as np
import pandas as pd
import time
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

from objective_function import MOOSEObjectiveFunction
from visualization import (
    plot_particle_distribution,
    plot_convergence_curve,
    update_gif_realtime
)
from data_class import (
    Parameters, 
    PSOConfig, 
    SwarmState, 
    IterationStateSummary, 
    IterationEvaluation,
    EvaluationHistory,
    OptimizationResult
)


class PSOOptimizer:
    """
    粒子群优化器（重构版）
    
    主要特性：
    - 清晰的方法划分
    - 增量保存数据
    - 收敛后继续输出
    - 完整的类型提示
    """
    
    def __init__(self, config: dict):
        """初始化优化器"""
        self._validate_config(config)
        self._initialize_components(config)
        self._initialize_storage()
        self._initialize_swarm()
        self._print_configuration()
    
    # ==================== 初始化方法 ====================
    
    def _validate_config(self, config: dict):
        """验证配置文件"""
        required_keys = ['parameters', 'pso']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
    
    def _initialize_components(self, config: dict):
        """初始化核心组件"""
        print("=" * 80)
        print("Particle Swarm Optimizer (Refactored)")
        print("=" * 80)
        
        # 目标函数
        self.objective_func = MOOSEObjectiveFunction(config)
        
        # 参数配置
        self.parameters = Parameters.from_dict(config['parameters'])
        
        # PSO 配置
        self.pso_config = PSOConfig.from_dict(config['pso'], self.parameters)
        
        # 工作目录
        self.work_dir = Path(config.get('work_dir', '.'))
        
        # 输出控制
        self.plot_interval = config.get('save_plot_interval', 0)
        
        # 缓存设置
        self.cache_enabled = self.objective_func.cache.enabled
    
    def _initialize_storage(self):
        """初始化数据存储"""
        # 历史记录（使用新的 EvaluationHistory 类）
        self.evaluation_history = EvaluationHistory()
        self.iteration_history: List[IterationStateSummary] = []
        
        # 状态变量
        self.printed_header = False
        self.pso_start_time = time.time()
        self.converged = False
        self.convergence_iteration: Optional[int] = None
        
        # 粒子群状态
        self.swarm: Optional[SwarmState] = None
        
        # 输出文件
        self.evaluation_file = self.work_dir / "pso_evaluation_history.csv"
        self.iteration_file = self.work_dir / "pso_iteration_history.csv"
        self.result_file = self.work_dir / "pso_optimization_result.yaml"
        
        self.particle_plots_dir = self.work_dir / "particle_plots"
        self.particle_plots_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        lb, ub = self.bounds
        n_particles = self.n_particles
        n_dims = self.n_dims
        
        # 随机初始化位置
        position = np.random.uniform(low=lb, high=ub, size=(n_particles, n_dims))
        
        # 初始化速度
        velocity_factor = 0.4
        velocity = np.random.uniform(
            low=-velocity_factor * np.abs(ub - lb),
            high=velocity_factor * np.abs(ub - lb),
            size=(n_particles, n_dims)
        )
        
        # 初始化最优值
        pbest_pos = position.copy()
        pbest_cost = np.full(n_particles, np.inf)
        cost = np.full(n_particles, np.inf)
        
        self.swarm = SwarmState(
            position=position,
            velocity=velocity,
            cost=cost,
            pbest_pos=pbest_pos,
            pbest_cost=pbest_cost,
            best_id=None,
            best_pos=None,
            best_cost=np.inf,
            prev_position=position.copy()
        )
    
    def _print_configuration(self):
        """打印配置信息"""
        print(self.parameters)
        print(self.pso_config)
    
    # ==================== 属性方法 ====================
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取参数边界"""
        params = self.pso_config.parameters
        return (params.lower_bounds, params.upper_bounds)
    
    @property
    def n_dims(self) -> int:
        """参数维度"""
        return self.pso_config.n_dims
    
    @property
    def n_particles(self) -> int:
        """粒子数量"""
        return self.pso_config.n_particles
    
    @property
    def max_iterations(self) -> int:
        """最大迭代数"""
        return self.pso_config.max_iterations
    
    # ==================== 主循环 ====================
    
    def run(self) -> OptimizationResult:
        """
        运行 PSO 优化
        
        返回：
            OptimizationResult: 优化结果对象
        """
        print("\nStarting optimization...")
        print("-" * 80)
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # 1. 执行迭代（评估粒子）
            iter_eval = self._execute_iteration(iteration)
            
            # 2. 更新粒子群状态
            iter_state = self._update_swarm_state(iteration, iter_eval, iter_start)
            
            # 3. 打印进度
            self._print_iteration_info(iter_state)
            
            # 4. 检查收敛（但不立即退出）
            just_converged = self._check_convergence(iteration)
            
            # 5. 输出结果（收敛时也输出）
            if self._should_output(iteration) or just_converged:
                self._save_iteration_data(iter_eval, iter_state)
                self._plot_iteration(iteration, iter_state, iter_eval)
            
            # 6. 收敛后退出
            if just_converged:
                break
            
            # 7. 更新粒子（最后一次迭代不更新）
            if iteration < self.max_iterations - 1:
                self._update_particles(iteration)
        
        # 构建并保存最终结果
        result = self._build_final_result()
        self._save_final_result(result)
        self._print_summary(result)
        
        return result
    
    # ==================== 迭代执行 ====================
    
    def _execute_iteration(self, iteration: int) -> IterationEvaluation:
        """
        执行单次迭代的评估
        
        参数：
            iteration: 迭代编号（从 0 开始）
            
        返回：
            IterationEvaluation: 评估结果
        """
        # 评估所有粒子
        eval_results = self.objective_func(self.swarm.position, iteration)
        self.swarm.cost = eval_results['cost']
        
        # 构建评估记录
        particle_ids = np.arange(self.n_particles)
        eval_ids = iteration * self.n_particles + particle_ids
        
        iter_eval = IterationEvaluation(
            iteration=iteration,
            eval_ids=eval_ids,
            particle_ids=particle_ids,
            parameters=self.swarm.position.copy(),
            costs=self.swarm.cost.copy(),
            elapsed_times=eval_results['elapsed_time'],
            statuses=np.array(eval_results['status'])
        )
        
        self.evaluation_history.append(iter_eval)
        return iter_eval
    
    def _update_swarm_state(self, iteration: int, iter_eval: IterationEvaluation,
                           iter_start_time: float) -> IterationStateSummary:
        """
        更新粒子群状态
        
        参数：
            iteration: 迭代编号
            iter_eval: 当前迭代的评估结果
            iter_start_time: 迭代开始时间
            
        返回：
            IterationStateSummary: 迭代状态摘要
        """
        # 更新个体最优
        better_mask = self.swarm.cost < self.swarm.pbest_cost
        self.swarm.pbest_cost[better_mask] = self.swarm.cost[better_mask]
        self.swarm.pbest_pos[better_mask] = self.swarm.position[better_mask]
        
        # 更新全局最优
        min_cost_idx = np.argmin(self.swarm.cost)
        if self.swarm.cost[min_cost_idx] < self.swarm.best_cost:
            self.swarm.best_cost = self.swarm.cost[min_cost_idx]
            self.swarm.best_pos = self.swarm.position[min_cost_idx].copy()
            self.swarm.best_id = int(iter_eval.eval_ids[min_cost_idx])
        
        # 创建迭代摘要
        iter_state = IterationStateSummary(
            iteration=iteration,
            global_best_cost=float(self.swarm.best_cost),
            global_best_eval_id=self.swarm.best_id,
            global_best_params=self.swarm.best_pos.tolist(),
            iter_best_cost=float(iter_eval.best_cost),
            iter_best_eval_id=int(iter_eval.best_eval_id),
            iter_best_params=iter_eval.best_parameters.tolist(),
            iter_elapsed=time.time() - iter_start_time,
            total_elapsed=time.time() - self.pso_start_time,
            avg_cost=float(iter_eval.avg_cost)
        )
        
        self.iteration_history.append(iter_state)
        return iter_state
    
    def _update_particles(self, iteration: int):
        """
        更新粒子速度和位置
        
        参数：
            iteration: 当前迭代编号
        """
        lb, ub = self.bounds
        
        # 动态惯性权重（当前 w_max == w_min，简化为常量）
        w = 0.8
        
        # 认知和社会分量（向量化）
        r1 = np.random.random(self.swarm.position.shape)
        r2 = np.random.random(self.swarm.position.shape)
        
        cognitive = self.pso_config.c1 * r1 * (self.swarm.pbest_pos - self.swarm.position)
        social = self.pso_config.c2 * r2 * (self.swarm.best_pos - self.swarm.position)
        
        # 保存上次位置（用于绘图）
        self.swarm.prev_position = self.swarm.position.copy()
        
        # 更新速度和位置
        self.swarm.velocity = w * self.swarm.velocity + cognitive + social
        new_position = self.swarm.position + self.swarm.velocity
        
        # 边界处理（向量化反弹法）
        lower_violation = new_position < lb
        upper_violation = new_position > ub
        
        # 反弹
        new_position = np.where(lower_violation, 2 * lb - new_position, new_position)
        new_position = np.where(upper_violation, 2 * ub - new_position, new_position)
        
        # 反弹后速度衰减
        self.swarm.velocity = np.where(
            lower_violation | upper_violation,
            -0.75 * self.swarm.velocity,
            self.swarm.velocity
        )
        
        # 最终裁剪（确保在边界内）
        self.swarm.position = np.clip(new_position, lb, ub)
    
    # ==================== 收敛检查 ====================
    
    def _check_convergence(self, iteration: int) -> bool:
        """
        检查是否收敛
        
        参数：
            iteration: 当前迭代编号
            
        返回：
            bool: 是否刚刚收敛（用于触发最后一次输出）
        """
        if self.converged:
            return False  # 已经收敛过了
        
        conv_window = self.pso_config.convergence_window
        conv_threshold = self.pso_config.convergence_threshold
        
        # 检查条件
        if (conv_window is None or 
            conv_threshold is None or
            len(self.iteration_history) < conv_window):
            return False
        
        # 提取最近的平均成本
        recent_avgs = [s.avg_cost for s in self.iteration_history[-conv_window:]]
        mean_avg = sum(recent_avgs) / conv_window
        
        # 判断收敛
        if mean_avg < conv_threshold:
            print(f"\n{'='*80}")
            print(f"CONVERGED at iteration {iteration}:")
            print(f"  Average cost: {mean_avg:.6e} < {conv_threshold:.6e}")
            print(f"{'='*80}\n")
            
            self.converged = True
            self.convergence_iteration = iteration
            return True  # 刚刚收敛
        
        return False
    
    # ==================== 输出控制 ====================
    
    def _should_output(self, iteration: int) -> bool:
        """判断是否应该输出"""
        if self.plot_interval == 0:
            return True  # 每次都输出
        
        return (
            iteration % self.plot_interval == 0 or
            iteration == 0 or
            iteration == self.max_iterations - 1
        )
    
    def _print_iteration_info(self, iter_state: IterationStateSummary):
        """打印迭代信息"""
        if not self.printed_header:
            print(IterationStateSummary.get_header(self.parameters.names))
            print(IterationStateSummary.get_separator(len(self.parameters.names)))
            self.printed_header = True
        
        print(iter_state)
    
    def _save_iteration_data(self, iter_eval: IterationEvaluation, 
                            iter_state: IterationStateSummary):
        """
        增量保存迭代数据
        
        参数：
            iter_eval: 评估结果
            iter_state: 迭代状态
        """
        # 增量保存评估历史
        eval_df = iter_eval.to_dataframe(self.parameters.names)
        mode = 'w' if not self.evaluation_file.exists() else 'a'
        header = not self.evaluation_file.exists()
        eval_df.to_csv(self.evaluation_file, mode=mode, header=header, index=False)
        
        # 增量保存迭代历史
        state_dict = {
            'iteration': iter_state.iteration,
            'global_best_cost': iter_state.global_best_cost,
            'global_best_eval_id': iter_state.global_best_eval_id,
            'iter_best_cost': iter_state.iter_best_cost,
            'iter_best_eval_id': iter_state.iter_best_eval_id,
            'avg_cost': iter_state.avg_cost,
            'iter_elapsed': iter_state.iter_elapsed,
            'total_elapsed': iter_state.total_elapsed,
            **{f'global_best_{name}': iter_state.global_best_params[i]
               for i, name in enumerate(self.parameters.names)},
            **{f'iter_best_{name}': iter_state.iter_best_params[i]
               for i, name in enumerate(self.parameters.names)}
        }
        
        mode = 'w' if iter_state.iteration == 0 else 'a'
        header = iter_state.iteration == 0
        pd.DataFrame([state_dict]).to_csv(
            self.iteration_file, mode=mode, header=header, index=False
        )
    
    def _plot_iteration(self, iteration: int, iter_state: IterationStateSummary,
                       iter_eval: IterationEvaluation):
        """
        绘制迭代图表
        
        参数：
            iteration: 迭代编号
            iter_state: 迭代状态
            iter_eval: 评估结果
        """
        # 提取历史数据
        cost_history = [s.global_best_cost for s in self.iteration_history]
        avg_cost_history = [s.avg_cost for s in self.iteration_history]
        
        # 绘制收敛曲线
        plot_convergence_curve(
            cost_history,
            avg_cost_history,
            iteration,
            self.work_dir
        )
        
        # 绘制粒子分布（仅 2D）
        if self.n_dims == 2:
            params = self.pso_config.parameters
            plot_particle_distribution(
                iter_eval,
                iter_state,
                self.swarm,
                params.names,
                (params.lower_bounds, params.upper_bounds),
                self.particle_plots_dir
            )
            
            if self.objective_func.realtime_gif:
                update_gif_realtime(self.work_dir)
    
    # ==================== 结果处理 ====================
    
    def _build_final_result(self) -> OptimizationResult:
        """构建最终结果"""
        # 构建参数字典
        best_params = {
            name: float(value) 
            for name, value in zip(self.parameters.names, self.swarm.best_pos)
        }
        
        # 构建缓存统计
        cache_stats = {}
        if self.cache_enabled:
            cache_stats = {
                'exact_hits': self.objective_func.cache.exact_hits,
                'near_hits': self.objective_func.cache.near_hits,
                'interp_hits': self.objective_func.cache.interp_hits,
                'misses': self.objective_func.cache.misses,
                'total_hits': (self.objective_func.cache.exact_hits + 
                             self.objective_func.cache.near_hits + 
                             self.objective_func.cache.interp_hits),
            }
        
        return OptimizationResult(
            best_cost=float(self.swarm.best_cost),
            best_params=best_params,
            best_eval_id=self.swarm.best_id,
            total_iterations=len(self.iteration_history),
            total_evaluations=len(self.evaluation_history) * self.n_particles,
            total_time=time.time() - self.pso_start_time,
            converged=self.converged,
            convergence_iteration=self.convergence_iteration,
            cache_enabled=self.cache_enabled,
            cache_stats=cache_stats
        )
    
    def _save_final_result(self, result: OptimizationResult):
        """保存最终结果"""
        with open(self.result_file, 'w') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False)
        print(f"\n✓ 优化结果已保存: {self.result_file.name}")
    
    def _print_summary(self, result: OptimizationResult):
        """打印优化摘要"""
        print(result)


# ==================== 函数式接口 ====================

def run_pso_optimization(config: dict) -> OptimizationResult:
    """
    运行 PSO 优化（函数式接口）
    
    参数：
        config: 配置字典
        
    返回：
        OptimizationResult: 优化结果对象
    """
    optimizer = PSOOptimizer(config)
    return optimizer.run()
