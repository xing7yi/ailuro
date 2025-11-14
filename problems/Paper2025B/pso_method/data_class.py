"""
数据类定义模块

- Parameter: 单个参数数据类
- Parameters: 参数集合数据类
- PSOConfig: PSO 配置数据类
- SwarmState: 粒子群状态数据类
- IterationState: 迭代状态数据类
- OptimizationResult: 优化结果数据类
""" 

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class EvaluationResults:
    """评估结果数据类

    """
    cost : List[float]
    



@dataclass
class Parameters:
    """参数集合数据类"""
    names: List[str]  # 参数名称列表
    lower_bounds: np.ndarray  # 参数下界列表
    upper_bounds: np.ndarray  # 参数上界列表
    descriptions: Optional[List[str]] = field(default_factory=list)  # 参数描述列表

    @classmethod
    def from_dict(cls, param_config: dict) -> 'Parameters':
        return cls(
            names=param_config['names'],
            lower_bounds= np.array(param_config['lower_bounds']),
            upper_bounds=np.array(param_config['upper_bounds']),
            descriptions=param_config.get('descriptions', [])
        )
    
    def __str__(self) -> str:
        """格式化输出参数信息"""
        lines = ["Parameter Configuration:"]
        for name, lb, ub, desc in zip(self.names, self.lower_bounds, self.upper_bounds, self.descriptions):
            lines.append(
                f"  {name}: {desc} "
                f"[{lb}, {ub}]"
            )
        return "\n".join(lines)

@dataclass(frozen=True)
class PSOConfig:
    """PSO 配置数据类
    
    封装所有 PSO 算法相关的配置参数。
    """
    # 参数配置
    # param_config = config['parameters']
    # param_names: List[str]
    # n_params: int = field(init=False)  # 派生字段
    # lower_bounds: np.ndarray
    # upper_bounds: np.ndarray
    n_dims: int = field(init=False) # 参数维度
    parameters: Parameters

    # 粒子群参数
    n_particles: int # 粒子数量
    max_iterations: int  # 最大迭代次数
    
    # PSO 算法参数
    cognitive_param: float  # 认知参数 (c1)
    social_param: float  # 社会参数 (c2)
    inertia_weight: float  # 惯性权重 (w)
    inertia_weight_decay: Optional[float] = None  # 惯性权重衰减因子（可选）
    
    # 收敛判断参数
    convergence_threshold: float  # 收敛阈值
    convergence_window: int  # 收敛窗口（迭代次数）
    
    @classmethod
    def from_dict(cls, pso_config: dict, parameters: Parameters) -> 'PSOConfig':
        """从配置字典创建 PSOConfig 对象
        
        参数：
            config: dict, 完整的配置字典

        返回：
            PSOConfig: PSO 配置对象
        """
        
        return cls(
            parameters = parameters,
            n_particles=pso_config['n_particles'],
            max_iterations=pso_config['max_iterations'],
            cognitive_param=pso_config['cognitive_param'],
            social_param=pso_config['social_param'],
            inertia_weight=pso_config['inertia_weight'],
            convergence_threshold=float(pso_config.get('convergence_threshold', 1e-6)),
            convergence_window=int(pso_config.get('convergence_window', 10)),
        )
    
    def get_options_dict(self) -> Dict[str, float]:
        """获取 PSO 算法选项字典
        
        返回：
            dict: 包含 c1, c2, w 的字典
        """
        return {
            'c1': self.cognitive_param,
            'c2': self.social_param,
            'w': self.inertia_weight
        }
    
    def __post_init__(self):
        """后初始化：计算派生字段"""
        self.n_dims = len(self.parameters.names)
    
    def __str__(self) -> str:
        """格式化输出配置信息"""
        return (
            f"PSO Optimization Configuration:\n"
            f"  Number of particles: {self.n_particles}\n"
            f"  Maximum iterations:  {self.max_iterations}\n"
            f"  Cognitive parameter (c1): {self.cognitive_param}\n"
            f"  Social parameter (c2):    {self.social_param}\n"
            f"  Inertia weight (w):       {self.inertia_weight}\n"
            f"  Convergence threshold:    {self.convergence_threshold}\n"
            f"  Convergence window:       {self.convergence_window}"
        )


@dataclass
class SwarmState:
    """粒子群状态数据类（仅保存当前状态）"""
    prev_position: np.ndarray # 上次位置
    position: np.ndarray  # 当前位置
    velocity: np.ndarray  # 当前速度
    cost: np.ndarray      # 当前成本
    pbest_pos: np.ndarray  # 个体最优位置
    pbest_cost: np.ndarray  # 个体最优成本
    best_pos: np.ndarray  # 全局最优位置
    best_cost: float = np.inf  # 全局最优成本
    best_id: int = None  # 全局最优评估ID


@dataclass
class IterationEvaluation:
    """单次迭代的评估结果集合
    
    存储一次迭代中所有粒子的评估数据，支持增量更新和批量操作。
    """
    iteration: int  # 迭代次数
    eval_ids: np.ndarray
    particle_ids: np.ndarray
    parameters: np.ndarray
    costs: np.ndarray
    elapsed_times: np.ndarray
    statuses: np.ndarray
    
    @property
    def n_particles(self) -> int:
        """粒子数量"""
        return len(self.eval_ids)
    
    @property
    def best_cost(self) -> float:
        """最优成本"""
        return np.min(self.costs) if len(self.costs) > 0 else np.inf
    
    @property
    def best_particle_id(self) -> int:
        """最优粒子ID"""
        return self.particle_ids[np.argmin(self.costs)] if len(self.costs) > 0 else -1
    
    @property
    def avg_cost(self) -> float:
        """平均成本"""
        return np.mean(self.costs) if len(self.costs) > 0 else np.nan
    
    @property
    def success_count(self) -> int:
        """成功评估数量"""
        return np.sum(self.statuses == 'success')
    
    @property
    def cache_hit_count(self) -> int:
        """缓存命中数量"""
        return np.sum(self.statuses == 'cache_hit')
    
    @property
    def total_elapsed_time(self) -> float:
        """总耗时"""
        return np.sum(self.elapsed_times)

    def to_dataframe(self, param_names: List[str]) -> pd.DataFrame:
        """转换为 DataFrame
        
        参数：
            param_names: 参数名称列表
            
        返回：
            DataFrame: 包含所有评估记录的表格
        """
        if len(self.eval_ids) == 0:
            columns = ['iteration', 'eval_id', 'particle_id'] + param_names + ['cost', 'elapsed_time', 'status']
            return pd.DataFrame(columns=columns)
        
        data = {
            'iteration': [self.iteration] *  len(self.eval_ids),
            'eval_id': self.eval_ids,
            'particle_id': self.particle_ids,
        }
        
        # 添加参数列
        for i, name in enumerate(param_names):
            data[name] = self.parameters[:, i]
        
        data.update({
            'cost': self.costs,
            'elapsed_time': self.elapsed_times,
            'status': self.statuses
        })
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """格式化输出"""
        if len(self.eval_ids) == 0:
            return f"Iteration {self.iteration}: Empty (0 particles)"
        
        return (
            f"Iteration {self.iteration}: {self.n_particles} particles\n"
            f"  Best cost:   {self.best_cost:.6f} (particle {self.best_particle_id})\n"
            f"  Avg cost:    {self.avg_cost:.6f}\n"
            f"  Success:     {self.success_count}/{self.n_particles}\n"
            f"  Cache hits:  {self.cache_hit_count}\n"
            f"  Total time:  {self.total_elapsed_time:.3f}s"
        )


@dataclass
class EvaluationHistory:
    """评估历史管理类
    
    管理所有迭代的评估记录，提供批量导出和统计功能。
    """
    evaluations: List[IterationEvaluation] = field(default_factory=list)
    
    def append(self, evaluation: IterationEvaluation):
        """添加一次迭代评估"""
        self.evaluations.append(evaluation)
    
    def to_dataframe(self, param_names: List[str]) -> pd.DataFrame:
        """转换所有历史为 DataFrame
        
        参数：
            param_names: 参数名称列表
            
        返回：
            DataFrame: 包含所有迭代评估记录的表格
        """
        if not self.evaluations:
            columns = ['iteration', 'eval_id', 'particle_id'] + param_names + ['cost', 'elapsed_time', 'status']
            return pd.DataFrame(columns=columns)
        
        dfs = [eval.to_dataframe(param_names) for eval in self.evaluations]
        return pd.concat(dfs, ignore_index=True)
    
    def save_to_csv(self, filepath: str, param_names: List[str]):
        """保存为 CSV 文件
        
        参数：
            filepath: CSV 文件路径
            param_names: 参数名称列表
        """
        df = self.to_dataframe(param_names)
        df.to_csv(filepath, index=False)
    
    def save_to_json(self, filepath: str, param_names: List[str]):
        """保存为 JSON 文件
        
        参数：
            filepath: JSON 文件路径
            param_names: 参数名称列表
        """
        df = self.to_dataframe(param_names)
        df.to_json(filepath, orient='records', indent=2)
    
    @property
    def cost_history(self) -> List[float]:
        """提取每次迭代的最优成本"""
        return [eval.best_cost for eval in self.evaluations]
    
    @property
    def avg_cost_history(self) -> List[float]:
        """提取每次迭代的平均成本"""
        return [eval.avg_cost for eval in self.evaluations]
    
    @property
    def n_iterations(self) -> int:
        """总迭代次数"""
        return len(self.evaluations)
    
    def __len__(self) -> int:
        """返回历史记录数量"""
        return len(self.evaluations)
    
    def __getitem__(self, index: int) -> IterationEvaluation:
        """支持索引访问"""
        return self.evaluations[index]


@dataclass
class IterationStateSummary:
    """迭代状态摘要数据类"""
    iteration: int  # 迭代次数
    
    # 全局最优信息
    global_best_cost: float  # 全局最优成本
    global_best_eval_id: int  # 全局最优评估ID
    global_best_params: List[float]  # 全局最优参数
    
    # 当前迭代最优信息
    iter_best_cost: float  # 当前迭代最优成本
    iter_best_eval_id: int  # 当前迭代最优评估ID
    iter_best_params: List[float]  # 当前迭代最优参数

    # 平均成本
    avg_cost: float  # 当前迭代所有粒子的平均成本
    
    # 时间信息
    iter_elapsed: float  # 本次迭代耗时（秒）
    total_elapsed: float  # 累计耗时（秒）
    

    def __str__(self):
        """格式化输出迭代状态"""
        global_params_str = ' '.join([f"{val:8.2f}" for val in self.global_best_params])
        iter_params_str = ' '.join([f"{val:8.2f}" for val in self.iter_best_params])
        
        return (f"{self.iteration:4d} "
                f"{self.global_best_cost:10.4f} {self.global_best_eval_id:6d} {global_params_str} "
                f"{self.iter_best_cost:10.4f} {self.iter_best_eval_id:6d} {iter_params_str} "
                f"{self.avg_cost:10.4f} "
                f"{self.iter_elapsed:8.3f} {self.total_elapsed:8.2f}")
    
    @staticmethod
    def get_header(param_names: List[str]) -> str:
        """生成表头"""
        param_headers = ' '.join([f"{name:>8}" for name in param_names])
        return (f"\n{'Iter':>4} {'GbCost':>10} {'GbID':>6} {param_headers} "
                f"{'ItCost':>10} {'ItID':>6} {param_headers} "
                f"{'AvgCost':>10} {'ItT(s)':>8} {'Total(s)':>8}")
    
    @staticmethod
    def get_separator(n_params: int) -> str:
        """生成分隔线"""
        return "-" * (4 + 10 + 6 + 8*n_params + 10 + 6 + 8*n_params + n_params*2 + 10 + 8 + 8 + 2)


@dataclass
class OptimizationResult:
    """优化结果数据类
    
    封装PSO优化的所有结果信息，包括最优解、迭代历史、缓存统计等。
    """
    # 最优解信息
    best_cost: float
    best_params: Dict[str, float]  # 已转换为字典格式 {param_name: value}
    best_eval_id: int
    
    # 优化统计
    total_iterations: int
    total_evaluations: int
    total_time: float  # 总耗时（秒）
    converged: bool  # 是否收敛
    convergence_iteration: Optional[int] = None  # 收敛时的迭代次数
    
    # 缓存统计（如果启用）
    cache_enabled: bool = False
    cache_stats: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        """格式化输出优化结果摘要"""
        lines = [
            "\n" + "=" * 80,
            "PSO Optimization Result Summary",
            "=" * 80,
            f"Best Cost:        {self.best_cost:.6f}",
            f"Best Parameters:  {self.best_params}",
            f"Best Eval ID:     {self.best_eval_id}",
            "-" * 80,
            f"Total Iterations: {self.total_iterations}",
            f"Total Evaluations: {self.total_evaluations}",
            f"Total Time:       {self.total_time:.2f} seconds",
            f"Converged:        {'Yes' if self.converged else 'No'}",
        ]
        
        if self.converged and self.convergence_iteration is not None:
            lines.append(f"Converged at Iteration: {self.convergence_iteration}")
        
        if self.cache_enabled:
            lines.append("-" * 80)
            lines.append("Cache Statistics:")
            for key, value in self.cache_stats.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            'best_cost': self.best_cost,
            'best_params': self.best_params,  # 已经是字典格式
            'best_eval_id': self.best_eval_id,
            'total_iterations': self.total_iterations,
            'total_evaluations': self.total_evaluations,
            'total_time': self.total_time,
            'converged': self.converged,
            'convergence_iteration': self.convergence_iteration,
            'cache_enabled': self.cache_enabled,
            'cache_stats': self.cache_stats
        }
    
