
import numpy as np
import time
from dataclasses import dataclass, field
from .tools import func_transformer
from pathlib import Path
import re

@dataclass
class PSOConfig:
    """PSO Algorithm Configuration"""
    n_dims: int
    n_particles: int
    max_iters: int
    w: float = 0.8  # inertia weight
    c1: float = 1.5  # cognitive coefficient
    c2: float = 1.5  # social coefficient
    lb: list[float] = None  # lower bounds
    ub: list[float] = None  # upper bounds

    cmode: str = 'avg_cost'  # convergence mode: 'best_cost' or 'avg_cost'
    cthreshold: float = 1  # convergence threshold
    cwindow: int = 20  # convergence window size

    def __post_init__(self):
        # 自动转换列表为 numpy 数组
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        assert self.ub.shape[0] == self.lb.shape[0] == self.n_dims, "Bounds shape mismatch with n_dims"

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            n_dims=config_dict.get('n_dims'),
            n_particles=config_dict.get('n_particles', 5),
            max_iters=config_dict.get('max_iters', 10),
            w=config_dict.get('w', 0.8),
            c1=config_dict.get('c1', 1.5),
            c2=config_dict.get('c2', 1.5),
            lb=np.array(config_dict.get('lb')),
            ub=np.array(config_dict.get('ub')),
            cmode=config_dict.get('cmode', 'avg_cost'),
            cthreshold=config_dict.get('cthreshold', 1),
            cwindow=config_dict.get('cwindow', 20),
        )
    
@dataclass
class ParticleSwarm:
    """Particle Swarm Optimization (PSO) Implementation"""
    cfg : PSOConfig

    position: np.ndarray
    
    velocity: np.ndarray 

    status: list = field(init=False)  # 评估状态列表 (n_particles,)
    
    cost: np.ndarray = field(init=False)     # 当前成本 (n_particles,)

    id: np.ndarray = field(init=False)    # 粒子ID (n_particles,)
    
    pbest_pos: np.ndarray = field(init=False)  # 个体最优位置 (n_particles, n_dims)
    pbest_cost: np.ndarray = field(init=False)  # 个体最优成本 (n_particles,)
    
    ibest_pos: np.ndarray = field(init=False)  # 迭代最优位置 (n_dims, )
    ibest_cost: float = field(init=False)  # 迭代最优成本
    ibest_id: int = field(init=False)  # 迭代最优评估ID 

    best_pos: np.ndarray = field(init=False)  # 全局最优位置 (n_dims,)
    best_cost: float = field(init=False)   # 全局最优成本
    best_id: int = field(init=False)  # 全局最优评估ID

    avg_cost: float = field(init=False)  # 平均成本
    
    prev_position: np.ndarray = field(init=False)  # 上次位置 (用于可视化轨迹)

    def __post_init__(self):
        # 初始化成本和ID数组
        n_particles = self.cfg.n_particles

        self.cost = np.full(n_particles, np.inf)
        self.status = [''] * n_particles
        self.id = np.arange(n_particles)

        # 初始化个体最优
        self.pbest_pos = self.position.copy()
        self.pbest_cost = self.cost.copy()
        
        # 初始化迭代最优
        min_idx = np.argmin(self.cost)
        self.ibest_pos = self.position[min_idx].copy()
        self.ibest_cost = np.inf
        self.ibest_id = -1  # 尚未设置
        
        # 初始化全局最优
        self.best_pos = self.position[min_idx].copy()
        self.best_cost = self.cost[min_idx]
        self.best_id = -1  # 尚未设置
        
        # 初始化上次位置
        self.prev_position = None


    def update_best(self):
        """更新粒子状态"""
        
        self.avg_cost = np.mean(self.cost)  # 更新平均成本

        # 更新个体最优
        better_mask = self.cost < self.pbest_cost
        self.pbest_cost[better_mask] = self.cost[better_mask]
        self.pbest_pos[better_mask] = self.position[better_mask]
        
        # 更新迭代最优
        min_idx = np.argmin(self.cost)
        self.ibest_cost = self.cost[min_idx]
        self.ibest_pos = self.position[min_idx].copy()
        self.ibest_id = self.id[min_idx]
        
        # 更新全局最优
        min_idx = np.argmin(self.cost)
        if self.cost[min_idx] < self.best_cost:
            self.best_cost = self.cost[min_idx]
            self.best_pos = self.position[min_idx].copy()
            self.best_id = self.id[min_idx] 

    def update_id(self):
        """更新粒子评估ID"""
        self.id += self.cfg.n_particles  # 更新评估ID
    
    def update_velocity(self):
        """更新粒子速度
        
        Args:
            w: 惯性权重
            c1: 认知系数 (个体学习因子)
            c2: 社会系数 (群体学习因子)
        """
        w,c1,c2 = self.cfg.w, self.cfg.c1, self.cfg.c2

        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        
        cognitive = c1 * r1 * (self.pbest_pos - self.position)
        social = c2 * r2 * (self.best_pos - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        """更新粒子位置并处理边界
        
        Args:
            lb: 下界
            ub: 上界
        """
        lb, ub = self.cfg.lb, self.cfg.ub
        # 保存当前位置用于可视化
        self.prev_position = self.position.copy()
        
        # 更新位置
        new_position = self.position + self.velocity
        
        # 边界处理（反弹）
        for dim in range(self.cfg.n_dims):
            # 下边界
            lower_violation = new_position[:, dim] < lb[dim]
            if np.any(lower_violation):
                new_position[lower_violation, dim] = 2 * lb[dim] - new_position[lower_violation, dim]
                self.velocity[lower_violation, dim] *= -0.75
            
            # 上边界
            upper_violation = new_position[:, dim] > ub[dim]
            if np.any(upper_violation):
                new_position[upper_violation, dim] = 2 * ub[dim] - new_position[upper_violation, dim]
                self.velocity[upper_violation, dim] *= -0.75
        
        # 最终裁剪确保在边界内
        self.position = np.clip(new_position, lb, ub)
   
class PSOOptimizer():
    """
    Do PSO (Particle swarm optimization) algorithm.
    """

    def __init__(self, func, config:PSOConfig, 
                 output_dir='pso_output', output_interval=1, 
                 callback=None, callback_interval=1, 
                 ):
        """
        初始化PSO优化器
        
        Args:
            func: 目标函数
            config: PSO配置参数
            callback: 回调函数，签名为 callback(optimizer, iter_num)
            callback_interval: 回调函数调用间隔（迭代次数）
        """
        self.cfg = config
        self.func = func_transformer(func)

        v_mag = 0.4 * np.abs(self.cfg.ub - self.cfg.lb)
        init_pos = np.random.uniform(self.cfg.lb, self.cfg.ub, size=(self.cfg.n_particles, self.cfg.n_dims))
        init_vel = np.random.uniform(-v_mag, v_mag, size=(self.cfg.n_particles, self.cfg.n_dims))
        self.swarm = ParticleSwarm(self.cfg, init_pos, init_vel)

        self.start_time = None
        self.elapsed_time = 0.0

        self.converged = False


        # 输出设置
        self.output_dir = output_dir
        self.output_interval = output_interval

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.particle_history_file = self.output_dir / "pso_particle_history.csv"
        self.iter_history_file = self.output_dir / "pso_iteration_history.log"

        # 清空文件内容,但不删除文件
        if self.iter_history_file.exists():
            with open(self.iter_history_file, 'w') as f:
                pass

        if self.particle_history_file.exists():
            with open(self.particle_history_file, 'w') as f:
                pass

        self.last_saved_iter = -1  # 上次保存的迭代编号
        
        # 回调函数设置
        self.callback = callback
        self.callback_interval = callback_interval
        self.verbose = True

        self.iter_history = {'iteration': [],
                             'best_cost': [],
                             'best_id': [],
                             'best_pos': [],
                             'iter_cost': [],                             
                             'iter_id': [],
                             'iter_pos': [],
                             'avg_cost': [],
                             'total_elapsed': []
                             }
        
        # 粒子级别的详细记录
        self.particle_history = {
            'iteration': [],
            'eval_id': [],
            'particle_id': [],
            'cost': []
        }
        # 动态添加位置列 x0, x1, ..., x{n_dims-1}
        for i in range(self.cfg.n_dims):
            self.particle_history[f'x{i}'] = []

        self.particle_history['status'] = []

        # 输出并行相关信息
        if hasattr(self.func, '_parallel_info'):
            info = self.func._parallel_info
            print(f"\n{'='*60}")
            print(f"  并行计算配置")
            print(f"{'='*60}")
            print(f"  模式: {info['mode']}")
            print(f"  使用核心数: {info['n_processes']}")
            print(f"  系统总核心数: {info['total_cores']}")
            
            print(f"{'='*60}\n")
        


    def print_statistics(self, iter_num, current_time, printed_header):
        """
        打印并记录统计信息到控制台和文件
        
        Args:
            iter_num: 当前迭代次数
            current_time: 当前时间
            printed_header: 是否已打印表头
        """
        # 构建表头
        data_line = ""
        if not printed_header:
            gb_param_names = [f"Gb_x{i}" for i in range(self.cfg.n_dims)]
            it_param_names = [f"It_x{i}" for i in range(self.cfg.n_dims)]
            gb_param_headers = ' '.join([f"{name:>8}" for name in gb_param_names])
            it_param_headers = ' '.join([f"{name:>8}" for name in it_param_names])
            header = (f"{'Iter':>4} {'GbCost':>10} {'GbID':>6} {gb_param_headers} "
                    f"{'ItCost':>10} {'ItID':>6} {it_param_headers} "
                    f"{'AvgCost':>10} {'Total(s)':>8}")
            data_line += header + '\n'

        
        # 构建数据行
        best_pos_str = ' '.join([f"{val:8.2f}" for val in self.swarm.best_pos])
        ibest_pos_str = ' '.join([f"{val:8.2f}" for val in self.swarm.ibest_pos])
        total_elapsed = current_time - self.start_time
        
        data_line += (f"{iter_num:4d} "
                    f"{self.swarm.best_cost:10.4f} {self.swarm.best_id:6d} {best_pos_str} "
                    f"{self.swarm.ibest_cost:10.4f} {self.swarm.ibest_id:6d} {ibest_pos_str} "
                    f"{self.swarm.avg_cost:10.4f} "
                    f"{total_elapsed:8.2f}")
        
        # 输出到控制台
        print(data_line)

        # 输出到文件

        if self.iter_history_file:
            with open(self.iter_history_file, 'a') as f:
                f.write(data_line + '\n')

        return data_line

        
    def record_iteration_history(self, iter_num, current_time):
        """记录迭代历史"""
        self.elapsed_time = current_time - self.start_time

        self.iter_history['iteration'].append(iter_num)
        self.iter_history['best_id'].append(self.swarm.best_id)
        self.iter_history['best_cost'].append(self.swarm.best_cost)
        self.iter_history['best_pos'].append(self.swarm.best_pos.copy())
        self.iter_history['iter_id'].append(self.swarm.ibest_id)
        self.iter_history['iter_pos'].append(self.swarm.ibest_pos.copy())
        self.iter_history['iter_cost'].append(self.swarm.ibest_cost)
        self.iter_history['avg_cost'].append(self.swarm.avg_cost)
        self.iter_history['total_elapsed'].append(self.elapsed_time)
    
    def record_particle_history(self, iter_num):
        """
        记录每个粒子的详细信息（向量化优化版）
        
        Args:
            iter_num: 当前迭代次数
        """
        n = self.cfg.n_particles
        
        # 向量化记录基本信息
        self.particle_history['iteration'].extend([iter_num] * n)
        self.particle_history['eval_id'].extend(self.swarm.id.tolist())
        self.particle_history['particle_id'].extend(range(n))
        self.particle_history['cost'].extend(self.swarm.cost.tolist())
        self.particle_history['status'].extend(self.swarm.status)
        
        # 向量化记录各维度位置
        for dim_idx in range(self.cfg.n_dims):
            self.particle_history[f'x{dim_idx}'].extend(
                self.swarm.position[:, dim_idx].tolist()
            )

    def output_particle_history(self, iter_start, iter_end):
        """
        输出粒子历史到文件（增量保存）
        
        Args:
            out_file: 输出文件路径
            iter_start: 起始迭代次数（包含）
            iter_end: 结束迭代次数（包含）
        """
        import pandas as pd

        out_file = self.particle_history_file

        # 计算在 particle_history 中的索引范围
        start_idx = iter_start * self.cfg.n_particles
        end_idx = (iter_end + 1) * self.cfg.n_particles
        
        # 提取指定范围的数据
        new_data = {}
        for key in self.particle_history.keys():
            new_data[key] = self.particle_history[key][start_idx:end_idx]
        
        df_new = pd.DataFrame(new_data)
        
        # 首次保存时写入表头，后续追加时不写表头
        mode = 'w' if iter_start == 0 else 'a'
        header = iter_start == 0
        df_new.to_csv(out_file, mode=mode, header=header, index=False)

    def check_convergence(self, mode, threshold, window):
        """
        检查优化是否收敛
        
        Args:
            threshold: 收敛阈值
            window: 检查窗口大小（最近几次迭代）
        Returns:
            bool: 是否收敛
        """

        if mode == 'best_cost':
            history = self.iter_history['best_cost']
        elif mode == 'avg_cost':
            history = self.iter_history['avg_cost']
        else:
            raise ValueError(f"未知的收敛模式: {mode}")

        if len(history) < window:
            return False
        # 从 iteration_history 提取最近的平均成本
        recent_costs = history[-window:]
        max_avg = max(recent_costs)
        min_avg = min(recent_costs)
        mean_avg = sum(recent_costs) / window

        if mean_avg < threshold:
            self.converged = True
            return True
        
        return False
    
    def run(self):
        """运行 PSO 优化算法"""

        printed_header = False
        self.start_time = time.time()

        # 打开日志文件
        # self.log_file = open('pso_statistics.log', 'w')
        

        for iter_num in range(self.cfg.max_iters):
            # 评估目标函数（传递 eval_id）
            self.swarm.cost, self.swarm.status = self.func(self.swarm.position, self.swarm.id)
            
            # 更新最优值
            self.swarm.update_best()

            current_time = time.time()
            
            # 打印进度
            if self.verbose:
                self.print_statistics(iter_num, current_time, printed_header)
                printed_header = True

            # 检查收敛
            is_converged = self.check_convergence(self.cfg.cmode, 
                                                  self.cfg.cthreshold,
                                                  self.cfg.cwindow)

            # 记录历史
            self.record_iteration_history(iter_num, current_time)
            
            # 记录粒子的详细信息
            self.record_particle_history(iter_num)

            # 增量保存粒子历史到文件
            if (iter_num + 1) % self.output_interval == 0 \
                    or iter_num == self.cfg.max_iters - 1 \
                    or is_converged:
                # 保存从上次保存位置到当前位置的数据
                iter_start = self.last_saved_iter + 1
                iter_end = iter_num
                self.output_particle_history(iter_start, iter_end)
                self.last_saved_iter = iter_num
            
            # 调用回调函数
            if self.callback is not None and (iter_num % self.callback_interval == 0 
                                              or iter_num == self.cfg.max_iters - 1
                                              or is_converged):
                self.callback(self, iter_num)

            if is_converged:
                print(f"优化在迭代 {iter_num} 时收敛。")
                break

            # 更新速度和位置
            self.swarm.update_velocity()
            self.swarm.update_position()
            self.swarm.update_id()
        
        return self.swarm.best_pos, self.swarm.best_cost
