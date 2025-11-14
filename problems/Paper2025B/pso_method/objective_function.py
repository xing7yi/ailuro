"""
MOOSE 目标函数模块

该模块包含 MOOSE 仿真的目标函数包装器，用于 PSO 优化。

主要功能：
- MOOSEObjectiveFunction: 包装 MOOSE 仿真，计算目标函数值
- 支持测试模式（使用简单函数替代仿真）
- 支持并行评估多个参数组合
- 集成缓存系统避免重复计算
- 自动保存评估历史
"""

import numpy as np
import time
import subprocess
import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cache import SimulationCache


def test_function_2d(x, y):
    """
    简单的2D测试函数，用于测试模式
    
    该函数在 (0.7, 0.65) 附近有最小值
    参数域: x, y ∈ [0, 1]
    """
    # 移动最优点到 (0.7, 0.65)
    x_opt, y_opt = 0.7, 0.65
    
    # 添加多个局部最小值
    result = (
        # 主要最小值 (0.7, 0.65)
        10 * ((x - x_opt)**2 + (y - y_opt)**2) +
        # 次要最小值 (0.3, 0.3)
        5 * np.exp(-20 * ((x - 0.3)**2 + (y - 0.3)**2)) +
        # 噪声项
        0.5 * np.sin(10*x) * np.sin(10*y)
    )
    
    return result


def rastrigin_2d(x, y):
    """
    2D Rastrigin 函数（经典优化测试函数）
    
    全局最优解在 (0.5, 0.5) 处，最小值为 0
    参数域: x, y ∈ [0, 1]
    
    特点：
    - 多峰函数，有大量局部最小值
    - 全局最优解明确
    - 所有函数值非负
    """
    A = 10
    # 将 [0, 1] 映射到 [-5.12, 5.12]（Rastrigin 标准域）
    x_scaled = (x - 0.5) * 10.24
    y_scaled = (y - 0.5) * 10.24
    
    # Rastrigin 函数
    result = (
        2 * A +
        (x_scaled**2 - A * np.cos(2 * np.pi * x_scaled)) +
        (y_scaled**2 - A * np.cos(2 * np.pi * y_scaled))
    )
    
    return result


class MOOSEObjectiveFunction:
    """
    MOOSE 仿真目标函数包装器
    
    该类封装了 MOOSE 有限元仿真的调用，并计算与实验数据的误差作为目标函数。
    
    主要特性：
    1. 支持测试模式（使用简单函数替代仿真，用于调试）
    2. 支持并行评估（ProcessPoolExecutor）
    3. 集成缓存系统（避免重复计算）
    4. 自动保存评估历史
    5. 计算 RMSE 作为目标函数值
    
    参数：
        config (dict): 配置字典，包含以下关键字段：
            - test_mode: 是否启用测试模式
            - executable: MOOSE 可执行文件路径
            - input_file: MOOSE 输入文件
            - work_dir: 工作目录
            - output_dir: 输出目录
            - objective_csv: 实验数据 CSV 文件路径
            - parameters: 参数配置（名称、路径、上下界）
            - cache: 缓存配置
            - use_parallel: 是否启用并行计算
            - max_workers: 最大并行进程数
    """
    
    def __init__(self, config):
        print("Initializing MOOSE Objective Function...")
        
        # 测试模式开关
        self.test_mode = config.get('test_mode', False)
        
        # 测试函数选择（'default', 'rastrigin'）
        self.test_function_type = config.get('test_function_type', 'default')
        
        # 可视化配置
        self.realtime_gif = config.get('realtime_gif', False)

        # MOOSE 配置
        moose_config = config.get('moose', {})
        
        self.executable = moose_config.get('executable', 'ailuro-opt')
        self.input_file = Path(moose_config.get('input_file', ''))
        self.work_dir = Path(moose_config.get('work_dir', '.'))
        self.output_dir = self.work_dir / Path(moose_config.get('output_dir', 'results'))
        self.mesh_file = moose_config.get('mesh_file', None)
        self.filebase_param = moose_config.get('filebase_param', 'out_name')
        self.timeout = moose_config.get('timeout', 300)
        self.max_disp_param = config.get('max_disp_param', 'uy_max')

        self.result_col_x = config.get('result_col_x', 'disp')
        self.result_col_y = config.get('result_col_y', 'force')

        # 实验数据（位移-力）CSV文件路径（相对 path 可能在 work_dir 中）
        self.objective_csv = Path(config.get('objective_csv')) if not self.test_mode else None

        # 根据实验数据自动确定压缩比
        self.max_disp_value = self.get_max_disp_from_csv() if not self.test_mode else 0

        # 插值使用的固定力点数量（在仿真与实验的共同力区间上等间距）
        self.n_interp_points = int(config.get('n_interp_points', 10))
        self.param_names = config['parameters']['names']
        self.param_paths = config['parameters']['paths']
        self.n_params = len(self.param_names)
        
        # 保存参数边界（用于归一化）
        self.lower_bounds = np.array(config['parameters']['lower_bounds'])
        self.upper_bounds = np.array(config['parameters']['upper_bounds'])
        
        self.eval_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.eval_history = []
        
        # 并行计算配置
        self.use_parallel = config.get('use_parallel', True)
        self.max_workers = config.get('max_workers', multiprocessing.cpu_count() // 2)
        
        # 缓存配置
        cache_config = config.get('cache', {})
        cache_enabled = cache_config.get('enabled', False)
        
        if cache_enabled:
            cache_file = self.work_dir / cache_config.get('file', 'simulation_cache.db')
            self.cache = SimulationCache(
                cache_file=cache_file,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                exact_decimals=cache_config.get('exact_decimals', 4),
                near_tol=cache_config.get('near_tolerance', 0.01),
                interp_method=cache_config.get('interpolation_method', 'nearest'),
                max_neighbors=cache_config.get('max_neighbors', 5),
                enabled=True,
                version=cache_config.get('version', '1.0')
            )
            print(f"  缓存系统: 启用")
            print(f"    - 文件: {cache_file}")
            print(f"    - 精确匹配: {cache_config.get('exact_decimals', 4)} 位小数")
            print(f"    - 邻近容差: {cache_config.get('near_tolerance', 0.01)}")
            print(f"    - 插值方法: {cache_config.get('interpolation_method', 'nearest')}")
        else:
            self.cache = SimulationCache(
                cache_file='dummy.db',
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                enabled=False
            )
            print(f"  缓存系统: 禁用")
        
        if self.test_mode:
            func_name = 'rastrigin_2d' if self.test_function_type == 'rastrigin' else 'test_function_2d'
            print(f"  *** 测试模式启用 *** (使用 {func_name} 替代 MOOSE 模拟)")
        
        print(f"  实时GIF更新: {'启用' if self.realtime_gif else '禁用（仅最终生成）'}")
        print(f"  并行计算: {'启用' if self.use_parallel else '禁用'}")
        if self.use_parallel:
            print(f"  最大并行数: {self.max_workers} (系统CPU核心数: {multiprocessing.cpu_count()})")

        # Clean and create output directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_dir.exists() and not self.test_mode:
            shutil.rmtree(self.output_dir)

    def __call__(self, particle_positions, iteration=None):
        """
        评估一批粒子的目标函数值
        
        参数：
            particle_positions: numpy数组，形状 (n_particles, n_params)
            iteration: 当前迭代次数（可选，用于记录）
        
        返回：
            costs: numpy数组，形状 (n_particles,)，每个粒子的目标函数值
        """
        n_particles = particle_positions.shape[0]
        costs = np.zeros(n_particles)
        elapsed_times = np.zeros(n_particles)
        statuses = [None] * n_particles


        
        if self.use_parallel and n_particles > 1:
            # 并行评估
            # print(f"\n并行评估 {n_particles} 个粒子 (使用 {min(self.max_workers, n_particles)} 个进程)...")
            
            # 预先分配评估ID（避免并行时ID冲突）
            eval_ids = list(range(self.eval_count + 1, self.eval_count + n_particles + 1))
            self.eval_count += n_particles
            
            with ProcessPoolExecutor(max_workers=min(self.max_workers, n_particles)) as executor:
                # 提交所有任务，传递预分配的eval_id、particle_id和iteration
                future_to_index = {
                    executor.submit(self.evaluate_single, particle_positions[i], eval_ids[i], i, iteration): i
                    for i in range(n_particles)
                }
                
                # 收集结果和历史记录
                # completed = 0
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        cost, elapsed_time, status = future.result()
                        costs[i] = cost
                        elapsed_times[i] = elapsed_time
                        statuses[i] = status
                        # self.eval_history.append(history_record)
                        
                        # # 在主进程中更新计数器（根据状态）
                        # if history_record['status'] == '✓':
                        #     self.success_count += 1
                        # elif history_record['status'] in ['✗', 'T']:
                        #     self.fail_count += 1
                        
                        # completed += 1
                        # print(f"  进度: {completed}/{n_particles} 完成")
                    except Exception as e:
                        print(f"  粒子 {i} 评估失败: {e}")
                        costs[i] = 1e10
                        elapsed_times[i] = 0
                        statuses[i] = 'E'
                        # # 为失败的评估也添加历史记录
                        # self.eval_history.append({
                        #     'eval_id': eval_ids[i],
                        #     'particle_id': i,
                        #     'iteration': iteration,
                        #     'parameters': particle_positions[i].tolist(),
                        #     'objective': 1e10,
                        #     'elapsed_time': 0,
                        #     'status': 'E'  # Error
                        # })
                        # self.fail_count += 1  # 异常也算失败
                        # completed += 1
        else:
            # 串行评估
            for i in range(n_particles):
                self.eval_count += 1
                cost, elapsed_time, status = self.evaluate_single(particle_positions[i], self.eval_count, i, iteration)
                costs[i] = cost
                # self.eval_history.append(history_record)
                
                # # 在主进程中更新计数器
                # if history_record['status'] == '✓':
                #     self.success_count += 1
                # elif history_record['status'] in ['✗', 'T']:
                #     self.fail_count += 1
        

        return  {
            'cost': costs,
            'elapsed_time': elapsed_times,
            'status': statuses
        }

    def get_max_disp_from_csv(self):
        """从实验数据 CSV 中获取最大位移值"""
        df = pd.read_csv(self.objective_csv)
        return df[self.result_col_x].max()

    def evaluate_single(self, parameters, eval_id, particle_id=None, iteration=None):
        """
        评估单个粒子的目标函数值
        
        如果 test_mode=True，使用测试函数（归一化参数）
        否则运行 MOOSE 仿真
        
        使用缓存加速重复或邻近评估
        
        参数：
            parameters: numpy数组，参数值
            eval_id: 评估ID
            particle_id: 粒子ID（可选）
            iteration: 迭代次数（可选）
        
        返回：
            objective: 目标函数值
            history_record: 评估历史记录字典
        """
        start_time = time.time()
        
        # ========== 1. 查询缓存 ==========
        found, cached_result, cache_source = self.cache.query(parameters)
        
        if found:
            objective = cached_result['objective']
            elapsed_time = time.time() - start_time
            
            # 确定状态标记
            if cache_source == 'exact':
                status = "C"  # Cache exact hit
            elif cache_source == 'nearest':
                status = "N"  # Nearest neighbor
            elif cache_source == 'interpolated':
                status = "I"  # Interpolated
            else:
                status = "C"
            
            history_record = {
                'eval_id': eval_id,
                'particle_id': particle_id,
                'iteration': iteration,
                'parameters': parameters.tolist(),
                'objective': objective,
                'elapsed_time': elapsed_time,
                'status': status
            }
            
            return objective, elapsed_time, status
        
        # ========== 2. 缓存未命中，执行实际评估 ==========
        
        # 测试模式：使用简单测试函数
        if self.test_mode:
            # 将参数归一化到 [0, 1]
            param_config = {'lower_bounds': self.lower_bounds, 'upper_bounds': self.upper_bounds}
            normalized_params = (parameters - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
            
            # 调用测试函数（假设2D参数）
            if len(normalized_params) == 2:
                if self.test_function_type == 'rastrigin':
                    objective = rastrigin_2d(normalized_params[0], normalized_params[1])
                else:
                    objective = test_function_2d(normalized_params[0], normalized_params[1])
            else:
                # 如果不是2D，使用简单的sphere函数
                objective = np.sum(normalized_params**2)
            
            elapsed_time = time.time() - start_time
            status = "✓"
            
            # 存储到缓存
            self.cache.store(parameters, objective, result_data=None, metadata={'mode': 'test'})
            
            # 记录历史
            history_record = {
                'eval_id': eval_id,
                'particle_id': particle_id,
                'iteration': iteration,
                'parameters': parameters.tolist(),
                'objective': objective,
                'elapsed_time': elapsed_time,
                'status': status
            }
            
            return objective, elapsed_time, status
        
        # MOOSE 仿真模式（原有代码）
        index_str = str(eval_id).zfill(4)
        out_name = f"run_{index_str}"

        cmd = [self.executable, "-i", self.input_file.name]

        for param_name, param_value in zip(self.param_paths, parameters):
            cmd.append(f"{param_name}={param_value:.2f}")
            out_name += f"_{param_name}_{param_value:.2f}"

        # use last part of output_dir
        file_base = self.output_dir / f"{out_name}"
        # print(f"file_base: {file_base}")
        cmd.append(f"{self.filebase_param}={file_base}")
        cmd.append(f"{self.max_disp_param}={-1*self.max_disp_value}")
        csv_file = Path(str(file_base) + '.csv')
        # print(f"csv_file: {csv_file}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                # cwd=self.work_dir,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            if result.returncode == 0:
                # print(f"✓ MOOSE simulation completed for parameters: {parameters}")
                objective = self._calculate_objective(csv_file)
                self.success_count += 1
                status = "✓"
            else:
                print(f"✗ MOOSE simulation failed for parameters: {parameters}")
                print(f"result.stderr:\n{result.stderr}")
                objective = 1e10
                self.fail_count += 1
                status = "✗"
        except subprocess.TimeoutExpired:
            elapsed_time = self.timeout
            print(f"✗ MOOSE simulation timed out after {self.timeout} seconds for parameters: {parameters}")
            objective = 1e10
            self.fail_count += 1
            status = "T"

        elapsed_time = time.time() - start_time
        
        # 存储到缓存（仅成功的评估）
        if status == "✓":
            metadata = {
                'executable': self.executable,
                'timeout': self.timeout,
                'n_interp_points': self.n_interp_points
            }
            self.cache.store(parameters, objective, result_data=None, metadata=metadata)
        
        # 记录历史
        history_record = {
            'eval_id': eval_id,
            'particle_id': particle_id,
            'iteration': iteration,
            'parameters': parameters.tolist(),
            'objective': objective,
            'elapsed_time': elapsed_time,
            'status': status
        }

        return objective, elapsed_time, status

    def _calculate_objective(self, csv_file):
        """
        计算目标函数（RMSE）

        过程：
        1. 读取仿真生成的 CSV（csv_file），包含位移和力列（由 self.result_col_x / self.result_col_y 指定）
        2. 读取实验 CSV（self.experimental_csv），并在仿真位移点上对实验力做插值
          （通常实验点多、仿真点少，因此将实验数据插值到仿真点上）
        3. 计算仿真力与插值后实验力的 RMSE 并返回（越小越好）

        返回：float（RMSE）；若出错或缺少数据则返回一个大的惩罚值 1e10
        """
        if not csv_file.exists():
            print(f"   未找到仿真输出文件: {csv_file}")
            return 1e10

        try:
            df_sim = pd.read_csv(csv_file)
        except Exception as e:
            print(f"   读取仿真输出 {csv_file.name} 失败: {e}")
            return 1e10

        # 检查列
        if self.result_col_x not in df_sim.columns or self.result_col_y not in df_sim.columns:
            print(f"   仿真输出缺少列: 期望 ( {self.result_col_x}, {self.result_col_y} ), 实际列: {list(df_sim.columns)}")
            return 1e10

        exp_path = self.objective_csv

        try:
            df_exp = pd.read_csv(exp_path)
        except Exception as e:
            print(f"   读取实验数据 {exp_path.name} 失败: {e}")
            return 1e10

        # 实验数据也必须包含相同的列名（位移, 力）
        if self.result_col_x not in df_exp.columns or self.result_col_y not in df_exp.columns:
            print(f"   实验数据缺少列: 期望 ( {self.result_col_x}, {self.result_col_y} ), 实验列: {list(df_exp.columns)}")
            return 1e10

        # 取位移和力数组（按力排序以便插值）
        x_sim = np.asarray(df_sim[self.result_col_x].astype(float))
        y_sim = np.asarray(df_sim[self.result_col_y].astype(float))

        x_exp = np.asarray(df_exp[self.result_col_x].astype(float))
        y_exp = np.asarray(df_exp[self.result_col_y].astype(float))

        if len(y_sim) < 2 or len(y_exp) < 2:
            print(f"   数据点不足以插值：仿真 {len(y_sim)}, 实验 {len(y_exp)}")
            return 1e10

        # 取位移和力数组（按位移排序以便插值）
        x_sim = np.asarray(df_sim[self.result_col_x].astype(float))
        y_sim = np.asarray(df_sim[self.result_col_y].astype(float))

        x_exp = np.asarray(df_exp[self.result_col_x].astype(float))
        y_exp = np.asarray(df_exp[self.result_col_y].astype(float))

        if len(x_sim) < 2 or len(x_exp) < 2:
            print(f"   数据点不足以插值：仿真 {len(x_sim)}, 实验 {len(x_exp)}")
            return 1e10

        # 按位移升序排序
        sim_sort = np.argsort(x_sim)
        x_sim_sorted = x_sim[sim_sort]
        y_sim_sorted = y_sim[sim_sort]

        exp_sort = np.argsort(x_exp)
        x_exp_sorted = x_exp[exp_sort]
        y_exp_sorted = y_exp[exp_sort]

        # 共同位移上限（取两者最大位移的较小者）
        xmax = min(x_sim_sorted[-1], x_exp_sorted[-1])
        if xmax <= 0:
            print(f"   无效的最大位移 xmax={xmax}")
            return 1e10

        # 按用户要求，位移最小值 xmin = xmax / n_points
        xmin = xmax / float(self.n_interp_points)

        # 在 [xmin, xmax] 上生成等间距位移点
        common_u = np.linspace(xmin, xmax, num=self.n_interp_points)

        try:
            y_sim_on_common = np.interp(common_u, x_sim_sorted, y_sim_sorted)
            y_exp_on_common = np.interp(common_u, x_exp_sorted, y_exp_sorted)
        except Exception as e:
            print(f"   插值失败: {e}")
            return 1e10

        # 在共同位移点上比较力的RMSE（因为插值后位移一致）
        mse = np.mean((y_sim_on_common - y_exp_on_common) ** 2)
        rmse = float(np.sqrt(mse))

        # print(f"   仿真点数: {len(x_sim)}, 实验点数: {len(x_exp)}, 共用位移区间: [{xmin:.6e}, {xmax:.6e}], 插值点: {self.n_interp_points}, RMSE(F)={rmse:.6e}")

        # 绘图：比较实验与仿真力-位移曲线，并标注用于插值的共同位移点
        try:
            fig, ax = plt.subplots(figsize=(5,3.6))
            # 原始曲线
            ax.plot(x_exp_sorted, y_exp_sorted, label='Experimental', color='C0', linewidth=1)
            ax.plot(x_sim_sorted, y_sim_sorted, label='Simulation', color='C1', linewidth=1)

            # 插值点（共同位移网格）
            ax.scatter(common_u, y_exp_on_common, marker='o', s=30, color='C0', facecolors='none', label='Interpolation Points (Exp)')
            ax.scatter(common_u, y_sim_on_common, marker='+', s=30, color='C1', label='Interpolation Points (Sim)')

            ax.set_xlabel(r'Compression ratio')
            ax.set_ylabel(r'Equivalent Stress (MPa)')
            ax.set_title(f'(RMSE={rmse:.3f})')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)

            # 保存图片到与 CSV 相同目录，文件名根据 csv_file 命名
            plot_path = csv_file.with_suffix('.png')
            # 确保目录存在
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
            # print(f"   已保存对比图: {plot_path}")
        except Exception as e:
            print(f"   绘图失败: {e}")

        return rmse
    
    def save_history(self, filename="pso_evaluation_history.csv"):
        """
        保存评估历史到 CSV 文件
        
        参数：
            filename: 输出文件名
        
        返回：
            df: pandas DataFrame，包含所有评估历史
        """
        df_data = []
        for record in self.eval_history:
            row = {
                'eval_id': record['eval_id'],
                'particle_id': record.get('particle_id'),
                'iteration': record.get('iteration')
            }
            for i, (name, val) in enumerate(zip(self.param_names, record['parameters'])):
                row[name] = val
            row['objective'] = record['objective']
            row['elapsed_time'] = record['elapsed_time']
            row['status'] = record['status']
            df_data.append(row)

        df = pd.DataFrame(df_data)
        output_path = self.work_dir / filename
        df.to_csv(output_path, index=False)
        return df
