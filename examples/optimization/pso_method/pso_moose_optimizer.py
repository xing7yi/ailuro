#!/usr/bin/env python3
"""
PSO (粒子群优化) + MOOSE 整合脚本

功能: 使用粒子群算法优化MOOSE模拟的材料参数

用法:
    python pso_moose_optimizer.py --config pso_config.yaml

作者: AI Assistant
日期: 2025-11-05
"""

import numpy as np
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import argparse
import yaml
import time
from datetime import datetime


class MOOSEObjectiveFunction:
    """
    MOOSE目标函数封装
    将MOOSE模拟作为黑盒优化目标函数
    """
    
    def __init__(self, config):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.executable = config.get('executable', 'ailuro-opt')
        self.input_file = Path(config['input_file'])  # 直接使用原始输入文件
        self.work_dir = Path(config.get('work_dir', 'pso_work'))
        self.timeout = config.get('timeout', 300)  # 5分钟默认超时
        
        # 参数配置 - 包含MOOSE路径映射
        self.param_names = config['parameters']['names']
        self.param_paths = config['parameters']['paths']  # MOOSE中的参数路径
        self.n_params = len(self.param_names)
        
        # 输出文件配置
        self.output_csv = config.get('output_csv', 'forward_out.csv')
        self.objective_column = config.get('objective_column', 'objective_value')
        
        # 统计信息
        self.eval_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.eval_history = []
        
        # 创建工作目录
        self.work_dir.mkdir(exist_ok=True)
        
        # 复制必要文件（网格、数据等）
        self._copy_auxiliary_files(config)
        
    def _copy_auxiliary_files(self, config):
        """复制网格文件等辅助文件到工作目录"""
        aux_files = config.get('auxiliary_files', [])
        for file in aux_files:
            src = Path(file)
            if src.exists():
                dst = self.work_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
                    print(f"  复制: {src.name}")
    
    def __call__(self, particle_positions):
        """
        批量评估粒子群的目标函数
        
        Args:
            particle_positions: shape (n_particles, n_params) 的numpy数组
            
        Returns:
            costs: shape (n_particles,) 的目标函数值数组
        """
        n_particles = particle_positions.shape[0]
        costs = np.zeros(n_particles)
        
        for i in range(n_particles):
            costs[i] = self.evaluate_single(particle_positions[i])
            
        return costs
    
    def evaluate_single(self, parameters):
        """
        评估单个参数组合
        
        Args:
            parameters: 参数向量 [p0, p1, p2, ...]
        
        Returns:
            objective_value: 目标函数值（越小越好）
        """
        self.eval_count += 1
        
        # 创建临时运行目录
        run_dir = self.work_dir / f"eval_{self.eval_count:05d}"
        run_dir.mkdir(exist_ok=True)
        
        # 创建软链接到必要文件
        self._link_auxiliary_files(run_dir)
        self._link_input_file(run_dir)
        
        # 构建命令：通过命令行参数覆盖材料参数
        cmd = [self.executable, "-i", self.input_file.name]
        
        # 添加参数覆盖
        for param_path, param_value in zip(self.param_paths, parameters):
            cmd.append(f"{param_path}={param_value}")
        
        # 运行MOOSE
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # 提取目标函数值
                objective = self._extract_objective(run_dir)
                self.success_count += 1
                status = "✓"
            else:
                print(f"\n⚠ 评估 {self.eval_count} 失败 (返回码: {result.returncode})")
                print(f"   参数: {parameters}")
                # 打印错误信息（如果有）
                if result.stderr:
                    print(f"   错误: {result.stderr[-500:]}")  # 最后500字符
                objective = 1e10  # 惩罚值
                self.fail_count += 1
                status = "✗"
                
        except subprocess.TimeoutExpired:
            elapsed = self.timeout
            print(f"\n⚠ 评估 {self.eval_count} 超时 (>{self.timeout}s)")
            print(f"   参数: {parameters}")
            objective = 1e10
            self.fail_count += 1
            status = "T"
        
        # 记录历史
        self.eval_history.append({
            'eval_id': self.eval_count,
            'parameters': parameters.tolist(),
            'objective': objective,
            'elapsed_time': elapsed,
            'status': status
        })
        
        # 输出进度
        param_str = ", ".join([f"{p:.2f}" for p in parameters])
        print(f"{status} 评估 {self.eval_count}/{self.success_count+self.fail_count}: "
              f"[{param_str}] → J={objective:.6e} ({elapsed:.1f}s)")
        
        # 清理（可选，节省磁盘空间）
        # self._cleanup(run_dir)
        
        return objective
    
    def _link_input_file(self, run_dir):
        """创建输入文件的软链接"""
        link_path = run_dir / self.input_file.name
        if not link_path.exists():
            try:
                link_path.symlink_to(self.input_file.resolve())
            except:
                shutil.copy(self.input_file, link_path)
    
    def _link_auxiliary_files(self, run_dir):
        """创建辅助文件的软链接"""
        for item in self.work_dir.iterdir():
            if item.is_file() and item.suffix in ['.msh', '.csv', '.e']:
                link_path = run_dir / item.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(item.resolve())
                    except:
                        # 软链接失败则复制
                        shutil.copy(item, link_path)
    
    def _extract_objective(self, run_dir):
        """从输出文件提取目标函数值"""
        csv_file = run_dir / self.output_csv
        
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                # 使用配置中指定的列名
                if self.objective_column in df.columns:
                    return float(df[self.objective_column].iloc[-1])
                else:
                    # 尝试查找包含objective的列
                    obj_cols = [col for col in df.columns if 'objective' in col.lower()]
                    if obj_cols:
                        return float(df[obj_cols[0]].iloc[-1])
            except Exception as e:
                print(f"   读取 {csv_file.name} 失败: {e}")
        
        print(f"   未找到目标函数值（期望文件: {self.output_csv}）")
        return 1e10
    
    def _cleanup(self, run_dir):
        """清理临时文件"""
        if run_dir.exists():
            shutil.rmtree(run_dir)
    
    def save_history(self, filename="pso_evaluation_history.csv"):
        """保存评估历史"""
        df_data = []
        for record in self.eval_history:
            row = {'eval_id': record['eval_id']}
            for i, (name, val) in enumerate(zip(self.param_names, record['parameters'])):
                row[name] = val
            row['objective'] = record['objective']
            row['elapsed_time'] = record['elapsed_time']
            row['status'] = record['status']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        output_path = self.work_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n评估历史已保存: {output_path}")
        return df


def run_pso_optimization(config):
    """
    运行PSO优化
    
    Args:
        config: 配置字典
    """
    print("="*80)
    print("PSO (粒子群优化) + MOOSE 优化")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 导入PySwarms
    try:
        from pyswarms.single.global_best import GlobalBestPSO
    except ImportError:
        print("\n错误: 未安装 pyswarms 库")
        print("请运行: pip install pyswarms")
        return None
    
    # 创建目标函数
    print("\n初始化目标函数...")
    objective_func = MOOSEObjectiveFunction(config)
    
    # 参数配置
    param_config = config['parameters']
    lower_bounds = np.array(param_config['lower_bounds'])
    upper_bounds = np.array(param_config['upper_bounds'])
    bounds = (lower_bounds, upper_bounds)
    
    print(f"\n优化参数:")
    for i, name in enumerate(param_config['names']):
        print(f"  {name}: [{lower_bounds[i]}, {upper_bounds[i]}]")
    
    # PSO配置
    pso_config = config['pso']
    n_particles = pso_config['n_particles']
    max_iters = pso_config['max_iterations']
    
    options = {
        'c1': pso_config['cognitive_param'],  # 认知参数
        'c2': pso_config['social_param'],     # 社会参数
        'w': pso_config['inertia_weight']     # 惯性权重
    }
    
    print(f"\nPSO配置:")
    print(f"  粒子数量: {n_particles}")
    print(f"  最大迭代: {max_iters}")
    print(f"  惯性权重: {options['w']}")
    print(f"  认知参数: {options['c1']}")
    print(f"  社会参数: {options['c2']}")
    
    # 创建优化器
    print(f"\n开始优化...")
    print("-"*80)
    
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(param_config['names']),
        options=options,
        bounds=bounds
    )
    
    # 运行优化
    start_time = time.time()
    cost, pos = optimizer.optimize(
        objective_func,
        iters=max_iters,
        verbose=True,
        n_processes=1  # 串行评估（MOOSE已经并行）
    )
    elapsed_time = time.time() - start_time
    
    # 结果
    print("\n" + "="*80)
    print("优化完成！")
    print("="*80)
    print(f"\n最优参数:")
    for name, value in zip(param_config['names'], pos):
        print(f"  {name} = {value:.4f}")
    print(f"\n最优目标函数: {cost:.6e}")
    print(f"\n统计信息:")
    print(f"  总评估次数: {objective_func.eval_count}")
    print(f"  成功评估: {objective_func.success_count}")
    print(f"  失败评估: {objective_func.fail_count}")
    print(f"  总计算时间: {elapsed_time/60:.1f} 分钟")
    print(f"  平均评估时间: {elapsed_time/objective_func.eval_count:.1f} 秒")
    
    # 保存历史
    df_history = objective_func.save_history()
    
    # 保存最优参数
    result_file = objective_func.work_dir / "pso_optimal_parameters.yaml"
    result = {
        'optimal_parameters': {name: float(val) for name, val in zip(param_config['names'], pos)},
        'optimal_cost': float(cost),
        'statistics': {
            'total_evaluations': objective_func.eval_count,
            'successful': objective_func.success_count,
            'failed': objective_func.fail_count,
            'total_time_minutes': elapsed_time/60,
            'average_time_seconds': elapsed_time/objective_func.eval_count
        }
    }
    
    with open(result_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    print(f"\n结果已保存: {result_file}")
    
    return pos, cost, df_history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PSO + MOOSE 优化')
    parser.add_argument('--config', default='pso_config.yaml',
                       help='配置文件路径 (default: pso_config.yaml)')
    args = parser.parse_args()
    
    # 加载配置
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        print(f"\n请创建配置文件，参考 pso_config_example.yaml")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 运行优化
    result = run_pso_optimization(config)
    
    if result:
        print(f"\n✓ 优化成功完成")
    else:
        print(f"\n✗ 优化失败")


if __name__ == "__main__":
    main()
