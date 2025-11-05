import argparse
from pathlib import Path
import yaml
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def rastrigin_3d(x):
    A = 10
    x = np.asarray(x)
    return A*3 + np.sum(x**2 - A * np.cos(2*np.pi*x), axis=-1)

class MOOSEObjectiveFunction:
    """Objective function wrapper for MOOSE simulations."""
    def __init__(self, config):
        print("Initializing MOOSE Objective Function...")
        self.executable = config.get('executable', 'ailuro-opt')
        self.input_file = Path(config['input_file'])
        self.work_dir = Path(config.get('work_dir', '.'))
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.filebase_param = config.get('filebase_param', 'out_name')
        self.timeout = config.get('timeout', 300)

        self.result_col_x = config.get('result_col_x', 'disp')
        self.result_col_y = config.get('result_col_y', 'force')

        # 实验数据（位移-力）CSV文件路径（相对 path 可能在 work_dir 中）
        # 支持在 config 中指定：objective_csv: 'path/to/exp.csv'
        self.objective_csv = Path(config.get('objective_csv'))
        # 插值使用的固定力点数量（在仿真与实验的共同力区间上等间距）
        self.n_interp_points = int(config.get('n_interp_points', 10))
        self.param_names = config['parameters']['names']
        self.param_paths = config['parameters']['paths']
        self.n_params = len(self.param_names)
        self.eval_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.eval_history = []
        
        # 并行计算配置
        self.use_parallel = config.get('use_parallel', True)
        self.max_workers = config.get('max_workers', multiprocessing.cpu_count() // 2)
        print(f"  并行计算: {'启用' if self.use_parallel else '禁用'}")
        if self.use_parallel:
            print(f"  最大并行数: {self.max_workers} (系统CPU核心数: {multiprocessing.cpu_count()})")

        # Combine work_dir and output_dir
        self.output_dir = self.work_dir / self.output_dir

        # Clean and create output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, particle_positions):
        n_particles = particle_positions.shape[0]
        costs = np.zeros(n_particles)
        
        if self.use_parallel and n_particles > 1:
            # 并行评估
            print(f"\n并行评估 {n_particles} 个粒子 (使用 {min(self.max_workers, n_particles)} 个进程)...")
            
            # 预先分配评估ID（避免并行时ID冲突）
            eval_ids = list(range(self.eval_count + 1, self.eval_count + n_particles + 1))
            self.eval_count += n_particles
            
            with ProcessPoolExecutor(max_workers=min(self.max_workers, n_particles)) as executor:
                # 提交所有任务，传递预分配的eval_id
                future_to_index = {
                    executor.submit(self.evaluate_single, particle_positions[i], eval_ids[i]): i
                    for i in range(n_particles)
                }
                
                # 收集结果和历史记录
                completed = 0
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        cost, history_record = future.result()
                        costs[i] = cost
                        self.eval_history.append(history_record)
                        completed += 1
                        print(f"  进度: {completed}/{n_particles} 完成")
                    except Exception as e:
                        print(f"  粒子 {i} 评估失败: {e}")
                        costs[i] = 1e10
                        # 为失败的评估也添加历史记录
                        self.eval_history.append({
                            'eval_id': eval_ids[i],
                            'parameters': particle_positions[i].tolist(),
                            'objective': 1e10,
                            'elapsed_time': 0,
                            'status': 'E'  # Error
                        })
                        completed += 1
        else:
            # 串行评估
            for i in range(n_particles):
                self.eval_count += 1
                cost, history_record = self.evaluate_single(particle_positions[i], self.eval_count)
                costs[i] = cost
                self.eval_history.append(history_record)
            
        return costs

    def evaluate_single(self, parameters, eval_id):
        # Here you would integrate with the MOOSE simulation
        # For now, we'll use a dummy objective function

        index_str = str(eval_id).zfill(4)
        out_name = f"run_{index_str}"

        cmd = [self.executable, "-i", self.input_file.name]

        for param_name, param_value in zip(self.param_paths, parameters):
            cmd.append(f"{param_name}={param_value:.2f}")
            out_name += f"_{param_name}_{param_value:.2f}"

        # use last part of output_dir
        file_base = self.output_dir.name + f"/{out_name}"
        cmd.append(f"{self.filebase_param}={file_base}")
        csv_file = self.work_dir / f"{file_base}.csv"

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ MOOSE simulation completed for parameters: {parameters}")
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
            elapsed = self.timeout
            print(f"✗ MOOSE simulation timed out after {self.timeout} seconds for parameters: {parameters}")
            objective = 1e10
            self.fail_count += 1
            status = "T"

        elapsed = time.time() - start_time
        # 记录历史
        history_record = {
            'eval_id': eval_id,
            'parameters': parameters.tolist(),
            'objective': objective,
            'elapsed_time': elapsed,
            'status': status
        }

        return objective, history_record

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

        exp_path = self.work_dir / self.objective_csv

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
        u_sim = np.asarray(df_sim[self.result_col_x].astype(float))
        f_sim = np.asarray(df_sim[self.result_col_y].astype(float))

        u_exp = np.asarray(df_exp[self.result_col_x].astype(float))
        f_exp = np.asarray(df_exp[self.result_col_y].astype(float))

        if len(f_sim) < 2 or len(f_exp) < 2:
            print(f"   数据点不足以插值：仿真 {len(f_sim)}, 实验 {len(f_exp)}")
            return 1e10

        # 取位移和力数组（按位移排序以便插值）
        u_sim = np.asarray(df_sim[self.result_col_x].astype(float))
        f_sim = np.asarray(df_sim[self.result_col_y].astype(float))

        u_exp = np.asarray(df_exp[self.result_col_x].astype(float))
        f_exp = np.asarray(df_exp[self.result_col_y].astype(float))

        if len(u_sim) < 2 or len(u_exp) < 2:
            print(f"   数据点不足以插值：仿真 {len(u_sim)}, 实验 {len(u_exp)}")
            return 1e10

        # 按位移升序排序
        sim_sort = np.argsort(u_sim)
        u_sim_sorted = u_sim[sim_sort]
        f_sim_sorted = f_sim[sim_sort]

        exp_sort = np.argsort(u_exp)
        u_exp_sorted = u_exp[exp_sort]
        f_exp_sorted = f_exp[exp_sort]

        # 共同位移上限（取两者最大位移的较小者）
        umax = min(u_sim_sorted[-1], u_exp_sorted[-1])
        if umax <= 0:
            print(f"   无效的最大位移 umax={umax}")
            return 1e10

        # 按用户要求，位移最小值 umin = umax / n_points
        umin = umax / float(self.n_interp_points)

        # 在 [umin, umax] 上生成等间距位移点
        common_u = np.linspace(umin, umax, num=self.n_interp_points)

        try:
            f_sim_on_common = np.interp(common_u, u_sim_sorted, f_sim_sorted)
            f_exp_on_common = np.interp(common_u, u_exp_sorted, f_exp_sorted)
        except Exception as e:
            print(f"   插值失败: {e}")
            return 1e10

        # 在共同位移点上比较力的RMSE（因为插值后位移一致）
        mse = np.mean((f_sim_on_common - f_exp_on_common) ** 2)
        rmse = float(np.sqrt(mse))

        print(f"   仿真点数: {len(u_sim)}, 实验点数: {len(u_exp)}, 共用位移区间: [{umin:.6e}, {umax:.6e}], 插值点: {self.n_interp_points}, RMSE(F)={rmse:.6e}")

        # 绘图：比较实验与仿真力-位移曲线，并标注用于插值的共同位移点
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6,4))
            # 原始曲线
            ax.plot(u_exp_sorted, f_exp_sorted, label='Experimental', color='C0', linewidth=1)
            ax.plot(u_sim_sorted, f_sim_sorted, label='Simulation', color='C1', linewidth=1)

            # 插值点（共同位移网格）
            ax.scatter(common_u, f_exp_on_common, marker='o', s=30, color='C0', facecolors='none', label='Interpolation Points (Exp)')
            ax.scatter(common_u, f_sim_on_common, marker='+', s=30, color='C1', label='Interpolation Points (Sim)')

            ax.set_xlabel(r'Displacement ($\mu$m)')
            ax.set_ylabel(r'Force (mN)')
            ax.set_title(f'(RMSE={rmse:.3e})')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)

            # 保存图片到与 CSV 相同目录，文件名根据 csv_file 命名
            plot_path = csv_file.with_suffix('.png')
            # 确保目录存在
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"   已保存对比图: {plot_path}")
        except Exception as e:
            print(f"   绘图失败: {e}")

        return rmse
    
    def save_history(self, filename="pso_evaluation_history.csv"):
        """Save evaluation history to a CSV file."""
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
        return df

def run_pso_optimization(config: dict):
    """
    Run PSO optimization
    
    Args:
        config (dict): Configuration dictionary
    """
    from pyswarms.single.global_best import GlobalBestPSO

    print("="*80)
    print(f"Particle Swarm Optimization (MOOSE)")
    print("="*80)

    print(f"\nRunning PSO Optimization...")

    # Create objective function
    objective_func = MOOSEObjectiveFunction(config)

    # Parameters configuration
    param_config = config['parameters']
    lower_bounds = np.array(param_config['lower_bounds'])
    upper_bounds = np.array(param_config['upper_bounds'])
    bounds = (lower_bounds, upper_bounds)
    print(f"\nParameter Configuration:")
    for i, name in enumerate(param_config['names']):
        print(f"  {name}: [{lower_bounds[i]}, {upper_bounds[i]}]")

    # PSO parameters
    pso_config = config['pso']
    n_particles = pso_config['n_particles']
    max_iters = pso_config['max_iterations']

    options = {
        'c1': pso_config['cognitive_param'],  # cognitive parameter
        'c2': pso_config['social_param'],     # social parameter
        'w': pso_config['inertia_weight']     # inertia weight
    }

    print(f"\nPSO Configuration:")
    print(f"  Number of particles: {n_particles}")
    print(f"  Maximum iterations:  {max_iters}")
    print(f"  Cognitive parameter: {options['c1']}")
    print(f"  Social parameter:    {options['c2']}")
    print(f"  Inertia weight:      {options['w']}")

    # Create PSO optimizer
    print("\nStarting optimization...")
    print("-"*80)

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(param_config['names']),
        options=options,
        bounds=bounds,
        ftol=1e-6,
        ftol_iter=5        
    )

    # 定义迭代回调函数，每次迭代后保存历史和绘图
    def iteration_callback(iteration, cost_history):
        """在每次迭代后调用，保存历史和绘制收敛曲线"""
        print(f"迭代 {iteration}: 最优成本 = {cost_history[-1]:.6e}")
        
        # 保存历史记录
        history_file = objective_func.work_dir / f"pso_evaluation_history_iter_{iteration:04d}.csv"
        df_history = objective_func.save_history(filename=history_file.name)
        
        # 绘制收敛曲线
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(cost_history, color='blue', linewidth=1.5)
        ax.scatter(len(cost_history)-1, cost_history[-1], color='red', s=50, zorder=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Cost")
        ax.set_title(f"PSO Convergence (Iter {iteration})")
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        plot_file = objective_func.work_dir / f"pso_convergence_iter_{iteration:04d}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close(fig)
        
        print(f"  已保存: {history_file.name}, {plot_file.name}")

    # Perform optimization with callback
    start_time = time.time()
    
    # 手动迭代循环以便在每次迭代后调用回调
    for i in range(max_iters):
        optimizer.optimize(objective_func, iters=1)
        iteration_callback(i + 1, optimizer.cost_history)
    
    cost = optimizer.cost_history[-1]
    pos = optimizer.pos
    
    elapsed_time = time.time() - start_time

    # Output results
    print("-"*80)
    print(f"Best cost (objective function value): {cost}")
    print(f"Best position (parameters):")
    for i, name in enumerate(param_config['names']):
        print(f"  {name} = {pos[i]:.6f}")

    print(f"Optimization completed in {elapsed_time:.2f} seconds.")

    
    # 保存最终历史
    final_history_file = "pso_evaluation_history_final.csv"
    df_history = objective_func.save_history(filename=final_history_file)
    print(f"最终历史记录已保存: {final_history_file}")
    
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
    print(f"\n最优参数已保存: {result_file}")

    # 绘制最终收敛曲线（高分辨率）
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(optimizer.cost_history, color='blue', linewidth=1.5)
    ax.scatter(len(optimizer.cost_history)-1, optimizer.cost_history[-1], color='red', s=80, zorder=5, label='Final Best')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.set_title("PSO Convergence Curve (Final)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    final_plot = objective_func.work_dir / "pso_convergence_final.png"
    plt.savefig(final_plot, dpi=300)
    plt.close(fig)
    print(f"最终收敛曲线已保存: {final_plot.name}")


    # # 3D Scatter plot of particle positions
    # final_positions = optimizer.pos_history[-1]   # shape = (n_particles, 3)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(final_positions[:,0], final_positions[:,1], final_positions[:,2], label='Particles')
    # ax.scatter(pos[0], pos[1], pos[2], marker='*', s=150, label='Best',)
    # ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    # ax.set_title("Final Particle Positions")
    # ax.legend()
    # plt.show()
    
    return pos, cost



def main():
    parser = argparse.ArgumentParser(description='PSO + MOOSE Optimization')
    parser.add_argument('--config', default='pso_config.yaml',
                       help='config file path (default: pso_config.yaml)')
    args = parser.parse_args()

    # Load configuration
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Error: config file does not exist: {config_file}")
        print(f"Please create the config file.")
        return

    print(f"Using config file: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Run Optimization
    run_pso_optimization(config)


if __name__ == "__main__":
    main()