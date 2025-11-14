"""
PSO 稳定性测试脚本

运行多次 PSO 优化，统计最优解的稳定性
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from pso_runner_class import run_pso_optimization


def run_stability_test(config_file, n_runs=100, output_dir=None):
    """
    运行多次 PSO 优化，统计稳定性
    
    参数：
        config_file: str, 配置文件路径
        n_runs: int, 运行次数
        output_dir: str, 输出目录（可选）
    """
    
    # 加载配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 准备输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"stability_test_{timestamp}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 记录所有运行结果
    results = []
    param_names = config['parameters']['names']
    
    print(f"开始稳定性测试: 运行 {n_runs} 次")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    for i in range(n_runs):
        print(f"\n[{i+1}/{n_runs}] 运行中...")
        
        # 临时修改工作目录，避免每次都写入同一个目录
        original_work_dir = config['moose'].get('work_dir', 'pso_work')
        config['moose']['work_dir'] = str(output_dir / f"run_{i+1:03d}")
        
        try:
            # 运行优化（返回 OptimizationResult 对象）
            result = run_pso_optimization(config)
            
            # 记录结果
            run_result = {
                'run_id': i + 1,
                'optimal_cost': result.best_cost,
                'n_iterations': result.total_iterations,
                'elapsed_time': result.total_time,
                'n_evaluations': result.total_evaluations,
                'converged': result.converged,
                'convergence_iteration': result.convergence_iteration if result.converged else None,
                'final_avg_cost': result.avg_cost_history[-1] if result.avg_cost_history else np.nan,
                'cache_hit_rate': result.cache_hit_rate if result.cache_enabled else 0.0,
            }
            
            # 记录每个参数的最优值
            for param_name, param_value in zip(param_names, result.best_params):
                run_result[f'param_{param_name}'] = param_value
            
            results.append(run_result)
            
            print(f"  ✓ 完成 - 最优成本: {result.best_cost:.6e}, "
                  f"迭代: {result.total_iterations}, 耗时: {result.total_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ 失败 - 错误: {e}")
            # 记录失败的运行
            run_result = {
                'run_id': i + 1,
                'optimal_cost': np.nan,
                'error': str(e)
            }
            results.append(run_result)
        
        # 恢复原始配置
        config['moose']['work_dir'] = original_work_dir
    
    # 转换为 DataFrame
    df_results = pd.DataFrame(results)
    
    # 保存详细结果
    results_file = output_dir / "stability_results.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n详细结果已保存: {results_file}")
    
    # 统计分析
    print("\n" + "="*80)
    print("稳定性统计分析")
    print("="*80)
    
    # 成功运行的数据
    df_success = df_results[df_results['optimal_cost'].notna()]
    n_success = len(df_success)
    n_failed = n_runs - n_success
    
    print(f"\n运行统计:")
    print(f"  总运行次数: {n_runs}")
    print(f"  成功次数: {n_success} ({100*n_success/n_runs:.1f}%)")
    print(f"  失败次数: {n_failed} ({100*n_failed/n_runs:.1f}%)")
    
    if n_success > 0:
        # 最优成本统计
        print(f"\n最优成本统计:")
        print(f"  均值: {df_success['optimal_cost'].mean():.6e}")
        print(f"  标准差: {df_success['optimal_cost'].std():.6e}")
        print(f"  最小值: {df_success['optimal_cost'].min():.6e}")
        print(f"  最大值: {df_success['optimal_cost'].max():.6e}")
        print(f"  中位数: {df_success['optimal_cost'].median():.6e}")
        print(f"  变异系数: {df_success['optimal_cost'].std() / df_success['optimal_cost'].mean():.4f}")
        
        # 迭代次数统计
        print(f"\n迭代次数统计:")
        print(f"  均值: {df_success['n_iterations'].mean():.1f}")
        print(f"  标准差: {df_success['n_iterations'].std():.1f}")
        print(f"  最小值: {df_success['n_iterations'].min()}")
        print(f"  最大值: {df_success['n_iterations'].max()}")
        
        # 运行时间统计
        print(f"\n运行时间统计 (秒):")
        print(f"  均值: {df_success['elapsed_time'].mean():.2f}")
        print(f"  标准差: {df_success['elapsed_time'].std():.2f}")
        print(f"  最小值: {df_success['elapsed_time'].min():.2f}")
        print(f"  最大值: {df_success['elapsed_time'].max():.2f}")
        
        # 参数统计
        print(f"\n最优参数统计:")
        for param_name in param_names:
            col_name = f'param_{param_name}'
            if col_name in df_success.columns:
                print(f"  {param_name}:")
                print(f"    均值: {df_success[col_name].mean():.6f}")
                print(f"    标准差: {df_success[col_name].std():.6f}")
                print(f"    最小值: {df_success[col_name].min():.6f}")
                print(f"    最大值: {df_success[col_name].max():.6f}")
        
        # 绘制统计图
        plot_stability_results(df_success, output_dir, param_names)
        
    # 保存统计摘要
    summary_file = output_dir / "stability_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PSO 稳定性测试摘要\n")
        f.write("="*80 + "\n\n")
        f.write(f"运行次数: {n_runs}\n")
        f.write(f"成功次数: {n_success} ({100*n_success/n_runs:.1f}%)\n")
        f.write(f"失败次数: {n_failed}\n\n")
        
        if n_success > 0:
            f.write("最优成本统计:\n")
            f.write(f"  均值 ± 标准差: {df_success['optimal_cost'].mean():.6e} ± {df_success['optimal_cost'].std():.6e}\n")
            f.write(f"  最小值: {df_success['optimal_cost'].min():.6e}\n")
            f.write(f"  最大值: {df_success['optimal_cost'].max():.6e}\n")
            f.write(f"  中位数: {df_success['optimal_cost'].median():.6e}\n")
            f.write(f"  变异系数: {df_success['optimal_cost'].std() / df_success['optimal_cost'].mean():.4f}\n")
    
    print(f"\n统计摘要已保存: {summary_file}")
    
    return df_results


def plot_stability_results(df, output_dir, param_names):
    """
    绘制稳定性测试结果图
    """
    
    # 1. 最优成本分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 成本分布
    ax = axes[0, 0]
    ax.hist(df['optimal_cost'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['optimal_cost'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["optimal_cost"].mean():.6e}')
    ax.axvline(df['optimal_cost'].median(), color='green', linestyle='--', 
               label=f'Median: {df["optimal_cost"].median():.6e}')
    ax.set_xlabel('Optimal Cost')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Optimal Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 迭代次数分布
    ax = axes[0, 1]
    ax.hist(df['n_iterations'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['n_iterations'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["n_iterations"].mean():.1f}')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 运行时间分布
    ax = axes[1, 0]
    ax.hist(df['elapsed_time'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['elapsed_time'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["elapsed_time"].mean():.2f}s')
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Elapsed Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 成本随运行序号变化
    ax = axes[1, 1]
    ax.plot(df['run_id'], df['optimal_cost'], 'o-', alpha=0.6, markersize=3)
    ax.axhline(df['optimal_cost'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["optimal_cost"].mean():.6e}')
    ax.fill_between(df['run_id'], 
                     df['optimal_cost'].mean() - df['optimal_cost'].std(),
                     df['optimal_cost'].mean() + df['optimal_cost'].std(),
                     alpha=0.2, color='red', label='±1 Std Dev')
    ax.set_xlabel('Run ID')
    ax.set_ylabel('Optimal Cost')
    ax.set_title('Optimal Cost vs Run ID')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "stability_distributions.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"分布图已保存: {plot_file}")
    
    # 2. 参数分布图
    n_params = len(param_names)
    if n_params > 0:
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        if n_params == 1:
            axes = [axes]
        
        for i, param_name in enumerate(param_names):
            col_name = f'param_{param_name}'
            if col_name in df.columns:
                ax = axes[i]
                ax.hist(df[col_name], bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(df[col_name].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df[col_name].mean():.6f}')
                ax.axvline(df[col_name].median(), color='green', linestyle='--', 
                          label=f'Median: {df[col_name].median():.6f}')
                ax.set_xlabel(param_name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {param_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / "parameter_distributions.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"参数分布图已保存: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='PSO Stability Test')
    parser.add_argument('--config', default='pso_config.yaml',
                       help='配置文件路径 (default: pso_config.yaml)')
    parser.add_argument('--n-runs', type=int, default=20,
                       help='运行次数 (default: 100)')
    parser.add_argument('--output-dir', default=None,
                       help='输出目录 (default: auto-generated)')
    args = parser.parse_args()
    
    # 检查配置文件
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        return
    
    # 运行稳定性测试
    run_stability_test(config_file, n_runs=args.n_runs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
