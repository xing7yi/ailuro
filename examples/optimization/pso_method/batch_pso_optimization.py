#!/usr/bin/env python3
"""
批量PSO优化脚本

对 experiment_csv 文件夹下的所有实验数据CSV文件，
批量执行PSO逆向参数识别。

用法：
    python batch_pso_optimization.py --template pso_config.yaml --exp-dir experiment_csv/normalized
"""

import argparse
import yaml
from pathlib import Path
from glob import glob
import shutil
import time
import pandas as pd
import copy
from pso_opt import run_pso_optimization


def create_batch_config(template_config, exp_csv_file, output_base_dir):
    """
    基于模板配置创建单个实验的配置
    
    Args:
        template_config: 模板配置字典
        exp_csv_file: 实验CSV文件路径（Path对象）
        output_base_dir: 输出根目录（Path对象）
    
    Returns:
        config: 修改后的配置字典
        work_dir: 该实验的工作目录
    """
    # 深拷贝避免嵌套字典/列表引用问题
    config = copy.deepcopy(template_config)
    
    # 从文件名提取标识（例如：316L_D27.485_CL_0004_norm.csv -> 316L_D27.485_CL_0004）
    stem = exp_csv_file.stem
    if stem.endswith('_norm'):
        stem = stem[:-5]  # 去掉 _norm 后缀
    elif stem.endswith('_normalized'):
        stem = stem[:-11]  # 去掉 _normalized 后缀
    
    # 创建该实验的工作目录
    work_dir = output_base_dir / f"pso_{stem}"
    
    # 更新配置
    config['work_dir'] = str(work_dir)
    config['objective_csv'] = str(exp_csv_file)
    
    return config, work_dir


def batch_optimize(template_file, exp_dir, output_base_dir, pattern="*.csv", 
                   skip_existing=False, max_files=None):
    """
    批量执行PSO优化
    
    Args:
        template_file: 模板配置文件路径
        exp_dir: 实验数据目录
        output_base_dir: 输出根目录
        pattern: CSV文件匹配模式
        skip_existing: 是否跳过已存在的结果
        max_files: 最大处理文件数（None表示全部）
    """
    
    # 加载模板配置
    with open(template_file, 'r') as f:
        template_config = yaml.safe_load(f)
    
    # 查找所有实验CSV文件
    exp_path = Path(exp_dir)
    exp_files = sorted(exp_path.glob(pattern))
    
    # 过滤掉脚本文件
    exp_files = [f for f in exp_files if not f.name.endswith('.py')]
    
    if len(exp_files) == 0:
        print(f"错误：在 {exp_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    if max_files is not None:
        exp_files = exp_files[:max_files]
    
    print("="*80)
    print("批量PSO参数识别")
    print("="*80)
    print(f"模板配置: {template_file}")
    print(f"实验数据目录: {exp_dir}")
    print(f"输出根目录: {output_base_dir}")
    print(f"找到 {len(exp_files)} 个实验数据文件")
    print("-"*80)
    
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 创建批量结果汇总文件
    summary_results = []
    
    for i, exp_file in enumerate(exp_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(exp_files)}] 处理实验: {exp_file.name}")
        print(f"{'='*80}")
        
        # 创建配置
        config, work_dir = create_batch_config(template_config, exp_file, output_base)
        
        # 检查是否已存在结果
        result_file = work_dir / "pso_optimal_parameters.yaml"
        if skip_existing and result_file.exists():
            print(f"✓ 跳过（结果已存在）: {work_dir}")
            # 读取已有结果
            with open(result_file, 'r') as f:
                result = yaml.safe_load(f)
            summary_results.append({
                'experiment': exp_file.name,
                'work_dir': str(work_dir),
                'status': 'skipped (existing)',
                **result.get('optimal_parameters', {}),
                'optimal_cost': result.get('optimal_cost', None)
            })
            continue
        
        # 保存当前实验的配置文件
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = work_dir / "pso_config_used.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)
        print(f"配置已保存: {config_file}")
        
        # 执行优化
        start_time = time.time()
        try:
            best_pos, best_cost = run_pso_optimization(config)
            elapsed = time.time() - start_time
            
            print(f"\n✓ 优化完成")
            print(f"  最优成本: {best_cost:.6e}")
            print(f"  最优参数: {best_pos}")
            print(f"  耗时: {elapsed/60:.2f} 分钟")
            
            # 记录结果
            param_dict = {name: float(val) for name, val in 
                         zip(config['parameters']['names'], best_pos)}
            summary_results.append({
                'experiment': exp_file.name,
                'work_dir': str(work_dir),
                'status': 'success',
                **param_dict,
                'optimal_cost': float(best_cost),
                'elapsed_minutes': elapsed/60
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ 优化失败: {e}")
            print(f"  耗时: {elapsed/60:.2f} 分钟")
            
            # 记录失败
            summary_results.append({
                'experiment': exp_file.name,
                'work_dir': str(work_dir),
                'status': f'failed: {str(e)}',
                'elapsed_minutes': elapsed/60
            })
    
    # 保存批量结果汇总
    print(f"\n{'='*80}")
    print("批量优化完成")
    print(f"{'='*80}")
    
    summary_file = output_base / "batch_pso_summary.csv"
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(summary_file, index=False)
    print(f"\n汇总结果已保存: {summary_file}")
    
    # 打印统计
    success_count = sum(1 for r in summary_results if r['status'] == 'success')
    fail_count = sum(1 for r in summary_results if r['status'].startswith('failed'))
    skip_count = sum(1 for r in summary_results if r['status'].startswith('skipped'))
    
    print(f"\n统计:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过: {skip_count}")
    print(f"  总计: {len(summary_results)}")
    
    # 打印成功案例的参数范围
    if success_count > 0:
        df_success = df_summary[df_summary['status'] == 'success']
        print(f"\n成功案例的参数范围:")
        param_names = config['parameters']['names']
        for param in param_names:
            if param in df_success.columns:
                print(f"  {param}: [{df_success[param].min():.2f}, {df_success[param].max():.2f}]")
        print(f"  optimal_cost: [{df_success['optimal_cost'].min():.6e}, {df_success['optimal_cost'].max():.6e}]")


def main():
    parser = argparse.ArgumentParser(
        description='批量PSO参数识别工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对 experiment_csv/normalized 目录下所有归一化数据执行PSO
  python batch_pso_optimization.py \\
      --template pso_config.yaml \\
      --exp-dir experiment_csv/normalized \\
      --output batch_pso_results
  
  # 跳过已有结果，只处理前3个文件（测试）
  python batch_pso_optimization.py \\
      --template pso_config.yaml \\
      --exp-dir experiment_csv/normalized \\
      --output batch_pso_results \\
      --skip-existing \\
      --max-files 3
  
  # 指定文件匹配模式
  python batch_pso_optimization.py \\
      --template pso_config.yaml \\
      --exp-dir experiment_csv/normalized \\
      --output batch_pso_results \\
      --pattern "316L_D27*_norm.csv"
        """
    )
    
    parser.add_argument('--template', '-t', 
                       default='pso_config.yaml',
                       help='模板配置文件路径（默认: pso_config.yaml）')
    
    parser.add_argument('--exp-dir', '-e',
                       required=True,
                       help='实验数据目录（包含CSV文件）')
    
    parser.add_argument('--output', '-o',
                       default='batch_pso_results',
                       help='输出根目录（默认: batch_pso_results）')
    
    parser.add_argument('--pattern', '-p',
                       default='*.csv',
                       help='CSV文件匹配模式（默认: *.csv）')
    
    parser.add_argument('--skip-existing', '-s',
                       action='store_true',
                       help='跳过已存在结果的实验')
    
    parser.add_argument('--max-files', '-m',
                       type=int,
                       default=None,
                       help='最大处理文件数（用于测试，默认: 全部）')
    
    args = parser.parse_args()
    
    # 检查模板文件
    if not Path(args.template).exists():
        print(f"错误: 模板配置文件不存在: {args.template}")
        return 1
    
    # 检查实验数据目录
    if not Path(args.exp_dir).exists():
        print(f"错误: 实验数据目录不存在: {args.exp_dir}")
        return 1
    
    # 执行批量优化
    batch_optimize(
        template_file=args.template,
        exp_dir=args.exp_dir,
        output_base_dir=args.output,
        pattern=args.pattern,
        skip_existing=args.skip_existing,
        max_files=args.max_files
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
