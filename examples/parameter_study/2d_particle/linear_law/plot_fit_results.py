#!/usr/bin/env python3
"""
批量绘制力-位移曲线及Yeoh模型拟合结果

用法:
    python plot_fit_results.py [file_base] [options]
    
选项:
    --num-plots N    绘制前N个样本 (默认: 全部)
    --grid-size NxM  网格布局 (默认: 自动)
    --save-individual 保存每个样本的单独图像
    --dpi DPI        图像分辨率 (默认: 100)
"""

import csv
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


def YeohModel_Power3(epsilon, a, b, c):
    """Yeoh 超弹性模型"""
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1
    I1 = np.power(lmbd, 2) + 2 * np.power(lmbd, -1)
    stress = 2 * (lmbd - np.power(lmbd, -2)) * (
        a + 2 * b * (I1 - 3) + 3 * c * np.power((I1 - 3), 2)
    )
    return stress


def read_runner_data(csv_file, config):
    """读取单个runner的CSV数据"""
    time_data = []
    displacement_data = []
    force_data = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                time = float(row.get('time', 0))
                disp = float(row.get(config['disp_col'], 0))
                force = float(row.get(config['force_col'], 0))
                
                if abs(disp) > 1e-6 and force > 0:
                    time_data.append(time)
                    displacement_data.append(abs(disp))
                    force_data.append(force)
            except (ValueError, KeyError):
                continue
    
    return np.array(displacement_data), np.array(force_data)


def plot_single_fit(runner_num, displacement, force, fit_params, config, ax=None):
    """绘制单个样本的拟合结果"""
    
    # 计算应变和应力
    epsilon = displacement / config['initial_height']
    stress = force / config['contact_area']
    
    # 生成拟合曲线
    epsilon_fit = np.linspace(epsilon.min(), epsilon.max(), 200)
    stress_fit = YeohModel_Power3(epsilon_fit, 
                                   fit_params['a'], 
                                   fit_params['b'], 
                                   fit_params['c'])
    displacement_fit = epsilon_fit * config['initial_height']
    force_fit = stress_fit * config['contact_area']
    
    # 如果没有提供ax，创建新图
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False
    
    # 绘制原始数据点
    ax.scatter(epsilon, stress, s=20, alpha=0.6, label='Original Data', color='blue')
    
    # 绘制拟合曲线
    ax.plot(epsilon_fit, stress_fit, 'r-', linewidth=2, label='Yeoh Fit')
    
    # 设置标签和标题
    ax.set_xlabel('Strain', fontsize=10)
    ax.set_ylabel('Stress (MPa)', fontsize=10)
    ax.set_title(f'Runner {runner_num}\n'
                 f'a={fit_params["a"]:.2e}, b={fit_params["b"]:.2e}, c={fit_params["c"]:.2e}\n'
                 f'R²={fit_params["r_squared"]:.4f}',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig
    else:
        return ax


def plot_batch(runner_files, fit_params_csv, config, args):
    """批量绘制"""
    
    print(f"\n读取拟合参数: {fit_params_csv}")
    
    # 读取拟合参数
    fit_params = {}
    with open(fit_params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runner = int(row['runner'])
            fit_params[runner] = {
                'a': float(row['a']),
                'b': float(row['b']),
                'c': float(row['c']),
                'r_squared': float(row['r_squared']),
                'filename': row['filename']
            }
    
    # 确定要绘制的样本
    total_samples = len(runner_files)
    num_to_plot = min(args.num_plots, total_samples) if args.num_plots else total_samples
    
    print(f"总样本数: {total_samples}")
    print(f"绘制样本数: {num_to_plot}")
    
    # 创建输出目录
    output_dir = Path('fit_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 如果需要保存单独图像
    if args.save_individual:
        print(f"\n保存单独图像到: {output_dir}/")
        for i, csv_file in enumerate(runner_files[:num_to_plot]):
            runner_num = extract_runner_number(csv_file)
            
            if runner_num not in fit_params:
                continue
            
            # 读取数据
            displacement, force = read_runner_data(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制
            fig = plot_single_fit(runner_num, displacement, force, 
                                 fit_params[runner_num], config)
            
            # 保存
            output_file = output_dir / f'runner{runner_num:03d}_fit.pdf'
            plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            
            if (i + 1) % 10 == 0:
                print(f"  已保存 {i + 1}/{num_to_plot} 个图像")
        
        print(f"完成! 所有图像已保存到 {output_dir}/")
    
    # 绘制网格图
    if args.grid_size:
        nrows, ncols = map(int, args.grid_size.split('x'))
    else:
        # 自动确定网格大小 - 默认使用5x5网格
        nrows, ncols = 5, 5
    
    plots_per_grid = nrows * ncols
    num_grids = (num_to_plot + plots_per_grid - 1) // plots_per_grid
    
    print(f"\n绘制网格图: {num_grids} 页，每页 {nrows}x{ncols}")
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * plots_per_grid
        end_idx = min(start_idx + plots_per_grid, num_to_plot)
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(ncols * 4, nrows * 3))
        
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 绘制当前页的子图
        plot_count = 0
        for i in range(start_idx, end_idx):
            csv_file = runner_files[i]
            runner_num = extract_runner_number(csv_file)
            
            if runner_num not in fit_params:
                continue
            
            # 读取数据
            displacement, force = read_runner_data(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制到子图
            ax = axes[plot_count]
            plot_single_fit(runner_num, displacement, force, 
                          fit_params[runner_num], config, ax=ax)
            
            plot_count += 1
        
        # 隐藏多余的子图
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # 保存网格图
        output_file = output_dir / f'fit_grid_{grid_idx+1:02d}.pdf'
        plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"  保存网格图: {output_file}")
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")
    print(f"图像保存位置: {output_dir}/")


def plot_statistics(fit_params_csv, config):
    """绘制拟合参数的统计分布"""
    
    print(f"\n绘制统计分布...")
    
    # 读取拟合参数
    data = []
    with open(fit_params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['fit_success'] == 'True':
                data.append({
                    'runner': int(row['runner']),
                    'a': float(row['a']),
                    'b': float(row['b']),
                    'c': float(row['c']),
                    'r_squared': float(row['r_squared']),
                })
    
    if not data:
        print("没有成功的拟合结果可绘制")
        return
    
    # 提取数据
    a_values = [d['a'] for d in data]
    b_values = [d['b'] for d in data]
    c_values = [d['c'] for d in data]
    r2_values = [d['r_squared'] for d in data]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 参数a的分布
    axes[0, 0].hist(a_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Parameter a', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Parameter a Distribution\nMean={np.mean(a_values):.2e}, Std={np.std(a_values):.2e}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 参数b的分布
    axes[0, 1].hist(b_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Parameter b', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title(f'Parameter b Distribution\nMean={np.mean(b_values):.2e}, Std={np.std(b_values):.2e}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 参数c的分布
    axes[1, 0].hist(c_values, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Parameter c', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'Parameter c Distribution\nMean={np.mean(c_values):.2e}, Std={np.std(c_values):.2e}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # R²的分布
    axes[1, 1].hist(r2_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('R-squared (Goodness of Fit)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title(f'Fit Quality Distribution\nMean={np.mean(r2_values):.4f}, Min={np.min(r2_values):.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path('fit_plots')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'parameters_distribution.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  保存统计图: {output_file}")
    plt.close(fig)


def extract_runner_number(filename):
    """从文件名提取runner编号"""
    match = re.search(r'runner(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def find_files(file_base=None):
    """查找需要的文件"""
    
    if file_base:
        runner_pattern = f"{file_base}_out_runner*.csv"
        fit_params = f"{file_base}_fit_params.csv"
    else:
        fit_files = glob.glob("*_fit_params.csv")
        if not fit_files:
            print("错误: 找不到 *_fit_params.csv 文件")
            return None, None
        
        fit_params = fit_files[0]
        file_base = fit_params.replace('_fit_params.csv', '')
        runner_pattern = f"{file_base}_out_runner*.csv"
    
    runner_files = sorted(glob.glob(runner_pattern), key=extract_runner_number)
    
    if not runner_files:
        print(f"错误: 找不到runner文件: {runner_pattern}")
        return None, None
    
    if not Path(fit_params).exists():
        print(f"错误: 找不到拟合参数文件: {fit_params}")
        return None, None
    
    return runner_files, fit_params


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='批量绘制力-位移曲线及Yeoh模型拟合结果'
    )
    parser.add_argument('file_base', nargs='?', default=None,
                       help='文件基础名 (例如: main_lh_sampler)')
    parser.add_argument('--num-plots', type=int, default=None,
                       help='绘制前N个样本 (默认: 全部)')
    parser.add_argument('--grid-size', type=str, default=None,
                       help='网格布局，例如: 4x4')
    parser.add_argument('--save-individual', action='store_true',
                       help='保存每个样本的单独图像')
    parser.add_argument('--dpi', type=int, default=100,
                       help='图像分辨率 (默认: 100)')
    parser.add_argument('--stats-only', action='store_true',
                       help='只绘制统计分布图')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'initial_height': 1.0,
        'contact_area': np.pi * (1 ** 2),
        'disp_col': 'disp_abs',
        'force_col': 'force'
    }
    
    # 查找文件
    runner_files, fit_params_csv = find_files(args.file_base)
    
    if not runner_files or not fit_params_csv:
        sys.exit(1)
    
    print(f"检测到 {len(runner_files)} 个runner文件")
    
    # 绘制统计分布
    plot_statistics(fit_params_csv, config)
    
    # 绘制拟合曲线
    if not args.stats_only:
        plot_batch(runner_files, fit_params_csv, config, args)

    
