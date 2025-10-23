#!/usr/bin/env python3
"""
绘制归一化应力-应变曲线及Yeoh模型拟合

用法:
    python plot_normalized_stress_strain.py [file_base] [options]
    
选项:
    --num-plots N        绘制前N个样本 (默认: 16)
    --grid-size NxM      网格布局 (默认: 4x4)
    --save-individual    保存每个样本的单独图像
    --format FORMAT      输出格式: png, pdf, both (默认: pdf)
    --dpi DPI           图像分辨率 (默认: 150)
    --output DIR        输出目录 (默认: normalized_stress_strain_plots)
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
    I1 = np.power(lmbd, 2) + 2*np.power(lmbd, -1)
    stress = 2 * (lmbd - np.power(lmbd, -2)) * (
        a + 2 * b * (I1 - 3) + 3 * c * np.power((I1 - 3), 2)
    )
    return stress


def read_runner_data(csv_file, config):
    """读取单个runner的CSV数据"""
    displacement_data = []
    force_data = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                disp = float(row.get(config['disp_col'], 0))
                force = float(row.get(config['force_col'], 0))
                
                if abs(disp) > 1e-6 and force > 0:
                    displacement_data.append(abs(disp))
                    force_data.append(force)
            except (ValueError, KeyError):
                continue
    
    return np.array(displacement_data), np.array(force_data)


def plot_single_normalized_curve(runner_num, displacement, force, fit_params, config, ax=None):
    """绘制单个样本的归一化应力-应变曲线"""
    
    # 计算应变和应力
    strain = displacement / config['initial_height']
    stress = force / config['contact_area']
    
    # 归一化（使用最大应力）
    max_stress = np.max(stress)
    normalized_stress = stress / max_stress
    
    # 生成拟合曲线
    strain_fit = np.linspace(strain.min(), strain.max(), 200)
    stress_fit = YeohModel_Power3(strain_fit,
                                   fit_params['a'],
                                   fit_params['b'],
                                   fit_params['c'])
    normalized_stress_fit = stress_fit / max_stress
    
    # 如果没有提供ax，创建新图
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False
    
    # 绘制原始数据点
    ax.scatter(strain, normalized_stress, s=20, alpha=0.6, 
              label='Original Data', color='blue')
    
    # 绘制拟合曲线
    ax.plot(strain_fit, normalized_stress_fit, 'r-', linewidth=2, 
           label='Yeoh Fit')
    
    # 设置标签和标题
    ax.set_xlabel('Strain (mm/mm)', fontsize=10)
    ax.set_ylabel('Normalized Stress (σ/σ_max)', fontsize=10)
    ax.set_title(f'Runner {runner_num}\n'
                 f'σ_max={max_stress:.2f} MPa, R²={fit_params["r_squared"]:.4f}\n'
                 f'YS={fit_params.get("yield_stress", 0):.1f} MPa, '
                 f'HC={fit_params.get("hardening_constant", 0):.1f} MPa',
                 fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    if standalone:
        plt.tight_layout()
        return fig
    else:
        return ax


def plot_batch(runner_files, fit_params_csv, complete_results_csv, config, args):
    """批量绘制"""
    
    print(f"\nReading fit parameters: {fit_params_csv}")
    
    # 读取拟合参数
    fit_params = {}
    with open(fit_params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runner = int(row['runner'])
            if row['fit_success'] == 'True':
                fit_params[runner] = {
                    'a': float(row['a']),
                    'b': float(row['b']),
                    'c': float(row['c']),
                    'r_squared': float(row['r_squared']),
                    'filename': row['filename']
                }
    
    # 读取完整结果（包含yield_stress和hardening_constant）
    if Path(complete_results_csv).exists():
        print(f"Reading complete results: {complete_results_csv}")
        with open(complete_results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                runner = int(row['runner'])
                if runner in fit_params:
                    fit_params[runner]['yield_stress'] = float(row['yield_stress'])
                    fit_params[runner]['hardening_constant'] = float(row['hardening_constant'])
    
    # 筛选出成功拟合的样本
    valid_runners = sorted(fit_params.keys())
    
    # 确定要绘制的样本
    total_samples = len(valid_runners)
    num_to_plot = min(args.num_plots, total_samples) if args.num_plots else total_samples
    
    print(f"Total successful fits: {total_samples}")
    print(f"Number to plot: {num_to_plot}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 如果需要保存单独图像
    if args.save_individual:
        print(f"\nSaving individual plots to: {output_dir}/")
        for i, runner_num in enumerate(valid_runners[:num_to_plot]):
            # 找到对应的CSV文件
            csv_file = None
            for f in runner_files:
                if f'runner{runner_num:03d}' in f:
                    csv_file = f
                    break
            
            if csv_file is None:
                continue
            
            # 读取数据
            displacement, force = read_runner_data(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制
            fig = plot_single_normalized_curve(runner_num, displacement, force, 
                                             fit_params[runner_num], config)
            
            # 保存
            if args.format in ['png', 'both']:
                output_file = output_dir / f'runner{runner_num:03d}_normalized.png'
                plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
            
            if args.format in ['pdf', 'both']:
                output_file = output_dir / f'runner{runner_num:03d}_normalized.pdf'
                plt.savefig(output_file, bbox_inches='tight')
            
            plt.close(fig)
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{num_to_plot} plots")
        
        print(f"Complete! All individual plots saved to {output_dir}/")
    
    # 绘制网格图
    if args.grid_size:
        nrows, ncols = map(int, args.grid_size.split('x'))
    else:
        nrows, ncols = 4, 4
    
    plots_per_grid = nrows * ncols
    num_grids = (num_to_plot + plots_per_grid - 1) // plots_per_grid
    
    print(f"\nGenerating grid plots: {num_grids} page(s), {nrows}x{ncols} each")
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * plots_per_grid
        end_idx = min(start_idx + plots_per_grid, num_to_plot)
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(ncols * 3.5, nrows * 3))
        
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # 绘制当前页的子图
        plot_count = 0
        for i in range(start_idx, end_idx):
            runner_num = valid_runners[i]
            
            # 找到对应的CSV文件
            csv_file = None
            for f in runner_files:
                if f'runner{runner_num:03d}' in f:
                    csv_file = f
                    break
            
            if csv_file is None:
                continue
            
            # 读取数据
            displacement, force = read_runner_data(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制到子图
            ax = axes[plot_count]
            plot_single_normalized_curve(runner_num, displacement, force, 
                                       fit_params[runner_num], config, ax=ax)
            
            plot_count += 1
        
        # 隐藏多余的子图
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # 保存网格图
        if args.format in ['png', 'both']:
            output_file = output_dir / f'normalized_grid_{grid_idx+1:02d}.png'
            plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
            print(f"  Saved grid plot: {output_file}")
        
        if args.format in ['pdf', 'both']:
            output_file = output_dir / f'normalized_grid_{grid_idx+1:02d}.pdf'
            plt.savefig(output_file, bbox_inches='tight')
            print(f"  Saved grid plot: {output_file}")
        
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}")
    print(f"Plots saved to: {output_dir}/")


def plot_overlay_all(runner_files, fit_params_csv, complete_results_csv, config, args):
    """将所有归一化曲线叠加在一起"""
    
    print(f"\nGenerating overlay plot...")
    
    # 读取拟合参数
    fit_params = {}
    with open(fit_params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runner = int(row['runner'])
            if row['fit_success'] == 'True':
                fit_params[runner] = {
                    'a': float(row['a']),
                    'b': float(row['b']),
                    'c': float(row['c']),
                    'r_squared': float(row['r_squared']),
                }
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    valid_runners = sorted(fit_params.keys())
    num_to_plot = min(args.num_plots if args.num_plots else len(valid_runners), 
                     len(valid_runners))
    
    # 应用步长间隔
    step = max(1, args.overlay_step)
    runners_to_plot = valid_runners[:num_to_plot:step]
    actual_num_plotted = len(runners_to_plot)
    
    print(f"  Plotting {actual_num_plotted} curves with step={step} (from {num_to_plot} total)")
    
    # 绘制所有曲线
    for i, runner_num in enumerate(runners_to_plot):
        # 找到对应的CSV文件
        csv_file = None
        for f in runner_files:
            if f'runner{runner_num:03d}' in f:
                csv_file = f
                break
        
        if csv_file is None:
            continue
        
        # 读取数据
        displacement, force = read_runner_data(csv_file, config)
        
        if len(displacement) == 0:
            continue
        
        # 计算归一化应力-应变
        strain = displacement / config['initial_height']
        stress = force / config['contact_area']
        max_stress = np.max(stress)
        normalized_stress = stress / max_stress
        
        # 绘制原始数据（半透明）
        # ax.scatter(strain, normalized_stress, s=1, alpha=0.2, marker='o', color='blue')
        ax.plot(strain, normalized_stress, '-', marker='o' , markersize=1, alpha=0.5, linewidth=1, color='blue')
        
        # 绘制拟合曲线（半透明）
        strain_fit = np.linspace(strain.min(), strain.max(), 100)
        stress_fit = YeohModel_Power3(strain_fit, 
                                      fit_params[runner_num]['a'], 
                                      fit_params[runner_num]['b'], 
                                      fit_params[runner_num]['c'])
        normalized_stress_fit = stress_fit / max_stress
        ax.plot(strain_fit, normalized_stress_fit, '--', alpha=0.6, linewidth=0.9, color='red')
    
    ax.set_xlabel('Strain', fontsize=14)
    ax.set_ylabel('Normalized Stress (σ/σ_max)', fontsize=14)
    step_info = f' (step={step})' if step > 1 else ''
    ax.set_title(f'Overlay of Normalized Stress-Strain Curves\n'
                 f'Blue: Original Data, Red: Yeoh Fit', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 保存
    if args.format in ['png', 'both']:
        output_file = output_dir / 'normalized_overlay_all.png'
        plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if args.format in ['pdf', 'both']:
        output_file = output_dir / 'normalized_overlay_all.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
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
        complete_results = f"{file_base}_complete_results.csv"
    else:
        fit_files = glob.glob("*_fit_params.csv")
        if not fit_files:
            print("Error: Cannot find *_fit_params.csv file")
            return None, None, None
        
        fit_params = fit_files[0]
        file_base = fit_params.replace('_fit_params.csv', '')
        runner_pattern = f"{file_base}_out_runner*.csv"
        complete_results = f"{file_base}_complete_results.csv"
    
    runner_files = sorted(glob.glob(runner_pattern), key=extract_runner_number)
    
    if not runner_files:
        print(f"Error: Cannot find runner files: {runner_pattern}")
        return None, None, None
    
    if not Path(fit_params).exists():
        print(f"Error: Cannot find fit params file: {fit_params}")
        return None, None, None
    
    return runner_files, fit_params, complete_results


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Plot normalized stress-strain curves with Yeoh fit'
    )
    parser.add_argument('file_base', nargs='?', default=None,
                       help='File base name (e.g., main_lh_sampler)')
    parser.add_argument('--num-plots', type=int, default=None,
                       help='Number of samples to plot (default: 16)')
    parser.add_argument('--grid-size', type=str, default='5x5',
                       help='Grid layout, e.g., 5x5')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual plots for each sample')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    parser.add_argument('--output', type=str, default='normalized_stress_strain_plots',
                       help='Output directory (default: normalized_stress_strain_plots)')
    parser.add_argument('--overlay', action='store_true',
                       help='Generate overlay plot with all curves')
    parser.add_argument('--overlay-only', action='store_true',
                       help='Only generate overlay plot')
    parser.add_argument('--overlay-step', type=int, default=1,
                       help='Step interval for overlay plot (e.g., 2 means plot every 2nd sample) (default: 1)')

    args = parser.parse_args()
    
    # 配置
    config = {
        'initial_height': 1.0,
        'contact_area': np.pi * (1.0)**2,
        'disp_col': 'disp_abs',
        'force_col': 'force'
    }
    
    # 查找文件
    runner_files, fit_params_csv, complete_results_csv = find_files(args.file_base)
    
    if not runner_files or not fit_params_csv:
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"Plotting Normalized Stress-Strain Curves")
    print(f"{'='*60}")
    print(f"Detected {len(runner_files)} runner files")
    print(f"Fit parameters: {fit_params_csv}")
    if Path(complete_results_csv).exists():
        print(f"Complete results: {complete_results_csv}")
    
    if args.overlay_only:
        # 只绘制叠加图
        plot_overlay_all(runner_files, fit_params_csv, complete_results_csv, config, args)
    else:
        # 绘制归一化曲线
        plot_batch(runner_files, fit_params_csv, complete_results_csv, config, args)
        
        # 如果需要，绘制叠加图
        if args.overlay:
            plot_overlay_all(runner_files, fit_params_csv, complete_results_csv, config, args)

