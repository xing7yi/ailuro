#!/usr/bin/env python3
"""
绘制 yield_stress 和 hardening_constant 与最大应力的关系

用法:
    python plot_stress_surface.py [csv_file] [options]
    
选项:
    --radius R       粒子半径 (mm) (默认: 1.0)
    --format FORMAT  输出格式: png, pdf, both (默认: pdf)
    --dpi DPI        图像分辨率 (默认: 150)
    --output DIR     输出目录 (默认: stress_plots)
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path
import argparse


def read_data(csv_file, particle_radius):
    """读取CSV数据并计算最大应力"""
    
    yield_stress_list = []
    hardening_constant_list = []
    max_stress_list = []
    runner_list = []
    fit_success_list = []
    
    contact_area = np.pi * particle_radius**2
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                runner = int(row['runner'])
                yield_stress = float(row['yield_stress'])
                hardening_constant = float(row['hardening_constant'])
                max_force = float(row['max_force'])
                fit_success = row['fit_success'] == 'True'
                
                # 计算最大应力
                max_stress = max_force / contact_area
                
                yield_stress_list.append(yield_stress)
                hardening_constant_list.append(hardening_constant)
                max_stress_list.append(max_stress)
                runner_list.append(runner)
                fit_success_list.append(fit_success)
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row {row.get('runner', '?')}: {e}")
                continue
    
    return {
        'runner': np.array(runner_list),
        'yield_stress': np.array(yield_stress_list),
        'hardening_constant': np.array(hardening_constant_list),
        'max_stress': np.array(max_stress_list),
        'fit_success': np.array(fit_success_list)
    }


def plot_3d_scatter(data, output_dir, file_format, dpi):
    """绘制3D散点图"""
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 分离成功和失败的拟合
    success_mask = data['fit_success']
    fail_mask = ~success_mask
    
    # 绘制成功的点
    if np.any(success_mask):
        scatter = ax.scatter(data['yield_stress'][success_mask], 
                           data['hardening_constant'][success_mask],
                           data['max_stress'][success_mask],
                           c=data['max_stress'][success_mask],
                           cmap='viridis',
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           label='Successful Fit')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Max Stress (MPa)', fontsize=12)
    
    # 绘制失败的点
    if np.any(fail_mask):
        ax.scatter(data['yield_stress'][fail_mask], 
                  data['hardening_constant'][fail_mask],
                  data['max_stress'][fail_mask],
                  c='red',
                  s=100,
                  alpha=0.5,
                  marker='x',
                  linewidth=2,
                  label='Failed Fit')
    
    ax.set_xlabel('Yield Stress (MPa)', fontsize=12, labelpad=10)
    ax.set_ylabel('Hardening Constant (MPa)', fontsize=12, labelpad=10)
    ax.set_zlabel('Max Stress (MPa)', fontsize=12, labelpad=10)
    ax.set_title('Relationship between Material Parameters and Max Stress', 
                 fontsize=14, pad=20)
    
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 保存
    if file_format in ['png', 'both']:
        output_file = output_dir / 'stress_3d_scatter.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / 'stress_3d_scatter.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_3d_surface(data, output_dir, file_format, dpi):
    """绘制3D曲面图（插值）"""
    
    # 只使用成功拟合的数据
    success_mask = data['fit_success']
    
    if np.sum(success_mask) < 4:
        print("Warning: Not enough successful data points for surface plot")
        return
    
    x = data['yield_stress'][success_mask]
    y = data['hardening_constant'][success_mask]
    z = data['max_stress'][success_mask]
    
    # 创建网格用于插值
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)
    
    # 插值
    ZI = griddata((x, y), z, (XI, YI), method='cubic')
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', alpha=0.7,
                          linewidth=0, antialiased=True)
    
    # 叠加原始数据点
    ax.scatter(x, y, z, c='red', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(surf, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Max Stress (MPa)', fontsize=12)
    
    ax.set_xlabel('Yield Stress (MPa)', fontsize=12, labelpad=10)
    ax.set_ylabel('Hardening Constant (MPa)', fontsize=12, labelpad=10)
    ax.set_zlabel('Max Stress (MPa)', fontsize=12, labelpad=10)
    ax.set_title('Interpolated Surface: Material Parameters vs Max Stress', 
                 fontsize=14, pad=0)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 保存
    if file_format in ['png', 'both']:
        output_file = output_dir / 'stress_3d_surface.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / 'stress_3d_surface.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_2d_contour(data, output_dir, file_format, dpi):
    """绘制2D等高线图"""
    
    # 只使用成功拟合的数据
    success_mask = data['fit_success']
    
    if np.sum(success_mask) < 4:
        print("Warning: Not enough successful data points for contour plot")
        return
    
    x = data['yield_stress'][success_mask]
    y = data['hardening_constant'][success_mask]
    z = data['max_stress'][success_mask]
    
    # 创建网格用于插值
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    
    # 插值
    ZI = griddata((x, y), z, (XI, YI), method='cubic')
    
    fig, ax = plt.subplots(figsize=(5.6, 4))
    
    # 绘制等高线填充
    contourf = ax.contourf(XI, YI, ZI, levels=20, cmap='viridis', alpha=0.8)
    
    # 绘制等高线
    contour = ax.contour(XI, YI, ZI, levels=10, colors='black', 
                        linewidths=0.5, alpha=0.4)
    ax.clabel(contour , inline=True, fontsize=8, fmt='%.0f')
    
    # 叠加原始数据点
    scatter = ax.scatter(x, y, c=z, s=50, cmap='viridis', 
                        edgecolors='black', linewidth=0.5, zorder=5)
    
    # 添加颜色条
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Max Stress (MPa)', fontsize=12)
    
    ax.set_xlabel('Yield Stress (MPa)', fontsize=12)
    ax.set_ylabel('Hardening Constant (MPa)', fontsize=12)
    ax.set_title('Contour Plot: Material Parameters vs Max Stress', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if file_format in ['png', 'both']:
        output_file = output_dir / 'stress_2d_contour.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / 'stress_2d_contour.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_individual_relationships(data, output_dir, file_format, dpi):
    """绘制单独的参数关系图"""
    
    success_mask = data['fit_success']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # yield_stress vs max_stress
    ax = axes[0]
    if np.any(success_mask):
        ax.scatter(data['yield_stress'][success_mask], 
                  data['max_stress'][success_mask],
                  c=data['hardening_constant'][success_mask],
                  cmap='viridis',
                  s=60,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Hardening Constant (MPa)', fontsize=10)
    
    if np.any(~success_mask):
        ax.scatter(data['yield_stress'][~success_mask], 
                  data['max_stress'][~success_mask],
                  c='red',
                  s=100,
                  alpha=0.5,
                  marker='x',
                  linewidth=2)
    
    ax.set_xlabel('Yield Stress (MPa)', fontsize=12)
    ax.set_ylabel('Max Stress (MPa)', fontsize=12)
    ax.set_title('Yield Stress vs Max Stress', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # hardening_constant vs max_stress
    ax = axes[1]
    if np.any(success_mask):
        ax.scatter(data['hardening_constant'][success_mask], 
                  data['max_stress'][success_mask],
                  c=data['yield_stress'][success_mask],
                  cmap='viridis',
                  s=60,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Yield Stress (MPa)', fontsize=10)
    
    if np.any(~success_mask):
        ax.scatter(data['hardening_constant'][~success_mask], 
                  data['max_stress'][~success_mask],
                  c='red',
                  s=100,
                  alpha=0.5,
                  marker='x',
                  linewidth=2)
    
    ax.set_xlabel('Hardening Constant (MPa)', fontsize=12)
    ax.set_ylabel('Max Stress (MPa)', fontsize=12)
    ax.set_title('Hardening Constant vs Max Stress', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if file_format in ['png', 'both']:
        output_file = output_dir / 'stress_individual_relationships.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / 'stress_individual_relationships.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def print_statistics(data):
    """打印统计信息"""
    
    success_mask = data['fit_success']
    n_total = len(data['runner'])
    n_success = np.sum(success_mask)
    n_fail = n_total - n_success
    
    print(f"\n{'='*60}")
    print(f"Data Statistics:")
    print(f"{'='*60}")
    print(f"Total samples: {n_total}")
    print(f"Successful fits: {n_success} ({100*n_success/n_total:.1f}%)")
    print(f"Failed fits: {n_fail} ({100*n_fail/n_total:.1f}%)")
    
    if n_success > 0:
        print(f"\nYield Stress (MPa):")
        print(f"  Range: [{data['yield_stress'][success_mask].min():.2f}, "
              f"{data['yield_stress'][success_mask].max():.2f}]")
        print(f"  Mean: {data['yield_stress'][success_mask].mean():.2f}")
        print(f"  Std: {data['yield_stress'][success_mask].std():.2f}")
        
        print(f"\nHardening Constant (MPa):")
        print(f"  Range: [{data['hardening_constant'][success_mask].min():.2f}, "
              f"{data['hardening_constant'][success_mask].max():.2f}]")
        print(f"  Mean: {data['hardening_constant'][success_mask].mean():.2f}")
        print(f"  Std: {data['hardening_constant'][success_mask].std():.2f}")
        
        print(f"\nMax Stress (MPa):")
        print(f"  Range: [{data['max_stress'][success_mask].min():.2f}, "
              f"{data['max_stress'][success_mask].max():.2f}]")
        print(f"  Mean: {data['max_stress'][success_mask].mean():.2f}")
        print(f"  Std: {data['max_stress'][success_mask].std():.2f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Plot relationship between material parameters and max stress'
    )
    parser.add_argument('csv_file', nargs='?', 
                       default='main_lh_sampler_complete_results.csv',
                       help='CSV file with results')
    parser.add_argument('--radius', type=float, default=1.0,
                       help='Particle radius in mm (default: 1.0)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    parser.add_argument('--output', type=str, default='stress_plots',
                       help='Output directory (default: stress_plots)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Plotting stress relationships")
    print(f"{'='*60}")
    print(f"Input file: {args.csv_file}")
    print(f"Particle radius: {args.radius} mm")
    print(f"Contact area: {np.pi * args.radius**2:.4f} mm²")
    print(f"Output directory: {output_dir}/")
    print(f"Output format: {args.format}")
    
    # 读取数据
    print(f"\nReading data...")
    data = read_data(args.csv_file, args.radius)
    
    # 打印统计信息
    print_statistics(data)
    
    # 绘制图形
    print("Generating plots...")
    
    plot_3d_scatter(data, output_dir, args.format, args.dpi)
    plot_3d_surface(data, output_dir, args.format, args.dpi)
    plot_2d_contour(data, output_dir, args.format, args.dpi)
    plot_individual_relationships(data, output_dir, args.format, args.dpi)
    
    print(f"\n{'='*60}")
    print(f"Complete! All plots saved to: {output_dir}/")
    print(f"{'='*60}\n")
