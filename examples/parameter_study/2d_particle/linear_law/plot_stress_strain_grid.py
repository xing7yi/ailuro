#!/usr/bin/env python3


import csv
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


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



def plot_single(runner_num, displacement, force, config, ax=None, material_props=None):
    """绘制单个样本的拟合结果"""
    
    # 计算应变和应力
    epsilon = displacement / config['initial_height']
    stress = force / config['contact_area']

    log_strain = -np.log(1-epsilon)
    log_stress = (1-epsilon) * stress

    # 如果没有提供ax，创建新图
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False
    
    # 绘制原始数据点
    ax.plot(log_strain, log_stress, 'o-', markersize=2, alpha=1, label='Original Data', color='blue')
    
    # 设置标签和标题
    ax.set_xlabel('Strain (log)', fontsize=10)
    ax.set_ylabel('Stress (MPa)', fontsize=10)
    
    # 构建标题（包含材料属性）
    title = f'Runner {runner_num}'
    if material_props and runner_num in material_props:
        props_str = format_material_props(material_props[runner_num])
        if props_str:
            title += f'\n{props_str}'
    
    ax.set_title(title, fontsize=9)
    # ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig
    else:
        return ax



def plot_batch(runner_files, config, args, material_props=None):
    """批量绘制"""
     
    # 确定要绘制的样本
    total_samples = len(runner_files)
    num_to_plot = min(args.num_plots, total_samples) if args.num_plots else total_samples
    
    print(f"总样本数: {total_samples}")
    print(f"绘制样本数: {num_to_plot}")
    
    # 创建输出目录
    output_dir = Path('stress_strain_plots')
    output_dir.mkdir(exist_ok=True)

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
            
            # 读取数据
            displacement, force = read_runner_data(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制到子图
            ax = axes[plot_count]
            plot_single(runner_num, displacement, force, config, ax=ax, material_props=material_props)
            
            plot_count += 1
        
        # 隐藏多余的子图
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # 保存网格图
        output_file = output_dir / f'response_curve_grid_{grid_idx+1:02d}.pdf'
        plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"  保存网格图: {output_file}")
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")
    print(f"图像保存位置: {output_dir}/")


def extract_runner_number(filename):
    """从文件名提取runner编号"""
    match = re.search(r'runner(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def load_material_properties(file_base):
    """加载材料属性数据"""
    # 尝试加载完整结果CSV（包含材料属性）
    complete_csv = f"{file_base}_complete_results.csv"
    
    if not Path(complete_csv).exists():
        print(f"警告: 找不到材料属性文件 {complete_csv}")
        return None
    
    material_props = {}
    try:
        with open(complete_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                runner_num = int(row.get('runner', -1))
                if runner_num >= 0:
                    # 提取材料属性（根据实际CSV文件调整列名）
                    props = {}
                    if 'yield_stress' in row:
                        props['yield_stress'] = float(row['yield_stress'])
                    if 'hardening_constant' in row:
                        props['hardening_constant'] = float(row['hardening_constant'])
                    if 'q' in row:
                        props['q'] = float(row['q'])
                    
                    material_props[runner_num] = props
        
        print(f"成功加载 {len(material_props)} 个样本的材料属性")
        return material_props
    except Exception as e:
        print(f"警告: 加载材料属性失败: {e}")
        return None


def format_material_props(props):
    """格式化材料属性为字符串"""
    if not props:
        return ""
    
    parts = []
    if 'yield_stress' in props:
        parts.append(f"σ_y={props['yield_stress']:.1f}")
    if 'hardening_constant' in props:
        parts.append(f"H={props['hardening_constant']:.1f}")
    if 'q' in props:
        parts.append(f"q={props['q']:.1f}")
    
    return ", ".join(parts)


def find_runner_files(file_base=None):
    """查找需要的文件"""
    
    if file_base:
        runner_pattern = f"{file_base}_out_runner*.csv"
    else:
        # 自动查找
        patterns = glob.glob("*_out_runner*.csv")
        if not patterns:
            print("错误: 找不到 *_out_runner*.csv 文件")
            return None, None
        
        # 推断file_base
        first_file = patterns[0]
        match = re.match(r'(.+?)_out_runner\d+\.csv', first_file)
        if match:
            file_base = match.group(1)
            pattern = f"{file_base}_out_runner*.csv"
        else:
            print(f"错误: 无法从文件名 {first_file} 推断 file_base")
            return None, None
    
    files = sorted(glob.glob(pattern), key=extract_runner_number)

    if not files:
        print(f"错误: 找不到匹配的文件: {pattern}")
        return None, None
    
    return files, file_base


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='批量绘制力-位移曲线'
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
    runner_files, detected_base = find_runner_files(args.file_base)

    if runner_files is None:
        exit(1)

    print(f"检测到 file_base: {detected_base}, 共 {len(runner_files)} 个runner文件")

    # 加载材料属性
    material_props = load_material_properties(detected_base)

    # 绘制曲线
    plot_batch(runner_files, config, args, material_props=material_props)

    
