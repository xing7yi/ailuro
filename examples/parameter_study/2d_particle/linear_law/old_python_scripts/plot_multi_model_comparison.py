#!/usr/bin/env python3
"""
可视化不同应变因子模型的拟合效果

从拟合结果CSV文件中读取数据，生成对比图：
1. 各模型拟合质量对比（箱线图、直方图）
2. 选定样本的拟合曲线对比
3. R²分布对比
4. AIC/BIC对比
5. 参数分布对比

用法:
    python plot_multi_model_comparison.py [file_base]
    
    可选参数:
    --runners N1 N2 N3    指定要绘制的runner编号（默认：自动选择）
    --format FORMAT       输出格式: png, pdf, both (默认: pdf)
    --dpi DPI            图像分辨率 (默认: 150)
"""

import sys
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re


# 模型信息
MODELS = {
    'yeoh': {'name': 'Yeoh', 'color': 'blue', 'label': r'Yeoh: $(\lambda - \lambda^{-2})$'},
    'linear': {'name': 'Linear', 'color': 'red', 'label': r'Linear: $(\lambda - 1)$'},
    # 'logarithmic': {'name': 'Logarithmic', 'color': 'green', 'label': r'Log: $\ln(\lambda^2)$'},
    # 'quadratic': {'name': 'Quadratic', 'color': 'purple', 'label': r'Quad: $(\lambda^2 - 1)$'},
    'cubic':{'name': 'Cubic Yeoh', 'color': 'purple', 'label': r'Cubic: $(\lambda - 1/\lambda^2)$'}
}


def compute_I1(lmbd):
    """计算第一不变量"""
    return np.power(lmbd, 2) + 2.0 / lmbd


def polynomial_term(lmbd, a, b, c):
    """多项式项"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


def model_yeoh(epsilon, a, b, c):
    """Yeoh模型"""
    lmbd = epsilon + 1.0
    return 2.0 * (lmbd - 1.0/(lmbd**2)) * polynomial_term(lmbd, a, b, c)


def model_linear(epsilon, a, b, c):
    """Linear模型"""
    lmbd = epsilon + 1.0
    return 2.0 * (lmbd - 1.0) * polynomial_term(lmbd, a, b, c)


def model_logarithmic(epsilon, a, b, c):
    """Logarithmic模型"""
    lmbd = epsilon + 1.0
    return 2.0 * np.log(lmbd) * polynomial_term(lmbd, a, b, c)


def model_quadratic(epsilon, a, b, c):
    """Quadratic模型"""
    lmbd = epsilon + 1.0
    return 2.0 * (lmbd**2 - 1.0) * polynomial_term(lmbd, a, b, c)

def model_cubic(epsilon, a, b, c):
    lmbd = epsilon + 1.0
    return 2.0 * (lmbd**1 - 1.0/(lmbd**3)) * polynomial_term(lmbd, a, b, c)


MODEL_FUNCTIONS = {
    'yeoh': model_yeoh,
    'linear': model_linear,
    'logarithmic': model_logarithmic,
    'quadratic': model_quadratic,
    'cubic':model_cubic
}


def read_fit_params(file_base, model_key):
    """读取某个模型的拟合参数"""
    csv_file = f"{file_base}_fit_params_{model_key}.csv"
    
    if not Path(csv_file).exists():
        print(f"警告: 找不到文件 {csv_file}")
        return None
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值
            for key in ['runner', 'a', 'b', 'c', 'r_squared', 'rmse', 
                       'rel_rmse', 'aic', 'bic', 'n_points']:
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = np.nan
            
            row['fit_success'] = row.get('fit_success', 'False') == 'True'
            data.append(row)
    
    return data


def read_original_data(csv_file, config):
    """读取原始CSV数据"""
    displacement_data = []
    force_data = []
    
    try:
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
        
        if len(displacement_data) < 5:
            return None, None
        
        displacement = np.array(displacement_data)
        force = np.array(force_data)
        
        epsilon = displacement / config['initial_height']
        stress = force / config['contact_area']
        
        return epsilon, stress
        
    except Exception as e:
        print(f"读取文件失败 {csv_file}: {e}")
        return None, None


def plot_r2_comparison(all_params, file_base, file_format='pdf', dpi=150):
    """绘制R²分布对比"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 箱线图
    data_for_box = []
    labels = []
    colors = []
    
    for model_key in MODELS.keys():
        if model_key in all_params and all_params[model_key]:
            r2_values = [p['r_squared'] for p in all_params[model_key] 
                        if p['fit_success'] and not np.isnan(p['r_squared'])]
            if r2_values:
                data_for_box.append(r2_values)
                labels.append(MODELS[model_key]['name'])
                colors.append(MODELS[model_key]['color'])
    
    bp = ax1.boxplot(data_for_box, tick_labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('R² (Coefficient of Determination)', fontsize=12)
    ax1.set_title('R² Distribution Across Models', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.9, 1.0])
    
    # 直方图叠加
    for model_key in MODELS.keys():
        if model_key in all_params and all_params[model_key]:
            r2_values = [p['r_squared'] for p in all_params[model_key] 
                        if p['fit_success'] and not np.isnan(p['r_squared'])]
            if r2_values:
                ax2.hist(r2_values, bins=30, alpha=0.5, 
                        color=MODELS[model_key]['color'],
                        label=MODELS[model_key]['name'],
                        density=True)
    
    ax2.set_xlabel('R²', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('R² Distribution (Normalized Histogram)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.9, 1.0])
    
    plt.tight_layout()
    
    if file_format in ['png', 'both']:
        plt.savefig(f'{file_base}_r2_comparison.png', dpi=dpi, bbox_inches='tight')
    if file_format in ['pdf', 'both']:
        plt.savefig(f'{file_base}_r2_comparison.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f"  已保存: {file_base}_r2_comparison")


def plot_aic_comparison(all_params, file_base, file_format='pdf', dpi=150):
    """绘制AIC对比"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_for_box = []
    labels = []
    colors = []
    
    for model_key in MODELS.keys():
        if model_key in all_params and all_params[model_key]:
            aic_values = [p['aic'] for p in all_params[model_key] 
                         if p['fit_success'] and not np.isnan(p['aic'])]
            if aic_values:
                data_for_box.append(aic_values)
                labels.append(MODELS[model_key]['name'])
                colors.append(MODELS[model_key]['color'])
    
    bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('AIC (Akaike Information Criterion)', fontsize=12)
    ax.set_title('AIC Comparison (Lower is Better)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if file_format in ['png', 'both']:
        plt.savefig(f'{file_base}_aic_comparison.png', dpi=dpi, bbox_inches='tight')
    if file_format in ['pdf', 'both']:
        plt.savefig(f'{file_base}_aic_comparison.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f"  已保存: {file_base}_aic_comparison")


def plot_parameter_distributions(all_params, file_base, file_format='pdf', dpi=150):
    """绘制参数分布对比"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    param_names = ['a', 'b', 'c']
    param_labels = ['Parameter a', 'Parameter b', 'Parameter c']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        
        for model_key in MODELS.keys():
            if model_key in all_params and all_params[model_key]:
                param_values = [p[param] for p in all_params[model_key] 
                              if p['fit_success'] and not np.isnan(p[param])]
                if param_values:
                    ax.hist(param_values, bins=20, alpha=0.5,
                           color=MODELS[model_key]['color'],
                           label=MODELS[model_key]['name'],
                           density=True)
        
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {label}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if file_format in ['png', 'both']:
        plt.savefig(f'{file_base}_param_distributions.png', dpi=dpi, bbox_inches='tight')
    if file_format in ['pdf', 'both']:
        plt.savefig(f'{file_base}_param_distributions.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f"  已保存: {file_base}_param_distributions")


def plot_fitting_curves(runner_nums, all_params, file_base, config,
                       file_format='pdf', dpi=150):
    """绘制选定runner的拟合曲线对比"""
    
    # 读取材料属性数据
    material_props = {}
    complete_results_file = f"{file_base}_complete_results.csv"
    if Path(complete_results_file).exists():
        try:
            with open(complete_results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    runner_id = int(float(row.get('runner', -1)))
                    if runner_id >= 0:
                        material_props[runner_id] = {
                            'yield_stress': float(row.get('yield_stress', 0)),
                            'hardening_constant': float(row.get('hardening_constant', 0))
                        }
        except Exception as e:
            print(f"  警告: 无法读取材料属性 {complete_results_file}: {e}")
    
    n_runners = len(runner_nums)
    fig, axes = plt.subplots(n_runners, 1, figsize=(7, 3.6*n_runners))
    
    if n_runners == 1:
        axes = [axes]
    
    for idx, runner_num in enumerate(runner_nums):
        ax = axes[idx]
        
        # 读取原始数据（尝试不同的文件名格式）
        csv_file = f"{file_base}_out_runner{runner_num:03d}.csv"
        if not Path(csv_file).exists():
            csv_file = f"{file_base}_out_runner{runner_num}.csv"
        
        epsilon_orig, stress_orig = read_original_data(csv_file, config)
        
        if epsilon_orig is None:
            ax.text(0.5, 0.5, f'Runner {runner_num}: No data',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 绘制原始数据
        ax.scatter(epsilon_orig, stress_orig, s=20, c='gray', alpha=0.5,
                  label='Original Data', marker='o', zorder=1)
        
        # 绘制各模型的拟合曲线
        epsilon_fit = np.linspace(0, np.max(epsilon_orig), 200)
        
        for model_key in MODELS.keys():
            if model_key not in all_params or not all_params[model_key]:
                continue
            
            # 找到对应runner的参数
            params = None
            for p in all_params[model_key]:
                if int(p['runner']) == runner_num:
                    params = p
                    break
            
            if params and params['fit_success']:
                a, b, c = params['a'], params['b'], params['c']
                r2 = params['r_squared']
                
                stress_fit = MODEL_FUNCTIONS[model_key](epsilon_fit, a, b, c)
                
                ax.plot(epsilon_fit, stress_fit, linewidth=2,
                       color=MODELS[model_key]['color'],
                       label=f"{MODELS[model_key]['name']} (R²={r2:.4f})",
                       zorder=2)
        
        ax.set_xlabel('Strain', fontsize=12)
        ax.set_ylabel('Stress', fontsize=12)
        
        # 构建标题，包含材料属性
        title = f'Runner {runner_num}: Multi-Model Fitting Comparison'
        if runner_num in material_props:
            ys = material_props[runner_num]['yield_stress']
            hc = material_props[runner_num]['hardening_constant']
            title += f'\n($\\sigma_y$={ys:.1f} MPa, $H$={hc:.1f} MPa)'
        
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_name = f'{file_base}_fitting_curves_comparison'
    if file_format in ['png', 'both']:
        plt.savefig(f'{output_name}.png', dpi=dpi, bbox_inches='tight')
    if file_format in ['pdf', 'both']:
        plt.savefig(f'{output_name}.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f"  已保存: {output_name}")


def plot_success_rate(all_params, file_base, file_format='pdf', dpi=150):
    """绘制成功率对比"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    success_rates = []
    colors = []
    
    for model_key in MODELS.keys():
        if model_key in all_params and all_params[model_key]:
            total = len(all_params[model_key])
            success = sum(1 for p in all_params[model_key] if p['fit_success'])
            rate = success / total * 100 if total > 0 else 0
            
            model_names.append(MODELS[model_key]['name'])
            success_rates.append(rate)
            colors.append(MODELS[model_key]['color'])
    
    bars = ax.bar(model_names, success_rates, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%',
               ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Fitting Success Rate by Model', fontsize=14)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if file_format in ['png', 'both']:
        plt.savefig(f'{file_base}_success_rate.png', dpi=dpi, bbox_inches='tight')
    if file_format in ['pdf', 'both']:
        plt.savefig(f'{file_base}_success_rate.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f"  已保存: {file_base}_success_rate")


def auto_select_runners(all_params, n_select=3):
    """自动选择代表性的runner编号"""
    
    # 使用yeoh模型的结果作为参考
    if 'yeoh' not in all_params or not all_params['yeoh']:
        return []
    
    successful = [p for p in all_params['yeoh'] if p['fit_success']]
    
    if len(successful) == 0:
        return []
    
    # 按R²排序
    successful.sort(key=lambda x: x['r_squared'], reverse=True)
    
    # 选择最好、中等、最差的
    selected = []
    if len(successful) >= 1:
        selected.append(int(successful[0]['runner']))  # 最好
    if len(successful) >= 2:
        selected.append(int(successful[len(successful)//2]['runner']))  # 中等
    if len(successful) >= 3:
        selected.append(int(successful[-1]['runner']))  # 最差
    
    return selected[:n_select]


if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(
        description='Visualize multi-model fitting comparison'
    )
    parser.add_argument('file_base', nargs='?', default=None,
                       help='Base name of the CSV files')
    parser.add_argument('--runners', nargs='+', type=int, default=None,
                       help='Runner numbers to plot (default: auto-select)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    
    args = parser.parse_args()
    
    # 查找file_base
    file_base = args.file_base
    if not file_base:
        patterns = glob.glob("*_fit_params_yeoh.csv")
        if patterns:
            match = re.match(r'(.+?)_fit_params_yeoh\.csv', patterns[0])
            if match:
                file_base = match.group(1)
        
        if not file_base:
            print("错误: 无法自动检测file_base，请手动指定")
            print("用法: python plot_multi_model_comparison.py file_base")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Multi-Model Fitting Visualization")
    print(f"{'='*70}")
    print(f"File base: {file_base}")
    print(f"Output format: {args.format}")
    
    # 读取所有模型的拟合参数
    print(f"\n读取拟合参数...")
    all_params = {}
    for model_key in MODELS.keys():
        params = read_fit_params(file_base, model_key)
        if params:
            all_params[model_key] = params
            print(f"  {MODELS[model_key]['name']:15s}: {len(params)} 样本")
    
    if not all_params:
        print("\n错误: 没有找到任何拟合参数文件")
        sys.exit(1)
    
    # 配置（用于读取原始数据）
    config = {
        'initial_height': 1.0,
        'contact_area': np.pi * (1.0 ** 2),
        'disp_col': 'disp_abs',
        'force_col': 'force'
    }
    
    # 生成各种对比图
    print(f"\n生成对比图...")
    
    plot_success_rate(all_params, file_base, args.format, args.dpi)
    plot_r2_comparison(all_params, file_base, args.format, args.dpi)
    plot_aic_comparison(all_params, file_base, args.format, args.dpi)
    plot_parameter_distributions(all_params, file_base, args.format, args.dpi)
    
    # 绘制拟合曲线
    runner_nums = args.runners
    if not runner_nums:
        runner_nums = auto_select_runners(all_params, n_select=3)
        print(f"\n自动选择的runner: {runner_nums}")
    else:
        print(f"\n用户指定的runner: {runner_nums}")
    
    if runner_nums:
        plot_fitting_curves(runner_nums, all_params, file_base, config,
                           args.format, args.dpi)
    
    print(f"\n{'='*70}")
    print(f"完成!")
    print(f"{'='*70}\n")
