#!/usr/bin/env python3
"""
绘制不同系数值在不同应变因子下的影响

结合系数变化和应变因子变体，创建矩阵式对比图：
- 行：不同的应变因子（Yeoh, Linear, Logarithmic, Quadratic）
- 列：变化的系数（a, b, c）
- 每条曲线使用各自的最大值归一化

用法:
    python plot_coefficient_strain_factor_matrix.py [options]
    
选项:
    --lambda-range MIN MAX  拉伸比范围 (默认: 1.0 1.5)
    --format FORMAT         输出格式: png, pdf, both (默认: pdf)
    --dpi DPI              图像分辨率 (默认: 150)
    --n-samples N          系数采样数量 (默认: 9)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compute_I1(lmbd):
    """计算第一不变量 I1"""
    lmbd = np.asarray(lmbd)
    I1 = np.power(lmbd, 2) + 2.0 / lmbd
    return I1


def strain_factor_yeoh(lmbd):
    """Yeoh原始应变因子: (λ - 1/λ²)"""
    return lmbd - 1.0 / (lmbd**2)


def strain_factor_linear(lmbd):
    """线性简化: (λ - 1)"""
    return lmbd - 1.0


def strain_factor_logarithmic(lmbd):
    """对数应变: ln(λ²)"""
    return np.log(lmbd**2)


def strain_factor_quadratic(lmbd):
    """二次形式: (λ² - 1)"""
    return lmbd**2 - 1.0


def polynomial_term(lmbd, a, b, c):
    """多项式项（Component 2）: a + 2b(I₁-3) + 3c(I₁-3)²"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


def compute_stress(lmbd, strain_factor_func, a, b, c):
    """计算完整应力"""
    strain_values = strain_factor_func(lmbd)
    poly = polynomial_term(lmbd, a, b, c)
    stress = 2.0 * strain_values * poly
    return stress


def plot_coefficient_strain_factor_matrix(lambda_min=1.0, lambda_max=1.5,
                                         output_name='coefficient_strain_factor_matrix',
                                         file_format='pdf', dpi=150, n_samples=9):
    """
    绘制系数-应变因子影响矩阵
    
    创建 4x3 子图网格：
    - 4行：4种应变因子（Yeoh, Linear, Logarithmic, Quadratic）
    - 3列：变化a, b, c系数（其他保持固定）
    - 每个子图显示多条归一化曲线（不同系数值）
    
    Parameters:
    -----------
    lambda_min : float
        最小拉伸比
    lambda_max : float
        最大拉伸比
    output_name : str
        输出文件名（不含扩展名）
    file_format : str
        输出格式: 'png', 'pdf', 'both'
    dpi : int
        PNG分辨率
    n_samples : int
        系数采样数量
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 定义应变因子
    strain_factors = [
        {'name': 'Yeoh', 'func': strain_factor_yeoh, 
         'label': r'$(\lambda - \lambda^{-2})$', 'color_base': 'blue'},
        {'name': 'Linear', 'func': strain_factor_linear, 
         'label': r'$(\lambda - 1)$', 'color_base': 'red'},
        {'name': 'Logarithmic', 'func': strain_factor_logarithmic, 
         'label': r'$\ln(\lambda^2)$', 'color_base': 'green'},
        {'name': 'Quadratic', 'func': strain_factor_quadratic, 
         'label': r'$(\lambda^2 - 1)$', 'color_base': 'purple'},
    ]
    
    # 定义系数范围
    a_range = np.linspace(0.5, 2.0, n_samples)
    b_range = np.linspace(-1.0, 1.0, n_samples)
    c_range = np.linspace(0.5, 1.5, n_samples)
    
    # 固定值
    a_fixed = 1.0
    b_fixed = -1.0
    c_fixed = 1.0
    
    # 创建colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    # 创建图形（4行3列）
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # 遍历每种应变因子（行）
    for row, sf_info in enumerate(strain_factors):
        strain_func = sf_info['func']
        sf_name = sf_info['name']
        sf_label = sf_info['label']
        
        # 列1：变化a，固定b和c
        ax = axes[row, 0]
        for i, a_val in enumerate(a_range):
            stress = compute_stress(lmbd, strain_func, a_val, b_fixed, c_fixed)
            max_stress = np.max(np.abs(stress))
            normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
            
            # 只显示部分标签避免拥挤
            if i % max(1, n_samples // 4) == 0:
                label = f'a={a_val:.2f}'
            else:
                label = None
            
            ax.plot(lmbd, normalized_stress, color=colors[i], linewidth=1.5, 
                   alpha=0.8, label=label)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'$\lambda$', fontsize=11)
        ax.set_ylabel(r'$\sigma/\sigma_{\max}$', fontsize=11)
        ax.set_title(f'{sf_name}: Varying a\n{sf_label} (b={b_fixed:.1f}, c={c_fixed:.1f})', 
                    fontsize=12)
        if row == 0:
            ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([lambda_min, lambda_max])
        ax.set_ylim([0, 1.1])
        
        # 列2：变化b，固定a和c
        ax = axes[row, 1]
        for i, b_val in enumerate(b_range):
            stress = compute_stress(lmbd, strain_func, a_fixed, b_val, c_fixed)
            max_stress = np.max(np.abs(stress))
            normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
            
            if i % max(1, n_samples // 4) == 0:
                label = f'b={b_val:.2f}'
            else:
                label = None
            
            ax.plot(lmbd, normalized_stress, color=colors[i], linewidth=1.5, 
                   alpha=0.8, label=label)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'$\lambda$', fontsize=11)
        ax.set_ylabel(r'$\sigma/\sigma_{\max}$', fontsize=11)
        ax.set_title(f'{sf_name}: Varying b\n{sf_label} (a={a_fixed:.1f}, c={c_fixed:.1f})', 
                    fontsize=12)
        if row == 0:
            ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([lambda_min, lambda_max])
        ax.set_ylim([0, 1.1])
        
        # 列3：变化c，固定a和b
        ax = axes[row, 2]
        for i, c_val in enumerate(c_range):
            stress = compute_stress(lmbd, strain_func, a_fixed, b_fixed, c_val)
            max_stress = np.max(np.abs(stress))
            normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
            
            if i % max(1, n_samples // 4) == 0:
                label = f'c={c_val:.2f}'
            else:
                label = None
            
            ax.plot(lmbd, normalized_stress, color=colors[i], linewidth=1.5, 
                   alpha=0.8, label=label)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'$\lambda$', fontsize=11)
        ax.set_ylabel(r'$\sigma/\sigma_{\max}$', fontsize=11)
        ax.set_title(f'{sf_name}: Varying c\n{sf_label} (a={a_fixed:.1f}, b={b_fixed:.1f})', 
                    fontsize=12)
        if row == 0:
            ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([lambda_min, lambda_max])
        ax.set_ylim([0, 1.1])
    
    plt.suptitle('Coefficient Variations Across Different Strain Factors (Normalized)', 
                fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.992])
    
    # 保存图像
    if file_format in ['png', 'both']:
        output_file = f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_single_coefficient_comparison(lambda_min=1.0, lambda_max=1.5,
                                      coefficient='a',
                                      output_name='single_coef_comparison',
                                      file_format='pdf', dpi=150, n_samples=9):
    """
    绘制单个系数变化在所有应变因子下的对比
    
    创建3个子图（对应a, b, c），每个子图包含所有4种应变因子的曲线族
    
    Parameters:
    -----------
    lambda_min : float
        最小拉伸比
    lambda_max : float
        最大拉伸比
    coefficient : str
        要变化的系数: 'a', 'b', 或 'c'
    output_name : str
        输出文件名（不含扩展名）
    file_format : str
        输出格式: 'png', 'pdf', 'both'
    dpi : int
        PNG分辨率
    n_samples : int
        系数采样数量
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 定义应变因子
    strain_factors = [
        {'name': 'Yeoh', 'func': strain_factor_yeoh, 
         'label': r'Yeoh: $(\lambda - \lambda^{-2})$', 'color': 'blue', 'ls': '-'},
        {'name': 'Linear', 'func': strain_factor_linear, 
         'label': r'Linear: $(\lambda - 1)$', 'color': 'red', 'ls': '-'},
        {'name': 'Logarithmic', 'func': strain_factor_logarithmic, 
         'label': r'Log: $\ln(\lambda^2)$', 'color': 'green', 'ls': '--'},
        {'name': 'Quadratic', 'func': strain_factor_quadratic, 
         'label': r'Quad: $(\lambda^2 - 1)$', 'color': 'purple', 'ls': '-.'},
    ]
    
    # 定义系数范围
    coef_ranges = {
        'a': np.linspace(0.5, 2.0, n_samples),
        'b': np.linspace(-1.0, 1.0, n_samples),
        'c': np.linspace(0.5, 1.5, n_samples)
    }
    
    # 固定值
    fixed_values = {'a': 1.0, 'b': -1.0, 'c': 1.0}
    
    # 创建图形（3个子图，每个对应一个系数）
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for col_idx, coef_name in enumerate(['a', 'b', 'c']):
        ax = axes[col_idx]
        coef_values = coef_ranges[coef_name]
        
        # 为每种应变因子绘制曲线族
        for sf_info in strain_factors:
            strain_func = sf_info['func']
            sf_name = sf_info['name']
            base_color = sf_info['color']
            ls = sf_info['ls']
            
            # 生成该应变因子的颜色渐变
            n_colors = len(coef_values)
            
            for i, coef_val in enumerate(coef_values):
                # 构建系数组合
                coeffs = fixed_values.copy()
                coeffs[coef_name] = coef_val
                
                stress = compute_stress(lmbd, strain_func, 
                                      coeffs['a'], coeffs['b'], coeffs['c'])
                max_stress = np.max(np.abs(stress))
                normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
                
                # 计算颜色深浅（从浅到深）
                alpha_val = 0.3 + 0.7 * (i / (n_colors - 1))
                
                # 只在中间值显示标签
                if i == n_colors // 2:
                    label = sf_info['label']
                else:
                    label = None
                
                ax.plot(lmbd, normalized_stress, color=base_color, 
                       linestyle=ls, linewidth=1.5, alpha=alpha_val, label=label)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
        ax.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=13)
        
        # 构建标题显示固定系数
        other_coefs = [c for c in ['a', 'b', 'c'] if c != coef_name]
        fixed_str = ', '.join([f'{c}={fixed_values[c]:.1f}' for c in other_coefs])
        coef_range_str = f'{coef_name}=[{coef_values[0]:.2f}, {coef_values[-1]:.2f}]'
        
        ax.set_title(f'Varying {coef_name.upper()}: {coef_range_str}\n' + 
                    f'Fixed: {fixed_str}', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([lambda_min, lambda_max])
        ax.set_ylim([0, 1.1])
    
    plt.suptitle('Coefficient Effects Across All Strain Factors (Light→Dark = Low→High coefficient)', 
                fontsize=15, y=1.00)
    plt.tight_layout()
    
    # 保存图像
    if file_format in ['png', 'both']:
        output_file = f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Plot coefficient effects across different strain factors'
    )
    parser.add_argument('--lambda-range', nargs=2, type=float, default=[1.0, 1.5],
                       metavar=('MIN', 'MAX'),
                       help='Stretch ratio range (default: 1.0 1.5)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    parser.add_argument('--n-samples', type=int, default=9,
                       help='Number of coefficient samples (default: 9)')
    
    args = parser.parse_args()
    
    lambda_min, lambda_max = args.lambda_range
    
    print(f"\n{'='*70}")
    print(f"Plotting Coefficient-Strain Factor Matrix")
    print(f"{'='*70}")
    print(f"Lambda range: [{lambda_min}, {lambda_max}]")
    print(f"Output format: {args.format}")
    print(f"Coefficient samples: {args.n_samples}")
    
    # 绘制完整矩阵（4x3网格）
    print(f"\nGenerating coefficient-strain factor matrix plot...")
    plot_coefficient_strain_factor_matrix(lambda_min, lambda_max, 
                                         'coefficient_strain_factor_matrix',
                                         args.format, args.dpi, args.n_samples)
    
    # 绘制单系数对比（3个子图，每个显示所有应变因子）
    print(f"\nGenerating single coefficient comparison plot...")
    plot_single_coefficient_comparison(lambda_min, lambda_max,
                                      output_name='single_coef_all_strain_factors',
                                      file_format=args.format, dpi=args.dpi,
                                      n_samples=args.n_samples)
    
    print(f"\n{'='*70}")
    print(f"Complete!")
    print(f"{'='*70}\n")
