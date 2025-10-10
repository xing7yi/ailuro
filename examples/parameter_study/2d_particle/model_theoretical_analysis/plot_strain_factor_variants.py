#!/usr/bin/env python3
"""
绘制不同形式的应变因子与Yeoh多项式的组合

比较不同的Component 1（应变因子）形式：
- Yeoh原始: 2(λ - 1/λ²)
- 简化线性: 2(λ - 1)
- 其他变体

保持Component 2（多项式项）不变

用法:
    python plot_strain_factor_variants.py [options]
    
选项:
    --lambda-range MIN MAX  拉伸比范围 (默认: 1.0 1.5)
    --format FORMAT         输出格式: png, pdf, both (默认: pdf)
    --dpi DPI              图像分辨率 (默认: 150)
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


def strain_factor_engineering(lmbd):
    """工程应变: (λ - 1)/1 = λ - 1"""
    return lmbd - 1.0


def strain_factor_logarithmic(lmbd):
    """对数应变: ln(λ)"""
    return np.log(lmbd**2)


def strain_factor_quadratic(lmbd):
    """二次形式: (λ² - 1)"""
    return lmbd**2 - 1.0

def strain_factor_cubic(lmbd):
    return lmbd - 1.0 / (lmbd**3)

def polynomial_term(lmbd, a, b, c):
    """多项式项（Component 2）: a + 2b(I₁-3) + 3c(I₁-3)²"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


def plot_strain_factor_comparison(lambda_min=1.0, lambda_max=1.5, 
                                  output_name='strain_factor_comparison',
                                  file_format='pdf', dpi=150):
    """
    绘制不同应变因子的比较
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算不同的应变因子
    strain_factors = {
        'Yeoh Original': {'func': strain_factor_yeoh, 'color': 'blue', 'ls': '-', 
                         'label': r'$(\lambda - \lambda^{-2})$ (Yeoh)'},
        'Linear': {'func': strain_factor_linear, 'color': 'red', 'ls': '-', 
                  'label': r'$(\lambda - 1)$ (Linear)'},
        # 'Logarithmic': {'func': strain_factor_logarithmic, 'color': 'green', 'ls': '--', 
        #                'label': r'$\ln(\lambda)$ (Logarithmic)'},
        'Quadratic': {'func': strain_factor_quadratic, 'color': 'purple', 'ls': '-.', 
                     'label': r'$(\lambda^2 - 1)$ (Quadratic)'},
        'Cubic': {'func': strain_factor_cubic, 'color': 'green', 'ls': '--', 
                     'label': r'$(\lambda - \lambda^{-3})$  (Cubic)'},           
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制每种应变因子
    for name, info in strain_factors.items():
        values = info['func'](lmbd)
        ax.plot(lmbd, values, linewidth=2.5, color=info['color'], 
               linestyle=info['ls'], label=info['label'])

    
    ax.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=14)
    ax.set_ylabel('Strain Factor Value', fontsize=14)
    ax.set_title('Comparison of Different Strain Factor Forms (Component 1)', 
                 fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([lambda_min, lambda_max])
    
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


def plot_model_variants_components(lambda_min=1.0, lambda_max=1.5,
                                   output_name='model_variants_components',
                                   file_format='pdf', dpi=150):
    """
    绘制不同应变因子形式与固定多项式项的组合
    仿照yeoh_model_components的布局
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 固定的多项式系数
    a, b, c = 1.0, -1.0, 1
    poly = polynomial_term(lmbd, a, b, c)
    
    # 定义要比较的应变因子
    strain_factors = [
        {'name': 'Yeoh', 'func': strain_factor_yeoh, 
         'label': r'$(\lambda - \lambda^{-2})$', 'color': 'blue'},
        {'name': 'Linear', 'func': strain_factor_linear, 
         'label': r'$(\lambda - 1)$', 'color': 'red'},
        {'name': 'Logarithmic', 'func': strain_factor_logarithmic, 
         'label': r'$\ln(\lambda)$', 'color': 'green'},
    ]
    
    # 创建图形（每个应变因子一组3个子图）
    n_variants = len(strain_factors)
    fig, axes = plt.subplots(n_variants, 3, figsize=(13, 3.8*n_variants))
    
    if n_variants == 1:
        axes = axes.reshape(1, -1)
    
    for row, sf_info in enumerate(strain_factors):
        strain_func = sf_info['func']
        strain_values = strain_func(lmbd)
        color = sf_info['color']
        sf_label = sf_info['label']
        sf_name = sf_info['name']
        
        # 计算完整应力
        stress = 2.0 * strain_values * poly
        
        # 归一化应力（按最大值）
        max_stress = np.max(np.abs(stress))
        normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
        
        # 第一列：应变因子
        ax1 = axes[row, 0]
        ax1.plot(lmbd, strain_values, color=color, linewidth=2.5, label=sf_label)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=12)
        ax1.set_ylabel('Strain Factor', fontsize=12)
        ax1.set_title(f'{sf_name} Model: Component 1\n{sf_label}', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([lambda_min, lambda_max])
        
        # 第二列：多项式项（所有行相同）
        ax2 = axes[row, 1]
        ax2.plot(lmbd, poly, 'purple', linewidth=2.5,
                label=rf'$a + 2b(I_1-3) + 3c(I_1-3)^2$' + f'\n(a={a}, b={b}, c={c})')
        ax2.axhline(y=a, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
                   label=f'Baseline: a={a}')
        ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=12)
        ax2.set_ylabel('Polynomial Term', fontsize=12)
        ax2.set_title(f'Component 2 (Same for all)\n' + 
                     r'$[a + 2b(I_1-3) + 3c(I_1-3)^2]$', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([lambda_min, lambda_max])
        
        # 第三列：归一化应力
        ax3 = axes[row, 2]
        ax3.plot(lmbd, normalized_stress, 'darkgreen', linewidth=2.5,
                label=f'{sf_name} Normalized Stress\n' + 
                      rf'$\sigma/\sigma_{{\max}}$ (max={max_stress:.2f})')
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=12)
        ax3.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=12)
        ax3.set_title(f'Complete {sf_name} Model (Normalized)\n' + 
                     r'$\sigma = 2 \times$ Component1 $\times$ Component2', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([lambda_min, lambda_max])
        ax3.set_ylim([0, 1.1])
    
    plt.suptitle('Comparison of Model Variants with Different Strain Factors', 
                fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
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


def plot_stress_comparison(lambda_min=1.0, lambda_max=1.5,
                           output_name='stress_comparison',
                           file_format='pdf', dpi=150):
    """
    比较不同应变因子形式产生的应力曲线
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 固定的多项式系数
    a, b, c = 1.0, -1.0, 1
    poly = polynomial_term(lmbd, a, b, c)
    
    # 定义要比较的应变因子
    strain_factors = [
        {'name': 'Yeoh', 'func': strain_factor_yeoh, 
         'label': r'$\sigma = 2(\lambda - \lambda^{-2}) \times [\cdots]$', 
         'color': 'blue', 'ls': '-'},
        {'name': 'Linear', 'func': strain_factor_linear, 
         'label': r'$\sigma = 2(\lambda - 1) \times [\cdots]$', 
         'color': 'red', 'ls': '-'},
        # {'name': 'Logarithmic', 'func': strain_factor_logarithmic, 
        #  'label': r'$\sigma = 2\ln(\lambda) \times [\cdots]$', 
        #  'color': 'green', 'ls': '--'},
        {'name': 'Quadratic', 'func': strain_factor_quadratic, 
         'label': r'$\sigma = 2(\lambda^2 - 1) \times [\cdots]$', 
         'color': 'purple', 'ls': '-.'},
        {'name': 'Cubic', 'func': strain_factor_cubic, 
         'label': r'$\sigma = 2(\lambda - \lambda^{-3}) \times [\cdots]$', 
         'color': 'green', 'ls': '--'}  
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 绘制归一化的应力曲线（每条曲线按各自最大值归一化）
    for sf_info in strain_factors:
        strain_values = sf_info['func'](lmbd)
        stress = 2.0 * strain_values * poly
        
        # 按各自最大值归一化
        max_stress = np.max(np.abs(stress))
        normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
        
        # 更新label显示原始最大值
        label_with_max = sf_info['label'] + f' (max={max_stress:.2f})'
        ax.plot(lmbd, normalized_stress, linewidth=1.5, color=sf_info['color'],
               linestyle=sf_info['ls'], label=label_with_max)
    
    
    ax.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=14)
    ax.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=14)
    ax.set_title(f'Normalized Stress Comparison\n' + 
                rf'$[\cdots] = a + 2b(I_1-3) + 3c(I_1-3)^2$ (a={a}, b={b}, c={c})', 
                fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([lambda_min, lambda_max])
    ax.set_ylim([0, 1.1])
    
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
        description='Plot strain factor variants with fixed polynomial term'
    )
    parser.add_argument('--lambda-range', nargs=2, type=float, default=[1.0, 1.5],
                       metavar=('MIN', 'MAX'),
                       help='Stretch ratio range (default: 1.0 1.5)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    
    args = parser.parse_args()
    
    lambda_min, lambda_max = args.lambda_range
    
    print(f"\n{'='*60}")
    print(f"Plotting Strain Factor Variants")
    print(f"{'='*60}")
    print(f"Lambda range: [{lambda_min}, {lambda_max}]")
    print(f"Output format: {args.format}")
    
    # 绘制应变因子比较
    print(f"\nGenerating strain factor comparison plot...")
    plot_strain_factor_comparison(lambda_min, lambda_max, 'strain_factor_comparison', 
                                  args.format, args.dpi)
    
    # 绘制模型变体组成部分（仿照yeoh_model_components）
    print(f"\nGenerating model variants components plot...")
    plot_model_variants_components(lambda_min, lambda_max, 'model_variants_components', 
                                   args.format, args.dpi)
    
    # 绘制应力比较
    print(f"\nGenerating stress comparison plot...")
    plot_stress_comparison(lambda_min, lambda_max, 'stress_comparison', 
                          args.format, args.dpi)
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}\n")
