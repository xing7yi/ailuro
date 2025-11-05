#!/usr/bin/env python3
"""
绘制第一不变量 I1 与拉伸比 lambda 的关系

I1 = lambda^2 + 2/lambda

用法:
    python plot_I1_invariant.py [options]
    
选项:
    --lambda-range MIN MAX  拉伸比范围 (默认: 1.0 2.0)
    --format FORMAT         输出格式: png, pdf, both (默认: pdf)
    --dpi DPI              图像分辨率 (默认: 150)
      parser.add_argument('--lambda-range', type=float, nargs=2, default=[1.0, 2.0],
                       metavar=('MIN', 'MAX'),
                       help='Lambda range (default: 1.0 2.0)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()ILE          输出文件名 (默认: I1_vs_lambda)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compute_I1(lmbd):
    """
    计算第一不变量 I1
    
    Parameters:
    -----------
    lmbd : array_like
        拉伸比 lambda
        
    Returns:
    --------
    I1 : ndarray
        第一不变量
    """
    lmbd = np.asarray(lmbd)
    I1 = np.power(lmbd, 2) + 2.0 / lmbd
    return I1


def compute_yeoh_stress(lmbd, a, b, c):
    """
    计算Yeoh模型的应力
    
    stress = 2 * (lambda - 1/lambda^2) * [a + 2*b*(I1-3) + 3*c*(I1-3)^2]
    
    Parameters:
    -----------
    lmbd : array_like
        拉伸比 lambda
    a, b, c : float
        Yeoh模型参数
        
    Returns:
    --------
    stress : ndarray
        应力
    """
    lmbd = np.asarray(lmbd)
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    
    # 应变不变量因子
    strain_factor = lmbd - 1.0 / (lmbd**2)
    
    # Yeoh多项式
    yeoh_poly = a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)
    
    # 完整应力
    stress = 2.0 * strain_factor * yeoh_poly
    
    return stress


def plot_I1_vs_lambda(lambda_min=1.0, lambda_max=2.0, output_name='I1_vs_lambda', 
                      file_format='pdf', dpi=150):
    """
    绘制 I1 vs lambda 关系图
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算I1
    I1 = compute_I1(lmbd)
    
    # 找到最小值点
    min_idx = np.argmin(I1)
    lambda_min_I1 = lmbd[min_idx]
    I1_min = I1[min_idx]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制主曲线
    ax.plot(lmbd, I1, 'b-', linewidth=2.5, label=r'$I_1 = \lambda^2 + \frac{2}{\lambda}$')
    
    # 标记最小值点
    ax.plot(lambda_min_I1, I1_min, 'ro', markersize=10, 
            label=f'Minimum at λ={lambda_min_I1:.3f}, $I_1$={I1_min:.3f}')
    
    # 添加垂直线标记关键点
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='λ=1 (undeformed)')
    ax.axhline(y=3.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='$I_1$=3 (reference)')
    
    # 设置标签和标题
    ax.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=14)
    ax.set_ylabel(r'First Invariant $I_1$', fontsize=14)
    ax.set_title(r'First Invariant $I_1$ vs Stretch Ratio $\lambda$', fontsize=16, pad=15)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # 设置图例
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # 设置坐标轴范围
    ax.set_xlim([lambda_min, lambda_max])
    
    # # 添加注释
    # ax.text(0.98, 0.02, 
    #         f'Note: Minimum $I_1$ occurs at λ = $2^{{1/3}}$ ≈ {2**(1/3):.4f}',
    #         transform=ax.transAxes,
    #         fontsize=10,
    #         verticalalignment='bottom',
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_I1_components(lambda_min=1, lambda_max=2.0, output_name='I1_components', 
                       file_format='pdf', dpi=150):
    """
    绘制 I1 各组成部分的贡献
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算各组分
    term1 = lmbd**2
    term2 = 2.0 / lmbd
    I1 = term1 + term2
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 上图：各组分
    ax1.plot(lmbd, term1, 'r-', linewidth=2, label=r'$\lambda^2$')
    ax1.plot(lmbd, term2, 'b-', linewidth=2, label=r'$\frac{2}{\lambda}$')
    ax1.plot(lmbd, I1, 'k-', linewidth=2.5, label=r'$I_1 = \lambda^2 + \frac{2}{\lambda}$')
    
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax1.set_ylabel('Value', fontsize=13)
    ax1.set_title(r'Components of First Invariant $I_1$', fontsize=14)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([lambda_min, lambda_max])
    
    # 下图：相对贡献（百分比）
    contribution1 = (term1 / I1) * 100
    contribution2 = (term2 / I1) * 100
    
    ax2.fill_between(lmbd, 0, contribution1, alpha=0.5, color='red', 
                     label=r'$\lambda^2$ contribution')
    ax2.fill_between(lmbd, contribution1, 100, alpha=0.5, color='blue', 
                     label=r'$\frac{2}{\lambda}$ contribution')
    
    ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax2.set_ylabel('Contribution (%)', fontsize=13)
    ax2.set_title(r'Relative Contributions to $I_1$', fontsize=14)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([lambda_min, lambda_max])
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_I1_minus_3_quadratic(lambda_min=1, lambda_max=2.0, output_name='I1_minus_3_quadratic',
                              file_format='pdf', dpi=150):
    """
    绘制 (I1-3) 作为自变量的二次函数关系
    显示常数项、一次项、二次项：1, (I1-3), (I1-3)^2 vs lambda
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算I1和相关量
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    I1_minus_3_squared = (I1 - 3.0)**2
    
    # 创建图形（3个子图）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # 上图：所有项叠加在一起 vs lambda（常数项=1, 一次项, 二次项）
    constant_term = np.ones_like(lmbd)
    ax1.plot(lmbd, constant_term, 'g-', linewidth=2.5, label=r'Constant: $1$')
    ax1.plot(lmbd, I1_minus_3, 'b-', linewidth=2.5, label=r'Linear: $(I_1 - 3)$')
    ax1.plot(lmbd, I1_minus_3_squared, 'r-', linewidth=2.5, label=r'Quadratic: $(I_1 - 3)^2$')
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax1.set_ylabel('Function Value', fontsize=13)
    ax1.set_title(r'Quadratic Polynomial Terms: $1$, $(I_1 - 3)$, $(I_1 - 3)^2$ vs $\lambda$', fontsize=14)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([lambda_min, lambda_max])
    
    # 中图：示例多项式组合 f = a + b(I1-3) + c(I1-3)^2
    # 使用几组典型系数
    a_val, b_val, c_val = 1.0, 2.0, 0.5
    poly_example = a_val * constant_term + b_val * I1_minus_3 + c_val * I1_minus_3_squared
    
    ax2.plot(lmbd, constant_term * a_val, 'g--', linewidth=1.5, alpha=0.7, 
             label=rf'$a \cdot 1$ (a={a_val})')
    ax2.plot(lmbd, I1_minus_3 * b_val, 'b--', linewidth=1.5, alpha=0.7, 
             label=rf'$b(I_1-3)$ (b={b_val})')
    ax2.plot(lmbd, I1_minus_3_squared * c_val, 'r--', linewidth=1.5, alpha=0.7, 
             label=rf'$c(I_1-3)^2$ (c={c_val})')
    ax2.plot(lmbd, poly_example, 'k-', linewidth=3, 
             label=rf'Total: $f = {a_val} + {b_val}(I_1-3) + {c_val}(I_1-3)^2$')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax2.set_ylabel('Function Value', fontsize=13)
    ax2.set_title(r'Example Quadratic Polynomial: $f = a + b(I_1-3) + c(I_1-3)^2$', fontsize=14)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([lambda_min, lambda_max])
    
    # 下图：(I1-3) 和 (I1-3)^2 双Y轴图
    color1 = 'tab:blue'
    ax3.plot(lmbd, I1_minus_3, color=color1, linewidth=2.5, label=r'$(I_1 - 3)$')
    ax3.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax3.set_ylabel(r'$(I_1 - 3)$', fontsize=13, color=color1)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 第二个Y轴 - (I1-3)^2
    ax3_twin = ax3.twinx()
    color2 = 'tab:red'
    ax3_twin.plot(lmbd, I1_minus_3_squared, color=color2, linewidth=2.5, 
                  linestyle='--', label=r'$(I_1 - 3)^2$')
    ax3_twin.set_ylabel(r'$(I_1 - 3)^2$', fontsize=13, color=color2)
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    
    ax3.set_title(r'Comparison: $(I_1 - 3)$ and $(I_1 - 3)^2$ on Separate Scales', fontsize=14)
    
    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    ax3.set_xlim([lambda_min, lambda_max])
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_polynomial_vs_lambda(lambda_min=1, lambda_max=2.0, output_name='polynomial_vs_lambda',
                              file_format='pdf', dpi=150):
    """
    绘制完整二次多项式 f(I1-3) = a + b(I1-3) + c(I1-3)^2 与 lambda 的关系
    使用多组系数展示不同的曲线形状
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算I1
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    
    # 定义多组系数来展示不同的多项式行为（只保留完整的二次多项式）
    coefficients = [
        {'a': 1.0, 'b': 1.0, 'c': 0.5, 'label': r'$f = 1 + (I_1-3) + 0.5(I_1-3)^2$', 'color': 'green', 'ls': '-'},
        {'a': 2.0, 'b': 1.5, 'c': 0.3, 'label': r'$f = 2 + 1.5(I_1-3) + 0.3(I_1-3)^2$', 'color': 'purple', 'ls': '-'},
        {'a': 0.5, 'b': 2.0, 'c': 0.8, 'label': r'$f = 0.5 + 2(I_1-3) + 0.8(I_1-3)^2$', 'color': 'orange', 'ls': '-'},
        {'a': 1.5, 'b': -0.5, 'c': 1.0, 'label': r'$f = 1.5 - 0.5(I_1-3) + (I_1-3)^2$', 'color': 'cyan', 'ls': '--'},
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制每组系数对应的曲线
    for coef in coefficients:
        a, b, c = coef['a'], coef['b'], coef['c']
        poly = a + b * I1_minus_3 + c * (I1_minus_3**2)
        ax.plot(lmbd, poly, linewidth=2.5, label=coef['label'], 
                color=coef['color'], linestyle=coef['ls'])
    
    # 添加参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=r'$\lambda=1$ (undeformed)')
    
    ax.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=14)
    ax.set_ylabel(r'Polynomial Value $f(I_1-3)$', fontsize=14)
    ax.set_title(r'Quadratic Polynomial $f(I_1-3) = a + b(I_1-3) + c(I_1-3)^2$ vs $\lambda$', 
                 fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([lambda_min, lambda_max])
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_polynomial_vs_I1_minus_3(lambda_min=1, lambda_max=2.0, 
                                  output_name='polynomial_vs_I1_minus_3',
                                  file_format='pdf', dpi=150):
    """
    绘制完整二次多项式 f(I1-3) = a + b(I1-3) + c(I1-3)^2 与 (I1-3) 的关系
    这是一个标准的二次函数图像
    
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
    """
    
    # 生成lambda值并计算对应的I1-3
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    
    # 定义多组系数来展示不同的多项式曲线
    coefficients = [
        {'a': 1.0, 'b': -1.0, 'c': 1.0, 'label': r'$f = 1$', 'color': 'gray', 'ls': '--'},
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制每组系数对应的曲线
    for coef in coefficients:
        a, b, c = coef['a'], coef['b'], coef['c']
        poly = a + 2*b * I1_minus_3 + 3*c * (I1_minus_3**2)
        ax.plot(I1_minus_3, poly, linewidth=2.5, label=coef['label'], 
                color=coef['color'], linestyle=coef['ls'])
    
    # 添加参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
              label=r'$I_1=3$ (undeformed)')
    
    ax.set_xlabel(r'$(I_1 - 3)$', fontsize=14)
    ax.set_ylabel(r'Polynomial Value $f(I_1-3)$', fontsize=14)
    ax.set_title(r'Quadratic Polynomial: $f(I_1-3) = a + 2b(I_1-3) + 3c(I_1-3)^2$', 
                 fontsize=15, pad=15)
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 标注lambda值在轴上（次要刻度）
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # 在特定的I1-3值处标注对应的lambda值
    lambda_markers = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    I1_markers = [compute_I1(l) - 3 for l in lambda_markers]
    ax2.set_xticks(I1_markers)
    ax2.set_xticklabels([f'λ={l:.1f}' for l in lambda_markers], fontsize=9)
    ax2.set_xlabel(r'Corresponding Stretch Ratio $\lambda$', fontsize=12, color='gray')
    ax2.tick_params(axis='x', colors='gray')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_yeoh_model_components(lambda_min=1, lambda_max=2.0, output_name='yeoh_model_components',
                               file_format='pdf', dpi=150):
    """
    绘制完整Yeoh模型的各个组成部分
    
    stress = 2 * (lambda - 1/lambda^2) * [a + 2*b*(I1-3) + 3*c*(I1-3)^2]
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算各个组分
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    
    # 应变不变量因子
    strain_factor = lmbd - 1.0 / (lmbd**2)
    
    # 多项式项（使用示例系数 a=1, b=1, c=0.5）
    a, b, c = 1, -1, 1
    poly_term = a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)
    
    # 完整应力
    stress = 2.0 * strain_factor * poly_term
    
    # 创建图形（3个子图）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 第一个子图：应变因子 (lambda - 1/lambda^2)
    ax1.plot(lmbd, strain_factor, 'b-', linewidth=2.5, 
            label=r'$(\lambda - \lambda^{-2})$')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax1.set_ylabel(r'Strain Factor $(\lambda - \lambda^{-2})$', fontsize=13)
    ax1.set_title(r'Yeoh Model Component 1: Strain Factor', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([lambda_min, lambda_max])
    
    # 第二个子图：多项式项
    ax2.plot(lmbd, poly_term, 'r-', linewidth=2.5,
            label=rf'$a + 2b(I_1-3) + 3c(I_1-3)^2$' + f'\n(a={a}, b={b}, c={c})')
    ax2.axhline(y=a, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
               label=f'Baseline: a={a}')
    ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax2.set_ylabel(r'Polynomial Term', fontsize=13)
    ax2.set_title(r'Yeoh Model Component 2: Polynomial $[a + 2b(I_1-3) + 3c(I_1-3)^2]$', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([lambda_min, lambda_max])
    
    # 第三个子图：完整应力
    ax3.plot(lmbd, stress, 'g-', linewidth=3,
            label=rf'Complete Yeoh Stress' + f'\n(a={a}, b={b}, c={c})')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax3.set_ylabel(r'Stress $\sigma$', fontsize=13)
    ax3.set_title(r'Complete Yeoh Model: $\sigma = 2(\lambda - \lambda^{-2})[a + 2b(I_1-3) + 3c(I_1-3)^2]$', 
                 fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([lambda_min, lambda_max])
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_yeoh_model_various_coefficients(lambda_min=1, lambda_max=2.0, 
                                         output_name='yeoh_model_various_coef',
                                         file_format='pdf', dpi=150,
                                         n_samples=11):
    """
    绘制不同系数组合的完整Yeoh模型应力曲线（归一化）
    创建3个子图，分别展示改变a、b、c系数时的影响（其他系数保持固定）
    每条曲线使用各自的最大值归一化
    
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
        在参数空间均匀采样的数量
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 定义参数范围
    a_range = np.linspace(0.5, 2.0, n_samples)    
    b_range = np.linspace(-1.0, 1.0, n_samples)
    c_range = np.linspace(0.5, 1.5, n_samples)
    
    # 固定值（当其他参数变化时使用）
    a_fixed = 1.5
    b_fixed = -1
    c_fixed = 1
    
    # 创建colormap用于不同曲线
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    # 创建图形（3个子图）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 14))
    
    # 子图1：改变a，固定b和c
    for i in range(n_samples):
        a = a_range[i]
        b = b_fixed
        c = c_fixed
        
        # 计算应力
        stress = compute_yeoh_stress(lmbd, a, b, c)
        
        # 按各自最大值归一化
        max_stress = np.max(np.abs(stress))
        normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
        
        # 绘制归一化曲线
        label = f'a={a:.2f}'
        ax1.plot(lmbd, normalized_stress, linewidth=2, label=label, 
                color=colors[i], alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax1.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=13)
    ax1.set_title(f'Varying coefficient a (b={b_fixed:.2f}, c={c_fixed:.2f} fixed)', 
                 fontsize=14, pad=10)
    ax1.legend(fontsize=9, loc='best', ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([lambda_min, lambda_max])
    ax1.set_ylim([0, 1.1])
    
    # 子图2：改变b，固定a和c
    for i in range(n_samples):
        a = a_fixed
        b = b_range[i]
        c = c_fixed
        
        # 计算应力
        stress = compute_yeoh_stress(lmbd, a, b, c)
        
        # 按各自最大值归一化
        max_stress = np.max(np.abs(stress))
        normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
        
        # 绘制归一化曲线
        label = f'b={b:.2f}'
        ax2.plot(lmbd, normalized_stress, linewidth=2, label=label, 
                color=colors[i], alpha=0.8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax2.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=13)
    ax2.set_title(f'Varying coefficient b (a={a_fixed:.2f}, c={c_fixed:.2f} fixed)', 
                 fontsize=14, pad=10)
    ax2.legend(fontsize=9, loc='best', ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([lambda_min, lambda_max])
    ax2.set_ylim([0, 1.1])
    
    # 子图3：改变c，固定a和b
    for i in range(n_samples):
        a = a_fixed
        b = b_fixed
        c = c_range[i]
        
        # 计算应力
        stress = compute_yeoh_stress(lmbd, a, b, c)
        
        # 按各自最大值归一化
        max_stress = np.max(np.abs(stress))
        normalized_stress = stress / max_stress if max_stress > 1e-10 else stress
        
        # 绘制归一化曲线
        label = f'c={c:.2f}'
        ax3.plot(lmbd, normalized_stress, linewidth=2, label=label, 
                color=colors[i], alpha=0.8)
    
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=13)
    ax3.set_ylabel(r'Normalized Stress $\sigma/\sigma_{\max}$', fontsize=13)
    ax3.set_title(f'Varying coefficient c (a={a_fixed:.2f}, b={b_fixed:.2f} fixed)', 
                 fontsize=14, pad=10)
    ax3.legend(fontsize=9, loc='best', ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([lambda_min, lambda_max])
    ax3.set_ylim([0, 1.1])
    
    # 总标题
    fig.suptitle(r'Normalized Yeoh Model: $\sigma = 2(\lambda - \lambda^{-2})[a + 2b(I_1-3) + 3c(I_1-3)^2]$' + 
                 f'\nIndividual coefficient variations ({n_samples} samples each)', 
                 fontsize=15, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


def plot_strain_vs_lambda(lambda_min=1, lambda_max=2.0, output_name='strain_vs_lambda',
                          file_format='pdf', dpi=150):
    """
    绘制应变 epsilon 与拉伸比 lambda 的关系
    epsilon = lambda - 1
    
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
    """
    
    # 生成lambda值
    lmbd = np.linspace(lambda_min, lambda_max, 1000)
    
    # 计算应变
    epsilon = lmbd - 1
    
    # 计算I1
    I1 = compute_I1(lmbd)
    
    # 创建图形（双Y轴）
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    color1 = 'tab:blue'
    ax1.set_xlabel(r'Stretch Ratio $\lambda$', fontsize=14)
    ax1.set_ylabel(r'Strain $\epsilon = \lambda - 1$', fontsize=14, color=color1)
    ax1.plot(lmbd, epsilon, color=color1, linewidth=2.5, label='Strain')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 第二个Y轴 - I1
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(r'First Invariant $I_1$', fontsize=14, color=color2)
    ax2.plot(lmbd, I1, color=color2, linewidth=2.5, linestyle='--', label='$I_1$')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 设置标题
    ax1.set_title(r'Strain and First Invariant vs Stretch Ratio', fontsize=16, pad=15)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    ax1.set_xlim([lambda_min, lambda_max])
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path('.')
    
    if file_format in ['png', 'both']:
        output_file = output_dir / f'{output_name}.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    if file_format in ['pdf', 'both']:
        output_file = output_dir / f'{output_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close(fig)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Plot first invariant I1 vs stretch ratio lambda'
    )
    parser.add_argument('--lambda-range', nargs=2, type=float, default=[1, 1.5],
                       metavar=('MIN', 'MAX'),
                       help='Stretch ratio range (default: 0.5 3.0)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['png', 'pdf', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Image resolution for PNG (default: 150)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    lambda_min, lambda_max = args.lambda_range
    
    # 切换到输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    import os
    os.chdir(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Plotting First Invariant I1")
    print(f"{'='*60}")
    print(f"Lambda range: [{lambda_min}, {lambda_max}]")
    print(f"Output format: {args.format}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # 绘制主图
    print(f"\nGenerating I1 vs lambda plot...")
    plot_I1_vs_lambda(lambda_min, lambda_max, 'I1_vs_lambda', args.format, args.dpi)
    
    # 绘制组分图
    print(f"\nGenerating I1 components plot...")
    plot_I1_components(lambda_min, lambda_max, 'I1_components', args.format, args.dpi)
    
    # 绘制应变关系图
    # print(f"\nGenerating strain vs lambda plot...")
    # plot_strain_vs_lambda(lambda_min, lambda_max, 'strain_vs_lambda', args.format, args.dpi)
    
    # 绘制(I1-3)的幂次关系图
    print(f"\nGenerating (I1-3) quadratic terms plot...")
    plot_I1_minus_3_quadratic(lambda_min, lambda_max, 'I1_minus_3_quadratic', args.format, args.dpi)
    
    # 绘制完整多项式与lambda的关系
    print(f"\nGenerating polynomial vs lambda plot...")
    plot_polynomial_vs_lambda(lambda_min, lambda_max, 'polynomial_vs_lambda', args.format, args.dpi)
    
    # 绘制完整多项式与(I1-3)的关系
    print(f"\nGenerating polynomial vs (I1-3) plot...")
    plot_polynomial_vs_I1_minus_3(lambda_min, lambda_max, 'polynomial_vs_I1_minus_3', args.format, args.dpi)
    
    # 绘制Yeoh模型各组成部分
    print(f"\nGenerating Yeoh model components plot...")
    plot_yeoh_model_components(lambda_min, lambda_max, 'yeoh_model_components', args.format, args.dpi)
    
    # 绘制不同系数的完整Yeoh模型
    print(f"\nGenerating Yeoh model with various coefficients plot...")
    plot_yeoh_model_various_coefficients(lambda_min, lambda_max, 'yeoh_model_various_coef', args.format, args.dpi)
    
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}\n")
    
    # 打印一些关键值
    print("Key values:")
    print(f"  At λ = 1.0 (undeformed): I1 = {compute_I1(1.0):.4f}, (I1-3) = {compute_I1(1.0)-3:.4f}")
    print(f"  At λ = 2.0: I1 = {compute_I1(2.0):.4f}, (I1-3) = {compute_I1(2.0)-3:.4f}")
    if lambda_max >= 3.0:
        print(f"  At λ = 3.0: I1 = {compute_I1(3.0):.4f}, (I1-3) = {compute_I1(3.0)-3:.4f}")
