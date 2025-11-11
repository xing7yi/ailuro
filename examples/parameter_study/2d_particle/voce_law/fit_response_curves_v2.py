#!/usr/bin/env python3
"""
改进的响应曲线拟合脚本 - 支持多种拟合模型选择

支持的模型:
1. yeoh:        σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
2. linear:      σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]
3. logarithmic: σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
4. quadratic:   σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]
5. yeoh4:       σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³]

用法:
    python fit_response_curves_v2.py
    
配置:
    在 config 字典中设置 'fitting_model' 参数来选择模型
"""

import numpy as np
import glob
import re
import csv
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt
import argparse


# ============================================================================
# 模型定义
# ============================================================================

def compute_I1(lmbd):
    """计算第一不变量 I₁ = λ² + 2/λ"""
    return np.power(lmbd, 2) + 2.0 / lmbd


def strain_factor_yeoh(lmbd):
    """Yeoh应变因子: (λ - λ⁻²)"""
    return lmbd - np.power(lmbd, -2)

def strain_factor_user_defined(lmbd, k):
    x = lmbd - 1
    return x*np.exp(k*(1-x)**2)

def polynomial_term_3rd(lmbd, a, b, c):
    """三阶多项式项: a + 2b(I₁-3) + 3c(I₁-3)²"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


def polynomial_term_4th(lmbd, a, b, c, d):
    """四阶多项式项: a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2) + 4*d*(I1_minus_3**3)

# 3阶模型定义
def model_yeoh_3rd(epsilon, a, b, c):
    """Yeoh 3阶模型"""
    lmbd = epsilon + 1.0
    strain = strain_factor_yeoh(lmbd)
    poly = polynomial_term_3rd(lmbd, a, b, c)
    return 2.0 * strain * poly

# 4阶模型定义
def model_yeoh_4th(epsilon, a, b, c, d):
    """Yeoh 4阶模型"""
    lmbd = epsilon + 1.0
    strain = strain_factor_yeoh(lmbd)
    poly = polynomial_term_4th(lmbd, a, b, c, d)
    return 2.0 * strain * poly

def model_user_defined(epsilon, a, b, c, k):
    """用户自定义模型示例"""
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1
    I1 = compute_I1(lmbd)
    strain_factor = strain_factor_user_defined(lmbd, k)
    stress = 2 * strain_factor * polynomial_term_3rd(lmbd, a, b, c)
    return stress

def model_user_defined2(epsilon, a, b, c):
    """用户自定义模型示例"""
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1
    I1 = np.power(lmbd, 2) + np.power(lmbd, -1)
    strain_factor = lmbd - np.power(lmbd, -3)
    polynomial_term = a + 2*b*(I1-3) + 3*c*(I1-3)**2
    stress = 2 * strain_factor * polynomial_term
    return stress

def model_swift(epsilon, k, eps_0, n):
    """Swift模型"""
    return k * (epsilon + eps_0)**n

def model_power(epsilon, A, B, C):
    """Power Law模型"""
    return A * np.power(epsilon, B) + C * epsilon

def model_power2(epsilon, A, B, C):
    """Power Law模型2"""
    return A * (1 - np.power(epsilon, -B)) + C * epsilon**2

def model_bilinear(eps, E, sigma0, H):
    eps_y = sigma0 / E  # 屈服点
    sigma = np.zeros_like(eps)
    elastic = eps <= eps_y
    plastic = eps > eps_y
    
    # 弹性阶段
    sigma[elastic] = E * eps[elastic]
    
    # 塑性阶段（线性硬化）
    eps_p = eps[plastic] - eps_y
    sigma[plastic] = sigma0 + H * eps_p
    
    return sigma

def model_voce(eps, E, sigma0, R0 , R_inf):
    eps_y = sigma0 / E  # 屈服点
    sigma = np.zeros_like(eps)
    elastic = eps <= eps_y
    plastic = eps > eps_y
    
    # 弹性阶段
    sigma[elastic] = E * eps[elastic]
    
    # 塑性阶段（平滑连接）
    eps_p = eps[plastic] - eps_y
    b = (E - R0)/R_inf
    sigma[plastic] = sigma0 + R0 * eps_p + R_inf * (1 - np.exp(-b * eps_p))
    
    return sigma

def model_reduced_voce(eps, E, sigma0, R0):
    eps_y = sigma0 / E  # 屈服点
    sigma = np.zeros_like(eps)
    elastic = eps <= eps_y
    plastic = eps > eps_y
    
    # 弹性阶段
    sigma[elastic] = E * eps[elastic]
    
    # 塑性阶段（线性硬化）
    eps_p = eps[plastic] - eps_y
    sigma[plastic] = sigma0 + R0 * eps_p
    
    return sigma

# 模型注册表
FITTING_MODELS = {
    'yeoh': {
        'name': 'Yeoh (3rd order)',
        'function': model_yeoh_3rd,
        'n_params': 3,
        'param_names': ['a', 'b', 'c'],
        'description': 'σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]'
    },
    'yeoh4': {
        'name': 'Yeoh (4th order)',
        'function': model_yeoh_4th,
        'n_params': 4,
        'param_names': ['a', 'b', 'c', 'd'],
        'description': 'σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³]'
    },
    'user_defined': {
        'name': 'User Defined Model',
        'function': model_user_defined,
        'n_params': 4,
        'param_names': ['a', 'b', 'c', 'k'],
        'description': 'σ = 2(λ - 1)e^{k(1-λ)²} × [a + 2b(I₁-3) + 3c(I₁-3)²]'
    },
    'user_defined2': {
        'name': 'User Defined Model 2',
        'function': model_user_defined2,
        'n_params': 3,
        'param_names': ['a', 'b', 'c'],
        'description': 'σ = 2(λ - λ⁻³) × [a + 2b(I₁-3) + 3c(I₁-3)²]'
    },
    'swift': {
        'name': 'Swift Model',
        'function': model_swift,
        'n_params': 3,
        'param_names': ['k', 'eps_0', 'n'],
        'description': 'σ = k(ε + ε₀)ⁿ'
    },
    'power': {
        'name': 'Power Law Model',
        'function': model_power,
        'n_params': 3,
        'param_names': ['A', 'B', 'C'],
        'description': 'σ = Aεᴮ + Cε'
    },
    'power2': {
        'name': 'Power Law Model 2',
        'function': model_power2,
        'n_params': 3,
        'param_names': ['A', 'B', 'C'],
        'description': 'σ = A(1 - ε⁻ᴮ) + Cε²'
    },
    'voce': {
        'name': 'Voce Model',
        'function': model_voce,
        'n_params': 4,
        'param_names': ['E', 'sigma0', 'R0', 'R_inf'],
        'description': 'Voce model with elastic-plastic transition'
    },
    'reduced_voce': {
        'name': 'Reduced Voce Model',
        'function': model_reduced_voce,
        'n_params': 3,
        'param_names': ['E', 'sigma0', 'R0'],
        'description': 'Reduced Voce model with linear hardening'
    },
    'bilinear': {
        'name': 'Bilinear Model',
        'function': model_bilinear,
        'n_params': 3,
        'param_names': ['E', 'sigma0', 'H'],
        'bounds': [[0, 0, 0], [np.inf, np.inf, np.inf]],
        'description': 'Bilinear elastic-plastic model with linear hardening'
    }

}


# ============================================================================
# 工具函数
# ============================================================================

def extract_runner_number(filename):
    """从文件名提取runner编号"""
    match = re.search(r'runner(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def find_runner_files(file_base=None):
    """查找所有runner CSV文件"""
    
    if file_base:
        pattern = f"{file_base}_out_runner*.csv"
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


# ============================================================================
# 拟合函数
# ============================================================================

def fit_time_series(csv_file, config):
    """
    拟合单个时间序列CSV文件
    
    参数:
        csv_file: CSV文件路径
        config: 配置字典，必须包含 'fitting_model' 键
    
    返回:
        dict: 包含拟合参数和拟合质量指标
    """
    # 获取拟合模型
    model_key = config.get('fitting_model', 'yeoh')
    
    if model_key not in FITTING_MODELS:
        raise ValueError(f"Unknown fitting model: {model_key}. "
                        f"Available models: {list(FITTING_MODELS.keys())}")
    
    model_info = FITTING_MODELS[model_key]
    fitting_func = model_info['function']
    param_names = model_info['param_names']
    n_params = model_info['n_params']
    bounds = model_info['bounds'] if 'bounds' in model_info else None
    
    try:
        # 读取CSV数据
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
                    
                    # 过滤掉初始点和无效点
                    if abs(disp) > 1e-6 and force > 0:
                        time_data.append(time)
                        displacement_data.append(abs(disp))  # 取绝对值
                        force_data.append(force)
                except (ValueError, KeyError):
                    continue
        
        # 转换为numpy数组
        displacement = np.array(displacement_data)
        force = np.array(force_data)
        
        # 检查最大位移是否达到预期值
        max_disp = np.max(displacement)
        min_required_disp = config.get('min_displacement', 0.4)
        
        if max_disp < min_required_disp:
            result = {
                'r_squared': np.nan, 
                'rmse': np.nan,
                'rel_rmse': np.nan,
                'n_points': len(displacement_data),
                'max_displacement': max_disp,
                'max_force': np.max(force),
                'fit_success': False,
                'message': f'Insufficient displacement: max={max_disp:.4f}mm < required={min_required_disp}mm',
                'model': model_key
            }
            # 添加参数（全为NaN）
            for pname in param_names:
                result[pname] = np.nan
            return result
        
        # 计算应变和应力
        epsilon = displacement / config['initial_height']
        stress = force / config['contact_area']

        if config['response_curve'] == 'raw':
            _x = displacement
            _y = force
        elif config['response_curve'] == 'nominal':
            _x = epsilon
            _y = stress
        elif config['response_curve'] == 'true':
            _x = -np.log(1 - epsilon)
            _y = (1 - epsilon) * stress

        
        # 执行拟合
        try:
            # 根据bounds是否存在来调用curve_fit
            if bounds is not None:
                popt, pcov = curve_fit(
                    fitting_func,
                    _x,
                    _y,
                    bounds=bounds,
                    maxfev=20000
                )
            else:
                popt, pcov = curve_fit(
                    fitting_func,
                    _x,
                    _y,
                    maxfev=20000
                )
            
            # 计算拟合质量
            y_pred = fitting_func(_x, *popt)
            residuals = _y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((_y - np.mean(_y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean(residuals**2))
            rel_rmse = rmse / np.mean(_y) if np.mean(_y) > 0 else np.inf

            result = {
                'r_squared': r_squared,
                'rmse': rmse,
                'rel_rmse': rel_rmse,
                'n_points': len(_x),
                'max_force': np.max(force),
                'max_displacement': np.max(displacement),
                'fit_success': True,
                'message': 'Success',
                'model': model_key
            }
            
            # 添加拟合参数
            for i, pname in enumerate(param_names):
                result[pname] = popt[i]
            
            return result
            
        except Exception as e:
            result = {
                'r_squared': np.nan, 
                'rmse': np.nan,
                'rel_rmse': np.nan,
                'n_points': len(_x),
                'max_force': np.max(force),
                'max_displacement': np.max(displacement),
                'fit_success': False,
                'message': f'Fitting failed: {str(e)}',
                'model': model_key
            }
            # 添加参数（全为NaN）
            for pname in param_names:
                result[pname] = np.nan
            return result
    
    except Exception as e:
        result = {
            'r_squared': np.nan, 
            'rmse': np.nan,
            'rel_rmse': np.nan,
            'n_points': 0,
            'max_force': np.nan,
            'max_displacement': np.nan,
            'fit_success': False,
            'message': f'Error reading file: {str(e)}',
            'model': model_key
        }
        # 添加参数（全为NaN）
        for pname in FITTING_MODELS[model_key]['param_names']:
            result[pname] = np.nan
        return result


def read_runner_data_for_plot(csv_file, config):
    """读取单个runner的CSV数据用于绘图"""
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


def plot_single_fit(runner_num, displacement, force, fit_params, config, model_info, ax=None):
    """绘制单个样本的拟合结果"""

    # 计算应变和应力
    epsilon = displacement / config['initial_height']
    stress = force / config['contact_area']

    if config['response_curve'] == 'raw':
        _x = displacement
        _y = force
    elif config['response_curve'] == 'nominal':
        _x = epsilon
        _y = stress
    elif config['response_curve'] == 'true':
        _x = -np.log(1 - epsilon)
        _y = (1 - epsilon) * stress

    # 生成拟合曲线
    x_fit = np.linspace(_x.min(), _x.max(), 200)
    
    # 使用对应模型的函数
    fitting_func = model_info['function']
    param_names = model_info['param_names']
    params = [fit_params[pname] for pname in param_names]
    y_fit = fitting_func(x_fit, *params)
    
    # 如果没有提供ax，创建新图
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False
    
    # 绘制原始数据点
    ax.scatter(_x, _y, s=20, alpha=0.6, label='Original Data', color='blue')
    
    # 绘制拟合曲线
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'{model_info["name"]} Fit')
    
    # 设置标签和标题
    ax.set_xlabel('Strain', fontsize=10)
    ax.set_ylabel('Stress (MPa)', fontsize=10)
    
    # 构建参数字符串
    param_list = [f'{pname}={fit_params[pname]:.2e}' for pname in param_names]
    grouped_params = [', '.join(param_list[i:i+2]) for i in range(0, len(param_list), 2)]
    param_str = '\n'.join(grouped_params)
    ax.set_title(f'Runner {runner_num}\n{param_str}\nR²={fit_params["r_squared"]:.4f}',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if standalone:
        plt.tight_layout()
        return fig
    else:
        return ax


def plot_statistics(results, model_info, output_dir):
    """绘制拟合参数的统计分布"""
    
    print(f"\n绘制统计分布...")
    
    # 提取成功的拟合结果
    successful = [r for r in results if r['fit_success']]
    
    if not successful:
        print("没有成功的拟合结果可绘制")
        return
    
    param_names = model_info['param_names']
    n_params = len(param_names)
    
    # 创建图形 - 参数 + R²
    ncols = 2
    nrows = (n_params + 2) // ncols  # +1 for R²
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = axes.flatten()
    
    # 绘制每个参数的分布
    for i, pname in enumerate(param_names):
        values = [r[pname] for r in successful]
        axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(f'Parameter {pname}', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(f'{pname} Distribution\nMean={np.mean(values):.2e}, Std={np.std(values):.2e}')
        axes[i].grid(True, alpha=0.3)
    
    # R²的分布
    r2_values = [r['r_squared'] for r in successful]
    axes[n_params].hist(r2_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[n_params].set_xlabel('R² (Goodness of Fit)', fontsize=12)
    axes[n_params].set_ylabel('Frequency', fontsize=12)
    axes[n_params].set_title(f'Fit Quality\nMean={np.mean(r2_values):.4f}, Min={np.min(r2_values):.4f}')
    axes[n_params].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(n_params + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    output_file = output_dir / 'parameters_distribution.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  保存统计图: {output_file}")
    plt.close(fig)


def plot_fitting_curves(runner_files, results, config, model_info, num_plots=None, 
                        save_individual=False, grid_size=None):
    """绘制拟合曲线"""
    
    print(f"\n绘制拟合曲线...")
    
    # 创建输出目录
    output_dir = Path('fit_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 构建runner_num到结果的映射
    results_dict = {r['runner']: r for r in results if r['fit_success']}
    
    # 确定要绘制的样本
    total_samples = len(runner_files)
    num_to_plot = min(num_plots, total_samples) if num_plots else total_samples
    
    print(f"  总样本数: {total_samples}")
    print(f"  绘制样本数: {num_to_plot}")
    
    # 如果需要保存单独图像
    if save_individual:
        print(f"\n  保存单独图像到: {output_dir}/")
        saved_count = 0
        for i, csv_file in enumerate(runner_files[:num_to_plot]):
            runner_num = extract_runner_number(csv_file)
            
            if runner_num not in results_dict:
                continue
            
            # 读取数据
            displacement, force = read_runner_data_for_plot(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制
            fig = plot_single_fit(runner_num, displacement, force, 
                                 results_dict[runner_num], config, model_info)
            
            # 保存
            model_key = config['fitting_model']
            output_file = output_dir / f'runner{runner_num:03d}_{model_key}_fit.pdf'
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            saved_count += 1
            
            if (saved_count) % 10 == 0:
                print(f"    已保存 {saved_count} 个图像")
        
        print(f"  完成! 共保存 {saved_count} 个单独图像")
    
    # 绘制网格图
    if grid_size:
        nrows, ncols = map(int, grid_size.split('x'))
    else:
        nrows, ncols = 5, 5  # 默认5x5网格
    
    plots_per_grid = nrows * ncols
    num_grids = (num_to_plot + plots_per_grid - 1) // plots_per_grid
    
    print(f"\n  绘制网格图: {num_grids} 页，每页 {nrows}x{ncols}")
    
    model_key = config['fitting_model']
    
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
            
            if runner_num not in results_dict:
                continue
            
            # 读取数据
            displacement, force = read_runner_data_for_plot(csv_file, config)
            
            if len(displacement) == 0:
                continue
            
            # 绘制到子图
            ax = axes[plot_count]
            plot_single_fit(runner_num, displacement, force, 
                          results_dict[runner_num], config, model_info, ax=ax)
            
            plot_count += 1
        
        # 隐藏多余的子图
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # 保存网格图
        output_file = output_dir / f'{model_key}_fit_grid_{grid_idx+1:02d}.pdf'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"    保存网格图 {grid_idx+1}/{num_grids}: {output_file}")
        plt.close(fig)
    
    print(f"\n  所有图像已保存到: {output_dir}/")


def process_all_runners(runner_files, file_base, config, plot_args=None):
    """处理所有runner文件并生成拟合参数CSV"""
    
    model_key = config.get('fitting_model', 'yeoh')
    model_info = FITTING_MODELS[model_key]
    
    output_csv = f"{file_base}_fit_params_{model_key}.csv"
    
    print(f"\n{'='*70}")
    print(f"拟合模型: {model_info['name']}")
    print(f"模型公式: {model_info['description']}")
    print(f"{'='*70}")
    print(f"\n找到 {len(runner_files)} 个runner文件")
    print(f"开始拟合...")
    
    results = []
    success_count = 0
    
    for i, csv_file in enumerate(runner_files):
        runner_num = extract_runner_number(csv_file)
        
        # 拟合当前文件
        fit_result = fit_time_series(csv_file, config)
        
        # 添加runner编号
        result_row = {
            'runner': runner_num,
            'filename': Path(csv_file).name,
            **fit_result
        }
        
        results.append(result_row)
        
        if fit_result['fit_success']:
            success_count += 1
        
        # 进度显示
        if (i + 1) % 10 == 0 or (i + 1) == len(runner_files):
            print(f"  已处理 {i + 1}/{len(runner_files)} 个文件 "
                  f"(成功: {success_count})")
    
    # 写入结果CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*70}")
        print(f"拟合完成!")
        print(f"{'='*70}")
        print(f"  总样本数: {len(results)}")
        print(f"  拟合成功: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"  拟合失败: {len(results) - success_count}")
        print(f"\n结果已保存到: {output_csv}")
        
        # 统计信息
        if success_count > 0:
            successful = [r for r in results if r['fit_success']]
            
            param_names = model_info['param_names']
            r2_values = [r['r_squared'] for r in successful]
            
            print(f"\n拟合参数统计:")
            for pname in param_names:
                values = [r[pname] for r in successful]
                print(f"  参数 {pname}:")
                print(f"    范围: [{np.min(values):.4e}, {np.max(values):.4e}]")
                print(f"    均值: {np.mean(values):.4e}")
                print(f"    标准差: {np.std(values):.4e}")
            
            print(f"\n拟合质量 (R²):")
            print(f"    范围: [{np.min(r2_values):.4f}, {np.max(r2_values):.4f}]")
            print(f"    均值: {np.mean(r2_values):.4f}")
            
            # 找出拟合质量最好和最差的
            best_idx = np.argmax(r2_values)
            worst_idx = np.argmin(r2_values)
            
            print(f"\n拟合质量最好: runner{successful[best_idx]['runner']} "
                  f"(R² = {successful[best_idx]['r_squared']:.4f})")
            print(f"拟合质量最差: runner{successful[worst_idx]['runner']} "
                  f"(R² = {successful[worst_idx]['r_squared']:.4f})")
        
        # 失败的样本
        if success_count < len(results):
            print(f"\n拟合失败的样本:")
            failed = [r for r in results if not r['fit_success']]
            for r in failed[:10]:  # 只显示前10个
                print(f"  runner{r['runner']}: {r['message']}")
            if len(failed) > 10:
                print(f"  ... 还有 {len(failed)-10} 个失败样本")
        
        # 绘图（如果请求）
        if plot_args and not plot_args.get('no_plot', False):
            print(f"\n{'='*70}")
            print(f"开始绘图...")
            print(f"{'='*70}")
            
            output_dir = Path('fit_plots')
            output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
            
            # 绘制统计分布
            if not plot_args.get('no_stats', False):
                plot_statistics(results, model_info, output_dir)
            
            # 绘制拟合曲线
            if not plot_args.get('curves_only', False):
                plot_fitting_curves(
                    runner_files, 
                    results, 
                    config, 
                    model_info,
                    num_plots=plot_args.get('num_plots', None),
                    save_individual=plot_args.get('save_individual', False),
                    grid_size=plot_args.get('grid_size', None)
                )
    
    else:
        print("错误: 没有数据可写入")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='响应曲线拟合程序 v2.0 - 支持多种模型和自动绘图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的拟合模型:
  yeoh        : Yeoh 3阶模型
  linear      : Linear 3阶模型
  logarithmic : Logarithmic 3阶模型
  quadratic   : Quadratic 3阶模型
  yeoh4       : Yeoh 4阶模型
  user_defined: 用户自定义模型

示例:
  # 基本拟合（默认yeoh模型）
  python fit_response_curves_v2.py
  
  # 指定模型
  python fit_response_curves_v2.py --model linear
  
  # 拟合并绘图
  python fit_response_curves_v2.py --plot
  
  # 拟合并绘制前100个样本的网格图
  python fit_response_curves_v2.py --plot --num-plots 100 --grid-size 4x4
  
  # 保存每个样本的单独图像
  python fit_response_curves_v2.py --plot --save-individual
  
  # 只绘制统计分布，不绘制拟合曲线
  python fit_response_curves_v2.py --plot --stats-only
        """
    )
    
    parser.add_argument('--model', type=str, default='voce',
                       choices=list(FITTING_MODELS.keys()),
                       help='拟合模型选择 (默认: voce)')
    parser.add_argument('--response-curve', type=str, default='true',
                       choices=['raw', 'nominal', 'true'],
                       help='响应曲线类型 (默认: nominal)')
    parser.add_argument('--no-plot', action='store_true',
                       help='拟合后不自动绘图')
    parser.add_argument('--num-plots', type=int, default=None,
                       help='绘制前N个样本 (默认: 250)')
    parser.add_argument('--grid-size', type=str, default=None,
                       help='网格布局，例如: 5x5 (默认: 5x5)')
    parser.add_argument('--save-individual', action='store_true',
                       help='保存每个样本的单独图像')
    parser.add_argument('--stats-only', action='store_true',
                       help='只绘制统计分布图，不绘制拟合曲线')
    parser.add_argument('--no-stats', action='store_true',
                       help='不绘制统计分布图')
    parser.add_argument('--file-base', type=str, default=None,
                       help='文件基础名 (默认: 自动检测)')
    
    args = parser.parse_args()
    
    # 配置参数
    config = {
        # 几何参数
        'initial_height': 1.0,          # 试样初始高度 (mm)
        'contact_area': np.pi * (1**2), # 接触面积 (mm²)
        
        # CSV列名
        'disp_col': 'disp_abs',         # 位移列名
        'force_col': 'force',           # 力列名
        
        # 拟合参数
        'min_displacement': 0.4,        # 最小位移要求 (mm)
        
        # 拟合模型选择
        'fitting_model': args.model,

        # 响应曲线类型
        'response_curve': args.response_curve,
    }
    
    # 绘图参数
    plot_args = {
        'no_plot': args.no_plot,
        'num_plots': args.num_plots,
        'grid_size': args.grid_size,
        'save_individual': args.save_individual,
        'curves_only': args.stats_only,
        'no_stats': args.no_stats,
    } if not args.no_plot else {'no_plot': True}
    
    print("\n" + "="*70)
    print("响应曲线拟合程序 v2.0")
    print("="*70)
    print("\n可用的拟合模型:")
    for key, info in FITTING_MODELS.items():
        marker = " ← 当前选择" if key == args.model else ""
        print(f"  '{key}': {info['name']}{marker}")
        print(f"           {info['description']}")
    
    # 查找文件
    runner_files, detected_base = find_runner_files(args.file_base)
    
    if runner_files is None:
        exit(1)
    
    print(f"\n检测到 file_base: {detected_base}")
    
    # 处理所有文件
    process_all_runners(runner_files, detected_base, config, plot_args)
    
    print("\n" + "="*70)
    print("程序完成!")
    print("="*70 + "\n")
