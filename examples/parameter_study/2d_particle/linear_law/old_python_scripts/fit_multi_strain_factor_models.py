#!/usr/bin/env python3
"""
使用不同应变因子模型拟合力-位移数据，并比较拟合效果

支持的模型:
1. Yeoh Model:        σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
2. Linear Model:      σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]
3. Logarithmic Model: σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
4. Quadratic Model:   σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]

用法:
    python fit_multi_strain_factor_models.py [file_base]
    
输出:
    - {file_base}_fit_params_yeoh.csv
    - {file_base}_fit_params_linear.csv
    - {file_base}_fit_params_logarithmic.csv
    - {file_base}_fit_params_quadratic.csv
    - {file_base}_fit_comparison.csv (所有模型的拟合质量对比)
    - fit_comparison_summary.txt (统计摘要)
"""

import csv
import sys
import glob
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
import re
from typing import Dict, List, Tuple, Callable


# ============================================================================
# 模型定义
# ============================================================================

def compute_I1(lmbd):
    """计算第一不变量 I₁ = λ² + 2/λ"""
    return np.power(lmbd, 2) + 2.0 / lmbd


def polynomial_term(lmbd, a, b, c):
    """多项式项: a + 2b(I₁-3) + 3c(I₁-3)²"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


def model_yeoh(epsilon, a, b, c):
    """
    Yeoh 模型: σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
    应变因子: (λ - λ⁻²)
    """
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1.0
    strain_factor = lmbd - 1.0 / (lmbd**2)
    poly = polynomial_term(lmbd, a, b, c)
    return 2.0 * strain_factor * poly


def model_linear(epsilon, a, b, c):
    """
    Linear 模型: σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]
    应变因子: (λ - 1)
    """
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1.0
    strain_factor = lmbd - 1.0
    poly = polynomial_term(lmbd, a, b, c)
    return 2.0 * strain_factor * poly


def model_logarithmic(epsilon, a, b, c):
    """
    Logarithmic 模型: σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]
    应变因子: ln(λ²) = 2ln(λ)
    """
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1.0
    strain_factor = np.log(lmbd**2)
    poly = polynomial_term(lmbd, a, b, c)
    return 2.0 * strain_factor * poly


def model_quadratic(epsilon, a, b, c):
    """
    Quadratic 模型: σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]
    应变因子: (λ² - 1)
    """
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1.0
    strain_factor = lmbd**2 - 1.0
    poly = polynomial_term(lmbd, a, b, c)
    return 2.0 * strain_factor * poly
def model_cubic(epsilon, a, b, c):
    lmbd = epsilon + 1.0
    return 2.0 * (lmbd**1 - 1.0/(lmbd**3)) * polynomial_term(lmbd, a, b, c)


# 模型字典
MODELS = {
    'yeoh': {
        'name': 'Yeoh',
        'func': model_yeoh,
        'description': 'σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
        'strain_factor': '(λ - λ⁻²)'
    },
    'linear': {
        'name': 'Linear',
        'func': model_linear,
        'description': 'σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
        'strain_factor': '(λ - 1)'
    },
    'logarithmic': {
        'name': 'Logarithmic',
        'func': model_logarithmic,
        'description': 'σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
        'strain_factor': 'ln(λ²)'
    },
    'quadratic': {
        'name': 'Quadratic',
        'func': model_quadratic,
        'description': 'σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
        'strain_factor': '(λ² - 1)'
    },
    'cubic': {
        'name': 'Cubic',
        'func': model_cubic,
        'description': 'σ = 2(λ - 1/λ^3) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
        'strain_factor': '(λ - 1/λ^3)'
    }
}


# ============================================================================
# 数据读取和预处理
# ============================================================================

def read_csv_data(csv_file: str, config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    读取CSV文件并提取位移-力数据
    
    返回:
        epsilon: 应变数组
        stress: 应力数组
        info: 数据信息字典
    """
    try:
        displacement_data = []
        force_data = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    disp = float(row.get(config['disp_col'], 0))
                    force = float(row.get(config['force_col'], 0))
                    
                    # 过滤有效点
                    if abs(disp) > 1e-6 and force > 0:
                        displacement_data.append(abs(disp))
                        force_data.append(force)
                except (ValueError, KeyError):
                    continue
        
        if len(displacement_data) < 5:
            return None, None, {
                'success': False,
                'message': f'Not enough valid data points ({len(displacement_data)} < 5)',
                'n_points': len(displacement_data)
            }
        
        displacement = np.array(displacement_data)
        force = np.array(force_data)
        
        # 检查最大位移
        max_disp = np.max(displacement)
        min_required_disp = config.get('min_displacement', 0.4)
        
        if max_disp < min_required_disp:
            return None, None, {
                'success': False,
                'message': f'Insufficient displacement: max={max_disp:.4f} < required={min_required_disp}',
                'n_points': len(displacement),
                'max_displacement': max_disp
            }
        
        # 计算应变和应力
        epsilon = displacement / config['initial_height']
        stress = force / config['contact_area']
        
        return epsilon, stress, {
            'success': True,
            'n_points': len(epsilon),
            'max_displacement': max_disp,
            'max_force': np.max(force),
            'max_stress': np.max(stress)
        }
        
    except Exception as e:
        return None, None, {
            'success': False,
            'message': f'Error reading file: {str(e)}',
            'n_points': 0
        }


# ============================================================================
# 拟合函数
# ============================================================================

def fit_single_model(epsilon: np.ndarray, stress: np.ndarray, 
                     model_func: Callable, model_name: str) -> Dict:
    """
    使用指定模型拟合数据
    
    返回:
        包含拟合参数和质量指标的字典
    """
    try:
        # 初始猜测
        n_init = min(10, len(epsilon))
        E_guess = np.polyfit(epsilon[:n_init], stress[:n_init], 1)[0]
        a_guess = max(E_guess / 2, 1.0)
        p0 = [a_guess, a_guess * 0.1, a_guess * 0.01]
        
        # 边界
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        
        # 拟合
        popt, pcov = curve_fit(
            model_func,
            epsilon,
            stress,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
            method='trf'
        )
        
        a_fit, b_fit, c_fit = popt
        
        # 计算拟合质量
        stress_pred = model_func(epsilon, *popt)
        residuals = stress - stress_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((stress - np.mean(stress))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean(residuals**2))
        rel_rmse = rmse / np.mean(stress) if np.mean(stress) > 0 else np.inf
        
        # 计算AIC和BIC
        n = len(epsilon)
        k = 3  # 参数数量
        log_likelihood = -n/2 * np.log(2*np.pi*ss_res/n) - n/2
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(n) - 2*log_likelihood
        
        return {
            'a': a_fit,
            'b': b_fit,
            'c': c_fit,
            'r_squared': r_squared,
            'rmse': rmse,
            'rel_rmse': rel_rmse,
            'aic': aic,
            'bic': bic,
            'fit_success': True,
            'message': 'Success'
        }
        
    except Exception as e:
        return {
            'a': np.nan,
            'b': np.nan,
            'c': np.nan,
            'r_squared': np.nan,
            'rmse': np.nan,
            'rel_rmse': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'fit_success': False,
            'message': f'Fitting failed: {str(e)}'
        }


def fit_all_models(csv_file: str, config: Dict) -> Dict[str, Dict]:
    """
    对单个CSV文件使用所有模型进行拟合
    
    返回:
        字典，键为模型名称，值为拟合结果
    """
    # 读取数据
    epsilon, stress, data_info = read_csv_data(csv_file, config)
    
    results = {}
    
    if not data_info['success']:
        # 数据读取失败，所有模型都返回失败
        for model_key in MODELS.keys():
            results[model_key] = {
                'a': np.nan, 'b': np.nan, 'c': np.nan,
                'r_squared': np.nan, 'rmse': np.nan, 'rel_rmse': np.nan,
                'aic': np.nan, 'bic': np.nan,
                'fit_success': False,
                'message': data_info['message'],
                'n_points': data_info.get('n_points', 0)
            }
        return results
    
    # 对每个模型进行拟合
    for model_key, model_info in MODELS.items():
        fit_result = fit_single_model(epsilon, stress, 
                                      model_info['func'], 
                                      model_info['name'])
        fit_result['n_points'] = data_info['n_points']
        fit_result['max_displacement'] = data_info['max_displacement']
        fit_result['max_force'] = data_info['max_force']
        fit_result['max_stress'] = data_info['max_stress']
        
        results[model_key] = fit_result
    
    return results


# ============================================================================
# 文件查找和编号提取
# ============================================================================

def extract_runner_number(filename: str) -> int:
    """从文件名提取runner编号"""
    match = re.search(r'runner(\d+)', filename)
    return int(match.group(1)) if match else -1


def find_runner_files(file_base: str = None) -> Tuple[List[str], str]:
    """查找所有runner CSV文件"""
    if file_base:
        pattern = f"{file_base}_out_runner*.csv"
    else:
        patterns = glob.glob("*_out_runner*.csv")
        if not patterns:
            print("错误: 找不到 *_out_runner*.csv 文件")
            return None, None
        
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
# 批量处理和结果输出
# ============================================================================

def process_all_runners(runner_files: List[str], file_base: str, config: Dict):
    """处理所有runner文件并生成结果"""
    
    print(f"\n{'='*70}")
    print(f"Multi-Model Fitting: {len(runner_files)} runner files")
    print(f"{'='*70}")
    print(f"Models: {', '.join([info['name'] for info in MODELS.values()])}")
    print(f"File base: {file_base}")
    print()
    
    # 存储所有结果
    all_results = {model_key: [] for model_key in MODELS.keys()}
    comparison_results = []
    
    # 处理每个文件
    for i, csv_file in enumerate(runner_files):
        runner_num = extract_runner_number(csv_file)
        
        # 拟合所有模型
        fit_results = fit_all_models(csv_file, config)
        
        # 保存各模型结果
        for model_key, fit_result in fit_results.items():
            result_row = {
                'runner': runner_num,
                'filename': Path(csv_file).name,
                **fit_result
            }
            all_results[model_key].append(result_row)
        
        # 创建对比行
        comparison_row = {
            'runner': runner_num,
            'filename': Path(csv_file).name,
            'n_points': fit_results['yeoh']['n_points']
        }
        
        # 添加每个模型的关键指标
        for model_key, model_info in MODELS.items():
            prefix = model_key
            comparison_row[f'{prefix}_r2'] = fit_results[model_key]['r_squared']
            comparison_row[f'{prefix}_rmse'] = fit_results[model_key]['rmse']
            comparison_row[f'{prefix}_aic'] = fit_results[model_key]['aic']
            comparison_row[f'{prefix}_bic'] = fit_results[model_key]['bic']
            comparison_row[f'{prefix}_success'] = fit_results[model_key]['fit_success']
        
        # 找出最佳模型（基于R²）
        r2_values = {k: v['r_squared'] for k, v in fit_results.items() 
                    if v['fit_success'] and not np.isnan(v['r_squared'])}
        if r2_values:
            best_model = max(r2_values, key=r2_values.get)
            comparison_row['best_model_r2'] = best_model
            comparison_row['best_r2_value'] = r2_values[best_model]
        else:
            comparison_row['best_model_r2'] = 'none'
            comparison_row['best_r2_value'] = np.nan
        
        # 找出最佳模型（基于AIC，越小越好）
        aic_values = {k: v['aic'] for k, v in fit_results.items() 
                     if v['fit_success'] and not np.isnan(v['aic'])}
        if aic_values:
            best_model_aic = min(aic_values, key=aic_values.get)
            comparison_row['best_model_aic'] = best_model_aic
            comparison_row['best_aic_value'] = aic_values[best_model_aic]
        else:
            comparison_row['best_model_aic'] = 'none'
            comparison_row['best_aic_value'] = np.nan
        
        comparison_results.append(comparison_row)
        
        # 进度显示
        if (i + 1) % 10 == 0 or (i + 1) == len(runner_files):
            print(f"  已处理 {i + 1}/{len(runner_files)} 个文件")
    
    # 写入各模型的详细结果
    print(f"\n保存各模型拟合参数...")
    for model_key, model_info in MODELS.items():
        output_csv = f"{file_base}_fit_params_{model_key}.csv"
        results = all_results[model_key]
        
        if results:
            fieldnames = list(results[0].keys())
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            success_count = sum(1 for r in results if r['fit_success'])
            print(f"  {model_info['name']:15s}: {output_csv:40s} "
                  f"(成功: {success_count}/{len(results)})")
    
    # 写入对比结果
    comparison_csv = f"{file_base}_fit_comparison.csv"
    if comparison_results:
        fieldnames = list(comparison_results[0].keys())
        with open(comparison_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_results)
        print(f"  Comparison     : {comparison_csv}")
    
    # 生成统计报告
    generate_summary_report(all_results, comparison_results, file_base)


def generate_summary_report(all_results: Dict, comparison_results: List[Dict], 
                           file_base: str):
    """生成统计摘要报告"""
    
    summary_file = f"{file_base}_fit_comparison_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Multi-Model Fitting Comparison Summary\n")
        f.write("="*70 + "\n\n")
        
        # 各模型统计
        f.write("Individual Model Statistics:\n")
        f.write("-"*70 + "\n\n")
        
        for model_key, model_info in MODELS.items():
            results = all_results[model_key]
            successful = [r for r in results if r['fit_success']]
            success_rate = len(successful) / len(results) * 100 if results else 0
            
            f.write(f"{model_info['name']} Model:\n")
            f.write(f"  Description: {model_info['description']}\n")
            f.write(f"  Strain Factor: {model_info['strain_factor']}\n")
            f.write(f"  Success Rate: {len(successful)}/{len(results)} ({success_rate:.1f}%)\n")
            
            if successful:
                r2_values = [r['r_squared'] for r in successful]
                rmse_values = [r['rmse'] for r in successful]
                aic_values = [r['aic'] for r in successful]
                
                f.write(f"  R² Statistics:\n")
                f.write(f"    Mean:   {np.mean(r2_values):.6f}\n")
                f.write(f"    Std:    {np.std(r2_values):.6f}\n")
                f.write(f"    Min:    {np.min(r2_values):.6f}\n")
                f.write(f"    Max:    {np.max(r2_values):.6f}\n")
                
                f.write(f"  RMSE Statistics:\n")
                f.write(f"    Mean:   {np.mean(rmse_values):.6f}\n")
                f.write(f"    Std:    {np.std(rmse_values):.6f}\n")
                
                f.write(f"  AIC Statistics:\n")
                f.write(f"    Mean:   {np.mean(aic_values):.6f}\n")
                f.write(f"    Min:    {np.min(aic_values):.6f}\n")
                
            f.write("\n")
        
        # 模型对比
        f.write("\nModel Comparison:\n")
        f.write("-"*70 + "\n\n")
        
        # 统计每个模型获胜次数（基于R²）
        best_model_counts_r2 = {}
        for model_key in MODELS.keys():
            best_model_counts_r2[model_key] = sum(
                1 for r in comparison_results 
                if r['best_model_r2'] == model_key
            )
        
        f.write("Best Model by R² (highest R²):\n")
        for model_key, count in sorted(best_model_counts_r2.items(), 
                                       key=lambda x: x[1], reverse=True):
            model_name = MODELS[model_key]['name']
            percentage = count / len(comparison_results) * 100 if comparison_results else 0
            f.write(f"  {model_name:15s}: {count:4d} times ({percentage:5.1f}%)\n")
        
        # 统计每个模型获胜次数（基于AIC）
        best_model_counts_aic = {}
        for model_key in MODELS.keys():
            best_model_counts_aic[model_key] = sum(
                1 for r in comparison_results 
                if r['best_model_aic'] == model_key
            )
        
        f.write("\nBest Model by AIC (lowest AIC):\n")
        for model_key, count in sorted(best_model_counts_aic.items(), 
                                       key=lambda x: x[1], reverse=True):
            model_name = MODELS[model_key]['name']
            percentage = count / len(comparison_results) * 100 if comparison_results else 0
            f.write(f"  {model_name:15s}: {count:4d} times ({percentage:5.1f}%)\n")
        
        # 平均R²对比
        f.write("\nAverage R² Comparison (successful fits only):\n")
        avg_r2 = {}
        for model_key in MODELS.keys():
            results = all_results[model_key]
            successful = [r for r in results if r['fit_success']]
            if successful:
                avg_r2[model_key] = np.mean([r['r_squared'] for r in successful])
            else:
                avg_r2[model_key] = 0
        
        for model_key, r2 in sorted(avg_r2.items(), key=lambda x: x[1], reverse=True):
            model_name = MODELS[model_key]['name']
            f.write(f"  {model_name:15s}: {r2:.6f}\n")
        
        # 平均AIC对比
        f.write("\nAverage AIC Comparison (successful fits only, lower is better):\n")
        avg_aic = {}
        for model_key in MODELS.keys():
            results = all_results[model_key]
            successful = [r for r in results if r['fit_success']]
            if successful:
                avg_aic[model_key] = np.mean([r['aic'] for r in successful])
            else:
                avg_aic[model_key] = np.inf
        
        for model_key, aic in sorted(avg_aic.items(), key=lambda x: x[1]):
            model_name = MODELS[model_key]['name']
            f.write(f"  {model_name:15s}: {aic:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n统计摘要已保存到: {summary_file}")
    
    # 在控制台显示关键信息
    print(f"\n{'='*70}")
    print(f"拟合对比摘要")
    print(f"{'='*70}")
    
    print(f"\n各模型成功率:")
    for model_key, model_info in MODELS.items():
        results = all_results[model_key]
        successful = [r for r in results if r['fit_success']]
        success_rate = len(successful) / len(results) * 100 if results else 0
        print(f"  {model_info['name']:15s}: {len(successful):3d}/{len(results)} "
              f"({success_rate:5.1f}%)")
    
    print(f"\n平均 R² (越高越好):")
    for model_key, r2 in sorted(avg_r2.items(), key=lambda x: x[1], reverse=True):
        model_name = MODELS[model_key]['name']
        print(f"  {model_name:15s}: {r2:.6f}")
    
    print(f"\n平均 AIC (越低越好):")
    for model_key, aic in sorted(avg_aic.items(), key=lambda x: x[1]):
        if aic == np.inf:
            continue
        model_name = MODELS[model_key]['name']
        print(f"  {model_name:15s}: {aic:.6f}")
    
    print(f"\n最佳模型统计 (基于 R²):")
    for model_key, count in sorted(best_model_counts_r2.items(), 
                                   key=lambda x: x[1], reverse=True):
        model_name = MODELS[model_key]['name']
        percentage = count / len(comparison_results) * 100 if comparison_results else 0
        print(f"  {model_name:15s}: {count:4d} 次 ({percentage:5.1f}%)")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    # 配置参数
    config = {
        'initial_height': 1.0,              # 试样初始高度 (mm)
        'contact_area': np.pi * (1.0 ** 2), # 接触面积 (mm²)
        'disp_col': 'disp_abs',             # 位移列名
        'force_col': 'force',               # 力列名
        'min_displacement': 0.4             # 最小位移要求 (mm)
    }
    
    # 解析命令行参数
    file_base = None
    if len(sys.argv) == 2:
        file_base = sys.argv[1]
    elif len(sys.argv) > 2:
        print("用法: python fit_multi_strain_factor_models.py [file_base]")
        print("\n示例:")
        print("  python fit_multi_strain_factor_models.py                    # 自动检测")
        print("  python fit_multi_strain_factor_models.py main_lh_sampler    # 指定 file_base")
        sys.exit(1)
    
    # 查找runner文件
    runner_files, detected_base = find_runner_files(file_base)
    
    if not runner_files:
        sys.exit(1)
    
    file_base = file_base or detected_base
    print(f"\n检测到 file_base: {file_base}")
    print(f"文件范围: {Path(runner_files[0]).name} 到 {Path(runner_files[-1]).name}")
    print(f"总文件数: {len(runner_files)}")
    
    # 处理所有文件
    process_all_runners(runner_files, file_base, config)
    
    print(f"\n{'='*70}")
    print(f"完成!")
    print(f"{'='*70}\n")
