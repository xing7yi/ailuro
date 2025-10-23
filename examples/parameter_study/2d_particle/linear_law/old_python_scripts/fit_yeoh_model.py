#!/usr/bin/env python3
"""
使用 Yeoh 模型拟合力-位移数据（时间序列CSV文件）

拟合函数:
    stress = 2*(λ - λ^-3) * (a + 2b(I₁-3) + 3c(I₁-3)²)
    其中: λ = ε + 1, I₁ = λ² + λ^-1

用法:
    python fit_yeoh_model.py [file_base]
    
    自动检测所有 {file_base}_out_runner*.csv 文件
    输出: {file_base}_fit_params.csv (包含所有样本的拟合参数)
"""

import csv
import sys
import glob
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
import re


def YeohModel_Power3(epsilon, a, b, c):
    """
    Yeoh 超弹性模型 (三阶)
    
    参数:
        epsilon: 工程应变 (无量纲)
        a, b, c: Yeoh 模型参数
    返回:
        应力 (与输入力的单位一致)
    """
    epsilon = np.asarray(epsilon)
    lmbd = epsilon + 1  # 伸长比 λ
    I1 = np.power(lmbd, 2) + 2 * np.power(lmbd, -1)  # 第一不变量
    stress = 2 * (lmbd - np.power(lmbd, -2)) * (
        a + 2 * b * (I1 - 3) + 3 * c * np.power((I1 - 3), 2)
    )
    return stress


def fit_time_series(csv_file, config):
    """
    拟合单个时间序列CSV文件
    
    参数:
        csv_file: CSV文件路径
        config: 配置字典
    
    返回:
        dict: 包含拟合参数和拟合质量指标
    """
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
        
        if len(displacement_data) < 5:
            return {
                'a': np.nan, 'b': np.nan, 'c': np.nan,
                'r_squared': np.nan, 'rmse': np.nan,
                'n_points': len(displacement_data),
                'max_displacement': max(displacement_data) if displacement_data else np.nan,
                'fit_success': False,
                'message': f'Not enough valid data points ({len(displacement_data)} < 5)'
            }
        
        # 转换为numpy数组
        displacement = np.array(displacement_data)
        force = np.array(force_data)
        
        # 检查最大位移是否达到预期值
        max_disp = np.max(displacement)
        min_required_disp = config.get('min_displacement', 0.4)  # 默认要求至少0.4mm
        
        if max_disp < min_required_disp:
            return {
                'a': np.nan, 'b': np.nan, 'c': np.nan,
                'r_squared': np.nan, 'rmse': np.nan,
                'n_points': len(displacement_data),
                'max_displacement': max_disp,
                'max_force': np.max(force),
                'fit_success': False,
                'message': f'Insufficient displacement: max={max_disp:.4f}mm < required={min_required_disp}mm'
            }
        
        # 计算应变和应力
        epsilon = displacement / config['initial_height']
        stress = force / config['contact_area']
        
        # 初始猜测
        # 使用前几个点估算线性弹性模量
        n_init = min(10, len(epsilon))
        E_guess = np.polyfit(epsilon[:n_init], stress[:n_init], 1)[0]
        a_guess = max(E_guess / 2, 1.0)  # 确保初始猜测为正
        
        p0 = [a_guess, a_guess * 0.1, a_guess * 0.01]
        
        # 设置合理的边界
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        
        # 执行拟合
        try:
            popt, pcov = curve_fit(
                YeohModel_Power3, 
                epsilon, 
                stress,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
                method='trf'  # Trust Region Reflective算法，对有界问题更稳定
            )
            
            a_fit, b_fit, c_fit = popt
            
            # 计算拟合质量
            stress_pred = YeohModel_Power3(epsilon, *popt)
            residuals = stress - stress_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((stress - np.mean(stress))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean(residuals**2))
            
            # 计算相对误差
            rel_rmse = rmse / np.mean(stress) if np.mean(stress) > 0 else np.inf
            
            return {
                'a': a_fit,
                'b': b_fit,
                'c': c_fit,
                'r_squared': r_squared,
                'rmse': rmse,
                'rel_rmse': rel_rmse,
                'n_points': len(epsilon),
                'max_force': np.max(force),
                'max_displacement': np.max(displacement),
                'fit_success': True,
                'message': 'Success'
            }
            
        except Exception as e:
            return {
                'a': np.nan, 'b': np.nan, 'c': np.nan,
                'r_squared': np.nan, 'rmse': np.nan,
                'rel_rmse': np.nan,
                'n_points': len(epsilon),
                'max_force': np.max(force) if len(force) > 0 else np.nan,
                'max_displacement': np.max(displacement) if len(displacement) > 0 else np.nan,
                'fit_success': False,
                'message': f'Fitting failed: {str(e)}'
            }
    
    except Exception as e:
        return {
            'a': np.nan, 'b': np.nan, 'c': np.nan,
            'r_squared': np.nan, 'rmse': np.nan,
            'rel_rmse': np.nan,
            'n_points': 0,
            'max_force': np.nan,
            'max_displacement': np.nan,
            'fit_success': False,
            'message': f'Error reading file: {str(e)}'
        }


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


def process_all_runners(runner_files, file_base, config):
    """处理所有runner文件并生成拟合参数CSV"""
    
    output_csv = f"{file_base}_fit_params.csv"
    
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
        
        print(f"\n{'='*60}")
        print(f"完成!")
        print(f"{'='*60}")
        print(f"  总样本数: {len(results)}")
        print(f"  拟合成功: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"  拟合失败: {len(results) - success_count}")
        print(f"\n结果已保存到: {output_csv}")
        
        # 统计信息
        if success_count > 0:
            successful = [r for r in results if r['fit_success']]
            
            a_values = [r['a'] for r in successful]
            b_values = [r['b'] for r in successful]
            c_values = [r['c'] for r in successful]
            r2_values = [r['r_squared'] for r in successful]
            
            print(f"\n拟合参数统计:")
            print(f"  参数 a:")
            print(f"    范围: [{np.min(a_values):.4e}, {np.max(a_values):.4e}]")
            print(f"    均值: {np.mean(a_values):.4e}")
            print(f"    标准差: {np.std(a_values):.4e}")
            
            print(f"  参数 b:")
            print(f"    范围: [{np.min(b_values):.4e}, {np.max(b_values):.4e}]")
            print(f"    均值: {np.mean(b_values):.4e}")
            print(f"    标准差: {np.std(b_values):.4e}")
            
            print(f"  参数 c:")
            print(f"    范围: [{np.min(c_values):.4e}, {np.max(c_values):.4e}]")
            print(f"    均值: {np.mean(c_values):.4e}")
            print(f"    标准差: {np.std(c_values):.4e}")
            
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
    
    else:
        print("错误: 没有数据可写入")


if __name__ == '__main__':
    # 配置参数（根据您的实际问题调整）
    config = {
        'initial_height': 1.0,  # 试样初始高度 (mm)
        'contact_area': np.pi * (1 ** 2),    # 接触面积 (mm²)，如果应力已归一化可设为1.0
        'disp_col': 'disp_abs',  # 位移列名
        'force_col': 'force'     # 力列
    }
    
    # 解析命令行参数
    file_base = None
    if len(sys.argv) == 2:
        file_base = sys.argv[1]
    elif len(sys.argv) > 2:
        print("用法: python fit_yeoh_model.py [file_base]")
        print("\n示例:")
        print("  python fit_yeoh_model.py                    # 自动检测")
        print("  python fit_yeoh_model.py main_lh_sampler    # 指定 file_base")
        sys.exit(1)
    
    # 查找runner文件
    runner_files, detected_base = find_runner_files(file_base)
    
    if not runner_files:
        sys.exit(1)
    
    file_base = file_base or detected_base
    print(f"\n检测到 file_base: {file_base}")
    print(f"文件范围: {Path(runner_files[0]).name} 到 {Path(runner_files[-1]).name}")
    
    # 处理所有文件
    process_all_runners(runner_files, file_base, config)
