#!/usr/bin/env python3
"""
合并拟合参数与采样参数

将 fit_params.csv 中的拟合结果与原始采样参数（从samples CSV或JSON）合并

用法:
    python merge_fit_with_samples.py [file_base]
    
    输入:
      - {file_base}_fit_params.csv (来自 fit_yeoh_model.py)
      - {file_base}_samples_sample_data_0000.csv (采样参数)
    输出:
      - {file_base}_complete_results.csv (合并后的完整结果)
"""

import csv
import json
import sys
import glob
from pathlib import Path


def find_files(file_base=None):
    """查找需要的文件"""
    
    if file_base:
        fit_params = f"{file_base}_fit_params_voce.csv"
        samples = f"{file_base}_samples_sample_data_0000.csv"
        output = f"{file_base}_complete_results.csv"
        
        if not Path(fit_params).exists():
            print(f"错误: 找不到文件 {fit_params}")
            print("请先运行 fit_yeoh_model.py 生成拟合参数文件")
            return None
        
        if not Path(samples).exists():
            print(f"警告: 找不到采样文件 {samples}")
            print("将只输出拟合参数，不包含原始采样参数")
            samples = None
        
        return fit_params, samples, output
    
    else:
        # 自动查找
        fit_files = glob.glob("*_fit_params_voce.csv")
        
        if not fit_files:
            print("错误: 找不到 *_fit_params_voce.csv 文件")
            print("请先运行 fit_yeoh_model.py")
            return None
        
        # 推断file_base
        fit_params = fit_files[0]
        file_base = fit_params.replace('_fit_params_voce.csv', '')
        samples = f"{file_base}_samples_sample_data_0000.csv"
        output = f"{file_base}_complete_results.csv"
        
        if not Path(samples).exists():
            print(f"警告: 找不到采样文件 {samples}")
            samples = None
        
        print(f"自动检测到 file_base: {file_base}")
        return fit_params, samples, output


def merge_results(fit_params_csv, samples_csv, output_csv):
    """合并拟合参数和采样参数"""
    
    # 列名映射（与combine_results.py保持一致）
    COLUMN_RENAME = {
        'grid_0': 'yield_stress',
        'grid_1': 'hardening_constant',
        'hypercube_0': 'yield_stress',
        'hypercube_1': 'hardening_constant',
    }
    
    print(f"\n读取拟合参数: {fit_params_csv}")
    
    # 1. 读取拟合参数
    fit_results = []
    with open(fit_params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fit_results.append(row)
    
    print(f"  找到 {len(fit_results)} 个拟合结果")
    
    # 2. 读取采样参数（如果存在）
    samples = []
    if samples_csv and Path(samples_csv).exists():
        print(f"\n读取采样参数: {samples_csv}")
        with open(samples_csv, 'r') as f:
            reader = csv.DictReader(f)
            param_names = reader.fieldnames
            for row in reader:
                # 重命名列
                renamed_row = {}
                for key, value in row.items():
                    new_key = COLUMN_RENAME.get(key, key)
                    renamed_row[new_key] = value
                samples.append(renamed_row)
        
        print(f"  找到 {len(samples)} 个采样点")
        print(f"  参数: {', '.join(param_names)}")
        
        if len(samples) != len(fit_results):
            print(f"\n警告: 采样点数({len(samples)})与拟合结果数({len(fit_results)})不匹配")
    
    # 3. 合并数据
    merged = []
    successful_count = 0
    failed_count = 0
    
    for i, fit_row in enumerate(fit_results):
        # 如果有采样参数，先添加采样参数
        if samples and i < len(samples):
            merged_row = samples[i].copy()
        else:
            merged_row = {}
        
        # 添加拟合结果（排除runner和filename，这些是索引信息）
        for key, value in fit_row.items():
            if key not in ['filename']:  # 保留runner作为索引
                merged_row[key] = value
        
        # 统计成功/失败
        if fit_row.get('fit_success', 'False') == 'True':
            successful_count += 1
        else:
            failed_count += 1
        
        merged.append(merged_row)
    
    # 4. 写入CSV
    if merged:
        fieldnames = list(merged[0].keys())
        
        # 调整列顺序：参数列在前，结果列在后
        param_cols = []
        result_cols = []
        
        for col in fieldnames:
            if col in ['runner', 'yield_stress', 'hardening_constant']:
                param_cols.append(col)
            else:
                result_cols.append(col)
        
        # 调整顺序
        ordered_fields = ['runner'] + [c for c in param_cols if c != 'runner'] + result_cols
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            writer.writerows(merged)
        
        print(f"\n{'='*60}")
        print(f"成功!")
        print(f"{'='*60}")
        print(f"合并后的数据已保存到: {output_csv}")
        print(f"  总行数: {len(merged)}")
        print(f"  拟合成功: {successful_count} ({successful_count/len(merged)*100:.1f}%)")
        print(f"  拟合失败: {failed_count} ({failed_count/len(merged)*100:.1f}%)")
        print(f"  列: {', '.join(ordered_fields)}")
        
        # 显示统计信息
        if samples:
            print(f"\n数据包含:")
            print(f"  - 采样参数 ({len(param_cols)-1} 列)")
            print(f"  - 拟合参数 (a, b, c)")
            print(f"  - 拟合质量指标 (r_squared, rmse)")
            print(f"  - 力学响应 (max_force, max_displacement)")
        
        # 显示拟合失败的样本信息
        if failed_count > 0:
            print(f"\n拟合失败的样本 (前10个):")
            failed_samples = [m for m in merged if m.get('fit_success', 'False') == 'False']
            for i, sample in enumerate(failed_samples[:10]):
                runner = sample.get('runner', '?')
                message = sample.get('message', 'Unknown error')
                max_disp = sample.get('max_displacement', 'N/A')
                print(f"  runner{runner}: {message} (max_disp={max_disp})")
            if len(failed_samples) > 10:
                print(f"  ... 还有 {len(failed_samples)-10} 个失败样本")
    else:
        print("错误: 没有数据可写入")


if __name__ == '__main__':
    # 解析命令行参数
    file_base = None
    if len(sys.argv) == 2:
        file_base = sys.argv[1]
    elif len(sys.argv) > 2:
        print("用法: python merge_fit_with_samples.py [file_base]")
        print("\n示例:")
        print("  python merge_fit_with_samples.py                    # 自动检测")
        print("  python merge_fit_with_samples.py main_lh_sampler    # 指定 file_base")
        sys.exit(1)
    
    # 查找文件
    files = find_files(file_base)
    if not files:
        sys.exit(1)
    
    fit_params_csv, samples_csv, output_csv = files
    
    # 合并结果
    merge_results(fit_params_csv, samples_csv, output_csv)
