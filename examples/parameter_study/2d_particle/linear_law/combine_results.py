#!/usr/bin/env python3
"""
合并采样参数和计算结果到单个 CSV 文件

用法:
    python combine_results.py [file_base]
    
    如果不指定 file_base，会自动查找当前目录下的文件
    如果指定了 file_base，会查找:
      - {file_base}_samples_sample_data_0000.csv
      - {file_base}_out.json
    输出为: {file_base}_combined.csv
"""

import csv
import json
import sys
from pathlib import Path
import glob


def combine_results(samples_csv, results_json, output_csv):
    """合并采样数据和计算结果"""
    
    # 列名映射（可自定义）
    COLUMN_RENAME = {
        'grid_0': 'yield_stress',
        'grid_1': 'hardening_constant',
        'hypercube_0': 'yield_stress',
        'hypercube_1': 'hardening_constant',
        'results:contact_pressure_avg:value': 'contact_pressure',
        'results:converged': 'converged',
        'results:force:value': 'max_force'
    }
    
    # 1. 读取采样数据 (CSV)
    samples = []
    with open(samples_csv, 'r') as f:
        reader = csv.DictReader(f)
        param_names = reader.fieldnames
        for row in reader:
            samples.append({k: float(v) for k, v in row.items()})
    
    print(f"读取了 {len(samples)} 个采样点，参数: {param_names}")
    
    # 2. 读取计算结果 (JSON)
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    # 找到最后一个时间步的结果
    if 'time_steps' not in data or len(data['time_steps']) == 0:
        print("错误: JSON 文件中没有 time_steps 数据")
        return
    
    last_step = data['time_steps'][-1]
    results = last_step.get('results', {})
    
    print(f"找到时间步 {last_step.get('time_step')}，时间 {last_step.get('time')}")
    
    # 提取结果向量
    result_vectors = {}
    for key, values in results.items():
        if isinstance(values, list):
            result_vectors[key] = values
            print(f"  结果 '{key}': {len(values)} 个值")
    
    # 3. 检查数据长度匹配
    num_samples = len(samples)
    for key, values in result_vectors.items():
        if len(values) != num_samples:
            print(f"警告: 结果 '{key}' 有 {len(values)} 个值，但采样点有 {num_samples} 个")
    
    # 4. 合并数据
    combined = []
    for i, sample in enumerate(samples):
        row = sample.copy()  # 从采样参数开始
        # 添加结果数据
        for key, values in result_vectors.items():
            if i < len(values):
                row[key] = values[i]
        combined.append(row)
    
    # 5. 重命名列（使用更友好的名称）
    renamed_combined = []
    for row in combined:
        renamed_row = {}
        for old_name, value in row.items():
            new_name = COLUMN_RENAME.get(old_name, old_name)
            renamed_row[new_name] = value
        renamed_combined.append(renamed_row)
    
    # 6. 写入 CSV
    if renamed_combined:
        fieldnames = list(renamed_combined[0].keys())
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(renamed_combined)
        
        print(f"\n成功! 合并后的数据已保存到: {output_csv}")
        print(f"  样本数: {len(combined)}")
        print(f"  列名: {', '.join(fieldnames)}")
    else:
        print("错误: 没有数据可写入")


def find_files(file_base=None):
    """自动查找采样和结果文件"""
    
    if file_base:
        # 使用指定的 file_base
        samples_pattern = f"{file_base}_samples_sample_data_0000.csv"
        results_pattern = f"{file_base}_out.json"
        output_csv = f"{file_base}_combined.csv"
        
        samples_files = glob.glob(samples_pattern)
        results_files = glob.glob(results_pattern)
        
        if not samples_files:
            print(f"错误: 找不到采样文件: {samples_pattern}")
            return None
        if not results_files:
            print(f"错误: 找不到结果文件: {results_pattern}")
            return None
            
        return samples_files[0], results_files[0], output_csv
    
    else:
        # 自动查找
        samples_files = glob.glob("*_samples_sample_data_0000.csv")
        results_files = glob.glob("*_out.json")
        
        if not samples_files:
            print("错误: 当前目录找不到采样文件 (*_samples_sample_data_0000.csv)")
            return None
        if not results_files:
            print("错误: 当前目录找不到结果文件 (*_out.json)")
            return None
        
        # 自动匹配所有可能的组合
        matched_pairs = []
        for samples_file in samples_files:
            base = samples_file.replace('_samples_sample_data_0000.csv', '')
            expected_results = f"{base}_out.json"
            if expected_results in results_files:
                matched_pairs.append({
                    'base': base,
                    'samples': samples_file,
                    'results': expected_results,
                    'output': f"{base}_combined.csv"
                })
        
        if not matched_pairs:
            print("\n错误: 无法自动匹配采样文件和结果文件")
            print("\n采样文件:")
            for f in samples_files:
                print(f"  - {f}")
            print("\n结果文件:")
            for f in results_files:
                print(f"  - {f}")
            print("\n请检查文件命名或手动指定 file_base:")
            print("  python combine_results.py <file_base>")
            return None
        
        # 只有一个匹配，直接使用
        if len(matched_pairs) == 1:
            pair = matched_pairs[0]
            print(f"找到匹配的文件组:")
            print(f"  基础名: {pair['base']}")
            print(f"  采样文件: {pair['samples']}")
            print(f"  结果文件: {pair['results']}")
            print(f"  输出文件: {pair['output']}")
            return pair['samples'], pair['results'], pair['output']
        
        # 多个匹配，让用户选择
        print(f"\n找到 {len(matched_pairs)} 组匹配的文件，请选择:")
        for i, pair in enumerate(matched_pairs):
            print(f"\n  {i+1}. {pair['base']}")
            print(f"     采样: {pair['samples']}")
            print(f"     结果: {pair['results']}")
        
        while True:
            try:
                choice = input(f"\n请输入选择 (1-{len(matched_pairs)}) 或按 Enter 使用第一个: ").strip()
                if not choice:
                    choice = 1
                else:
                    choice = int(choice)
                
                if 1 <= choice <= len(matched_pairs):
                    pair = matched_pairs[choice - 1]
                    print(f"\n选择了: {pair['base']}")
                    return pair['samples'], pair['results'], pair['output']
                else:
                    print(f"请输入 1 到 {len(matched_pairs)} 之间的数字")
            except ValueError:
                print("无效输入，请输入数字")
            except (KeyboardInterrupt, EOFError):
                print("\n\n用户取消")
                return None


if __name__ == '__main__':
    # 解析命令行参数
    file_base = None
    if len(sys.argv) == 2:
        file_base = sys.argv[1]
    elif len(sys.argv) > 2:
        print("用法: python combine_results.py [file_base]")
        print("\n示例:")
        print("  python combine_results.py                    # 自动查找")
        print("  python combine_results.py main_grid_sampler  # 指定 file_base")
        sys.exit(1)
    
    # 查找文件
    files = find_files(file_base)
    if not files:
        sys.exit(1)
    
    samples_csv, results_json, output_csv = files
    
    # 合并结果
    combine_results(samples_csv, results_json, output_csv)
