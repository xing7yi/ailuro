#!/usr/bin/env python3
"""
提取 main_csv_forward_OptimizationReporter_*.csv 文件中的参数值并汇总
"""
import pandas as pd
import glob
import os
import re

def extract_parameters():
    # 获取所有 csv_forward 文件
    pattern = "main_csv_forward_OptimizationReporter_*.csv"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"未找到匹配的文件: {pattern}")
        return
    
    print(f"找到 {len(files)} 个文件")
    
    # 存储结果
    results = []
    
    for file in files:
        # 从文件名提取序号
        match = re.search(r'_(\d{4})\.csv$', file)
        if not match:
            continue
        
        file_no = int(match.group(1))
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file)
            
            # 提取 parameter_results 列
            if 'parameter_results' not in df.columns:
                print(f"警告: {file} 中没有 parameter_results 列")
                continue
            
            params = df['parameter_results'].values
            
            # 根据参数数量创建记录
            if len(params) == 2:
                # 两个参数的情况
                results.append({
                    'No': file_no,
                    'P0': params[0],
                    'P1': params[1],
                    'P2': None
                })
            elif len(params) == 3:
                # 三个参数的情况
                results.append({
                    'No': file_no,
                    'P0': params[0],
                    'P1': params[1],
                    'P2': params[2]
                })
            else:
                print(f"警告: {file} 参数数量异常: {len(params)}")
                
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
            continue
    
    if not results:
        print("没有成功提取任何数据")
        return
    
    # 创建DataFrame
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('No')
    
    # 读取 main_csv_forward.csv 并添加 objective_value
    try:
        obj_df = pd.read_csv('main_csv_forward.csv')
        if 'OptimizationReporter/objective_value' in obj_df.columns:
            # 提取 objective_value 列
            obj_values = obj_df['OptimizationReporter/objective_value'].values
            # 根据 time 列匹配（time 对应 No）
            if 'time' in obj_df.columns:
                obj_dict = dict(zip(obj_df['time'].astype(int), obj_values))
                result_df['objective_value'] = result_df['No'].map(obj_dict)
            else:
                # 如果没有time列，按照索引顺序匹配
                result_df['objective_value'] = obj_values[:len(result_df)]
            print("成功添加 objective_value 列")
        else:
            print("警告: main_csv_forward.csv 中没有找到 objective_value 列")
    except FileNotFoundError:
        print("警告: 未找到 main_csv_forward.csv 文件")
    except Exception as e:
        print(f"读取 main_csv_forward.csv 时出错: {e}")
    
    # 保存到新文件
    output_file = "optimization_parameters_summary.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"\n成功提取 {len(result_df)} 个迭代的参数")
    print(f"结果已保存到: {output_file}")
    print(f"\n前5行预览:")
    print(result_df.head())
    print(f"\n最后5行预览:")
    print(result_df.tail())
    
    # 显示统计信息
    print(f"\n参数统计:")
    print(result_df.describe())

if __name__ == "__main__":
    extract_parameters()
