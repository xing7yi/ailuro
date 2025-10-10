#!/usr/bin/env python3
"""
汇总并比较不同应变因子模型的拟合效果

从拟合参数文件中读取数据，生成详细的统计比较报告

用法:
    python summarize_model_comparison.py [file_base]
"""

import sys
import csv
import glob
import numpy as np
from pathlib import Path
import re


# 模型信息
MODELS = {
    'yeoh': 'Yeoh',
    'linear': 'Linear',
    'logarithmic': 'Logarithmic',
    'quadratic': 'Quadratic'
}


def read_fit_params(file_base, model_key):
    """读取某个模型的拟合参数"""
    csv_file = f"{file_base}_fit_params_{model_key}.csv"
    
    if not Path(csv_file).exists():
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


def compute_statistics(values):
    """计算统计量"""
    if not values:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan
        }
    
    arr = np.array(values)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75)
    }


def generate_summary(all_params, file_base):
    """生成综合统计报告"""
    
    output_file = f"{file_base}_model_comparison_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("多模型拟合效果对比统计报告\n")
        f.write("="*80 + "\n\n")
        
        # 1. 成功率统计
        f.write("1. 拟合成功率\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<15} {'总样本数':<10} {'成功数':<10} {'成功率':<10}\n")
        f.write("-"*80 + "\n")
        
        for model_key, model_name in MODELS.items():
            if model_key not in all_params or not all_params[model_key]:
                continue
            
            total = len(all_params[model_key])
            success = sum(1 for p in all_params[model_key] if p['fit_success'])
            rate = success / total * 100 if total > 0 else 0
            
            f.write(f"{model_name:<15} {total:<10} {success:<10} {rate:>8.2f}%\n")
        
        f.write("\n\n")
        
        # 2. R²统计
        f.write("2. R² (决定系数) 统计\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<15} {'均值':<10} {'标准差':<10} {'最小值':<10} "
                f"{'中位数':<10} {'最大值':<10}\n")
        f.write("-"*80 + "\n")
        
        r2_comparison = {}
        for model_key, model_name in MODELS.items():
            if model_key not in all_params or not all_params[model_key]:
                continue
            
            r2_values = [p['r_squared'] for p in all_params[model_key] 
                        if p['fit_success'] and not np.isnan(p['r_squared'])]
            
            stats = compute_statistics(r2_values)
            r2_comparison[model_key] = stats
            
            f.write(f"{model_name:<15} "
                   f"{stats['mean']:>9.6f} "
                   f"{stats['std']:>9.6f} "
                   f"{stats['min']:>9.6f} "
                   f"{stats['median']:>9.6f} "
                   f"{stats['max']:>9.6f}\n")
        
        # R²最佳模型
        best_r2_model = max(r2_comparison.items(), 
                           key=lambda x: x[1]['mean'])[0]
        f.write(f"\n最佳模型（R²均值）: {MODELS[best_r2_model]}\n")
        
        f.write("\n\n")
        
        # 3. AIC统计（越小越好）
        f.write("3. AIC (赤池信息准则) 统计 [越小越好]\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<15} {'均值':<12} {'标准差':<12} {'最小值':<12} "
                f"{'中位数':<12} {'最大值':<12}\n")
        f.write("-"*80 + "\n")
        
        aic_comparison = {}
        for model_key, model_name in MODELS.items():
            if model_key not in all_params or not all_params[model_key]:
                continue
            
            aic_values = [p['aic'] for p in all_params[model_key] 
                         if p['fit_success'] and not np.isnan(p['aic'])]
            
            stats = compute_statistics(aic_values)
            aic_comparison[model_key] = stats
            
            f.write(f"{model_name:<15} "
                   f"{stats['mean']:>11.2f} "
                   f"{stats['std']:>11.2f} "
                   f"{stats['min']:>11.2f} "
                   f"{stats['median']:>11.2f} "
                   f"{stats['max']:>11.2f}\n")
        
        # AIC最佳模型
        best_aic_model = min(aic_comparison.items(), 
                            key=lambda x: x[1]['mean'])[0]
        f.write(f"\n最佳模型（AIC均值）: {MODELS[best_aic_model]}\n")
        
        f.write("\n\n")
        
        # 4. RMSE统计
        f.write("4. RMSE (均方根误差) 统计\n")
        f.write("-"*80 + "\n")
        f.write(f"{'模型':<15} {'均值':<12} {'标准差':<12} {'最小值':<12} "
                f"{'中位数':<12} {'最大值':<12}\n")
        f.write("-"*80 + "\n")
        
        rmse_comparison = {}
        for model_key, model_name in MODELS.items():
            if model_key not in all_params or not all_params[model_key]:
                continue
            
            rmse_values = [p['rmse'] for p in all_params[model_key] 
                          if p['fit_success'] and not np.isnan(p['rmse'])]
            
            stats = compute_statistics(rmse_values)
            rmse_comparison[model_key] = stats
            
            f.write(f"{model_name:<15} "
                   f"{stats['mean']:>11.4f} "
                   f"{stats['std']:>11.4f} "
                   f"{stats['min']:>11.4f} "
                   f"{stats['median']:>11.4f} "
                   f"{stats['max']:>11.4f}\n")
        
        f.write("\n\n")
        
        # 5. 参数统计
        f.write("5. 拟合参数统计\n")
        f.write("-"*80 + "\n")
        
        for param in ['a', 'b', 'c']:
            f.write(f"\n参数 {param.upper()}:\n")
            f.write(f"{'模型':<15} {'均值':<12} {'标准差':<12} {'最小值':<12} "
                   f"{'中位数':<12} {'最大值':<12}\n")
            f.write("-"*80 + "\n")
            
            for model_key, model_name in MODELS.items():
                if model_key not in all_params or not all_params[model_key]:
                    continue
                
                param_values = [p[param] for p in all_params[model_key] 
                              if p['fit_success'] and not np.isnan(p[param])]
                
                stats = compute_statistics(param_values)
                
                f.write(f"{model_name:<15} "
                       f"{stats['mean']:>11.4e} "
                       f"{stats['std']:>11.4e} "
                       f"{stats['min']:>11.4e} "
                       f"{stats['median']:>11.4e} "
                       f"{stats['max']:>11.4e}\n")
        
        f.write("\n\n")
        
        # 6. 综合评价
        f.write("6. 综合评价\n")
        f.write("-"*80 + "\n")
        f.write("基于不同指标的最佳模型:\n\n")
        
        f.write(f"  • 成功率最高: {max([(k, sum(1 for p in v if p['fit_success'])/len(v)) for k,v in all_params.items()], key=lambda x: x[1])[0].capitalize()}\n")
        f.write(f"  • R²均值最高: {MODELS[best_r2_model]}\n")
        f.write(f"  • AIC均值最低: {MODELS[best_aic_model]}\n")
        
        # 计算综合得分（标准化后加权平均）
        f.write(f"\n综合得分 (标准化后): 成功率×0.3 + R²×0.4 + (1-标准化AIC)×0.3\n\n")
        
        scores = {}
        success_rates = {k: sum(1 for p in v if p['fit_success'])/len(v) 
                        for k, v in all_params.items()}
        r2_means = {k: v['mean'] for k, v in r2_comparison.items()}
        aic_means = {k: v['mean'] for k, v in aic_comparison.items()}
        
        # 标准化
        max_aic = max(aic_means.values())
        min_aic = min(aic_means.values())
        
        for model_key in MODELS.keys():
            if model_key not in all_params:
                continue
            
            norm_aic = 1.0 - (aic_means[model_key] - min_aic) / (max_aic - min_aic) if max_aic != min_aic else 1.0
            score = 0.3 * success_rates[model_key] + 0.4 * r2_means[model_key] + 0.3 * norm_aic
            scores[model_key] = score
            
            f.write(f"  {MODELS[model_key]:<15}: {score:.6f}\n")
        
        best_overall = max(scores.items(), key=lambda x: x[1])[0]
        f.write(f"\n综合最佳模型: {MODELS[best_overall]}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n统计摘要已保存到: {output_file}")
    return output_file


if __name__ == '__main__':
    # 查找file_base
    file_base = None
    if len(sys.argv) > 1:
        file_base = sys.argv[1]
    else:
        patterns = glob.glob("*_fit_params_yeoh.csv")
        if patterns:
            match = re.match(r'(.+?)_fit_params_yeoh\.csv', patterns[0])
            if match:
                file_base = match.group(1)
    
    if not file_base:
        print("错误: 无法自动检测file_base，请手动指定")
        print("用法: python summarize_model_comparison.py file_base")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"多模型拟合效果统计分析")
    print(f"{'='*70}")
    print(f"File base: {file_base}\n")
    
    # 读取所有模型的拟合参数
    print(f"读取拟合参数...")
    all_params = {}
    for model_key, model_name in MODELS.items():
        params = read_fit_params(file_base, model_key)
        if params:
            all_params[model_key] = params
            print(f"  {model_name:15s}: {len(params)} 样本")
    
    if not all_params:
        print("\n错误: 没有找到任何拟合参数文件")
        sys.exit(1)
    
    # 生成统计摘要
    print(f"\n生成统计摘要...")
    output_file = generate_summary(all_params, file_base)
    
    print(f"\n{'='*70}")
    print(f"完成!")
    print(f"{'='*70}\n")
