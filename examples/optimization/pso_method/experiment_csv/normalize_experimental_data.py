#!/usr/bin/env python3
"""
将实验数据的力和位移转换为半径归一化的无量纲值

归一化方法：
- 归一化应力：force_norm_MPa = F / (π * R²)
- 压缩比：cmpr_ratio = u / D

其中：
- F: 力 (mN)
- R: 压头半径 (mm)
- u: 位移 (μm)
- D: 试样直径 (mm，从文件名中提取）

文件名格式：...D<直径>_...
例如：316L_D21.0_CL_0001.csv → 直径 = 21.0 mm
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import re


def extract_diameter_from_filename(filename):
    """
    从文件名中提取直径
    文件名格式：...D<直径>_...
    例如：316L_D21.0_CL_0001.csv → 21.0
    
    Args:
        filename: 文件名（str 或 Path）
    
    Returns:
        diameter: float, 直径 (mm)；如果未找到返回 None
    """
    filename = str(filename)
    
    # 匹配 D 后面跟数字（可能包含小数点）直到下一个下划线
    match = re.search(r'D(\d+\.?\d*)_', filename)
    
    if match:
        return 1e-3*float(match.group(1))
    else:
        return None


def normalize_data(input_file, output_file, diameter_mm, 
                   col_disp='disp', col_force='force'):
    """
    将实验数据归一化
    
    Args:
        input_file: 输入 CSV 文件路径
        output_file: 输出 CSV 文件路径
        radius_mm: 压头半径 (mm)
        diameter_mm: 试样直径 (mm)
        col_disp: 位移列名
        col_force: 力列名
    
    Returns:
        success: bool, 是否成功
    """
    
    # 读取数据
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"  错误: 读取文件失败 - {e}")
        return False
    
    # 检查列是否存在
    if col_disp not in df.columns or col_force not in df.columns:
        print(f"  错误: 数据缺少必要的列")
        print(f"    期望列: {col_disp}, {col_force}")
        print(f"    实际列: {list(df.columns)}")
        return False
    
    # 提取数据
    disp_um = df[col_disp].values  # 位移 (μm)
    force_mN = df[col_force].values  # 力 (mN)
    
    # 单位转换
    disp_mm = disp_um / 1000.0  # μm -> mm
    force_N = force_mN / 1000.0  # mN -> N
    
    # 计算归一化值
    # force_norm_MPa = F / (π * R²)
    area_mm2 = np.pi * (diameter_mm/2)**2
    force_norm_MPa = force_N / area_mm2
    
    # 压缩比 = u / D
    cmpr_ratio = disp_mm / diameter_mm
    
    # 创建新的 DataFrame
    df_normalized = pd.DataFrame({
        col_disp: disp_um,  # 保留原始位移 (μm)
        col_force: force_mN,  # 保留原始力 (mN)
        'force_norm_MPa': force_norm_MPa,  # 归一化应力 (MPa)
        'cmpr_ratio': cmpr_ratio  # 压缩比 (无量纲)
    })
    
    # 如果原始数据有其他列，也保留
    for col in df.columns:
        if col not in [col_disp, col_force]:
            df_normalized[col] = df[col]
    
    # 保存到文件
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_normalized.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"  错误: 保存文件失败 - {e}")
        return False


def batch_normalize(input_pattern, output_dir,
                    col_disp='disp', col_force='force', suffix='_normalized'):
    """
    批量归一化多个文件（自动从文件名提取直径）
    
    Args:
        input_pattern: 输入文件模式（支持通配符）
        output_dir: 输出目录
        col_disp: 位移列名
        col_force: 力列名
        suffix: 输出文件后缀
    
    Returns:
        results: dict, 处理结果统计
    """
    
    # 查找匹配的文件
    input_files = glob(input_pattern)
    
    if len(input_files) == 0:
        print(f"未找到匹配的文件: {input_pattern}")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print("="*80)
    print(f"批量归一化工具")
    print("="*80)
    print(f"找到 {len(input_files)} 个文件")
    print(f"试样直径: 从文件名自动提取（格式 D<直径>_）")
    print("-"*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, input_file in enumerate(input_files, 1):
        input_path = Path(input_file)
        print(f"\n[{i}/{len(input_files)}] 处理: {input_path.name}")
        
        # 从文件名提取直径
        diameter_mm = extract_diameter_from_filename(input_path.name)
        
        if diameter_mm is None:
            print(f"  ✗ 跳过：无法从文件名提取直径")
            skip_count += 1
            continue
        
        print(f"  提取直径: {diameter_mm} mm")
        
        output_file = output_path / f"{input_path.stem}{suffix}.csv"
        
        success = normalize_data(
            input_file=input_file,
            output_file=output_file,
            diameter_mm=diameter_mm,
            col_disp=col_disp,
            col_force=col_force
        )
        
        if success:
            print(f"  ✓ 已保存到: {output_file.name}")
            success_count += 1
        else:
            print(f"  ✗ 处理失败")
            fail_count += 1
    
    print("\n" + "="*80)
    print(f"批量处理完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  跳过: {skip_count}")
    print(f"  输出目录: {output_path}")
    print("="*80)
    
    return {
        'success': success_count,
        'failed': fail_count,
        'skipped': skip_count
    }


if __name__ == "__main__":
    # 默认配置
    INPUT_PATTERN = "*.csv"  # 输入文件模式
    OUTPUT_DIR = "normalized"  # 输出目录
    COL_DISP = 'disp_um'  # 位移列名
    COL_FORCE = 'force_mN'  # 力列名
    SUFFIX = '_norm'  # 输出文件后缀
    
    # 执行批量归一化
    batch_normalize(
        input_pattern=INPUT_PATTERN,
        output_dir=OUTPUT_DIR,
        col_disp=COL_DISP,
        col_force=COL_FORCE,
        suffix=SUFFIX
    )
