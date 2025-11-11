#!/usr/bin/env python3
"""
快速测试PSO优化器 - 使用命令行参数覆盖

测试单次MOOSE评估是否工作
"""

import subprocess
from pathlib import Path

def test_single_evaluation():
    """测试单次评估"""
    
    print("="*60)
    print("测试单次MOOSE评估（命令行参数覆盖）")
    print("="*60)
    
    # 测试参数（使用输入文件中的顶层变量 p0, p1, p2）
    test_params = {
        'p0': 450,   # yield_stress
        'p1': 850,   # hardening_constant
        'p2': 320    # q
    }
    
    # 检查输入文件是否存在
    input_file = Path("particle_friction_plastic_voce.i")
    if not input_file.exists():
        print(f"\n✗ 错误: 找不到输入文件 {input_file}")
        print(f"   当前目录: {Path.cwd()}")
        print(f"   可用文件:")
        for f in Path.cwd().glob("*.i"):
            print(f"     - {f.name}")
        return
    
    # 构建命令（使用顶层变量，更简洁）
    cmd = [
        "ailuro-opt",
        "-i", str(input_file),
        f"p0={test_params['p0']}",
        f"p1={test_params['p1']}",
        f"p2={test_params['p2']}"
    ]
    
    print(f"\n测试参数:")
    for name, value in test_params.items():
        print(f"  {name} = {value}")
    
    print(f"\n执行命令:")
    print(" ".join(cmd))
    
    print(f"\n运行中...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ 模拟成功完成！")
            
            # 尝试读取输出（查找所有CSV文件）
            csv_files = list(Path.cwd().glob("*.csv"))
            if csv_files:
                import pandas as pd
                print(f"\n  找到输出文件:")
                for csv_file in csv_files:
                    print(f"    - {csv_file.name}")
                
                # 使用第一个CSV文件
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)
                
                print(f"\n  分析 {csv_file.name}:")
                print(f"    行数: {len(df)}")
                print(f"    列名: {list(df.columns)}")
                
                # 查找objective相关列
                obj_cols = [col for col in df.columns if 'objective' in col.lower()]
                if obj_cols:
                    obj_value = df[obj_cols[0]].iloc[-1]
                    print(f"\n  ✓ 目标函数值: {obj_value:.6e} (列: {obj_cols[0]})")
                else:
                    print(f"\n  ⚠ 未找到包含'objective'的列")
            else:
                print(f"\n  ⚠ 未找到CSV输出文件")
        else:
            print(f"\n✗ 模拟失败 (返回码: {result.returncode})")
            print(f"\n标准错误输出:")
            print(result.stderr[-1000:])  # 最后1000字符
            
    except subprocess.TimeoutExpired:
        print("\n✗ 模拟超时")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")

if __name__ == "__main__":
    test_single_evaluation()
