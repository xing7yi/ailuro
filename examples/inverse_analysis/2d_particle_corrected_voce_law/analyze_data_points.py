#!/usr/bin/env python3
"""
分析逆向分析中数据点配置的合理性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取测量数据
measurement_file = Path("ObjectiveCSV/extracted_disp_force.csv")
data = pd.read_csv(measurement_file)

print("=" * 80)
print("当前数据点配置分析")
print("=" * 80)

print(f"\n数据点数量: {len(data)} 个")
print(f"\n数据详情:")
print(data.to_string(index=False))

# 计算时间间隔
time_intervals = np.diff(data['time'].values)
print(f"\n时间间隔分析:")
print(f"  平均间隔: {np.mean(time_intervals):.2f}")
print(f"  最小间隔: {np.min(time_intervals):.2f}")
print(f"  最大间隔: {np.max(time_intervals):.2f}")
print(f"  间隔标准差: {np.std(time_intervals):.2f}")

# 分析力-位移曲线的非线性特征
forces = data['force'].values
disps = -data['disp_y'].values  # 负值表示压缩

# 计算刚度变化 (ΔF/Δu)
stiffnesses = np.diff(forces) / np.diff(disps)
print(f"\n刚度变化分析 (表征塑性行为):")
for i, k in enumerate(stiffnesses):
    print(f"  段 {i+1} ({data['time'].iloc[i]:.1f}->{data['time'].iloc[i+1]:.1f}): K = {k:.1f} N/mm")

stiffness_change = (stiffnesses.max() - stiffnesses.min()) / stiffnesses.max() * 100
print(f"  刚度变化率: {stiffness_change:.1f}%")

# 计算二阶导数（曲率）来识别非线性区域
if len(forces) >= 3:
    curvatures = np.diff(stiffnesses) / np.diff(disps[:-1])
    print(f"\n曲率分析 (二阶导数，识别非线性变化剧烈区域):")
    for i, c in enumerate(curvatures):
        print(f"  段 {i+1}: 曲率 = {c:.1f} N/mm²")

# 读取优化结果
opt_result = pd.read_csv("main_csv_forward.csv")
print(f"\n优化收敛情况:")
print(f"  初始目标函数值: {opt_result['OptimizationReporter/objective_value'].iloc[0]:.6f}")
print(f"  最终目标函数值: {opt_result['OptimizationReporter/objective_value'].iloc[-1]:.6f}")
print(f"  优化迭代次数: {len(opt_result)}")

# 相对误差分析
rel_error = np.sqrt(opt_result['OptimizationReporter/objective_value'].iloc[-1]) / np.mean(disps) * 100
print(f"  位移相对误差: {rel_error:.2f}%")

print("\n" + "=" * 80)
print("数据点配置建议")
print("=" * 80)

# 基于塑性行为分析给出建议
if stiffness_change > 50:
    print("\n✓ 材料呈现显著塑性硬化 (刚度变化 > 50%)")
    print("  → 建议增加数据点以更好地捕捉非线性行为")
else:
    print("\n✓ 材料刚度变化较小 (刚度变化 < 50%)")
    print("  → 当前数据点可能已足够")

# 基于时间间隔均匀性分析
interval_uniformity = np.std(time_intervals) / np.mean(time_intervals)
if interval_uniformity > 0.3:
    print(f"\n✓ 时间间隔不均匀 (变异系数 = {interval_uniformity:.2f})")
    print("  → 建议使用更均匀的时间间隔分布")
else:
    print(f"\n✓ 时间间隔相对均匀 (变异系数 = {interval_uniformity:.2f})")

# 基于优化收敛情况分析
if opt_result['OptimizationReporter/objective_value'].iloc[-1] > 1e-3:
    print(f"\n⚠ 目标函数值较大 (> 1e-3)")
    print("  → 可能需要增加数据点或调整优化算法")
elif opt_result['OptimizationReporter/objective_value'].iloc[-1] < 1e-5:
    print(f"\n✓ 目标函数收敛良好 (< 1e-5)")
    print("  → 当前数据点配置可能已经合适")
else:
    print(f"\n✓ 目标函数值中等 (1e-5 ~ 1e-3)")
    print("  → 增加数据点可能提升精度")

# 建议的数据点配置
print("\n推荐的数据点配置方案:")
print("\n方案1: 保守增加 (7-8个点)")
print("  - 在塑性屈服区域(初期)增加1-2个点")
print("  - 在硬化过渡区域(中期)增加1个点")
print("  - 保持当前终点")
print("  - 优点: 计算成本增加有限 (~40%)")
print("  - 适用: 当前结果已较好，需要微调精度")

print("\n方案2: 显著增加 (10-12个点)")
print("  - 时间均匀分布或位移均匀分布")
print("  - 优点: 更精确捕捉整个加载过程")
print("  - 缺点: 计算成本翻倍")
print("  - 适用: 需要高精度参数识别或材料模型验证")

print("\n方案3: 非均匀分布 (8-10个点)")
print("  - 在曲率大的区域加密数据点")
print("  - 在线性区域稀疏采样")
print("  - 优点: 最优的信息/成本比")
print("  - 适用: 已知材料大致行为特征")

# 可视化当前数据点分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 力-位移曲线
ax1.plot(disps, forces, 'o-', linewidth=2, markersize=10, label='Measured Points')
ax1.set_xlabel('Displacement (mm)', fontsize=12)
ax1.set_ylabel('Force (N)', fontsize=12)
ax1.set_title(f'Current Data Point Distribution ({len(data)} points)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 添加时间标注
for i, (d, f, t) in enumerate(zip(disps, forces, data['time'])):
    ax1.annotate(f't={t:.1f}', (d, f), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

# 刚度变化
time_mid = (data['time'].values[:-1] + data['time'].values[1:]) / 2
ax2.plot(time_mid, stiffnesses, 's-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Stiffness K (N/mm)', fontsize=12)
ax2.set_title('Stiffness Evolution (Plastic Hardening Feature)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_points_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存: data_points_analysis.png")

print("\n" + "=" * 80)
