#!/usr/bin/env python3
"""
简单的力-位移曲线绘制程序
绘制particle_friction_plastic_voce_csv.csv中的力-位移关系
"""

import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file = 'particle_friction_plastic_voce_csv.csv'
data = pd.read_csv(csv_file)

# 提取位移和力数据
displacement = data['disp_abs']
force = data['force']

# 创建图形
plt.figure(figsize=(5, 4))
plt.plot(displacement, force, 'b-', linewidth=2, label='Force-Displacement')

# 设置标签和标题
plt.xlabel('Displacement (mm)', fontsize=12)
plt.ylabel('Force (N)', fontsize=12)
plt.title('Force-Displacement Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('force_displacement_curve.png', dpi=300, bbox_inches='tight')
print(f"图形已保存为 'force_displacement_curve.png'")

# 显示图形
