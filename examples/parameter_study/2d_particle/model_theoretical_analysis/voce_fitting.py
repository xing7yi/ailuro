import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import csv
csv_file = "../linear_law/main_lh_sampler_out_runner006.csv"

# 使用csv模块读取数据（支持列名）
_disp = []
_force = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            disp = float(row.get('disp_abs', 0))
            force = float(row.get('force', 0))
            
            # 过滤有效数据点
            if abs(disp) > 1e-6 and force > 0:
                _disp.append(abs(disp))
                _force.append(force)
        except (ValueError, KeyError):
            continue

_disp = np.array(_disp)
_force = np.array(_force)

initial_height = 1.0  # 初始高度，单位与位移一致 (mm)
contact_area = np.pi   # 接触面积，单位与力一致 (mm²)
strain = _disp / initial_height
stress = _force / contact_area
# 转换为对数应变和真应力
_x = -np.log(1-strain)  # 对数应变
_y = (1 - strain) * stress # 真应力


# --------------------------
# 定义 Voce 模型（包含弹性 + 塑性段）
# --------------------------
def voce_model(eps, E, sigma0, R0 , R_inf):
    eps_y = sigma0 / E  # 屈服点
    sigma = np.zeros_like(eps)
    elastic = eps <= eps_y
    plastic = eps > eps_y
    
    # 弹性阶段
    sigma[elastic] = E * eps[elastic]
    
    # 塑性阶段（平滑连接）
    eps_p = eps[plastic] - eps_y
    b = (E - R0)/R_inf
    sigma[plastic] = sigma0 + R0 * eps_p + R_inf * (1 - np.exp(-b * eps_p))
    
    return sigma

def bilinear_model(eps, E, sigma0, H):
    eps_y = sigma0 / E  # 屈服点
    sigma = np.zeros_like(eps)
    elastic = eps <= eps_y
    plastic = eps > eps_y
    
    # 弹性阶段
    sigma[elastic] = E * eps[elastic]
    
    # 塑性阶段（线性硬化）
    eps_p = eps[plastic] - eps_y
    sigma[plastic] = sigma0 + H * eps_p
    
    return sigma

# --------------------------
# 生成或加载实验数据（此处用模拟数据）
# --------------------------
# E_true, sigma0_true, sigmas_true, C_true = 200e3, 250, 400, 50
# eps = np.linspace(0, 0.02, 200)
# sigma_true = voce_model(eps, E_true, sigma0_true, sigmas_true, C_true)
# sigma_data = sigma_true + np.random.normal(0, 5, len(eps))  # 添加少量噪声

eps = _x
sigma_data = _y

# --------------------------
# 拟合参数
# --------------------------
# p0 = [150e3, 200, 350, 30]  # 初始猜测
# bounds = ([1e4, 10, 50, 1], [1e6, 1000, 2000, 500])  # 参数范围
popt, pcov = curve_fit(voce_model, eps, sigma_data
        # , p0=p0, bounds=bounds
        )

E_fit, sigma0_fit, R0_fit, R_inf_fit = popt
# 计算拟合曲线
sigma_fit = voce_model(eps, *popt)
print(f"拟合结果:")
print(f"E = {E_fit:.1f} MPa")
print(f"sigma0 = {sigma0_fit:.2f} MPa")
print(f"R0 = {R0_fit:.2f} MPa")
print(f"R_inf = {R_inf_fit:.2f} MPa")
# calculate b
b_fit = (E_fit - R0_fit) / R_inf_fit
print(f"b = {b_fit:.2f}")

# r_square
residuals = sigma_data - sigma_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((sigma_data - np.mean(sigma_data))**2)
r_square = 1 - (ss_res / ss_tot)
print(f"R^2 = {r_square:.4f}")




# Bilinear 模型拟合（可选）
# p0_bi = [150e3, 200, 1000]  # 初始猜测
# bounds_bi = ([1e4, 10, 100], [1e6, 1000, 5000])  # 参数范围
popt_bi, pcov_bi = curve_fit(bilinear_model, eps, sigma_data)
E_bi, sigma0_bi, H_bi = popt_bi
print(f"\nBilinear 拟合结果:")
print(f"E = {E_bi:.1f} MPa")
print(f"sigma0 = {sigma0_bi:.2f} MPa")
print(f"H = {H_bi:.2f} MPa")
sigma_bi_fit = bilinear_model(eps, *popt_bi)

# --------------------------
# 可视化对比
# --------------------------
plt.figure(figsize=(6,4))
plt.plot(_x, _y, '--o', lw=2.5, ms=9, mew=2, mfc='white', c='C0')  # MEASURED DATA
plt.plot(eps, sigma_fit, 'C1', label='Voce Fit')
plt.plot(eps, sigma_bi_fit, 'C2--', label='Bilinear Fit')
# plt.plot(eps, sigma_true, 'k--', label='真实曲线', linewidth=1)
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("voce_fitting.pdf")
