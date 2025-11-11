import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import csv

# ==== 输入实验数据 ====
# 假设你已有实验数据 e (strain), s (stress)
# 示例数据（请替换成你的真实数据）
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

_x = -np.log(1 - strain)
_y = (1 - strain) * stress

# ==== 定义 Voce + 弹性耦合模型 ====
def voce_implicit(sigma, eps, E, sigma_y, sigma_s, eps0):
    """Voce模型的隐式方程残差"""
    return sigma - (sigma_s - (sigma_s - sigma_y) *
                    np.exp(-(eps - sigma / E) / eps0))

def sigma_from_eps(eps, E, sigma_y, sigma_s, eps0):
    """给定应变eps, 用牛顿迭代求解应力sigma"""
    sigma = E * eps  # 初始猜测：纯弹性
    for _ in range(30):
        f = voce_implicit(sigma, eps, E, sigma_y, sigma_s, eps0)
        df = 1 - (sigma_s - sigma_y) * np.exp(-(eps - sigma / E) / eps0) * (1 / (E * eps0))
        sigma -= f / df
    return sigma

def model_stress(eps, E, sigma_y, sigma_s, eps0):
    return np.array([sigma_from_eps(e, E, sigma_y, sigma_s, eps0) for e in eps])

# ==== 残差函数 ====
def residual(params, eps, s_exp):
    E, sigma_y, sigma_s, eps0 = params
    s_pred = model_stress(eps, E, sigma_y, sigma_s, eps0)
    return s_pred - s_exp

# ==== 初始猜测 ====
E0 = 200000.0   # MPa
sigma_y0 = 1000.0
sigma_s0 = 1300.0
eps0_0 = 0.15
p0 = [E0, sigma_y0, sigma_s0, eps0_0]

# ==== 拟合 ====
result = least_squares(residual, p0, args=(_x, _y), bounds=(0, np.inf))
E, sigma_y, sigma_s, eps0 = result.x

print(f"Fitted parameters:")
print(f"E = {E:.1f} MPa")
print(f"σ_y = {sigma_y:.1f} MPa")
print(f"σ_s = {sigma_s:.1f} MPa")
print(f"ε₀ = {eps0:.4f}")

# ==== 绘图比较 ====
y_fit = model_stress(_x, E, sigma_y, sigma_s, eps0)

plt.figure(figsize=(7,5))
plt.plot(_x, _y, 'bo', label='Experimental')
plt.plot(_x, y_fit, 'r-', lw=2, label='Voce fit')
plt.xlabel("Strain (log)")
plt.ylabel("Stress (MPa)")
plt.title(f"Voce Fit: σ_y={sigma_y:.1f}, σ_s={sigma_s:.1f}, E={E/1000:.1f} GPa")
plt.legend()
plt.grid(True)
plt.savefig("voce_fit_implicit.pdf")
