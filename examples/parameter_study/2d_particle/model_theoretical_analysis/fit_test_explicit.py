import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# ==== 输入实验数据 ====
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

# _x = -np.log(1 - _x)
# _y = np.log(_y)
# _y = np.log(_y)
# _y =  1/ np.log(stress)
# 计算导数
# 使用移动窗口线性回归来平滑地计算导数
window_size = 21  # 窗口大小，必须是奇数
_dy = np.zeros_like(_y)
half_window = window_size // 2

for i in range(len(_x)):
    # 定义窗口边界
    start = max(0, i - half_window)
    end = min(len(_x), i + half_window + 1)
    
    # 提取窗口内的数据
    x_window = _x[start:end]
    y_window = _y[start:end]
    
    # 对窗口内的数据进行线性拟合 (y = mx + c)
    # m (斜率) 就是该点的导数估计值
    if len(x_window) > 1: # 至少需要两个点才能拟合
        A = np.vstack([x_window, np.ones(len(x_window))]).T
        m, c = np.linalg.lstsq(A, y_window, rcond=None)[0]
        _dy[i] = m
    elif len(x_window) == 1: # 如果窗口只有一个点，无法计算斜率
        # 使用前一个或后一个点的导数，或者用0填充
        if i > 0:
            _dy[i] = _dy[i-1]
        else:
            _dy[i] = 0 # 无法确定第一个点的导数

# ==== 定义显式Voce模型 ====


# def voce_explicit_simple(eps, sigma_y, sigma_s, eps0):
#     """
#     简化的显式Voce模型（不考虑弹性阶段）
#     直接使用Voce公式
    
#     σ = σ_s - (σ_s - σ_y) * exp(-ε/ε₀)
#     """
#     return sigma_s - (sigma_s - sigma_y) * np.exp(-eps / eps0)

def power_law_hardening(eps, sigma_y, K, n):
    """
    幂律硬化模型（另一种显式模型）
    
    σ = σ_y + K * ε^n
    """
    return sigma_y + K * np.power(eps, n)


# ==== 选择要使用的模型 ====
print("选择拟合模型:")
model_choice = 0

# ==== 拟合 ====

# if model_choice == 2:
#     # 简化Voce模型
#     p0 = [500, 1500, 0.1]  # 初始猜测: sigma_y, sigma_s, eps0
#     bounds = ([100, 500, 0.01], [2000, 5000, 1.0])
    
#     popt, pcov = curve_fit(voce_explicit_simple, _x, _y, p0=p0, bounds=bounds, maxfev=10000)
#     sigma_y, sigma_s, eps0 = popt
    
#     print(f"\n简化Voce模型拟合参数:")
#     print(f"σ_y = {sigma_y:.1f} MPa")
#     print(f"σ_s = {sigma_s:.1f} MPa")
#     print(f"ε₀ = {eps0:.4f}")
    
#     y_fit = voce_explicit_simple(_x, sigma_y, sigma_s, eps0)
#     model_name = "Simple Voce"
#     param_str = f"σ_y={sigma_y:.1f}, σ_s={sigma_s:.1f}, ε₀={eps0:.4f}"
if model_choice == 4:
    # 幂律硬化模型
    p0 = [500, 1000, 0.5]  # 初始猜测: sigma_y, K, n
    bounds = ([100, 0, 0.1], [2000, 10000, 2.0])
    
    popt, pcov = curve_fit(power_law_hardening, _x, _y, p0=p0, bounds=bounds, maxfev=10000)
    sigma_y, K, n = popt
    
    print(f"\n幂律硬化模型拟合参数:")
    print(f"σ_y = {sigma_y:.1f} MPa")
    print(f"K = {K:.1f} MPa")
    print(f"n = {n:.4f}")
    
    y_fit = power_law_hardening(_x, sigma_y, K, n)
    model_name = "Power Law"
    param_str = f"σ_y={sigma_y:.1f}, K={K:.1f}, n={n:.4f}"

# ==== 计算拟合质量 ====
# residuals = _y - y_fit
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((_y - np.mean(_y))**2)
# r_squared = 1 - (ss_res / ss_tot)
# rmse = np.sqrt(np.mean(residuals**2))

# print(f"\n拟合质量:")
# print(f"R² = {r_squared:.6f}")
# print(f"RMSE = {rmse:.4f} MPa")

# ==== 绘图比较 ====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：拟合结果（双y轴）
# 主y轴：真实应力
ax1.plot(_x, _y, 'bo', markersize=4, alpha=0.6, label='Experimental Data')
# ax1.plot(_x, y_fit, 'r-', linewidth=2.5, label=f'{model_name} Fit')
ax1.set_xlabel("Logarithmic Strain", fontsize=12)
ax1.set_ylabel("True Stress (MPa)", fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

# 副y轴：导数
ax1_twin = ax1.twinx()
ax1_twin.plot(_x, _dy, 'r--', linewidth=1.5, label='Derivative dσ/dε')
ax1_twin.set_ylabel("Derivative dσ/dε (MPa)", fontsize=12, color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')

# ax1.set_title(f"{model_name} Fit\n{param_str}\nR²={r_squared:.6f}", fontsize=13)

# 右图：残差分析
# ax2.plot(_x, residuals, 'go', markersize=4, alpha=0.6)
# ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
# ax2.axhline(y=2*rmse, color='orange', linestyle=':', linewidth=1, label=f'±2×RMSE')
# ax2.axhline(y=-2*rmse, color='orange', linestyle=':', linewidth=1)
# ax2.set_xlabel("Logarithmic Strain", fontsize=12)
# ax2.set_ylabel("Residuals (MPa)", fontsize=12)
# ax2.set_title(f"Residual Analysis\nRMSE={rmse:.4f} MPa", fontsize=13)

# ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fit_explicit.pdf")
print(f"\n图像已保存: fit_test_explicit.pdf")

# ==== 保存拟合结果到CSV ====
# output_csv = "fit_explicit_results.csv"
# with open(output_csv, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['strain', 'stress_exp', 'stress_fit', 'residual'])
#     for i in range(len(_x)):
#         writer.writerow([_x[i], _y[i], y_fit[i], residuals[i]])

# print(f"拟合数据已保存: {output_csv}")

plt.show()
