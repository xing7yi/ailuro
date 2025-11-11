# RAMBERG-OSGOOD CURVE FIT OF STRESS-STRAIN DATA
# AUTHOR: Frédéric Martin - Quadco Engineering
# DATE: 17/04/2020
# WEBSITE: https://www.quadco.engineering


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

# E = _y[1] / _x[1]  # CALCULATE YOUNG'S MODULUS
E = 3000

# MATERIAL DATA
# strain = np.array([0, 500/210000, 0.003749347, 0.005234888, 0.007631515, 0.011446096, 0.017384888,
#                    0.026405375, 0.039775799])
# stress = np.array([0, 500, 550, 600, 660, 720, 780, 840, 900])

# E = stress[1] / strain[1]  # CALCULATE YOUNG'S MODULUS


# FITTING RAMBERG-OSGOOD EQUATION
def test_func(x, K, n):
    return x / E + np.power(x / K, n)

def test_func_linear(x, k1, k2):
    return x / k1 +  x / k2

# p0 = [E, 500, 5]  # Initial estimate for [E, K, n]
c, cov = curve_fit(test_func_linear, _y, _x)

# PRINT CURVE FIT PARAMETERS
print()
print('-' * 28)
print(' Ramberg-Osgood parameters')
print('-' * 28)
# print(f' E = {c[0]} \n K = {c[1]} \n n = {c[2]}')
print(f' K = {c[0]} \n n = {c[1]}')
print('-' * 28)


# CREATE DATA ARRAY FOR THE FITTED CURVE
e = test_func_linear(np.linspace(0, _y[-1]), c[0], c[1])  # STRAIN
s = np.linspace(0, _y[-1])  # STRESS


# PLOT DATA AND FITTED CURVE
plt.figure(1, figsize=(8, 6))
plt.plot(e, s, lw=3, c='C1')  # PLOT FITTED CURVE
plt.plot(_x, _y, '--o', lw=2.5, ms=9, mew=2, mfc='white', c='C0')  # MEASURED DATA
plt.grid()
plt.xlabel(r'strain $\epsilon$ [mm/mm]')
plt.ylabel(r'stress $\sigma$ [MPa]')
plt.title(f'Ramberg-Osgood curve fit (K = {np.round(c[0], 2)} | n = {np.round(c[1], 3)})\n')
plt.legend(['Ramberg-Osgood fit', 'Measured stress-strain data'])
plt.plot()
plt.savefig('ramberg-osgood-en.png', dpi=600)


plt.figure(2, figsize=(12, 8))
plt.plot(e, s, lw=3, c='C1')  # PLOT FITTED CURVE
plt.plot(_x, _y, '--o', lw=2.5, ms=9, mew=2, mfc='white', c='C0')  # MEASURED DATA
plt.grid()
plt.xlabel(r'rek $\epsilon$ [mm/mm]')
plt.ylabel(r'spanning $\sigma$ [MPa]')
plt.title(f'Ramberg-Osgood curve fit (K = {np.round(c[0], 2)} | n = {np.round(c[1], 3)})\n')
plt.legend(['Ramberg-Osgood fit', 'Spanning-rek meetdata'])
plt.plot()
plt.savefig('ramberg-osgood-nl.pdf')

plt.show()
