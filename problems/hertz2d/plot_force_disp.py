import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def E_eq(E1, nu1, E2, nu2):
    return 1 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)

# Tatara理论
def tatara_disp(f,E_eq,R):
    v = 0.3
    a = (3/4 * f * R / E_eq ) ** (1/3)
    term1 = (3/4 * f / (E_eq * np.sqrt(R))) ** (2/3)
    term2 = 1/(a**2+4*R**2)**(1/2)
    term3 = 2*R**2/((1-v)*(a**2+4*R**2)**(3/2))
    return term1 - f/(np.pi*E_eq) * (term2 + term3)


# Hertz接触理论计算接触力
def hertz_force(u, E_eq, R):
    f = (4/3.) * E_eq * np.sqrt(R) * u**(3/2.)
    return f

def read_csv_data(file_path):
    data = pd.read_csv(file_path, skiprows=1, header=None)
    disp = data[1] * -1000  # mm to um
    force = data[2] * 1000  # N to mN
    return disp, force

if __name__ == "__main__":
    # Hertz接触理论参数
    E1 = 1e5 # 100 GPa
    nu1 = 0.3
    E2 = 1e7 # 10000 GPa
    nu2 = 0.3
    E_eq_value = E_eq(E1, nu1, E2, nu2)  # 等效弹性模量

    R = 1.0 # 半径 mm
    u_hertz = np.linspace(0, 0.5, 100)  # 位移范围 mm
    f_hertz = hertz_force(u_hertz, E_eq_value, R)
    u_hertz *= 1000  # 转换为um
    f_hertz *= 1000  # 转换为mN

    # Tatara理论计算接触力
    f_tatara = np.linspace(0, 1.2e3, 100)  # 力范围
    u_tatara = tatara_disp(f_tatara, E_eq_value, R)
    u_tatara *= 1000  # 转换为um
    f_tatara *= 1000  # 转换为mN


    # # 读取CSV文件
    # disp_1, force_1 = read_csv_data('hertz_linear_elastic_out.csv')
    # disp_2, force_2 = read_csv_data('hertz_linear_hardening_out.csv')
    # disp_3, force_3 = read_csv_data('hertz_linear_hardening_mesh_refine1_penalty_1e12_dt_0.1_out.csv')
    # disp_4, force_4 = read_csv_data('hertz_test.csv')
    # # disp_half_space, force_half_space = read_csv_data('hertz_elasticity_half_space_out.csv')
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    # ax.plot(disp_1[::5], force_1[::5], marker='^', markersize=1, linewidth=1, color='blue', label='FEA elastic')
    # ax.plot(disp_2[::1], force_2[::1], marker='v', markersize=1, linewidth=1, color='green', alpha=1, label='FEA plastic')
    # # ax.plot(disp_half_space, force_half_space, marker='o', markersize=5, linewidth=1, color='orange', label='FEA half space')
    # # ax.plot(u_hertz, f_hertz, linestyle='--', color='red', label='Hertz Theory')  # mm to um
    # ax.plot(u_tatara, f_tatara, linestyle=':', color='purple', label='Tatara Theory')   
    # ax.set_xlabel('Displacement (um)')
    # ax.set_ylabel('Force (mN)')
    # ax.set_title('Force-Displacement Curve')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig('force_disp.pdf')


    disp_4, force_4 = read_csv_data('hertz_test.csv')
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    ax.plot(disp_4[::1], force_4[::1], marker='o', markersize=1, linewidth=1, color='green', alpha=1, label='FEA plastic')
    ax.set_xlabel('Displacement (um)')
    ax.set_ylabel('Force (mN)')
    ax.set_title('Force-Displacement Curve')
    fig.tight_layout()
    fig.savefig('power_hardening_law_force_disp.pdf')

