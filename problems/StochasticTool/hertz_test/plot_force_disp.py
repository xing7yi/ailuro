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
    data = pd.read_csv(file_path)
    disp = np.abs(data["disp"]) * 1000  # mm to um
    force_nodal = np.abs(data["spec_nodalsum"]) * 1  # N to mN
    force_sideset = np.abs(data["spec_reaction"]) * 1  # N to mN
    return {"disp": disp, "force_nodal": force_nodal, "force_sideset": force_sideset}

if __name__ == "__main__":
    pressure_data = read_csv_data('hertz2d_pressure_load.csv')
    disp_data = read_csv_data('hertz2d_disp_load.csv')
    test_data = read_csv_data('hertz2d_pressure.csv')


    # Hertz接触理论参数
    E1 = 100e3 # 100 GPa
    nu1 = 0.3
    E2 = 1e7 # 10000 GPa
    nu2 = 0.3
    E_eq_value = E_eq(E1, nu1, E2, nu2)  # 等效弹性模量

    R = 1.0 # 半径 mm
    u_hertz = np.linspace(0, pressure_data['disp'].max()/1000, 100)  # 位移范围 mm
    f_hertz = hertz_force(u_hertz, E_eq_value, R)
    u_hertz *= 1000  # 转换为um
    # f_hertz *= 1000  # 转换为mN


    # Tatara理论计算接触力
    f_tatara = np.linspace(0, f_hertz.max(), 100)  # 力范围 0 - 1.6 N
    u_tatara = tatara_disp(f_tatara, E_eq_value, R)
    u_tatara *= 1000  # 转换为um
    # f_tatara *= 1000  # 转换为mN

    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    ax.plot(test_data["disp"], test_data["force_nodal"], marker='o', markersize=2, color='green', label='FEA Test')
    # ax.plot(pressure_data["disp"], pressure_data["force_nodal"], marker='o', markersize=3, color='blue', label='FEA Pressure')
    # ax.plot(disp_data["disp"], disp_data["force_nodal"], marker='o',markersize=3, color='red', label='FEA Displacement')
    # ax.plot(u_hertz, f_hertz, linestyle='--', color='green', label='Hertz Theory')
    # ax.plot(u_tatara, f_tatara, linestyle=':', color='orange', label='Tatara Theory')

    ax.set_xlabel('Displacement (um)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force-Displacement Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig('plot_force_disp.pdf')


    # # 批量绘制main_out_sub0.csv -main_out_sub4.csv 的力-位移曲线
    # fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2.8))
    
    # colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # for i in range(5):

    #     file_path = f'main_out_sub{i}.csv'
    #     batch_results = read_csv_data(file_path)
    #     ax2.plot(batch_results["disp"], batch_results["force"], 
    #             marker='o', markersize=1, linewidth=1, 
    #             color=colors[i], alpha=0.8, 
    #             label=f'Sub{i}')
    
    # ax2.set_xlabel('Displacement (um)')
    # ax2.set_ylabel('Force (N)')
    # ax2.set_title('Batch Force-Displacement Curves')
    # ax2.legend()
    # fig2.tight_layout()
    # fig2.savefig('batch_force_disp.pdf')
    
    # plt.show()

