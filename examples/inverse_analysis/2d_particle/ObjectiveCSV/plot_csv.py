import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


if __name__ == "__main__":
    # file = list(Path('.').glob('*.csv'))
    df = pd.read_csv("316L_CL_0004_D27.485_T11.7.csv")
    time , disp , force = np.array(df["time"]), np.array(df["disp"]), np.array(df["force"])

    fig,ax = plt.subplots(1,1, figsize=(4, 3.6))
    ax.plot(time, force, marker='o', markersize=1, linewidth=1, color='blue', label='Experiment Result')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig("316L_CL_0004_D27.485_T11.7_force_time.pdf")

    # # 1. 以位移为自变量建立力的插值
    # F = interp1d(df["disp"], df["force"], kind="cubic", fill_value="extrapolate")

    # # 2. 构造新的时间（保持原始跨度，节点数 N 可自定）
    # N = len(time)
    # print("Original data points:", N)
    # time_new = np.linspace(time.iloc[0], time.iloc[-1], N)

    # # 位移线性增长
    # disp_new = np.linspace(disp.iloc[0], disp.iloc[-1], N)

    # # 3. 由插值得到新的力
    # force_new = F(disp_new)

    # # 4. 写回 CSV
    # out = pd.DataFrame({
    #     "time": np.round(time_new, 2),
    #     "disp": np.round(disp_new, 2),
    #     "force": np.round(force_new, 2)
    # })
    # out.to_csv("316L_CL_0004_D27.485_T11.7_resampled.csv", index=False)

    # # 5. 绘图对比
    # fig,ax = plt.subplots(1,1, figsize=(4, 3.6))
    # ax.plot(disp[::5], force[::5], marker='o', markersize=2, linewidth=1, color='blue', label='Original FEA Result')
    # ax.plot(disp_new[::5], force_new[::5], marker='s', markersize=2, linewidth=0, color='red', alpha=0.5, label='Resampled FEA Result')
    # ax.set_xlabel('Displacement (mm)')
    # ax.set_ylabel('Force (N)')
    # ax.legend()
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig("resampled.pdf")


    # 额外：提取特定时间点的力和位移
    time_index = np.array([2.5,5,7.5,10,11.55])
    # corresponding force and displacement values
    disp_values = np.array([disp[time==t][0] for t in time_index])
    force_values = np.array([force[time==t][0] for t in time_index])
    print("Extracted displacement values:", disp_values)
    # save to csv
    output_df = pd.DataFrame({
        "time": np.round(time_index, 2),
        "disp_y": np.round(-1e0 * disp_values / 2, 7),
        "x_coord": [0.0]*len(time_index),
        "y_coord": [0.0137425]*len(time_index),
        "z_coord": [0.0]*len(time_index),
        "force": np.round(force_values, 3)
    })
    output_df.to_csv("extracted_disp_force.csv", index=False)


    # 读取模拟结果
    # df2 = pd.read_csv("../particle_friction_plastic_voce_out_p0_186.42_p1_823.72_p3_810.csv")
    df2 = pd.read_csv("../particle_friction_plastic_voce_p0_400_p1_800_p2_300.csv")
    df3 = pd.read_csv("No_3_D_0.027485_CR_0.4805_E_100000_C1_400_C2_800_C3_300_C4_10.csv")
    time2, disp2, force2 = np.array(df2["time"]), np.array(df2["disp_abs"]), np.array(df2["force"])
    disp3, force3 = np.array(df3["displacement"]), np.array(df3["force"])

    # 绘制力-位移曲线,并标注特定时间点
    fig,ax = plt.subplots(1,1, figsize=(4, 3.6))
    ax.plot(1e0 * disp / 2, force, marker='o', markersize=1, linewidth=1, color='blue', label='Experiment Result')    
    ax.plot(1e0 * disp_values / 2, force_values, 'rs', markersize=6, label='Extracted Points')
    ax.plot(1e3 * disp2, 1e3*force2, 'g^', markersize=2, label='MOOSE Result')
    ax.plot(1e0 * disp3, 1e0*force3, 'mD', markersize=2, label='ANSYS Result')
    ax.set_xlabel('Displacement (mm)')
    ax.set_ylabel('Force (N)')
    ax.legend()
    plt.tight_layout()
    plt.savefig("316L_CL_0004_D27.485_T11.7_force_disp.pdf")

    # 



