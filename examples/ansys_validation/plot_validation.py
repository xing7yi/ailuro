import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    csv_folder = Path("./csv_files")
    file = list(csv_folder.glob('*.csv'))

    df_ansys_voce = pd.read_csv("./csv_files/No_1_D_0.027485_CR_0.4805_E_193000_C1_400_C2_800_C3_300_C4_10.csv")
    time_ansys_voce , disp_ansys_voce , force_ansys_voce = np.array(df_ansys_voce["time"]), np.array(df_ansys_voce["displacement"]), np.array(df_ansys_voce["force"])

    df_moose_voce = pd.read_csv("./csv_files/particle_friction_plastic_voce_p0_400_p1_800_p2_300_p3_10.csv")
    time_moose_voce , disp_moose_voce , force_moose_voce = np.array(df_moose_voce["time"]), np.array(df_moose_voce["disp_abs"]), np.array(df_moose_voce["force"])

    df_ansys_biso = pd.read_csv("./csv_files/No_1_D_0.027485_CR_0.4805_E_193000_C1_400_C2_800.csv")
    time_ansys_biso , disp_ansys_biso , force_ansys_biso = np.array(df_ansys_biso["time"]), np.array(df_ansys_biso["displacement"]), np.array(df_ansys_biso["force"])

    df_moose_biso = pd.read_csv("./csv_files/particle_friction_plastic_biso_p0_400_p1_800.csv")
    time_moose_biso , disp_moose_biso , force_moose_biso = np.array(df_moose_biso["time"]), np.array(df_moose_biso["disp_abs"]), np.array(df_moose_biso["force"])

    fig,axes = plt.subplots(1,2, figsize=(7, 3.6))
    ax = axes[0]
    ax.plot(disp_ansys_voce, force_ansys_voce, marker='o', markersize=1, linewidth=1, color='blue', label='ANSYS Result')
    ax.plot(1e3*disp_moose_voce, 1e3*force_moose_voce, marker='s', markersize=1, linewidth=1, color='red', label='MOOSE Result')
    ax.set_title('VOCE Model')

    ax = axes[1]
    ax.plot(disp_ansys_biso, force_ansys_biso, marker='o', markersize=1, linewidth=1, color='blue', label='ANSYS Result')
    ax.plot(1e3*disp_moose_biso, 1e3*force_moose_biso, marker='s', markersize=1, linewidth=1, color='red', label='MOOSE Result')
    ax.set_title('BISO Model')

    for ax in axes:
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Force (N)')
        ax.legend()

    plt.tight_layout()
    plt.savefig("ansys_validation_contact_pressure_force_disp.pdf")