import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    time = np.array(data["time"])
    disp = np.abs(data["disp_abs"])
    force = np.abs(data["force"])
    return time, disp, force

if __name__ == "__main__":
    file = list(Path('.').glob('*.csv'))
    _, disp, force = read_csv_data("particle_friction_plastic_voce_out.csv")
    _, disp2, force2 = read_csv_data("main_out_forward0.csv")

    fig,ax = plt.subplots(1,1, figsize=(4, 3.6))
    ax.plot(disp, force, marker='o', markersize=1, linewidth=1, color='blue', label='FEA Result')
    # ax.plot(disp2, force2, marker='o', markersize=1, linewidth=1, color='red', label='FEA Result 2')
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Force (N)')
    ax.legend()
    plt.tight_layout()
    plt.savefig("particle_friction_plastic_voce_force_disp.pdf")
