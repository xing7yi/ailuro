import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    disp = data["disp"] * 1000  # mm to um
    force = data["force"] * -1  # N to mN
    return {"disp": disp, "force": force}

if __name__ == "__main__":
    
    voce_results = read_csv_data('sub_out.csv')

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.6))
    ax.plot(voce_results["disp"], voce_results["force"], marker='o', markersize=1, linewidth=1, color='blue', alpha=1, label='FEA Voce')
    ax.set_xlabel('Displacement (um)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force-Displacement Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig('plot_force_disp.pdf')


    # 批量绘制main_out_sub0.csv -main_out_sub4.csv 的力-位移曲线
    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2.8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(5):

        file_path = f'main_out_sub{i}.csv'
        batch_results = read_csv_data(file_path)
        ax2.plot(batch_results["disp"], batch_results["force"], 
                marker='o', markersize=1, linewidth=1, 
                color=colors[i], alpha=0.8, 
                label=f'Sub{i}')
    
    ax2.set_xlabel('Displacement (um)')
    ax2.set_ylabel('Force (N)')
    ax2.set_title('Batch Force-Displacement Curves')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig('batch_force_disp.pdf')
    
    # plt.show()

