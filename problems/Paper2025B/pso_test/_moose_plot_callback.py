from simuopt.moose_interface import MOOSEObjectiveFunction
from pathlib import Path

# 使用非交互式后端，避免多线程环境下的 GUI 问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 增加图形数量警告阈值（PSO 优化可能创建很多图形）
plt.rcParams['figure.max_open_warning'] = 100

def create_moose_callback():
    def plot_curve(f: MOOSEObjectiveFunction, sim_csv: Path, objective: float, plot_data: dict):
        """MOOSE 回调函数示例
        
        Args:
            f: MOOSE目标函数实例
            sim_csv: 仿真结果CSV文件路径
            objective: 目标函数值（RMSE）
            plot_data: 包含绘图数据的字典，包含以下键：
                - sim_x, sim_y: 仿真数据
                - ref_x, ref_y: 参考数据
                - common_x: 插值点
                - sim_y_interp, ref_y_interp: 插值后的数据
        """
        if plot_data is None:
            return
            
        # print(f"  当前评估参数: {f.current_params}, 目标函数值: {f.current_objective}")
        # 绘图：比较实验与仿真力-位移曲线，并标注用于插值的共同位移点
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(5,3.6))
            # 原始曲线
            ax.plot(plot_data['sim_x'], plot_data['sim_y'], label='Simulation', color='C0', linewidth=1)
            ax.plot(plot_data['ref_x'], plot_data['ref_y'], label='Reference', color='C1', linewidth=1)

            # 插值点（共同位移网格）
            ax.scatter(plot_data['common_x'], plot_data['sim_y_interp'], marker='o', s=30, color='C0', facecolors='none', label='Interpolation Points (Sim)')
            ax.scatter(plot_data['common_x'], plot_data['ref_y_interp'], marker='+', s=30, color='C1', label='Interpolation Points (Ref)')
            ax.set_xlabel(r'Compression ratio')
            ax.set_ylabel(r'Equivalent Stress (MPa)')
            ax.set_title(f'(RMSE={objective:.3f})')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)

            # 保存图片到与 CSV 相同目录，文件名根据 csv_file 命名
            plot_path = sim_csv.with_suffix('.png')
            # 确保目录存在
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=300)
            # print(f"   已保存对比图: {plot_path}")
        except Exception as e:
            print(f"   绘图失败: {e}")
        finally:
            # 确保图形被关闭，即使发生异常
            if fig is not None:
                plt.close(fig)        
    return plot_curve
