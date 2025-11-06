import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



def calculate_rmse_with_interpolation(sim_file, exp_file, n_interp_points=10, 
                                       col_x='disp', col_y='force',
                                       verbose=True, plot=False, output_dir=None):
    """
    计算仿真与实验数据之间的 RMSE（使用插值方法）
    
    Args:
        sim_file: 仿真数据 CSV 文件路径
        exp_file: 实验数据 CSV 文件路径
        n_interp_points: 插值点数量
        col_x: 位移列名
        col_y: 力列名
        verbose: 是否打印详细信息
        plot: 是否绘图
        output_dir: 图片输出目录（如果 plot=True）
        
    Returns:
        rmse: float，RMSE 值；如果出错返回 None
    """
    
    # 读取仿真数据
    try:
        df_sim = pd.read_csv(sim_file)
    except Exception as e:
        if verbose:
            print(f"   读取仿真数据 {sim_file} 失败: {e}")
        return None
    
    # 检查列
    if col_x not in df_sim.columns or col_y not in df_sim.columns:
        if verbose:
            print(f"   仿真数据缺少列: 期望 ({col_x}, {col_y}), 实际列: {list(df_sim.columns)}")
        return None
    
    # 读取实验数据
    try:
        df_exp = pd.read_csv(exp_file)
    except Exception as e:
        if verbose:
            print(f"   读取实验数据 {exp_file} 失败: {e}")
        return None
    
    # 检查列
    if col_x not in df_exp.columns or col_y not in df_exp.columns:
        if verbose:
            print(f"   实验数据缺少列: 期望 ({col_x}, {col_y}), 实际列: {list(df_exp.columns)}")
        return None
    
    # 提取数据
    u_sim = np.asarray(df_sim[col_x].astype(float))
    f_sim = np.asarray(df_sim[col_y].astype(float))
    
    u_exp = np.asarray(df_exp[col_x].astype(float))
    f_exp = np.asarray(df_exp[col_y].astype(float))
    
    # 检查数据点数量
    if len(u_sim) < 2 or len(u_exp) < 2:
        if verbose:
            print(f"   数据点不足以插值：仿真 {len(u_sim)}, 实验 {len(u_exp)}")
        return None
    
    # 按位移升序排序
    sim_sort = np.argsort(u_sim)
    u_sim_sorted = u_sim[sim_sort]
    f_sim_sorted = f_sim[sim_sort]
    
    exp_sort = np.argsort(u_exp)
    u_exp_sorted = u_exp[exp_sort]
    f_exp_sorted = f_exp[exp_sort]
    
    # 共同位移上限（取两者最大位移的较小者）
    umax = min(u_sim_sorted[-1], u_exp_sorted[-1])
    if umax <= 0:
        if verbose:
            print(f"   无效的最大位移 umax={umax}")
        return None
    
    # 位移最小值 umin = umax / n_points
    umin = umax / float(n_interp_points)
    
    # 在 [umin, umax] 上生成等间距位移点
    common_u = np.linspace(umin, umax, num=n_interp_points)
    
    # 插值
    try:
        f_sim_on_common = np.interp(common_u, u_sim_sorted, f_sim_sorted)
        f_exp_on_common = np.interp(common_u, u_exp_sorted, f_exp_sorted)
    except Exception as e:
        if verbose:
            print(f"   插值失败: {e}")
        return None
    
    # 计算 RMSE
    mse = np.mean((f_sim_on_common - f_exp_on_common) ** 2)
    rmse = float(np.sqrt(mse))
    
    if verbose:
        print(f"   仿真点数: {len(u_sim)}, 实验点数: {len(u_exp)}")
        print(f"   共用位移区间: [{umin:.6e}, {umax:.6e}]")
        print(f"   插值点数: {n_interp_points}")
        print(f"   RMSE(Force) = {rmse:.6e}")
    
    # 绘图
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 原始曲线
        ax.plot(u_exp_sorted, f_exp_sorted, label='Experimental', 
                color='C0', linewidth=1.5)
        ax.plot(u_sim_sorted, f_sim_sorted, label='Simulation', 
                color='C1', linewidth=1.5)
        
        # 插值点
        ax.scatter(common_u, f_exp_on_common, marker='o', s=30, 
                  color='C0', facecolors='none', label='Interpolation Points (Exp)')
        ax.scatter(common_u, f_sim_on_common, marker='+', s=30, 
                  color='C1', label='Interpolation Points (Sim)')
        
        ax.set_xlabel(r'Displacement ($\mu$m)')
        ax.set_ylabel(r'Force (mN)')
        ax.set_title(f'RMSE = {rmse:.3e}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        
        # 保存图片
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            sim_name = Path(sim_file).stem
            plot_path = output_dir / f"{sim_name}_rmse_comparison.png"
        else:
            plot_path = Path(sim_file).with_suffix('.png')
        
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        
        if verbose:
            print(f"   已保存对比图: {plot_path}")
    
    return rmse


if __name__ == "__main__":
    # sim_file = "p0_400_p1_800_p2_300_p3_10_remake.csv"
    sim_file = "run_0916_p0_320.39_p1_1048.81_p2_288.37_p3_20.63.csv"
    exp_file = "316L_CL_0004_D27.485_T11.55.csv"

    calculate_rmse_with_interpolation(sim_file, exp_file, n_interp_points=10, 
                                       col_x='disp_um', col_y='force_mN',
                                       verbose=True, plot=True, output_dir="rmse_plots")
    