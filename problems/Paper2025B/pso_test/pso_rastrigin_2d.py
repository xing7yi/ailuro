
from simuopt.pso import PSOOptimizer, PSOConfig
from _pso_plot_callback import create_pso_callback
from _pso_particle_animation import create_pso_animation
import numpy as np
import yaml

def rastrigin_func(x):
    """Rastrigin 函数，适用于 2D"""
    x0, x1 = x
    A = 10
    return A * 2 + (x0**2 - A * np.cos(2 * np.pi * x0)) + (x1**2 - A * np.cos(2 * np.pi * x1))

def pso_rastrigin_2d():
    pso_config = PSOConfig(
        n_dims=2,
        n_particles=20,
        max_iters=200,
        w=0.8,
        c1=1.5,
        c2=1.5,
        lb=[-5.12, -5.12],
        ub=[5.12, 5.12],
        cmode='avg_cost',
        cthreshold=1e-5,
        cwindow=10,
        position_init_method='lhs'  # 使用拉丁超立方采样初始化粒子位置
    )
    pso_callback = create_pso_callback()
    pso = PSOOptimizer(rastrigin_func, pso_config, 
                       output_dir='pso_rastrigin_2d', 
                       callback=pso_callback,callback_interval=10)
    return pso

if __name__ == "__main__":
    pso = pso_rastrigin_2d()
    pso.run()
    # 输出结果到yaml文件
    result = {
        'total evaluations': pso.particle_history['eval_id'][-1] + 1,
        'total iterations': pso.iter_history['iteration'][-1] + 1,
        'total elapsed time (s)': pso.elapsed_time,
        'best pos': pso.swarm.best_pos.tolist(),
        'best cost': float(pso.swarm.best_cost),  # 转换为 Python float
        'best eval_id': int(pso.swarm.best_id)   # 转换为 Python int
    }
    with open(pso.output_dir / 'optimal_result.yaml', 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    # create_pso_animation(pso, frames_per_iter=10)
