import yaml
from simuopt.pso import PSOOptimizer, PSOConfig
from simuopt.moose_interface import MOOSEObjectiveFunction,MOOSEConfig
from _moose_plot_callback import create_moose_callback
from _pso_plot_callback import create_pso_callback
from _pso_particle_animation import create_pso_animation

xlabel = 'Yield Strength (MPa)'
ylabel = 'Tangent Modulus (MPa)'
zlabel = 'RMSE'

def moose_eval():

    work_dir = 'sim_validation_biso'
    moose_config = MOOSEConfig(
        work_dir=work_dir,
        input_file='moose_files/particle_friction_plastic_voce.i',
        param_names=['p0','p1'],
        param_paths=['p0','p1'],
        filebase_param='out_name',
        csv_col_x='cmpr_ratio',
        csv_col_y='force_norm_MPa',
        reference_csv='moose_files/obj_csv/E100_p0_600_p1_1500_p2_300_p3_50.csv',
    )

    moose_callback = create_moose_callback()
    moose_func = MOOSEObjectiveFunction(moose_config, callback=moose_callback)

    moose_func.mode = 'multithreading'


    pso_config = PSOConfig(
        n_dims=2,
        n_particles=25,
        max_iters=200,
        w=0.8,
        c1=1.5,
        c2=1.5,
        lb=[10, 0 ],
        ub=[2000, 4000],
        cmode='avg_cost',
        cthreshold=20,
        cwindow=10,
        position_init_method='lhs'
    )
    pso_callback = create_pso_callback()
    pso = PSOOptimizer(moose_func, pso_config, 
                       output_dir=work_dir, 
                       callback=pso_callback)
    return pso

if __name__ == "__main__":
    pso = moose_eval()

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

    # 绘制gif动画（优化版，不需要传入 func）
    create_pso_animation(pso, xlabel, ylabel, zlabel, frames_per_iter=1)
