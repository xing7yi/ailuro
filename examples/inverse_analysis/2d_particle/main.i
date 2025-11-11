measurement_csv = ObjectiveCSV/extracted_disp_force.csv

[Optimization]
[]

[OptimizationReporter]
  type = GeneralOptimization
  objective_name = objective_value
  parameter_names = 'parameter_results'
  num_values = '3'
  initial_condition = '400 800 300'
  lower_bounds = '10 0 0'
  upper_bounds = '1200 2000 800'
[]
# [OptimizationReporter]
#   type = GeneralOptimization
#   objective_name = objective_value
#   parameter_names = 'parameter_results'
#   num_values = '2'
#   initial_condition = '400 300'
#   lower_bounds = '10 10'
#   upper_bounds = '1200 800'
# []

[Reporters]
  [main]
    type = OptimizationData
    measurement_file = ${measurement_csv}
    file_time = time
    file_value = disp_y
    file_xcoord = x_coord
    file_ycoord = y_coord
    file_zcoord = z_coord
  []
[]

# [Executioner]
#   type = Optimize
#   tao_solver = taobqnls
#   petsc_options_iname = '-tao_gttol -tao_max_it'
#   #petsc_options_value = '1e-5 100' #use this to get results for paper
#   petsc_options_value = '1e-8 5'
#   solve_on = 'NONE'
#   verbose = true
# []

# [Executioner]
#   type = Optimize
#   tao_solver = taolmvm
#   petsc_options_iname = '-tao_fd_gradient -tao_fd_delta -tao_gatol'
#   petsc_options_value = 'true 1e-8 0.1'
#   verbose = true
# []
# [Executioner]
#   type = Optimize
#   tao_solver = taonm
#   petsc_options_iname = '-tao_gatol -tao_nm_lambda'
#   petsc_options_value = '1e-8 50.0'  # lambda控制初始simplex扩张
#   verbose = true
#   output_optimization_iterations = true  # 启用迭代输出
# []
# [Executioner]
#   type = Optimize
#   tao_solver = taolmvm  # 或 taobqnls
#   petsc_options_iname = '-tao_fd_gradient -tao_fd_delta -tao_gatol'
#   petsc_options_value = 'true 10.0 1e-4'
#   verbose = true
#   output_optimization_iterations = true  # 启用迭代输出
# []
# [Executioner]
#   type = Optimize
#   tao_solver = taobqnls
#   petsc_options_iname = '-tao_fd_gradient -tao_fd_delta -tao_gatol'
#   petsc_options_value = 'true 10.0 1e-10'
#   verbose = true
#   output_optimization_iterations = true  # 启用迭代输出
# []
# [Executioner]
#   type = Optimize
#   tao_solver = taobqnls
#   petsc_options_iname = '-tao_gttol -tao_max_it'
#   #petsc_options_value = '1e-5 100' #use this to get results for paper
#   petsc_options_value = '1e-5 40'
#   solve_on = 'NONE'
#   verbose = true
# []
[Executioner]
  type = Optimize
  tao_solver = taoblmvm  # 或 taobqnls
  petsc_options_iname = '-tao_fd_gradient -tao_fd_delta -tao_gatol'
  petsc_options_value = 'true 10.0 1e-4'
  verbose = true
  output_optimization_iterations = true  # 启用迭代输出
[]
[MultiApps]
  [forward]
    type = FullSolveMultiApp
    input_files = forward_particle_friction_plastic_voce_.i
    execute_on = FORWARD
    ignore_solve_not_converge = true
  []
[]

[Transfers]
  [to_forward_measurements]
    type = MultiAppReporterTransfer
    to_multi_app = forward
    from_reporters = 'main/measurement_xcoord
                      main/measurement_ycoord
                      main/measurement_zcoord
                      main/measurement_time
                      main/measurement_values
                      OptimizationReporter/parameter_results'
    to_reporters = 'measure_data/measurement_xcoord
                    measure_data/measurement_ycoord
                    measure_data/measurement_zcoord
                    measure_data/measurement_time
                    measure_data/measurement_values
                    params/parameter_results'
  []

  [from_forward_objective]
    type = MultiAppReporterTransfer
    from_multi_app = forward
    from_reporters = 'measure_data/misfit_values
                      measure_data/objective_value'
    to_reporters = 'main/misfit_values
                    OptimizationReporter/objective_value'
  []
[]

[Reporters]
  [opt_status]
    type = OptimizationInfo
  []
[]

[Outputs]
  # csv = true
  [csv_forward]
    type = CSV
    execute_on = 'FORWARD'
  []
[]