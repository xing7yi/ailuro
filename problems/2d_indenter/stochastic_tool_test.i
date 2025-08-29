[StochasticTools]
  auto_create_executioner = false
[]

[Transfers]
  [sub]
    type = SamplerParameterTransfer
    to_multi_app = sub
    sampler = sample
    parameters = 'Materials/tensor/youngs_modulus'
    check_multiapp_execute_on = false
  []
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    input_files = indenter_rz_fine.i
    sampler = sample
    mode = batch-reset
  []
[]

[Samplers]
  [sample]
    type = MonteCarlo
    num_rows = 3
    distributions = 'uniform_e'
    execute_on = INITIAL # create random numbers on initial and use them for each timestep
  []
[]

[Distributions]
  [uniform_e]
    type = Uniform
    lower_bound = 10
    upper_bound = 1000
  []
[]

[Executioner]
  type = Transient
#   num_steps = 5
  dt = 0.01
  end_time = 10 # Executioner    
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
[]