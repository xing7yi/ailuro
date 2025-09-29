[StochasticTools]
  auto_create_executioner = false
[]

[Transfers]
  [sub]
    type = SamplerParameterTransfer
    to_multi_app = sub
    sampler = sample
    parameters = 'Materials/voce_plasticity/yield_stress'
    execute_on = INITIAL
    check_multiapp_execute_on = false
  []
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    input_files = hertz_voce_hardening.i
    sampler = sample
  []
[]

[Samplers]
  [sample]
    type = MonteCarlo
    num_rows = 10
    distributions = 'uniform_ys'
    execute_on = INITIAL # create random numbers on initial and use them for each timestep
  []
[]

[Distributions]
  [uniform_ys]
    type = Uniform
    lower_bound = 10
    upper_bound = 1000
  []
  [uniform_tm]
    type = Uniform
    lower_bound = 10
    upper_bound = 2000
  []
[]

[Executioner]
  type = Transient
  dt = 0.01
  end_time = 3.0 # Executioner
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
[]
