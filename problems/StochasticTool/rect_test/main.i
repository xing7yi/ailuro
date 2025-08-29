[StochasticTools]
  auto_create_executioner = false
[]

[Transfers]
  [sub]
    type = SamplerParameterTransfer
    to_multi_app = sub
    sampler = sample
    parameters = 'Materials/voce_plasticity/yield_stress Materials/voce_plasticity/isotropic_hardening_constant'
    execute_on = INITIAL
    check_multiapp_execute_on = false
  []
[]

[MultiApps]
  [sub]
    type = SamplerFullSolveMultiApp
    input_files = sub.i
    sampler = sample
  []
[]

[Samplers]
  [sample]
    type = MonteCarlo
    num_rows = 5
    distributions = 'uniform_ys uniform_tm'
    execute_on = INITIAL # create random numbers on initial and use them for each timestep
  []
[]

[Distributions]
  [uniform_ys]
    type = Uniform
    lower_bound = 10
    upper_bound = 500
  []
  [uniform_tm]
    type = Uniform
    lower_bound = 10
    upper_bound = 1000
  []
[]

[Executioner]
  type = Transient
  dt = 1
  end_time = 2 # Executioner
[]

[Outputs]
  execute_on = 'INITIAL TIMESTEP_END'
[]
