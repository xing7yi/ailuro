[StochasticTools]
  auto_create_executioner = false
[]

[Transfers]
  [parameters]
    type = SamplerParameterTransfer
    to_multi_app = sub
    sampler = sample
    parameters = 'Materials/plasticity/yield_stress
                  Materials/plasticity/hardening_constant
                  Materials/plasticity/q
                  Materials/plasticity/b'
    # parameters = 'Materials/elasticity/youngs_modulus'
    execute_on = 'initial'
    check_multiapp_execute_on = false
  []

  [results]
    type = SamplerReporterTransfer
    from_multi_app = sub
    sampler = sample
    stochastic_reporter = str
    from_reporter = 'force/value'
    # execute_on = 'timestep_end'
  []
[]

[Reporters]
  [str]
    type = StochasticReporter
    parallel_type = ROOT
  []
  # [stats]
  #   type = StatisticsReporter
  #   reporters = 'str/results:disp:value'
  #   compute = 'mean'
  #   ci_method = 'percentile'
  #   ci_levels = '0.05 0.95'    
  # []
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
    distributions = 'uniform_ys uniform_tm uniform_q uniform_b'
    execute_on = INITIAL # create random numbers on initial and use them for each timestep
  []
  # [sample]
  #   type = MonteCarlo
  #   num_rows = 5    
  #   distributions = 'uniform_e'
  #   execute_on = INITIAL
  # []
[]

[Distributions]
  [uniform_e]
    type = Uniform
    lower_bound = 100e3
    upper_bound = 500e3
  []
  [uniform_ys]
    type = Uniform
    lower_bound = 10
    upper_bound = 1000
  []
  [uniform_tm]
    type = Uniform
    lower_bound = 10
    upper_bound = 1000
  []
  [uniform_q]
    type = Uniform
    lower_bound = 0
    upper_bound = 1000
  []
  [uniform_b]
    type = Uniform
    lower_bound = 0
    upper_bound = 1000
  []    
[]

[Executioner]
  type = Transient
  dt = 1
  end_time = 1 # Executioner
[]

[Outputs]
  # execute_on = 'FINAL'
  [out]
    type = JSON
  []  
[]
