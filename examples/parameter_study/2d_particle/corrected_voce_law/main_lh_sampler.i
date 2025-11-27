[StochasticTools]
[]

[Distributions]
  [ys]
    type = Uniform
    lower_bound = 50
    upper_bound = 1500
  []
  [tm]
    type = Uniform
    lower_bound = 0
    upper_bound = 2000
  []
  [q]
    type = Uniform
    lower_bound = 0
    upper_bound = 1000
  []
  [b]
    type = Uniform
    lower_bound = 1
    upper_bound = 1000
  []
[]

[Samplers]
  [hypercube]
    type = LatinHypercube
    num_rows = 80
    distributions = 'ys tm q b'
  []
[]

[MultiApps]
  [runner]
    type = SamplerFullSolveMultiApp
    sampler = hypercube
    input_files = 'particle_friction_plastic_voce.i'
    mode = batch-reset
    ignore_solve_not_converge = true
  []
[]

[Transfers]
  [parameters]
    type = SamplerParameterTransfer
    to_multi_app = runner
    sampler = hypercube
    parameters = 'Materials/stress_specimen/yield_stress Materials/stress_specimen/hardening_constant Materials/stress_specimen/q Materials/stress_specimen/b'
  []
  [results]
    type = SamplerReporterTransfer
    from_multi_app = runner
    sampler = hypercube
    stochastic_reporter = results
    from_reporter = 'force_norm_MPa/value'
  []
[]

[VectorPostprocessors]
  [sample_data]
    type = SamplerData
    sampler = hypercube
    execute_on = 'INITIAL'
  []
[]

[Reporters]
  [results]
    type = StochasticReporter
    parallel_type = ROOT # gather all ranks data to rank 0
  []
  # [stats]
  #   type = StatisticsReporter
  #   reporters = 'results/results:force:value results/results:contact_pressure_avg:value'
  #   compute = 'mean stddev'
  #   ci_method = 'percentile'
  #   ci_levels = '0.05 0.95'
  # []
[]

[Outputs]
  [out]
    type = JSON
    execute_on = 'FINAL'
  []
  [samples]
    type = CSV
    execute_on = 'INITIAL'
    execute_reporters_on = 'NONE'
  []
[]