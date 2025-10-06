[ParameterStudy]
  input = particle_friction_plastic_voce.i
  parameters = 'Materials/stress_specimen/yield_stress Materials/stress_specimen/hardening_constant'
  quantities_of_interest = 'force/value contact_pressure_avg/value'

  sampling_type = lhs
  num_samples = 8
  distributions = 'uniform uniform'
  uniform_lower_bound = '50 0'
  uniform_upper_bound = '1500 2000'
[]