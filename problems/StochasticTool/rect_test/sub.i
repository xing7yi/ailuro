[Controls]
  [stochastic]
    type = SamplerReceiver
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  [simple_mesh]
    type = FileMeshGenerator
    file = rectangle.msh
  []
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    add_variables = true
    incremental = true
    strain = FINITE
  []
[]

#
# Added boundary/loading conditions
# https://mooseframework.inl.gov/modules/solid_mechanics/tutorials/introduction/step02.html
#
[BCs]
  [left_x]
    type = DirichletBC
    variable = disp_x
    boundary = Left_Edge
    value = 0
  []

  [bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = Bottom_Edge
    value = 0
  []
  [Pressure]
    [top]
      boundary = Top_Edge
      function = 1e2*t
    []
  []
[]

[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 100e3
    poissons_ratio = 0.3
  []
  [voce_plasticity]
    type = ControllableCombinedNonlinearHardeningPlasticity
    # type = CombinedNonlinearHardeningPlasticity
    yield_stress = 200
    isotropic_hardening_constant = 100
    q = 10
    b = 10
  []  
  [return_stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = 'voce_plasticity'
  []  
  # [stress]
  #   type = ComputeLinearElasticStress
  # []
[]

# consider all off-diagonal Jacobians for preconditioning
[Preconditioning]
  [SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  # we chose a direct solver here
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  end_time = 2
  dt = 1
[]

[Outputs]
  exodus = true
[]