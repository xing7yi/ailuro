[Controls]
  [stochastic]
    type = SamplerReceiver
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]

  type = GeneratedMesh
  dim = 2
  nx = 10
  ny = 20
  xmax = 1
  ymax = 2

[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    add_variables = true
    incremental = true
    strain = FINITE
    save_in = 'react_x react_y'
  []
[]

[AuxVariables]
  [react_x]
  []
  [react_y]
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
    boundary = left
    value = 0
  []

  [bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [top]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = 1e-2*t
  []
  # [Pressure]
  #   [top]
  #     boundary = top
  #     function = 1e2*t
  #   []
  # []
[]

[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 200e3
    poissons_ratio = 0.3
  []
  [plasticity]
    type = IsotropicVoceLawHardeningStressUpdate
    # type = CombinedNonlinearHardeningPlasticity
    yield_stress = 200  # Default value, will be overridden by StochasticTools
    hardening_constant = 1000  # Default value, will be overridden by StochasticTools
    q = 200
    b = 500
  []  
  [return_stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = 'plasticity'
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
  solve_type = 'PJFNK'
  
  # 改进的非线性求解器设置
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-8
  nl_max_its = 50
  
  # # 线搜索设置
  # line_search = 'bt'

  # 稳健的线性求解器
  petsc_options_iname = '-pc_type '
  petsc_options_value = 'lu       '
  
  start_time = 0
  end_time = 1.0
  dt = 0.05
[]

[Postprocessors]
  [disp]
    type = PointValue
    variable = disp_y
    point = '0 1 0 '
  []
  # [bottom_reaction_y]
  #   type = NodalSum
  #   variable = react_y
  #   boundary = bottom
  # []
  [force]
    type = SidesetReaction
    boundary = bottom
    direction = '0 1 0'
    stress_tensor = stress
  []
[]

[Outputs]
  csv = true
  [out]
    type = Exodus
    elemental_as_nodal = true
    time_step_interval = 1
  []  
[]