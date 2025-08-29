[Controls]
  [stochastic]
    type = SamplerReceiver
  []
[]

[GlobalParams]
  # volumetric_locking_correction = true
  displacements = 'disp_x disp_y'
[]

[Problem]
  type = ReferenceResidualProblem
  reference_vector = 'ref'
  extra_tag_vectors = 'ref'
[]

[Mesh]
  [simple_mesh]
    type = FileMeshGenerator
    # file = ../hertz_contact_refine1.msh
    file = ../hertz_contact_structural_ref_1.msh
  []
  uniform_refine = 0  # 全局细分1次，单元数增加4倍
  allow_renumbering = false
  coord_type = RZ  # 使用RZ坐标系:轴对称
  rz_coord_axis = y # RZ坐标系中的对称轴
[]

[Functions]
  [disp_y]
    type = PiecewiseLinear
    x = '0.  1e-5     10.0'
    y = '0.  -0.1   -0.6'
  []
[]

[AuxVariables]
  [saved_x]
  []
  [saved_y]
  []
[]

[Physics/SolidMechanics/QuasiStatic]
  [indenter]
    block = 'Indenter_Body'
    add_variables = true
    strain = SMALL
    use_automatic_differentiation = true  # 添加这行！
    incremental = true
    extra_vector_tags = 'ref'  # 在RZ坐标系中，extra_vector_tags用于参考残差
    # generate_output = '
    #   vonmises_stress
    # '
    save_in = 'saved_x saved_y'    
  []
  [specimen]
    block = 'Specimen_Body'
    add_variables = true
    incremental = true
    use_automatic_differentiation = true  # 添加这行！
    strain = FINITE  # 改为有限应变以配合FiniteStrainPlasticMaterial
    extra_vector_tags = 'ref'  # 在RZ坐标系中，extra_vector_tags用于参考残差
    # generate_output = '
    #   stress_xx stress_xy stress_yy stress_yx 
    #   strain_xx strain_xy strain_yy strain_yx 
    #   max_principal_stress mid_principal_stress min_principal_stress
    #   vonmises_stress maxshear_stress
    #   plastic_strain_xx plastic_strain_xy plastic_strain_yy plastic_strain_yx
    # '
    # generate_output = '
    #   stress_xx stress_yy 
    #   strain_xx strain_yy 
    #   vonmises_stress
    #   plastic_strain_xx plastic_strain_yy
    # '    
    save_in = 'saved_x saved_y'    
  []
[]


[BCs]
  [left_x_specimen]
    type = ADDirichletBC
    variable = disp_x
    boundary = Specimen_Left_Edge
    value = 0
  []

  [bottom_y_specimen]
    type = ADDirichletBC
    variable = disp_y
    boundary = Specimen_Bottom_Edge
    value = 0
  []

  [disp_y_load]
    type = ADFunctionDirichletBC
    variable = disp_y
    boundary = Indenter_Top_Edge
    function = disp_y
  []

  [left_x_indenter]
    type = ADDirichletBC
    variable = disp_x
    boundary = Indenter_Left_Edge
    value = 0
  []

[]

[Contact]
  [contact]
    secondary = Specimen_Top_Edge
    primary = Indenter_Bottom_Edge
    model = frictionless
    # # Investigate von Mises stress at the edge
    correct_edge_dropping = true
    formulation = mortar
    c_normal = 1e4
    # c_tangential = 0
    # penalty = 1e9
    # formulation = penalty
    # normal_smoothing_distance = 0.1
  []
[]

[Materials]
  # 压头保持弹性    
  [et_i]
    type = ADComputeIsotropicElasticityTensor
    block = 'Indenter_Body'
    youngs_modulus = 1e7 # E = 10000 GPa
    poissons_ratio = 0.3
  []
  [stress_indenter]
    type = ADComputeLinearElasticStress
    block = 'Indenter_Body'
  []


  [et_s]
    type = ADComputeIsotropicElasticityTensor
    block = 'Specimen_Body'
    youngs_modulus = 100e3 # E = 100 GPa
    poissons_ratio = 0.3
  []
  # [linear_plasticity]
  #   type = ADIsotropicPlasticityStressUpdate
  #   yield_stress = 200  # 屈服应力 (MPa)
  #   hardening_constant = 1000  # 线性硬化斜率 (MPa)
  # []

  [voce_plasticity]
    type = ADControllableCombinedNonlinearHardeningPlasticity
    block = 'Specimen_Body'    
    yield_stress = 200
    isotropic_hardening_constant = 1000
    q = 10
    b = 10
  []

  [return_stress]
    type = ADComputeMultipleInelasticStress
    block = 'Specimen_Body'
    inelastic_models = 'voce_plasticity'
  []

  # [stress_specimen]
  #   type = ComputeFiniteStrainElasticStress
  #   block = 'Specimen_Body'
  # []

  # # 为试样使用塑性材料
  # [power_law_hardening]
  #   type = IsotropicPowerLawHardeningStressUpdate
  #   block = 'Specimen_Body'
  #   strength_coefficient = 70.0  #K
  #   strain_hardening_exponent = 0.5 #n
  # []
  # [radial_return_stress]
  #   type = ComputeMultipleInelasticStress
  #   block = 'Specimen_Body'    #contact_secondary_subdomain
  #   inelastic_models = 'power_law_hardening'
  #   tangent_operator = elastic
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
  # solve_type = 'PJFNK'
  petsc_options = '-snes_ksp_ew'
  petsc_options_iname = '-pc_type -snes_linesearch_type -pc_factor_shift_type -pc_factor_shift_amount'
  petsc_options_value = 'lu       basic                 NONZERO               1e-15'
  solve_type = 'NEWTON'
  # petsc_options = '-snes_monitor -ksp_monitor'
  # petsc_options_iname = '-pc_type -pc_factor_mat_solver_type -snes_linesearch_type'
  # petsc_options_value = 'lu       mumps                     bt'

  line_search = 'none'
  automatic_scaling = true
  nl_abs_tol = 2.0e-07  # non-linear absolute tolerance
  nl_rel_tol = 2.0e-07  # non-linear relative tolerance
  l_max_its = 40        # linear maximum iterations
  l_abs_tol = 1e-08     # linear absolute tolerance
  l_tol = 1e-08         # linear tolerance
  start_time = 0.0

  end_time = 10 # Executioner  

  # dt = 0.025  # 初始时间步长
  dtmin = 1e-5

  [TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 10           # 减少目标迭代次数，提高效率
    iteration_window = 2
    dt = 0.01                        # 初始步长
    growth_factor = 1.2
    cutback_factor = 0.75
    cutback_factor_at_failure = 0.5

  []
[]



[Postprocessors]
  [disp]
    type = PointValue
    variable = disp_y
    point = '0 1 0 '
  []
  [reaction_force_y_bottom]
    type = NodalSum
    variable = saved_y  # 需要在Physics中启用save_in
    boundary = Specimen_Bottom_Edge
  []  

[]

[Outputs]
  # file_base = hertz_voce_q10_b10
  [out]
    type = Exodus
    elemental_as_nodal = true
    time_step_interval = 1
    # file_base = hertz_linear_hardening_mesh_refine1_penalty_1e12_dt_0.1
  []
  csv = true
[]

