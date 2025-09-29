[GlobalParams]
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
        file = ../hertz_contact_structural_ref_1.msh
    []
    uniform_refine = 0
    allow_renumbering = false
    coord_type = RZ
    rz_coord_axis = y
    patch_update_strategy = ITERATION
[]


[AuxVariables]
    [saved_x]
    []
    [saved_y]
    []
[] # AuxVariables

[Physics/SolidMechanics/QuasiStatic]
    [all]
        block = 'Specimen_Body Indenter_Body'
        add_variables = true
        strain = FINITE
        extra_vector_tags = 'ref'
        generate_output = '
          strain_xx strain_xy strain_xz
          strain_yx strain_yy strain_yz
          strain_zx strain_zy strain_zz
          stress_xx stress_xy stress_xz
          stress_yx stress_yy stress_yz
          stress_zx stress_zy stress_zz
          vonmises_stress'
        save_in = 'saved_x saved_y'
        use_automatic_differentiation = true
    []
[]

[BCs]
    [left_x]
        type = ADDirichletBC
        variable = disp_x
        boundary = 'Specimen_Left_Edge Indenter_Left_Edge'
        value = 0
    []


    [btm_spec_y]
        type = ADDirichletBC
        variable = disp_y
        boundary = Specimen_Bottom_Edge
        value = 0
    []

    # [top_ind_pressure]
    #     type = Pressure
    #     variable = disp_y
    #     boundary = Indenter_Top_Edge
    #     function = pressure
    # []

    [disp_y_load]
        type = ADFunctionDirichletBC
        variable = disp_y
        boundary = Indenter_Top_Edge
        function = disp_y
    []
[]

[Functions]
    [pressure]
        type = PiecewiseLinear
        x = '0.  1.  2.'
        y = '0.  1   1'
        scale_factor = 138.7045  # 980.4443N / (π*2^2)
    []
    [disp_y]
        type = PiecewiseLinear
        x = '0.  1.    2.'
        y = '0.  -0.5 -0.5'
    []
[]


[Contact]
    [contact_penalty_friction]
        secondary = Specimen_Top_Edge
        primary = Indenter_Bottom_Edge
        model = COULOMB
        friction_coefficient = 0.05   # 更低的摩擦系数
        formulation = PENALTY         # 改用PENALTY方法，更稳定
        penalty = 1e6                 # 适中的penalty参数
        normalize_penalty = true
        # tangential_tolerance = 1e-3   # 切向容差
    []
[]

[Materials]
    # Indenter
    [elasticity_tensor_indenter]
        type = ADComputeIsotropicElasticityTensor
        block = 'Indenter_Body'
        youngs_modulus = 1e8
        poissons_ratio = 0.2
    []
    [stress_indenter]
        type = ADComputeFiniteStrainElasticStress
        block = 'Indenter_Body'
    []

    # Specimen
    [elasticity_tensor_specimen]
        type = ADComputeIsotropicElasticityTensor
        block = 'Specimen_Body'
        youngs_modulus = 1e5
        poissons_ratio = 0.25
    []
    [stress_specimen]
        # type = ComputeFiniteStrainElasticStress
        type = ADIsotropicLinearHardeningStressUpdate
        block = 'Specimen_Body'
        yield_stress = 200
        hardening_constant = 100
    []
    [return_stress]
        type = ADComputeMultipleInelasticStress
        block = 'Specimen_Body'
        inelastic_models = 'stress_specimen'
    []
[] # Materials


[Preconditioning]
  [SMP]
    type = SMP
    full = true
  []
[] # Preconditioning

[Executioner]
    type = Transient
    solve_type = 'NEWTON'              # 使用NEWTON求解器，对摩擦接触更稳定
    petsc_options = '-snes_ksp_ew'

    # 针对摩擦接触优化的求解器选项
    # petsc_options_iname = '-pc_type -pc_factor_mat_solver_type -snes_linesearch_type -snes_linesearch_damping'
    # petsc_options_value = 'lu       mumps                    bt                     0.8'
    petsc_options_iname = '-pc_type -snes_linesearch_type -pc_factor_shift_type -pc_factor_shift_amount'
    petsc_options_value = 'lu       basic                 NONZERO               1e-15'
    line_search = 'none'               # 使用backtracking line search
    # automatic_scaling = true         # 启用自动缩放
    # compute_scaling_once = false     # 每步重新计算缩放
    
    nl_abs_tol = 2.0e-07            # 放松非线性绝对容差
    nl_rel_tol = 2.0e-07            # 放松非线性相对容差
    l_abs_tol = 1e-08               # 线性绝对容差
    l_tol = 1e-08                   # 线性容差
    nl_max_its = 100                # 增加非线性最大迭代次数
    l_max_its = 50                  # 增加线性最大迭代次数

    start_time = 0.0
    end_time = 1.0
    dtmin = 1e-10                   # 减小最小时间步长

    [TimeStepper]
        type = IterationAdaptiveDT
        optimal_iterations = 8        # 减少目标迭代次数
        iteration_window = 2
        dt = 1e-5                     # 更小的初始步长
        growth_factor = 1.2          # 更保守的增长因子
        cutback_factor = 0.75          # 更激进的回退因子
        cutback_factor_at_failure = 0.5
    []

    [Predictor]
        type = SimplePredictor
        scale = 1.0
    []    
[]

[Postprocessors]
    [disp]
        type = PointValue
        variable = disp_y
        point = '0 1 0'
    []
    # [disp]
    #     type = NodalVariableValue
    #     variable = disp_y
    #     nodeid = 157
    # []
    [spec_nodalsum]
        type = NodalSum
        variable = saved_y
        boundary = Specimen_Bottom_Edge
    []
    [spec_reaction]
        type = ADSidesetReaction
        boundary = Specimen_Bottom_Edge
        direction = '0 1 0'
        stress_tensor = stress
        use_displaced_mesh = true
    []
    [force_ratio]
        type = ParsedPostprocessor
        pp_names = 'spec_nodalsum spec_reaction'
        expression = 'spec_reaction / (spec_nodalsum)'
    []
[]

[Outputs]
    file_base = hertz2d_ad
    csv = true
    [out]
        type = Exodus
        elemental_as_nodal = true
        time_step_interval = 1
    []
[]
