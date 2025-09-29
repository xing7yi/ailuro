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
    uniform_refine = 1
    allow_renumbering = false
    coord_type = RZ
    rz_coord_axis = y
[]


[AuxVariables]
    [saved_x]
    []
    [saved_y]
    []
[] # AuxVariables

[Physics/SolidMechanics/QuasiStatic]
    [all]
        add_variables = true
        strain = FINITE
        extra_vector_tags = 'ref'
        save_in = 'saved_x saved_y'
        generate_output = '
          strain_xx strain_xy strain_xz
          strain_yx strain_yy strain_yz
          strain_zx strain_zy strain_zz
          stress_xx stress_xy stress_xz
          stress_yx stress_yy stress_yz
          stress_zx stress_zy stress_zz
          vonmises_stress'
    []
[]

[BCs]
    [left_spec_x]
        type = DirichletBC
        variable = disp_x
        boundary = Specimen_Left_Edge
        value = 0
    []

    [left_ind_x]
        type = DirichletBC
        variable = disp_x
        boundary = Indenter_Left_Edge
        value = 0
    []

    [btm_spec_y]
        type = DirichletBC
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
        type = FunctionDirichletBC
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
    # [contact_coulomb]
    #     secondary = Specimen_Top_Edge
    #     primary = Indenter_Bottom_Edge
    #     model = COULOMB
    #     friction_coefficient = 0.2
    #     formulation = PENALTY
    #     penalty = 4e6
    #     normal_smoothing_distance = 0.1
    # []    
    [contact_penalty]
        secondary = Specimen_Top_Edge
        primary = Indenter_Bottom_Edge
        model = frictionless
        formulation = PENALTY
        normalize_penalty = true
        penalty = 1e10
        # tangential_tolerance = 1e-3
    []
    # [contact_mortar]
    #     secondary = Specimen_Top_Edge
    #     primary = Indenter_Bottom_Edge
    #     model = frictionless
    #     formulation = MORTAR
    #     c_normal = 1e4
    #     normalize_penalty = true
    # []
[]

[Materials]
    # Indenter
    [elasticity_tensor_indenter]
        type = ComputeIsotropicElasticityTensor
        block = 'Indenter_Body'
        youngs_modulus = 1e8
        poissons_ratio = 0.25
    []
    [stress_indenter]
        type = ComputeFiniteStrainElasticStress
        block = 'Indenter_Body'
    []

    # Specimen
    [elasticity_tensor_specimen]
        type = ComputeIsotropicElasticityTensor
        block = 'Specimen_Body'
        youngs_modulus = 1e5
        poissons_ratio = 0.25
    []
    [stress_specimen]
        # type = ComputeFiniteStrainElasticStress
        type = IsotropicLinearHardeningStressUpdate
        block = 'Specimen_Body'
        yield_stress = 200
        hardening_constant = 100
    []
    [return_stress]
        type = ComputeMultipleInelasticStress
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
    solve_type = 'PJFNK'
    petsc_options = '-snes_ksp_ew'

    petsc_options_iname = '-pc_type -snes_linesearch_type -pc_factor_shift_type -pc_factor_shift_amount'
    petsc_options_value = 'lu       basic                 NONZERO               1e-15'
    # petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
    # petsc_options_value = 'lu       superlu_dist'
    line_search = 'none'
    # automatic_scaling = true
    # compute_scaling_once = false
    nl_abs_tol = 1.0e-07  # non-linear absolute tolerance
    nl_rel_tol = 1.0e-07  # non-linear relative tolerance
    l_max_its = 100        # linear maximum iterations
    l_abs_tol = 1e-08     # linear absolute tolerance
    l_tol = 1e-08         # linear tolerance

    start_time = 0.0
    end_time = 1.0
    dt = 0.01
    dtmin = 1e-8

    # [TimeStepper]
    #     type = IterationAdaptiveDT
    #     optimal_iterations = 10          # 减少目标迭代次数，提高效率
    #     iteration_window = 2
    #     dt = 0.01                        # 初始步长
    #     growth_factor = 1.2
    #     cutback_factor = 0.75
    #     cutback_factor_at_failure = 0.5
    # []
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
        type = SidesetReaction
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
    file_base = hertz2d_pressure
    csv = true
    [out]
        type = Exodus
        elemental_as_nodal = true
        time_step_interval = 1
    []
[]
