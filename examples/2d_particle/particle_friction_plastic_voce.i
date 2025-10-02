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
        file = ./mesh_files/particle_refine_lvl_1.msh
    []
    uniform_refine = 0
    allow_renumbering = false
    coord_type = RZ
    rz_coord_axis = y
[]


[AuxVariables]
    [saved_x]
    []
    [saved_y]
    []
    [contact_active]
        family = LAGRANGE
        order = FIRST
    []
[]

[AuxKernels]
  [contact_active_aux]
    type = ParsedAux
    variable = contact_active
    boundary = Specimen_Top_Edge
    coupled_variables = 'contact_pressure'
    expression = 'if(contact_pressure-1e-5, 1, 0)'
  []
[]

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
        use_finite_deform_jacobian = true
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

    [disp_x_top]
        type = ADDirichletBC
        variable = disp_x
        boundary = Indenter_Top_Edge
        value = 0
    []

    [disp_y_load]
        type = ADFunctionDirichletBC
        variable = disp_y
        boundary = Indenter_Top_Edge
        function = disp_y
    []
[]

[Functions]
    [disp_y]
        type = PiecewiseLinear
        x = '0.  1.    2.'
        y = '0.  -0.5 0'
    []
[]

[Contact]
    [contact]
        secondary = Specimen_Top_Edge
        primary = Indenter_Bottom_Edge
        model = COULOMB
        friction_coefficient = 0.1
        formulation = tangential_penalty
        normalize_penalty = true
        penalty = 1e7
        capture_tolerance = 0.0001
    []
[]

[Dampers]
    [contact_slip]
        type = ContactSlipDamper
        primary = Indenter_Bottom_Edge
        secondary = Specimen_Top_Edge
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
        type = ADIsotropicVoceLawHardeningStressUpdate
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
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type  -pc_factor_mat_solver_type -pc_factor_shift_type -pc_factor_shift_amount'
    petsc_options_value = 'lu    superlu_dist                  NONZERO               1e-15'
    line_search = 'none'

    l_max_its = 60
    nl_max_its = 50
    dt = 0.01
    dtmin = 0.00001
    end_time = 1.0
    nl_rel_tol = 1e-8
    nl_abs_tol = 1e-6
    l_tol = 1e-3

    [TimeStepper]
        type = IterationAdaptiveDT
        optimal_iterations = 8
        iteration_window = 2
        dt = 0.005
        growth_factor = 1.2
        cutback_factor = 0.75
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
    [disp_abs]
        type = ParsedPostprocessor
        pp_names = 'disp'
        expression = 'abs(disp)'
    []
    [force]
        type = NodalSum
        variable = saved_y
        boundary = Specimen_Bottom_Edge
    []
    # 平均von Mises应力
    [spec_avg_vonmises]
        type = ElementAverageValue
        variable = vonmises_stress
        block = Specimen_Body
    []
    
    [contact_pressure_integral]
        type = SideIntegralVariablePostprocessor
        variable = contact_pressure
        boundary = Specimen_Top_Edge
        use_displaced_mesh = true
    []
    
    [contact_pressure_max]
        type = NodalExtremeValue
        variable = contact_pressure
        value_type = max
        boundary = Specimen_Top_Edge
    []

    [contact_active_area]
        type = SideIntegralVariablePostprocessor
        variable = contact_active
        boundary = Specimen_Top_Edge
        use_displaced_mesh = true
    []
    [contact_pressure_avg]
        type = ParsedPostprocessor
        pp_names = 'contact_pressure_integral contact_active_area'
        expression = 'contact_pressure_integral / contact_active_area'
    []
    [plot_force_disp]
        type = PlotPostprocessor
        x_variable = disp_abs
        y_variable = force
        plot_title = 'Force vs Displacement'
        output_file = 'force_disp.png'
        x_label = 'Displacement ($\\mu m$)'
        y_label = 'Force (N)'
        real_time_plot = true
        plot_frequency = 1
        style = 'b-'
    []
[]

[Outputs]
    # file_base = particle_friction_plastic_voce
    csv = true
    [out]
        type = Exodus
        elemental_as_nodal = true
        time_step_interval = 1
    []
[]
