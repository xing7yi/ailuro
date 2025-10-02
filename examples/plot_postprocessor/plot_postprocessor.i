
[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    [mesh]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 10
        ny = 10
        xmax = 1
        ymax = 1
    []
[]

[AuxVariables]
    [saved_x]
    []
    [saved_y]
    []
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    add_variables = true
    strain = SMALL
    save_in = 'saved_x saved_y'
  []
[]

[Materials]
    [elasticity_tensor]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 1.0e5
        poissons_ratio = 0.3
    []
    [stress]
        type = ComputeLinearElasticStress
    []
[]

[BCs]
    [fix_bottom]
        type = DirichletBC
        variable = disp_y
        boundary = 'bottom'
        value = 0
    []
    [fix_left]
        type = DirichletBC
        variable = disp_x
        boundary = 'left'
        value = 0
    []
    [compress_top]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'top'
        function = -0.01*t
    []
[]

[Executioner]
    type = Transient
    solve_type = NEWTON
    petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
    petsc_options_value = 'lu    superlu_dist'
    end_time = 1
    dt = 0.05
[]

[Postprocessors]
    [disp]
        type = PointValue
        variable = disp_y
        point = '0 1 0'
    []
    [force]
        type = NodalSum
        variable = saved_y
        boundary = 'bottom'
    []
    [plot_force_disp]
        type = PlotPostprocessor
        pp_names = 'force'
        x_variable = disp
        plot_title = 'Force vs Displacement'
        output_file = 'force_disp.png'
        x_label = 'Displacement ($\\mu m$)'
        y_label = 'Force (N)'
        real_time_plot = true
        plot_frequency = 2
    []
[]

[Outputs]
    csv = true
    exodus = true
[]