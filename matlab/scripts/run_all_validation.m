function master_report = run_all_validation()
% RUN_ALL_VALIDATION Master validation orchestrator for SIH26126 Phase 7.
% Executes Experiments A through I, collects metrics, and exports validation reports.

    fprintf('==============================================================================\n');
    fprintf('           SIH26126 PHASE 7: MASTER VALIDATION ORCHESTRATOR\n');
    fprintf('==============================================================================\n\n');

    % Setup paths
    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..');
    addpath(genpath(root_dir));

    % 1. Environment & Artifact Audit
    env = check_matlab_environment();
    artifact_report = audit_project_artifacts();

    % 2. Run Experiments A through I
    fprintf('\n================== RUNNING EXPERIMENTS A THROUGH I ==================\n\n');

    % Experiment A: Coordinate convention verification
    fprintf('[EXPERIMENT A] Coordinate Convention Verification\n');
    res_A = check_coordinate_conventions();

    % Experiment B: BEV synthetic geometric validation
    fprintf('\n[EXPERIMENT B] BEV Synthetic Geometric Validation\n');
    res_B1 = validate_bev_geometry();
    res_B2 = validate_bev_resolution();
    res_B3 = compute_bev_transform_error();

    % Experiment C: Plane-fitting and slope validation
    fprintf('\n[EXPERIMENT C] Plane-Fitting and Slope Validation\n');
    res_C1 = validate_plane_fitting();
    res_C2 = validate_slope_estimation();
    res_C3 = validate_step_height();

    % Experiment D: Costmap consistency analysis
    fprintf('\n[EXPERIMENT D] Costmap Consistency Analysis\n');
    res_D1 = cross_validate_costmap();
    res_D2 = validate_obstacle_inflation();

    % Experiment E: Temporal fusion analysis
    fprintf('\n[EXPERIMENT E] Temporal Fusion Analysis\n');
    res_E1 = validate_temporal_fusion();
    res_E2 = validate_motion_compensation();

    % Experiment F: Terrain detour path analysis
    fprintf('\n[EXPERIMENT F] Terrain Detour Path Analysis (Section 61)\n');
    res_F = validate_terrain_detour();

    % Experiment G: Dynamic obstacle trajectory analysis
    fprintf('\n[EXPERIMENT G] Dynamic Obstacle Trajectory Analysis (Section 60)\n');
    res_G = analyze_dynamic_obstacle();

    % Experiment H: Python vs. MATLAB metric consistency
    fprintf('\n[EXPERIMENT H] Python vs. MATLAB Metric Consistency\n');
    res_H = compare_python_matlab_metrics();

    % Experiment I: ATE/RPE synthetic framework verification
    fprintf('\n[EXPERIMENT I] ATE / RPE Synthetic Framework Verification\n');
    res_I = test_synthetic_slam_metrics();

    % 3. Aggregate Master Report
    master_report = struct();
    master_report.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    master_report.environment = env;
    master_report.experiment_A_coordinates = res_A;
    master_report.experiment_B_bev = struct('geometry', res_B1, 'resolution', res_B2, 'transform_error', res_B3);
    master_report.experiment_C_terrain = struct('plane_fit', res_C1, 'slope', res_C2, 'step_height', res_C3);
    master_report.experiment_D_costmap = struct('integrity', res_D1, 'inflation', res_D2);
    master_report.experiment_E_temporal = struct('flicker_reduction', res_E1, 'motion_compensation', res_E2);
    master_report.experiment_F_detour = res_F;
    master_report.experiment_G_dynamic_obstacle = res_G;
    master_report.experiment_H_consistency = res_H;
    master_report.experiment_I_slam = res_I;

    % Save results
    report_dir = fullfile(root_dir, 'results', 'reports');
    if ~exist(report_dir, 'dir')
        mkdir(report_dir);
    end
    save(fullfile(report_dir, 'master_validation_report.mat'), 'master_report');

    fprintf('\n==============================================================================\n');
    fprintf('  ALL 9 EXPERIMENTS COMPLETED SUCCESSFULLY.\n');
    fprintf('  Master Validation Report saved to: matlab/results/reports/master_validation_report.mat\n');
    fprintf('==============================================================================\n\n');
end
