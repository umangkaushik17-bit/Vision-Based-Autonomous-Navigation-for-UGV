function all_passed = run_matlab_test_suite()
% RUN_MATLAB_TEST_SUITE Lightweight standalone test runner executing all Phase 7 tests.
% Does not depend on optional Navigation or Computer Vision toolboxes.

    fprintf('==============================================================================\n');
    fprintf('           SIH26126 PHASE 7: MASTER MATLAB TEST SUITE\n');
    fprintf('==============================================================================\n\n');

    % Ensure root and subdirectories are on MATLAB search path
    root_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    addpath(genpath(root_dir));

    tests = {
        'test_coordinates',        @test_coordinates_func;
        'test_bev_geometry',       @test_bev_geometry_func;
        'test_terrain_geometry',   @test_terrain_geometry_func;
        'test_costmap_integrity',  @test_costmap_integrity_func;
        'test_ate_rpe',            @test_ate_rpe_func;
        'test_planner_metrics',    @test_planner_metrics_func
    };

    N_tests = size(tests, 1);
    test_results = false(N_tests, 1);
    test_times_ms = zeros(N_tests, 1);

    for i = 1:N_tests
        t_name = tests{i, 1};
        t_func = tests{i, 2};

        fprintf('[TEST %d/%d] Running %s ...\n', i, N_tests, t_name);
        t0 = tic;
        try
            ok = t_func();
            elapsed = toc(t0) * 1000.0;
            test_results(i) = ok;
            test_times_ms(i) = elapsed;
            if ok
                fprintf('  -> PASS (%.1f ms)\n\n', elapsed);
            else
                fprintf('  -> FAIL (%.1f ms)\n\n', elapsed);
            end
        catch ME
            elapsed = toc(t0) * 1000.0;
            test_results(i) = false;
            test_times_ms(i) = elapsed;
            fprintf('  -> ERROR: %s (%.1f ms)\n\n', ME.message, elapsed);
        end
    end

    fprintf('==============================================================================\n');
    fprintf('                          TEST SUMMARY RESULTS\n');
    fprintf('==============================================================================\n');
    passed_count = sum(test_results);
    fprintf('  Total Tests : %d\n', N_tests);
    fprintf('  Passed      : %d\n', passed_count);
    fprintf('  Failed      : %d\n', N_tests - passed_count);
    fprintf('  Success Rate: %.1f%%\n', (passed_count / N_tests) * 100.0);
    fprintf('==============================================================================\n\n');

    all_passed = (passed_count == N_tests);
end


%% --- Individual Test Functions ---

function ok = test_coordinates_func()
    res_conv = check_coordinate_conventions();
    res_ext = validate_extrinsics();
    ok = res_conv.all_passed && res_ext.is_valid;
end

function ok = test_bev_geometry_func()
    res_geom = validate_bev_geometry();
    res_res = validate_bev_resolution();
    res_err = compute_bev_transform_error();
    ok = res_geom.is_valid && res_res.is_consistent && res_err.is_valid;
end

function ok = test_terrain_geometry_func()
    res_plane = validate_plane_fitting();
    res_slope = validate_slope_estimation();
    res_step = validate_step_height();
    ok = res_plane.is_valid && res_slope.is_valid && res_step.is_valid;
end

function ok = test_costmap_integrity_func()
    res_cm = cross_validate_costmap();
    res_infl = validate_obstacle_inflation();
    res_temp = validate_temporal_fusion();
    res_mot = validate_motion_compensation();
    ok = res_cm.is_valid && res_infl.is_valid && res_temp.is_stable && res_mot.is_valid;
end

function ok = test_ate_rpe_func()
    res_slam = test_synthetic_slam_metrics();
    ok = res_slam.is_valid;
end

function ok = test_planner_metrics_func()
    res_paths = analyze_global_paths();
    res_detour = validate_terrain_detour();
    res_dyn = analyze_dynamic_obstacle();
    res_score = audit_dwa_scoring();
    res_cross = compare_python_matlab_metrics();
    ok = res_paths.is_valid && res_detour.is_valid && res_dyn.is_valid && res_score.is_valid && res_cross.is_valid;
end
