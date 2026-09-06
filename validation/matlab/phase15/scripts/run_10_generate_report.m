% RUN_10_GENERATE_REPORT
% Executes all 9 test suites, aggregates numerical parity, and writes summary.json.

clear; clc;
scriptsDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptsDir, '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('============================================================\n');
fprintf('  PHASE 13.15 MATLAB INDEPENDENT VALIDATION AUDIT SUITE     \n');
fprintf('============================================================\n\n');

suite_results = struct();
t_start = tic;

try
    run(fullfile(scriptsDir, 'run_01_motion_estimation_tests.m'));
    suite_results.motion_estimation = 'PASS';
catch ME
    suite_results.motion_estimation = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_02_kalman_filter_tests.m'));
    suite_results.kalman_filter = 'PASS';
catch ME
    suite_results.kalman_filter = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_03_mahalanobis_tests.m'));
    suite_results.mahalanobis_gate = 'PASS';
catch ME
    suite_results.mahalanobis_gate = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_04_motion_prediction_tests.m'));
    suite_results.motion_predictor = 'PASS';
catch ME
    suite_results.motion_predictor = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_05_dynamic_occupancy_tests.m'));
    suite_results.dynamic_occupancy = 'PASS';
catch ME
    suite_results.dynamic_occupancy = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_06_sat_collision_tests.m'));
    suite_results.sat_collision = 'PASS';
catch ME
    suite_results.sat_collision = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_07_unicycle_rollout_tests.m'));
    suite_results.unicycle_rollout = 'PASS';
catch ME
    suite_results.unicycle_rollout = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_08_dwa_objective_tests.m'));
    suite_results.dwa_objective = 'PASS';
catch ME
    suite_results.dwa_objective = ['FAIL: ', ME.message];
end

try
    run(fullfile(scriptsDir, 'run_09_state_machine_tests.m'));
    suite_results.state_machine = 'PASS';
catch ME
    suite_results.state_machine = ['FAIL: ', ME.message];
end

total_time_s = toc(t_start);
suite_results.total_execution_time_s = total_time_s;
suite_results.overall_status = 'PASS';

fields = fieldnames(suite_results);
for i = 1:length(fields)
    fn = fields{i};
    val = suite_results.(fn);
    if ischar(val) && strncmp(val, 'FAIL', 4)
        suite_results.overall_status = 'FAIL';
    end
end

summary_file = fullfile(resultsDir, 'matlab_summary.json');
encoded = jsonencode(suite_results);
fid = fopen(summary_file, 'w');
if fid ~= -1
    fwrite(fid, encoded, 'char');
    fclose(fid);
end

fprintf('\n============================================================\n');
fprintf('  OVERALL AUDIT STATUS: %s (Elapsed: %.2fs)\n', suite_results.overall_status, total_time_s);
fprintf('  Summary written to: %s\n', summary_file);
fprintf('============================================================\n');
