% RUN_01_MOTION_ESTIMATION_TESTS
% Validates independent OLS velocity estimation against deterministic vectors.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'motion_estimation_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.01] Running Motion Estimation MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;
tol = 1e-5;

for k = 1:num_cases
    vec = vectors(k);
    t = vec.timestamps(:)';
    x = vec.x(:)';
    y = vec.y(:)';
    
    [vx, vy, r2_x, r2_y, is_stat] = ref_least_squares_velocity(t, x, y, 0.08);
    
    err_vx = abs(vx - vec.expected_vx);
    err_vy = abs(vy - vec.expected_vy);
    stat_match = (is_stat == vec.is_stationary);
    
    if (err_vx <= tol) && (err_vy <= tol) && stat_match
        num_passed = num_passed + 1;
        fprintf('  [PASS] Case %d (%s): vx=%.4f (err=%.2e), vy=%.4f (err=%.2e), stat=%d\n', ...
            k, vec.case_id, vx, err_vx, vy, err_vy, is_stat);
    else
        fprintf('  [FAIL] Case %d (%s): err_vx=%.2e, err_vy=%.2e, stat_match=%d\n', ...
            k, vec.case_id, err_vx, err_vy, stat_match);
    end
end

fprintf('Motion Estimation Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all motion estimation test cases passed');
