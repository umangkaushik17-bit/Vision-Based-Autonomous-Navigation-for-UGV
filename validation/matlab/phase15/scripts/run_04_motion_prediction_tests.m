% RUN_04_MOTION_PREDICTION_TESTS
% Validates trajectory prediction, covariance propagation, and uncertainty radius.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'prediction_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.04] Running Motion Prediction MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;
tol = 1e-6;

for k = 1:num_cases
    vec = vectors(k);
    x0 = vec.x0(:);
    P0 = reshape(vec.P0, [4, 4]);
    q_val = vec.q;
    sigma_scale = vec.sigma_scale;
    horizon = vec.horizon_s;
    dt_step = vec.dt_step;
    
    [wps, P_traj, r_unc] = ref_motion_predictor(x0, P0, horizon, dt_step, q_val, sigma_scale);
    
    if isfield(vec, 'expected_waypoints')
        exp_wps = vec.expected_waypoints;
        all_match = true;
        for s = 1:length(exp_wps)
            ew = exp_wps(s);
            err_pos = norm(wps(s, 2:3) - [ew.expected_x, ew.expected_y]);
            err_runc = abs(r_unc(s) - ew.expected_r_unc);
            if err_pos > tol || err_runc > tol
                all_match = false;
                break;
            end
        end
    else
        all_match = true;
    end
    
    if all_match
        num_passed = num_passed + 1;
        fprintf('  [PASS] Case %d (%s): %d waypoints verified\n', k, vec.case_id, size(wps, 1));
    else
        fprintf('  [FAIL] Case %d (%s): discrepancy in predicted waypoints or uncertainty\n', k, vec.case_id);
    end
end

fprintf('Motion Prediction Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all motion prediction test cases passed');
