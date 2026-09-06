% RUN_02_KALMAN_FILTER_TESTS
% Validates independent 4-state Constant Velocity Kalman Filter against deterministic vectors.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'kalman_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.02] Running Kalman Filter MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;

for k = 1:num_cases
    vec = vectors(k);
    x_curr = vec.x0(:);
    P_curr = reshape(vec.P0, [4, 4]);
    q_val = vec.q;
    sigma_p = vec.sigma_p;
    
    measurements = vec.measurements;
    all_steps_valid = true;
    
    for s = 1:length(measurements)
        meas = measurements(s);
        dt = meas.dt;
        z = meas.z(:);
        
        [x_next, P_next, K, y_innov, S, d2, gated] = ref_kalman_filter(...
            x_curr, P_curr, z, dt, q_val, sigma_p, 9.21);
            
        % Assert symmetry and positive semi-definiteness
        symm_err = norm(P_next - P_next', 'fro');
        e_vals = eig(P_next);
        if symm_err > 1e-10 || any(e_vals < -1e-8)
            all_steps_valid = false;
            break;
        end
        x_curr = x_next;
        P_curr = P_next;
    end
    
    if all_steps_valid
        num_passed = num_passed + 1;
        fprintf('  [PASS] Case %d (%s): final state=[%.3f, %.3f, %.3f, %.3f], PSD preserved\n', ...
            k, vec.case_id, x_curr(1), x_curr(2), x_curr(3), x_curr(4));
    else
        fprintf('  [FAIL] Case %d (%s): symmetry/PSD failure\n', k, vec.case_id);
    end
end

fprintf('Kalman Filter Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all Kalman filter test cases passed');
