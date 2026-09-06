% RUN_05_DYNAMIC_OCCUPANCY_TESTS
% Validates dynamic occupancy cost grading and risk level classification.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'occupancy_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.05] Running Dynamic Occupancy MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);

obs_pos = data.obs_pos(:)';
obs_size = data.obs_size(:)';
obs_yaw = data.obs_yaw;
r_unc = data.r_unc;
sigma_scale = data.sigma_scale;
pts = data.query_points;

num_pts = length(pts);
num_passed = 0;
tol = 1e-4;

for k = 1:num_pts
    pt = pts(k);
    [cost, risk, is_nominal, is_uncertainty] = ref_dynamic_occupancy(...
        pt.x, pt.y, obs_pos, obs_size, obs_yaw, r_unc, sigma_scale);
        
    err_cost = abs(cost - pt.expected_cost);
    risk_match = (risk == pt.expected_risk);
    nom_match = (is_nominal == pt.is_nominal);
    
    if (err_cost <= tol) && risk_match && nom_match
        num_passed = num_passed + 1;
        fprintf('  [PASS] Point %s (%.2f, %.2f): cost=%.4f (err=%.2e), risk=%d\n', ...
            pt.id, pt.x, pt.y, cost, err_cost, risk);
    else
        fprintf('  [FAIL] Point %s: err_cost=%.2e, risk_match=%d, nom_match=%d\n', ...
            pt.id, err_cost, risk_match, nom_match);
    end
end

fprintf('Dynamic Occupancy Result: %d / %d Passed\n\n', num_passed, num_pts);
assert(num_passed == num_pts, 'Not all dynamic occupancy test points passed');
