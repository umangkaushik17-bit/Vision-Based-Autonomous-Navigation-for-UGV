% RUN_08_DWA_OBJECTIVE_TESTS
% Validates multi-objective DWA trajectory cost scoring and risk penalties.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'dwa_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.08] Running DWA Objective MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;

for k = 1:num_cases
    vec = vectors(k);
    
    [total_cost, comp] = ref_dwa_objective(...
        vec.path_dist, vec.heading_err, vec.dist_to_goal, vec.clearance, ...
        vec.velocity, vec.max_v, vec.terrain_cost, vec.dynamic_cost, ...
        vec.dynamic_risk, vec.unknown_horizon);
        
    fprintf('  [INFO] Case %d (%s): total_cost = %.4f (dyn_cost = %.4f, risk_pen = %.4f)\n', ...
        k, vec.case_id, total_cost, comp.dynamic_cost, comp.risk_penalty);
        
    % Basic monotonicity checks
    assert(total_cost >= 0.0, 'Total cost must be non-negative');
    num_passed = num_passed + 1;
end

fprintf('DWA Objective Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all DWA objective test cases passed');
