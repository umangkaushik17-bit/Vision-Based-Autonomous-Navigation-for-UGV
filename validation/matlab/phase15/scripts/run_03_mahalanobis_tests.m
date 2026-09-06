% RUN_03_MAHALANOBIS_TESTS
% Validates Mahalanobis distance squared and chi-square gating against analytical references.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'mahalanobis_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.03] Running Mahalanobis Gating MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;
tol = 1e-6;

for k = 1:num_cases
    vec = vectors(k);
    y = vec.y(:);
    S = vec.S;
    thresh = vec.chi2_threshold;
    
    [d2, is_gated] = ref_mahalanobis_gate(y, S, thresh);
    
    err_d2 = abs(d2 - vec.expected_d2);
    gate_match = (is_gated == vec.expected_gated);
    
    if (err_d2 <= tol) && gate_match
        num_passed = num_passed + 1;
        fprintf('  [PASS] Case %d (%s): d2=%.4f (err=%.2e), gated=%d\n', ...
            k, vec.case_id, d2, err_d2, is_gated);
    else
        fprintf('  [FAIL] Case %d (%s): err_d2=%.2e, gate_match=%d\n', ...
            k, vec.case_id, err_d2, gate_match);
    end
end

fprintf('Mahalanobis Gating Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all Mahalanobis gating test cases passed');
