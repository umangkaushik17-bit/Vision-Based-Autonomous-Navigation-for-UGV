% RUN_06_SAT_COLLISION_TESTS
% Validates Separating Axis Theorem (SAT) oriented bounding box collision checking.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'collision_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.06] Running SAT Collision MATLAB Tests ===\n');

raw = fileread(dataPath);
data = jsondecode(raw);
vectors = data.vectors;

num_cases = length(vectors);
num_passed = 0;

for k = 1:num_cases
    vec = vectors(k);
    r_pos = vec.robot_pos(:)';
    r_size = vec.robot_size(:)';
    r_yaw = vec.robot_yaw;
    o_pos = vec.obs_pos(:)';
    o_size = vec.obs_size(:)';
    o_yaw = vec.obs_yaw;
    margin = vec.margin;
    
    is_col = ref_sat_collision(r_pos, r_size, r_yaw, o_pos, o_size, o_yaw, margin);
    
    if is_col == vec.expected_collision
        num_passed = num_passed + 1;
        fprintf('  [PASS] Case %d (%s): collision=%d (expected=%d)\n', ...
            k, vec.case_id, is_col, vec.expected_collision);
    else
        fprintf('  [FAIL] Case %d (%s): collision=%d (expected=%d)\n', ...
            k, vec.case_id, is_col, vec.expected_collision);
    end
end

fprintf('SAT Collision Result: %d / %d Passed\n\n', num_passed, num_cases);
assert(num_passed == num_cases, 'Not all SAT collision test cases passed');
