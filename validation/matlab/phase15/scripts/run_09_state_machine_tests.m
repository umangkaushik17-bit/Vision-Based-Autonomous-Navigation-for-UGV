% RUN_09_STATE_MACHINE_TESTS
% Validates behavioral state transitions, dwell time, timeouts, and replanner trigger hierarchy.

clear; clc;
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'reference'));
behPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'behavior_vectors.json');
repPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'replanner_vectors.json');
resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

fprintf('=== [Phase 15.09] Running State Machine & Replanner MATLAB Tests ===\n');

% 1. Behavioral State Machine
raw_beh = fileread(behPath);
data_beh = jsondecode(raw_beh);
beh_vectors = data_beh.vectors;

beh_passed = 0;
for k = 1:length(beh_vectors)
    bv = beh_vectors(k);
    [prop, next_st, trans] = ref_behavior_thresholds(...
        bv.planner_status_is_stop, bv.emergency_stop, bv.valid_trajectories, ...
        bv.dyn_collision, bv.unknown_horizon, bv.unknown_policy, bv.best_w, ...
        bv.dyn_rejected, bv.dyn_risk, bv.clearing_obstacle, bv.current_state, ...
        bv.dt_state, bv.min_dwell_time_s, bv.wait_elapsed_s, bv.wait_timeout_s, ...
        bv.recovery_count, bv.recovery_threshold);
        
    prop_match = strcmp(prop, bv.expected_proposed);
    next_match = strcmp(next_st, bv.expected_next);
    trans_match = strcmp(trans, bv.expected_transition);
    
    if prop_match && next_match && trans_match
        beh_passed = beh_passed + 1;
        fprintf('  [PASS] Beh Case %d (%s): prop=%s, next=%s, trans=%s\n', ...
            k, bv.case_id, prop, next_st, trans);
    else
        fprintf('  [FAIL] Beh Case %d (%s): prop_m=%d, next_m=%d, trans_m=%d\n', ...
            k, bv.case_id, prop_match, next_match, trans_match);
    end
end
assert(beh_passed == length(beh_vectors), 'Not all behavioral state tests passed');

% 2. Replanner Trigger Hierarchy
raw_rep = fileread(repPath);
data_rep = jsondecode(raw_rep);
rep_vectors = data_rep.vectors;

rep_passed = 0;
for k = 1:length(rep_vectors)
    rv = rep_vectors(k);
    [replan_trig, trig_type] = ref_replanner_triggers(...
        rv.is_goal_reached, rv.behavior_state, rv.is_deadlock, rv.blocked_duration_s, ...
        rv.persistent_threshold_s, rv.is_path_valid, rv.path_invalid_cycles, ...
        rv.debounce_threshold, rv.dt_window_s, rv.window_threshold_s, ...
        rv.net_progress_m, rv.min_progress_m);
        
    replan_match = (replan_trig == rv.expected_replan);
    trig_match = strcmp(trig_type, rv.expected_trigger);
    
    if replan_match && trig_match
        rep_passed = rep_passed + 1;
        fprintf('  [PASS] Replan Case %d (%s): trig=%d (%s)\n', ...
            k, rv.case_id, replan_trig, trig_type);
    else
        fprintf('  [FAIL] Replan Case %d (%s): replan_match=%d, trig_match=%d\n', ...
            k, rv.case_id, replan_match, trig_match);
    end
end
assert(rep_passed == length(rep_vectors), 'Not all replanner trigger tests passed');

fprintf('State Machine & Replanner Result: PASS (Beh: %d/%d, Replan: %d/%d)\n\n', ...
    beh_passed, length(beh_vectors), rep_passed, length(rep_vectors));
