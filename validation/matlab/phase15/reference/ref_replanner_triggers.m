function [replan_triggered, trigger_type] = ref_replanner_triggers(...
    is_goal_reached, behavior_state, is_deadlock, blocked_duration_s, ...
    persistent_threshold_s, is_path_valid, path_invalid_cycles, ...
    debounce_threshold, dt_window_s, window_threshold_s, net_progress_m, min_progress_m)
% REF_REPLANNER_TRIGGERS Independent reference model of global replanner trigger hierarchy.
%
% Syntax:
%   [replan_triggered, trigger_type] = ref_replanner_triggers(...
%       is_goal_reached, behavior_state, is_deadlock, blocked_duration_s, ...
%       persistent_threshold_s, is_path_valid, path_invalid_cycles, ...
%       debounce_threshold, dt_window_s, window_threshold_s, net_progress_m, min_progress_m)
%
% Outputs:
%   replan_triggered - Boolean flag indicating if global replanning is requested
%   trigger_type     - Deterministic trigger classification string

    if nargin < 5 || isempty(persistent_threshold_s), persistent_threshold_s = 3.0; end
    if nargin < 8 || isempty(debounce_threshold), debounce_threshold = 2; end
    if nargin < 10 || isempty(window_threshold_s), window_threshold_s = 4.0; end
    if nargin < 12 || isempty(min_progress_m), min_progress_m = 0.20; end

    % Guard 1: Goal reached
    if is_goal_reached
        replan_triggered = false;
        trigger_type = 'NONE';
        return;
    end

    % Guard 2: Temporary active motion with valid path
    if (strcmp(behavior_state, 'PROCEED') || strcmp(behavior_state, 'AVOID')) && is_path_valid
        replan_triggered = false;
        trigger_type = 'NONE';
        return;
    end

    % Trigger A: Dynamic Deadlock event
    if is_deadlock
        replan_triggered = true;
        trigger_type = 'DYNAMIC_DEADLOCK';
        return;
    end

    % Trigger B: Persistent blockage in STOP / WAIT
    if (strcmp(behavior_state, 'STOP') || strcmp(behavior_state, 'WAIT')) && ...
       (blocked_duration_s >= persistent_threshold_s)
        replan_triggered = true;
        trigger_type = 'PERSISTENT_BLOCKAGE';
        return;
    end

    % Trigger D: Path invalidated (debounced)
    if ~is_path_valid
        if (path_invalid_cycles + 1) >= debounce_threshold
            replan_triggered = true;
            trigger_type = 'PATH_INVALIDATED';
            return;
        end
    end

    % Trigger C: Insufficient progress over window
    if (dt_window_s >= window_threshold_s) && (net_progress_m < min_progress_m)
        replan_triggered = true;
        trigger_type = 'INSUFFICIENT_PROGRESS';
        return;
    end

    replan_triggered = false;
    trigger_type = 'NONE';
end
