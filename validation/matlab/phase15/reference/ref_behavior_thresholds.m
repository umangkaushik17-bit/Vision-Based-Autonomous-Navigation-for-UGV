function [proposed_state, next_state, transition] = ref_behavior_thresholds(...
    planner_status_is_stop, emergency_stop, valid_trajectories, ...
    dyn_collision, unknown_horizon, unknown_policy, best_w, dyn_rejected, ...
    dyn_risk, clearing_obstacle, current_state, dt_state, min_dwell_time_s, ...
    wait_elapsed_s, wait_timeout_s, recovery_count, recovery_threshold)
% REF_BEHAVIOR_THRESHOLDS Independent reference model of behavioral state transition rules.
%
% Syntax:
%   [proposed_state, next_state, transition] = ref_behavior_thresholds(...
%       planner_status_is_stop, emergency_stop, valid_trajectories, ...
%       dyn_collision, unknown_horizon, unknown_policy, best_w, dyn_rejected, ...
%       dyn_risk, clearing_obstacle, current_state, dt_state, min_dwell_time_s, ...
%       wait_elapsed_s, wait_timeout_s, recovery_count, recovery_threshold)
%
% Outputs:
%   proposed_state - Raw candidate behavioral intention ('PROCEED', 'AVOID', 'WAIT', 'STOP')
%   next_state     - Arbitrated state after dwell, hysteresis, and timeouts
%   transition     - Transition action ('HOLD', 'HOLD_DWELL', 'HOLD_RECOVERY', or '<OLD> -> <NEW>')

    if nargin < 13 || isempty(min_dwell_time_s), min_dwell_time_s = 0.20; end
    if nargin < 15 || isempty(wait_timeout_s), wait_timeout_s = 3.0; end
    if nargin < 17 || isempty(recovery_threshold), recovery_threshold = 2; end

    % 1. Determine Proposed State
    if planner_status_is_stop || emergency_stop || (valid_trajectories == 0)
        if clearing_obstacle && (strcmp(current_state, 'WAIT') || strcmp(current_state, 'PROCEED') || strcmp(current_state, 'AVOID'))
            proposed_state = 'WAIT';
        else
            proposed_state = 'STOP';
        end
    elseif dyn_collision
        proposed_state = 'STOP';
    elseif unknown_horizon && strcmp(unknown_policy, 'REJECT')
        if clearing_obstacle
            proposed_state = 'WAIT';
        else
            proposed_state = 'STOP';
        end
    else
        is_deviating = (abs(best_w) > 0.06) || (dyn_rejected > 0) || ...
                       strcmp(dyn_risk, 'HIGH') || strcmp(dyn_risk, 'MEDIUM');
        if ~is_deviating && (strcmp(dyn_risk, 'FREE') || strcmp(dyn_risk, 'LOW')) && ~unknown_horizon
            proposed_state = 'PROCEED';
        else
            if clearing_obstacle && strcmp(dyn_risk, 'HIGH')
                proposed_state = 'WAIT';
            else
                proposed_state = 'AVOID';
            end
        end
    end

    % 2. Apply Arbitration, Dwell, Hysteresis, and Timeouts
    wait_timed_out = strcmp(current_state, 'WAIT') && (wait_elapsed_s >= wait_timeout_s);

    if strcmp(proposed_state, 'STOP')
        if ~strcmp(current_state, 'STOP')
            next_state = 'STOP';
            transition = [current_state, ' -> STOP'];
        else
            next_state = 'STOP';
            transition = 'HOLD';
        end
    elseif wait_timed_out
        next_state = 'STOP';
        transition = 'WAIT -> STOP';
    elseif (dt_state < min_dwell_time_s) && ~strcmp(proposed_state, current_state)
        next_state = current_state;
        transition = 'HOLD_DWELL';
    elseif (strcmp(current_state, 'STOP') || strcmp(current_state, 'WAIT')) && ...
           (strcmp(proposed_state, 'PROCEED') || strcmp(proposed_state, 'AVOID'))
        if (recovery_count + 1) >= recovery_threshold
            next_state = proposed_state;
            transition = [current_state, ' -> ', proposed_state];
        else
            next_state = current_state;
            transition = 'HOLD_RECOVERY';
        end
    elseif ~strcmp(proposed_state, current_state)
        next_state = proposed_state;
        transition = [current_state, ' -> ', proposed_state];
    else
        next_state = current_state;
        transition = 'HOLD';
    end
end
