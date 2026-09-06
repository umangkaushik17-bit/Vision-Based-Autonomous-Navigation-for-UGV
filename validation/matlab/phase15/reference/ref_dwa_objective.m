function [total_score, components] = ref_dwa_objective(v, omega, end_x, end_y, end_yaw, local_target, min_cte, clearance_m, terrain_cost, dyn_penalty, dyn_clearance, weights, params)
% REF_DWA_OBJECTIVE Independent DWA objective score evaluator in MATLAB.
%
% Inputs:
%   v, omega       - Candidate velocities
%   end_x, end_y   - Endpoint coordinates of rollout trajectory
%   end_yaw        - Endpoint heading of rollout trajectory
%   local_target   - [tx, ty] target coordinate
%   min_cte        - Cross-track error to global path
%   clearance_m    - Static minimum obstacle clearance
%   terrain_cost   - Accumulated/average terrain traversal cost
%   dyn_penalty    - Penalty for dynamic risk zone penetration
%   dyn_clearance  - Normalized dynamic clearance [0, 1]
%   weights        - [w_path, w_heading, w_dist, w_clear, w_vel, w_terrain, w_dyn_clear]
%   params         - [max_v, pred_horizon, d_curr]
%
% Outputs:
%   total_score    - Composite objective score
%   components     - Structure containing individual sub-scores

    if nargin < 12 || isempty(weights)
        % Default weights from Phase 12 / Phase 13
        weights = [1.2, 1.0, 1.5, 0.8, 0.6, 0.5, 1.0];
    end
    if nargin < 13 || isempty(params)
        params = [0.8, 1.5, 2.0]; % [max_v, pred_horizon, d_curr]
    end

    max_v = params(1);
    pred_horizon = params(2);
    d_curr = params(3);

    w_path      = weights(1);
    w_heading   = weights(2);
    w_dist      = weights(3);
    w_clear     = weights(4);
    w_vel       = weights(5);
    w_terrain   = weights(6);
    w_dyn_clear = weights(7);

    % 1. Path alignment score
    path_score = 1.0 / (1.0 + max(0.0, min_cte));

    % 2. Heading score to local target
    tx = local_target(1); ty = local_target(2);
    target_yaw = atan2(ty - end_y, tx - end_x);
    heading_err = abs(atan2(sin(target_yaw - end_yaw), cos(target_yaw - end_yaw)));
    heading_score = (pi - heading_err) / pi;

    % 3. Distance and progress score
    d_end = hypot(tx - end_x, ty - end_y);
    progress_score = (d_curr - d_end) / max(0.1, max_v * pred_horizon);
    dist_score = 0.5 * (1.0 / (1.0 + d_end)) + 0.5 * max(0.0, progress_score);

    % 4. Clearance score
    clearance_score = min(max(0.0, clearance_m), 1.50) / 1.50;

    % 5. Velocity score
    vel_factor = max(0.0, cos(heading_err));
    if v >= 0
        vel_score = (v / max(0.1, max_v)) * vel_factor;
    else
        vel_score = -0.5 * (abs(v) / max(0.1, max_v));
    end

    % 6. Terrain score
    terrain_score = 1.0 / (1.0 + max(0.0, terrain_cost));

    % Composite score
    total_score = w_path * path_score + ...
                  w_heading * heading_score + ...
                  w_dist * dist_score + ...
                  w_clear * clearance_score + ...
                  w_vel * vel_score + ...
                  w_terrain * terrain_score - ...
                  dyn_penalty + ...
                  w_dyn_clear * dyn_clearance;

    components = struct(...
        'path', path_score, ...
        'heading', heading_score, ...
        'dist', dist_score, ...
        'clearance', clearance_score, ...
        'vel', vel_score, ...
        'terrain', terrain_score, ...
        'dyn_penalty', dyn_penalty, ...
        'dyn_clearance', dyn_clearance, ...
        'total', total_score);
end
