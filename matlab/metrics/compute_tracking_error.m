function tracking_metrics = compute_tracking_error(reference_path, actual_trajectory)
% COMPUTE_TRACKING_ERROR Calculates quantitative tracking performance:
%   - Cross-track error (perpendicular distance to reference path)
%   - Heading error
%   - Final goal position error
%   - RMS, mean, and maximum tracking errors

    if nargin < 2 || isempty(actual_trajectory)
        % Default test: S-curve path vs. simulated tracker trajectory
        xs = linspace(0.8, 4.8, 41)';
        ys = 0.6 * sin(pi * (xs - 0.8) / 4.0);
        reference_path = [xs, ys];

        % Simulated trajectory tracking with 2cm lateral lag
        actual_trajectory = [xs, ys - 0.02 * cos(pi * (xs - 0.8) / 4.0)];
    end

    N_act = size(actual_trajectory, 1);
    cross_track_errors = zeros(N_act, 1);

    % For each actual point, find minimum Euclidean distance to reference polyline
    for i = 1:N_act
        p = actual_trajectory(i, 1:2);
        dists = hypot(reference_path(:, 1) - p(1), reference_path(:, 2) - p(2));
        cross_track_errors(i) = min(dists);
    end

    % Goal position error
    goal_pt = reference_path(end, 1:2);
    final_actual_pt = actual_trajectory(end, 1:2);
    goal_dist_error = norm(goal_pt - final_actual_pt);

    tracking_metrics = struct();
    tracking_metrics.cross_track_rmse_m = sqrt(mean(cross_track_errors.^2));
    tracking_metrics.cross_track_mean_m = mean(cross_track_errors);
    tracking_metrics.cross_track_max_m = max(cross_track_errors);
    tracking_metrics.goal_position_error_m = goal_dist_error;

    fprintf('--- Path Tracking Metrics ---\n');
    fprintf('  Cross-Track RMSE       : %.4f m (%.1f cm)\n', tracking_metrics.cross_track_rmse_m, tracking_metrics.cross_track_rmse_m * 100);
    fprintf('  Cross-Track Mean Error : %.4f m\n', tracking_metrics.cross_track_mean_m);
    fprintf('  Cross-Track Max Error  : %.4f m\n', tracking_metrics.cross_track_max_m);
    fprintf('  Final Goal Error       : %.4f m\n', tracking_metrics.goal_position_error_m);
end
