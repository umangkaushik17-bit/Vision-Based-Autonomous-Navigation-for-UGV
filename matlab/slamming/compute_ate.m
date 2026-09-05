function ate_metrics = compute_ate(traj_gt, traj_est, do_align)
% COMPUTE_ATE Calculates Absolute Trajectory Error (ATE) between reference and estimated trajectories.
% Inputs:
%   traj_gt:  (N, 2) or (N, 3) reference positions [x, y] or [x, y, z]
%   traj_est: (N, 2) or (N, 3) estimated positions
%   do_align: (optional boolean) whether to apply rigid alignment first (default: true)
% Outputs:
%   ate_metrics: struct containing rmse, mean, median, std, max, errors array

    if nargin < 3
        do_align = true;
    end

    if do_align
        traj_eval = align_trajectories(traj_gt, traj_est);
    else
        traj_eval = traj_est;
    end

    diffs = traj_gt - traj_eval;
    errors = sqrt(sum(diffs.^2, 2));

    ate_metrics = struct();
    ate_metrics.rmse = sqrt(mean(errors.^2));
    ate_metrics.mean = mean(errors);
    ate_metrics.median = median(errors);
    ate_metrics.std = std(errors);
    ate_metrics.max = max(errors);
    ate_metrics.num_points = length(errors);
    ate_metrics.point_errors = errors;
end
