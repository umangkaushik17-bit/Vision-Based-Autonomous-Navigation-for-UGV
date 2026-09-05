function rpe_metrics = compute_rpe(traj_gt, traj_est, delta_step)
% COMPUTE_RPE Calculates Relative Pose Error (RPE) over a fixed interval delta_step.
% Inputs:
%   traj_gt:     (N, 2) or (N, 3) reference positions
%   traj_est:    (N, 2) or (N, 3) estimated positions
%   delta_step:  index interval for delta evaluation (default: 10 steps = 1.0s at 10Hz)
% Outputs:
%   rpe_metrics: struct containing trans_rmse, trans_mean, trans_std, num_pairs

    if nargin < 3
        delta_step = 10;
    end

    N = size(traj_gt, 1);
    if N <= delta_step
        error('Trajectory length must be strictly greater than delta_step');
    end

    num_pairs = N - delta_step;
    rpe_errors = zeros(num_pairs, 1);

    for i = 1:num_pairs
        j = i + delta_step;
        delta_gt  = traj_gt(j, :) - traj_gt(i, :);
        delta_est = traj_est(j, :) - traj_est(i, :);
        rpe_errors(i) = norm(delta_gt - delta_est);
    end

    rpe_metrics = struct();
    rpe_metrics.delta_step = delta_step;
    rpe_metrics.rmse = sqrt(mean(rpe_errors.^2));
    rpe_metrics.mean = mean(rpe_errors);
    rpe_metrics.median = median(rpe_errors);
    rpe_metrics.std = std(rpe_errors);
    rpe_metrics.max = max(rpe_errors);
    rpe_metrics.num_pairs = num_pairs;
    rpe_metrics.errors = rpe_errors;
end
